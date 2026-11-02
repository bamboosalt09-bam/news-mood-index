# 세션이 켜져 있는 동안 도는 주기 수집 (2026-10-31). 운영체제 예약 없이 백그라운드로 실행 → 세션이 끝나면 같이 멈춤.
# - 2분마다: S&P 500·나스닥 실시간 값 (장중일 때만) → data/us_realtime.tsv (미국 현지 체결 시각, 지수, 값)
# - 10분마다: 한국어·영어 지수용 제목 어제·오늘 치 (collect.py daily, 새 제목만 덧붙임) → web/build.py로 지수 갱신 (새 제목만 판정). 요청 제한이 걸리면 1시간 동안 30분 간격
# - 30분마다: 섹터용 제목 (collect.py sector)
# - 10분마다: 코스피·코스닥 1분 시세를 data/{kospi,kosdaq}_minute.tsv에 합침 (네이버는 최근 6거래일만 남김)
# python 실험/주기수집.py   (로그: data/수집로그/주기수집.log)
import json, time, datetime as dt, subprocess, sys, urllib.request
from pathlib import Path
HOME = Path(__file__).resolve().parent.parent
LOG = HOME / "data" / "수집로그" / "주기수집.log"
UA = {"User-Agent": "Mozilla/5.0"}
get = lambda u: urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30).read()


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def us_realtime():
    path = HOME / "data" / "us_realtime.tsv"
    if not path.exists():
        path.write_text("미국시각\t지수\t값\n", encoding="utf-8")
    seen = set(path.read_text(encoding="utf-8").splitlines()[-50:])
    new = []
    for code in (".INX", ".IXIC"):
        d = json.loads(get(f"https://polling.finance.naver.com/api/realtime/worldstock/index/{code}"))["datas"][0]
        if d.get("marketStatus") == "OPEN":
            line = f"{d['localTradedAt']}\t{code}\t{d['closePriceRaw']}"
            if line not in seen:
                new.append(line + "\n")
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(new)
    return len(new)


def kr_minute():
    for code, name in (("KOSPI", "kospi"), ("KOSDAQ", "kosdaq")):
        path = HOME / "data" / f"{name}_minute.tsv"
        old = {l.split("\t")[0]: l.split("\t")[1] for l in path.read_text(encoding="utf-8").splitlines()[1:]} if path.exists() else {}
        end = dt.date.today().strftime("%Y%m%d")
        for r in json.loads(get(f"https://api.stock.naver.com/chart/domestic/index/{code}/minute?startDateTime=202610010900&endDateTime={end}1530")):
            t = r["localDateTime"]
            old[f"{t[:4]}-{t[4:6]}-{t[6:8]} {t[8:10]}:{t[10:12]}"] = str(r["currentPrice"])
        path.write_text(f"시각\t{code}\n" + "".join(f"{t}\t{v}\n" for t, v in sorted(old.items())), encoding="utf-8")
    return len(old)


SLOW = {"until": 0.0}  # 구글 뉴스가 요청 제한(429·503)을 걸면 1시간 동안 지수용 수집을 30분 간격으로 늦춤 (우회하지 않음)


def collect(modes):
    out = []
    for mode, lang in [(m, l) for m in modes for l in ("ko", "en")]:
        r = subprocess.run([sys.executable, str(HOME / "collect.py"), mode, "1", lang, "-1"], cwd=HOME, capture_output=True, text=True, encoding="utf-8")
        out.append((r.stdout.strip().splitlines() or ["?"])[-1])
        if "HTTP Error 429" in r.stdout or "HTTP Error 503" in r.stdout:
            SLOW["until"] = time.time() + 3600
            out.append("요청 제한 → 1시간 동안 30분 간격")
    return out


def headlines():  # 지수용 제목 (10분마다) → 지수 갱신
    out = collect(("daily",))
    # 제목을 모은 뒤 지수도 갱신 (새 제목만 판정 → web/public/data/latest.json, 실시간 칸) — 2026-11-02 사용자 결정
    b = subprocess.run([sys.executable, str(HOME / "web" / "build.py")], cwd=HOME, capture_output=True, text=True, encoding="utf-8",
                       env=dict(__import__("os").environ, OMP_NUM_THREADS="4", PYTHONIOENCODING="utf-8"))
    out.append("지수 " + ("갱신" if b.returncode == 0 else f"실패 {b.returncode}: {(b.stderr.strip().splitlines() or ['?'])[-1][:120]}"))
    return " / ".join(out)


def sectors():  # 섹터용 제목 (30분마다, 검색어가 많아서)
    return " / ".join(collect(("sector",)))


last = {"us": 0.0, "head": 0.0, "sector": 0.0, "kr": 0.0}
log("시작")
while True:
    now = time.time()
    head_every = 1800 if now < SLOW["until"] else 600
    for key, every, fn in (("us", 120, us_realtime), ("head", head_every, headlines), ("sector", 1800, sectors), ("kr", 600, kr_minute)):
        if now - last[key] >= every:
            last[key] = now
            try:
                log(f"{key}: {fn()}")
            except Exception as e:  # 한 번 실패해도 루프는 계속
                log(f"{key} 실패: {e!r}")
    time.sleep(20)
