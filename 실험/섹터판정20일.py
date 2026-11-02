# Haiku 섹터 판정 vs 규칙 라벨, 10/30 앞 한국어 20 시장일 (docs/실험기록.md 2026-11-01 사전 등록)
# python 실험/섹터판정20일.py make  → 실험/결과/섹터판정20일/입력_<날>.txt (labeler용) + 섹터판정20일.tsv (규칙 라벨·excess)
# python 실험/섹터판정20일.py score → 실험/결과/섹터판정20일_haiku.json을 읽어 A·B·C 채점
import sys, json, collections
from pathlib import Path
HOME = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HOME / "web"), str(HOME)]
import numpy as np
import build as b, sectorjudge as sj, sectors as S

OUT = HOME / "실험" / "결과" / "섹터판정20일"
TSV = HOME / "실험" / "결과" / "섹터판정20일.tsv"
HAIKU = HOME / "실험" / "결과" / "섹터판정20일_haiku.json"
LAST, N, TH = "2026-10-30", 20, 0.5


def make():
    c = b.CONF["ko"]
    days, C, chg = b.prices(c["price"])
    IT = b.items("ko", c["col"])
    test = [d for d in days if d < LAST][-N:]
    fixed, orig_scoop, got = b.FIXED["ko"], b.scoop, {}
    b.FIXED["ko"] = tuple(x for _, x, _ in S.load("ko"))  # 모든 섹터를 판정 대상으로
    def scoop(lang, per, by, days_, d):  # 화면에 나왔을 특종은 원래 FIXED로 따로 기록
        b.FIXED["ko"], keep = fixed, b.FIXED["ko"]
        got["scoop"] = orig_scoop(lang, per, by, days_, d)
        b.FIXED["ko"] = keep
        return None
    b.scoop = scoop
    sj.judge = lambda lang, market, d, items: got.__setitem__("items", items) or {n: {"label": sj.rule(v["z"], v["mood"]), "line": None, "model": "dry-run"} for n, v in items.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = ["날짜\t섹터\t제목\t평균\tz\t규칙\texcess\t화면"]
    for d in test:
        r = b.sectors("ko", days, IT, d, chg[days.index(d)])
        cards = {x["name"]: x for x in r["list"]}
        items = {n: v for n, v in got["items"].items() if n in cards}
        sc = got["scoop"][0] if got["scoop"] else ""
        (OUT / f"입력_{d}.txt").write_text(sj.SYSTEM + "\n\n# 입력\n" + sj.payload(c["name"], d, items), encoding="utf-8")
        for n, x in cards.items():
            ex = (x["move"] or {}).get("excess")
            rows.append(f"{d}\t{n}\t{x['titles']}\t{x['mood']}\t{x['mood_rel']}\t{x['label']}\t{'' if ex is None else ex}\t{int(n in fixed or n == sc)}")
        print(d, len(cards), "섹터, 특종", sc or "없음", flush=True)
    TSV.write_text("\n".join(rows) + "\n", encoding="utf-8")


def rk(v):  # 순위 (동점은 평균 순위)
    v = np.asarray(v, float); r = np.empty(len(v)); r[np.argsort(v, kind="stable")] = np.arange(len(v))
    for u in np.unique(v):
        r[v == u] = r[v == u].mean()
    return r


def grade(label, ex):
    if label == "엇갈림" or ex is None:
        return None
    sign = 1 if label == "좋은 소식 많음" else -1
    return "헛경보" if abs(ex) < TH else "맞음" if sign * ex > 0 else "틀림"


def score():
    hk = json.loads(HAIKU.read_text(encoding="utf-8"))
    rows = [l.split("\t") for l in TSV.read_text(encoding="utf-8").splitlines()[1:]]
    val = {"좋은 소식 많음": 1, "엇갈림": 0, "나쁜 소식 많음": -1}
    res = {k: collections.defaultdict(lambda: collections.Counter()) for k in "ABC"}
    pairs = collections.defaultdict(lambda: collections.defaultdict(list))
    for d, n, t, mood, z, rule, ex, shown in rows:
        if not ex:
            continue
        ex, mood = float(ex), float(mood)
        h = hk.get(d, {}).get(n)
        if h is None:
            print("Haiku 판정 없음:", d, n); continue
        c = h if val[h] * mood >= 0 else "엇갈림"
        for k, lab in (("A", rule), ("B", h), ("C", c)):
            for part in ("전부",) + (("화면",) if shown == "1" else ()):
                res[k][part][grade(lab, ex) or "엇갈림"] += 1
                pairs[k][part].append((val[lab], ex))
    for part in ("전부", "화면"):
        print(f"\n[{part}]")
        for k, name in (("A", "규칙"), ("B", "Haiku"), ("C", "합의")):
            r = res[k][part]; xs = pairs[k][part]
            rho = float(np.corrcoef(rk([a for a, _ in xs]), rk([e for _, e in xs]))[0, 1]) if len(xs) > 2 else float("nan")
            print(f"{k} {name}: 맞음 {r['맞음']} 틀림 {r['틀림']} 헛경보 {r['헛경보']} 엇갈림 {r['엇갈림']} (맞음−틀림 {r['맞음'] - r['틀림']}), 순위상관 {rho:+.3f}, n {len(xs)}")
    a = res["A"]["전부"]
    for k in "BC":
        r = res[k]["전부"]
        print(f"{k} 통과: {r['틀림'] <= a['틀림'] and r['맞음'] - r['틀림'] > a['맞음'] - a['틀림']}")


if __name__ == "__main__":
    {"make": make, "score": score}[sys.argv[1]]()
