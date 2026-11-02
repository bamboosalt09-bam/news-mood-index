# 미국 장을 둘로 나눠 영어 뉴스(미국 관점) 견주기 (2026-10-29): 밤사이(전 거래일 16:00 ~ 오늘 09:30, 미국 동부) 뉴스 ↔ 시가 갭,
# 장중(09:30 ~ 16:00) 뉴스 ↔ 시가→종가 등락. 시세는 네이버 일별 시가·종가 (장중 분 자료는 받을 곳이 없음).
# 모델은 판정 창 주력 en_v2-감성의 미국 관점. 제목 시각은 UTC (동부 = UTC−4, 서머타임 끝나는 11월 첫 일요일 전까지).
# python 실험/장중_미국.py [시작일=2026-10-01] [시장=.INX]
import sys, json, math, datetime as dt, urllib.request
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
sys.stdout.reconfigure(encoding="utf-8")
START = sys.argv[1] if len(sys.argv) > 1 else "2026-10-01"
CODE = sys.argv[2] if len(sys.argv) > 2 else ".INX"
MIN_N = 5
u = f"https://api.stock.naver.com/chart/foreign/index/{CODE}/day?startDateTime={START.replace('-', '')}0000&endDateTime={dt.date.today():%Y%m%d}2359"
bars = {f"{r['localDate'][:4]}-{r['localDate'][4:6]}-{r['localDate'][6:]}": (r["openPrice"], r["closePrice"])
        for r in json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=20).read())}
days = sorted(bars)
et = lambda t: t - dt.timedelta(hours=4 if t < dt.datetime(2026, 11, 1, 6) else 5)  # UTC → 미국 동부
seen, items = set(), []
for l in open(j.HOME / "headlines" / "en_daily.tsv", encoding="utf-8"):
    p = l.split("\t")
    if len(p) < 3 or p[0][:2] != "20" or p[2] in seen or p[0][11:16] in ("", "07:00", "08:00", "00:00") or p[0] < START:
        continue
    seen.add(p[2])
    items.append((et(dt.datetime.strptime(p[0][:16], "%Y-%m-%d %H:%M")), p[2]))
P, m = j.predict("en_v2-감성", [x for _, x in items], 0)  # 0 = 미국 관점
ts = [x for _, x in items]
s = np.where(j.level_only(ts), np.nan, j.soften(ts, j.guard(ts, j.score_of(P, m), m.get("guard"), m.get("guard_lexicon"))[0])[0])
sc = dict(zip(ts, s))
rows = []
for a, b in zip(days, days[1:]):
    o, c = bars[b]
    night = (dt.datetime.fromisoformat(a + " 16:00"), dt.datetime.fromisoformat(b + " 09:30"))
    sess = (dt.datetime.fromisoformat(b + " 09:30"), dt.datetime.fromisoformat(b + " 16:00"))
    pick = lambda w: [sc[x] for t, x in items if w[0] <= t < w[1] and not np.isnan(sc[x])]
    vn, vs = pick(night), pick(sess)
    rows.append((b, np.mean(vn) if len(vn) >= MIN_N else np.nan, len(vn), np.mean(vs) if len(vs) >= MIN_N else np.nan, len(vs),
                 100 * math.log(o / bars[a][1]), 100 * math.log(c / o)))
N, S, gap, intra = (np.array([r[k] for r in rows], float) for k in (1, 3, 5, 6))


def corr(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    n = int(ok.sum())
    if n < 4:
        return float("nan"), n, float("nan")
    r = float(np.corrcoef(x[ok], y[ok])[0, 1])
    return r, n, math.erfc(abs(r * math.sqrt((n - 2) / max(1 - r * r, 1e-12))) / math.sqrt(2))


print(f"{CODE} {days[0]}~{days[-1]}, 시각 있는 영어 제목 {len(items):,}개")
for nm, x, y in (("밤사이 뉴스 ↔ 시가 갭", N, gap), ("장중 뉴스 ↔ 장중 등락 (시가→종가)", S, intra),
                 ("밤사이 뉴스 → 그날 장중 등락", N, intra), ("장중 뉴스 → 다음 날 시가 갭", S, np.r_[gap[1:], np.nan]),
                 ("하루 전체 뉴스 ↔ 종가 등락", np.nanmean(np.c_[N, S], 1), gap + intra)):
    r, n, p = corr(x, y)
    print(f"  {nm:32} {r:+.2f} (p={p:.3f}, {n}일)")
print("  날짜 | 밤사이 뉴스(개수) | 장중 뉴스(개수) | 갭% | 장중%")
for r in rows:
    print(f"  {r[0]} | {r[1]:+.2f} ({r[2]}) | {r[3]:+.2f} ({r[4]}) | {r[5]:+.2f} | {r[6]:+.2f}")
