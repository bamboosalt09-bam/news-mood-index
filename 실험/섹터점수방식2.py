# 섹터 점수 2차: 업체 규모(시가총액) 무게 (2026-10-31, docs/실험기록.md 같은 날 '섹터 점수 2차'). 정답 = 시가총액 무게 업체 바구니.
# python 실험/섹터점수방식2.py   → 실험/결과/섹터점수방식2.json·.log
import sys, json, collections
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
sys.path.insert(0, r"C:\english_folder\somethings\web")
sys.argv = sys.argv[:1]
import judge as j
import index as ix
import build as b
import sectors as S
sys.stdout.reconfigure(encoding="utf-8")
START, END = "2025-11-03", "2026-10-30"

days, _, kchg = b.prices("코스피")
ev = [i for i, d in enumerate(days) if START <= d <= END]
E = [days[i] for i in ev]
kospi = kchg[ev]
px = collections.defaultdict(dict)
for l in open(j.HOME / "data" / "섹터업체주가.tsv", encoding="utf-8").read().splitlines()[1:]:
    p = l.split("\t"); px[p[1]][p[0]] = float(p[3])
RET = {n: np.array([100 * (d[x] / d[days[i - 1]] - 1) if d.get(x) and d.get(days[i - 1]) else np.nan for i, x in zip(ev, E)]) for n, d in px.items()}
CAP = {l.split("\t")[0]: float(l.split("\t")[2]) for l in open(j.HOME / "data" / "섹터업체시총.tsv", encoding="utf-8").read().splitlines()[1:]}

D = S.load("ko")
home = {co: s for _, s, co in D if co}
IT = b.items("ko", "최종")
by = b.assign("ko", days, IT, ("daily", "sector"), since="2025-10-25")
IND = collections.defaultdict(lambda: collections.defaultdict(list))   # 섹터 → 날 → 업계 제목 점수
COM = collections.defaultdict(lambda: collections.defaultdict(list))   # 업체 → 날 → 점수
ALL = collections.defaultdict(lambda: collections.defaultdict(list))   # 섹터 → 날 → 모든 제목 점수 (V0)
for d in E:
    for t, it, _ in by.get(d, []):
        ss, cos = S.tag(t, D, "ko")
        for s in ss:
            ALL[s][d].append(it["s"])
            if not [x for x in cos if home[x] == s]:
                IND[s][d].append(it["s"])
        for co in cos:
            COM[co][d].append(it["s"])


def wavg(vals, w):
    m = np.isfinite(vals) & (w > 0)
    return float(np.average(vals[m], weights=w[m])) if m.any() else np.nan


def sector_vals(s, v):
    mem = [n for n, sec in home.items() if sec == s and n in CAP]
    w = np.array([CAP[n] for n in mem])
    out = []
    for d in E:
        n = np.array([len(COM[c][d]) for c in mem])
        m = np.array([np.mean(COM[c][d]) if COM[c][d] else np.nan for c in mem])
        if v in ("W2", "W4"):
            m = m * n / (n + 3)
        comp = wavg(m, w) if n.sum() >= 3 else np.nan
        if v == "V0":
            out.append(np.mean(ALL[s][d]) if len(ALL[s][d]) >= 5 else np.nan); continue
        if v in ("W1", "W2"):
            out.append(comp); continue
        ind = np.mean(IND[s][d]) if IND[s][d] else np.nan
        comp_any = wavg(m, w) if n.sum() else np.nan
        x = np.nanmean([ind, comp_any]) if np.isfinite([ind, comp_any]).any() else np.nan
        out.append(x if len(IND[s][d]) + n.sum() >= 5 else np.nan)
    return np.array(out)


r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
VS = ["V0", "W1", "W2", "W3", "W4"]
OUT = {"섹터": {}}
for s in ALL:
    mem = [n for n, sec in home.items() if sec == s and n in CAP and n in RET]
    R = np.vstack([RET[n] for n in mem]); w = np.array([CAP[n] for n in mem])
    capb = np.array([wavg(R[:, k], w) for k in range(len(E))])
    eqb = np.nanmean(R, 0)
    r = {"무게 상위": sorted(((n, round(CAP[n] / w.sum(), 2)) for n in mem), key=lambda x: -x[1])[:3]}
    for v in VS:
        x = sector_vals(s, v)
        r[v] = {"날수": int(np.isfinite(x).sum()), "초과": r3(ix.corr(x, capb - kospi)[0]), "같은날": r3(ix.corr(x, capb)[0]),
                "같은무게초과": r3(ix.corr(x, eqb - kospi)[0])}
    OUT["섹터"][s] = r
med = lambda v, k: float(np.nanmedian([OUT["섹터"][s][v][k] for s in OUT["섹터"] if OUT["섹터"][s][v][k] is not None]))
summ = {v: {"초과_중앙값": r3(med(v, "초과")), "같은날_중앙값": r3(med(v, "같은날")), "같은무게초과_중앙값": r3(med(v, "같은무게초과")), "날수_중앙값": med(v, "날수")} for v in VS}
best = max(VS, key=lambda v: summ[v]["초과_중앙값"])
pick = max([v for v in VS if summ[best]["초과_중앙값"] - summ[v]["초과_중앙값"] <= 0.02], key=lambda v: summ[v]["날수_중앙값"])
if summ[pick]["초과_중앙값"] - summ["V0"]["초과_중앙값"] < 0.05:
    pick = "V0"
OUT["요약"], OUT["고름"] = summ, pick
print(f"기간 {E[0]} ~ {E[-1]} ({len(E)}거래일). 정답 = 시가총액 무게 업체 바구니 (2026-10-31 시총 고정)")
for v in VS:
    print(f"  {v}: 초과 {summ[v]['초과_중앙값']:+.3f}  같은날 {summ[v]['같은날_중앙값']:+.3f}  (같은 무게 바구니 초과 {summ[v]['같은무게초과_중앙값']:+.3f})  날수 {summ[v]['날수_중앙값']:.0f}")
print(f"고름: {pick}\n\n섹터별 초과 상관 (" + " / ".join(VS) + "), 날수 V0/W1, 시총 무게 상위")
for s, r in OUT["섹터"].items():
    print(f"  {s:14} " + " / ".join(f"{r[v]['초과']:+.2f}" if r[v]["초과"] is not None else "  –  " for v in VS) + f"   {r['V0']['날수']}/{r['W1']['날수']}   {r['무게 상위']}")
json.dump(OUT, open(j.HOME / "실험" / "결과" / "섹터점수방식2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
