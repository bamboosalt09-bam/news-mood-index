# 제목 적은 섹터는 7일 평균 (2026-10-31, docs/실험기록.md 같은 날). 정답 = 시가총액 무게 업체 바구니 초과 등락.
# python 실험/섹터7일.py   → 실험/결과/섹터7일.json·.log
import sys, json, collections, datetime as dt
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
by = b.assign("ko", days, b.items("ko", "최종"), ("daily", "sector"), since="2025-10-20")
pool = [x for x in days if "2025-10-24" <= x <= END]
ALL = collections.defaultdict(lambda: collections.defaultdict(list))
for d in pool:
    for t, it, _ in by.get(d, []):
        for s in S.tag(t, D, "ko")[0]:
            ALL[s][d].append(it["s"])
r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
OUT = {}
for s, per in ALL.items():
    mem = [n for n, sec in home.items() if sec == s and n in CAP and n in RET]
    R = np.vstack([RET[n] for n in mem]); w = np.array([CAP[n] for n in mem])
    ex = np.array([np.average(R[np.isfinite(R[:, k]), k], weights=w[np.isfinite(R[:, k])]) if np.isfinite(R[:, k]).any() else np.nan for k in range(len(E))]) - kospi
    n1 = np.array([len(per[d]) for d in E])
    x1 = np.array([np.mean(per[d]) if len(per[d]) >= 5 else np.nan for d in E])
    x7, n7, ex7 = [], [], []
    for k, d in enumerate(E):
        lo = (dt.date.fromisoformat(d) - dt.timedelta(6)).isoformat()
        win = [x for x in pool if lo <= x <= d]
        v = [sc for x in win for sc in per[x]]
        n7.append(len(v)); x7.append(np.mean(v) if len(v) >= 10 else np.nan)
        ks = [kk for kk, x in enumerate(E) if lo <= x <= d]
        ex7.append(np.nansum(ex[ks]) if len(ks) == len([x for x in win]) else np.nan)  # 7일 안 거래일이 기간 안에 다 있을 때만
    x7, n7, ex7 = map(np.array, (x7, n7, ex7))
    last = slice(-60, None)
    r = {"하루": {"상관": r3(ix.corr(x1, ex)[0]), "값있는날": round(float(np.isfinite(x1).mean()), 2), "20개넘는날_최근60": round(float((n1[last] >= 20).mean()), 2)},
         "7일": {"상관": r3(ix.corr(x7, ex7)[0]), "값있는날": round(float(np.isfinite(x7).mean()), 2), "20개넘는날_최근60": round(float((n7[last] >= 20).mean()), 2)}}
    few = r["하루"]["20개넘는날_최근60"] < 0.8
    use7 = few and r["7일"]["20개넘는날_최근60"] >= 0.8 and r["7일"]["상관"] is not None and r["하루"]["상관"] is not None and r["7일"]["상관"] >= r["하루"]["상관"] - 0.05
    r["적은섹터"], r["판단"] = few, ("7일" if use7 else "하루")
    OUT[s] = r
print(f"기간 {E[0]} ~ {E[-1]}. 정답 = 시가총액 무게 업체 바구니 초과 등락 (7일 값은 7일 합)")
print("섹터 | 하루: 상관 · 20개 넘는 날(최근60) | 7일: 상관 · 20개 넘는 날(최근60) | 적은 섹터 | 판단")
for s, r in OUT.items():
    a, c = r["하루"], r["7일"]
    print(f"  {s:14} | {a['상관']:+.2f} · {a['20개넘는날_최근60']:.0%} | {c['상관']:+.2f} · {c['20개넘는날_최근60']:.0%} | {'예' if r['적은섹터'] else '-'} | {r['판단']}")
json.dump(OUT, open(j.HOME / "실험" / "결과" / "섹터7일.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
