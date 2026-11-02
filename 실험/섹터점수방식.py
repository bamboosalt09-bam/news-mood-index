# 섹터 점수 방식 고르기 (2026-10-31, docs/실험기록.md 같은 날 '섹터 점수 방식 고르기'). 정답 기준 = 사전 업체 주가 바구니 (ETF는 참고).
# python 실험/섹터점수방식.py   → 실험/결과/섹터점수방식.json·.log
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
START, END, MINT = "2025-11-03", "2026-10-30", 5
ETF = {"반도체": ["KODEX 반도체"], "자동차": ["KODEX 자동차"], "금융": ["KODEX 은행", "KODEX 증권"], "바이오": ["KODEX 바이오", "KODEX 헬스케어"],
       "2차전지": ["TIGER 2차전지테마"], "건설": ["KODEX 건설"], "소재(화학·철강·정유)": ["KODEX 철강", "KODEX 에너지화학"],
       "조선·방산": ["TIGER 조선TOP10", "PLUS K방산"], "전력·원전": ["HANARO 원자력iSelect"], "유통·소비": ["KODEX 필수소비재", "TIGER 화장품"],
       "인터넷·게임": ["NAVER", "카카오"]}

lang, c = "ko", b.CONF["ko"]
days, _, kchg = b.prices(c["price"])
ev = [i for i, d in enumerate(days) if START <= d <= END]
E = [days[i] for i in ev]
kospi = kchg[ev]


def returns(path, col):  # {이름: 등락률 배열 (E 위)}
    px = collections.defaultdict(dict)
    for l in open(j.HOME / "data" / path, encoding="utf-8").read().splitlines()[1:]:
        p = l.split("\t"); px[p[col]][p[0]] = float(p[3])
    out = {}
    for n, d in px.items():
        prev = [d.get(days[i - 1]) for i in ev]; cur = [d.get(x) for x in E]
        out[n] = np.array([100 * (b_ / a - 1) if a and b_ else np.nan for a, b_ in zip(prev, cur)])
    return out


CO = returns("섹터업체주가.tsv", 1)
EF = returns("섹터ETF.tsv", 1)
D = S.load(lang)
home = {co: s for _, s, co in D if co}
IT = b.items(lang, c["col"])
by = b.assign(lang, days, IT, ("daily", "sector"), since="2025-10-25")
# 섹터·날 → 갈래별 [(제목, it, 링크)], 업체·날 → [(점수, 시황?)]
SEC = collections.defaultdict(lambda: collections.defaultdict(lambda: {"업계": [], "업체시황": [], "업체기타": []}))
COM = collections.defaultdict(lambda: collections.defaultdict(list))
for d in E:
    for t, it, link in by.get(d, []):
        ss, cos = S.tag(t, D, lang)
        for s in ss:
            mine = [x for x in cos if home[x] == s]
            SEC[s][d]["업계" if not mine else "업체시황" if it["mkt"] else "업체기타"].append((t, it, link))
        for co in cos:
            COM[co][d].append((it["s"], it["mkt"]))


def val(g, v):  # 후보별 하루 값, (값, 쓴 제목 수)
    if v == "V0": L = g["업계"] + g["업체시황"] + g["업체기타"]
    elif v == "V1": L = g["업계"]
    else: L = g["업계"] + g["업체시황"]
    if v == "V4":
        w = [1.0] * len(L) + [0.3] * len(g["업체기타"]); L = L + g["업체기타"]
        return (float(np.average([it["s"] for _, it, _ in L], weights=w)) if len(L) >= MINT else np.nan), len(L)
    if len(L) < MINT:
        return np.nan, len(L)
    if v == "V3":
        return float(np.mean([x["score"] for x in b.clusters(L)])), len(L)
    return float(np.mean([it["s"] for _, it, _ in L])), len(L)


VS = ["V0", "V1", "V2", "V3", "V4"]
OUT = {"섹터": {}, "업체": {}}
med = lambda xs: float(np.nanmedian([x for x in xs if x is not None])) if xs else None
r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
for s, per in SEC.items():
    mem = [n for n, sec in home.items() if sec == s and n in CO]
    basket = np.nanmean(np.vstack([CO[n] for n in mem]), 0)
    etf = np.nanmean(np.vstack([EF[n] for n in ETF[s]]), 0)
    r = {"업체바구니": mem, "갈래별제목수_하루평균": {k: round(np.mean([len(per[d][k]) for d in E]), 1) for k in ("업계", "업체시황", "업체기타")}}
    for v in VS:
        x, n = map(np.array, zip(*[val(per[d], v) for d in E]))
        r[v] = {"날수": int(np.isfinite(x).sum()), "하루제목": round(float(n.mean()), 1),
                "같은날": r3(ix.corr(x, basket)[0]), "초과": r3(ix.corr(x, basket - kospi)[0]),
                "ETF같은날": r3(ix.corr(x, etf)[0]), "ETF초과": r3(ix.corr(x, etf - kospi)[0]),
                "초과_10~19개날": r3(ix.corr(np.where((n >= 10) & (n < 20), x, np.nan), basket - kospi)[0]) if ((n >= 10) & (n < 20) & np.isfinite(x)).sum() > 20 else None,
                "초과_20개이상날": r3(ix.corr(np.where(n >= 20, x, np.nan), basket - kospi)[0]) if ((n >= 20) & np.isfinite(x)).sum() > 20 else None}
    OUT["섹터"][s] = r
summ = {v: {"초과_중앙값": r3(med([OUT["섹터"][s][v]["초과"] for s in OUT["섹터"]])), "같은날_중앙값": r3(med([OUT["섹터"][s][v]["같은날"] for s in OUT["섹터"]])),
            "ETF초과_중앙값": r3(med([OUT["섹터"][s][v]["ETF초과"] for s in OUT["섹터"]])),
            "날수_중앙값": med([OUT["섹터"][s][v]["날수"] for s in OUT["섹터"]])} for v in VS}
best = max(VS, key=lambda v: summ[v]["초과_중앙값"])
close = [v for v in VS if summ[best]["초과_중앙값"] - summ[v]["초과_중앙값"] <= 0.02]
pick = max(close, key=lambda v: summ[v]["날수_중앙값"])
if summ[pick]["초과_중앙값"] - summ["V0"]["초과_중앙값"] < 0.05:
    pick = "V0"
lo = med([OUT["섹터"][s][pick]["초과_10~19개날"] for s in OUT["섹터"]]); hi = med([OUT["섹터"][s][pick]["초과_20개이상날"] for s in OUT["섹터"]])
OUT["요약"] = summ
OUT["고름"] = {"후보": pick, "문턱": 10 if lo is not None and hi is not None and hi - lo <= 0.05 else 20, "초과_10~19개날_중앙값": r3(lo), "초과_20개이상날_중앙값": r3(hi)}

# 업체별: 하루 3제목 이상인 날 60일 이상
for co, per in COM.items():
    if co not in CO:
        continue
    row = {}
    for k, f in (("전체", lambda m: True), ("시황", lambda m: m), ("기타", lambda m: not m)):
        x = np.array([np.mean([sc for sc, m in per[d] if f(m)]) if sum(f(m) for _, m in per[d]) >= 3 else np.nan for d in E])
        row[k] = {"날수": int(np.isfinite(x).sum()), "같은날": r3(ix.corr(x, CO[co])[0]) if np.isfinite(x).sum() >= 30 else None,
                  "초과": r3(ix.corr(x, CO[co] - kospi)[0]) if np.isfinite(x).sum() >= 30 else None}
    if row["전체"]["날수"] >= 60:
        OUT["업체"][co] = row

print(f"기간 {E[0]} ~ {E[-1]} ({len(E)}거래일), 하루 값은 제목 {MINT}개 이상. 정답 = 사전 업체 주가 바구니 (같은 무게)")
print("\n후보별 11개 섹터 중앙값 (초과 = 바구니 − 코스피):")
for v in VS:
    print(f"  {v}: 초과 {summ[v]['초과_중앙값']:+.3f}  같은날 {summ[v]['같은날_중앙값']:+.3f}  (ETF 초과 {summ[v]['ETF초과_중앙값']:+.3f})  날수 {summ[v]['날수_중앙값']:.0f}")
print(f"\n고름: {OUT['고름']}")
print("\n섹터별 초과 상관 (V0 / V1 / V2 / V3 / V4), 하루 갈래별 제목 수 업계·업체시황·업체기타:")
for s, r in OUT["섹터"].items():
    print(f"  {s:14} " + " / ".join(f"{r[v]['초과']:+.2f}" if r[v]["초과"] is not None else "  –  " for v in VS) + f"   날수 {r['V0']['날수']}/{r['V2']['날수']}   {r['갈래별제목수_하루평균']}")
print("\n업체별 (하루 3제목 이상 60일↑): 초과 상관 전체 / 시황만 / 기타만 (날수)")
for co, r in sorted(OUT["업체"].items(), key=lambda x: -x[1]["전체"]["날수"]):
    f = lambda k: (f"{r[k]['초과']:+.2f}" if r[k]["초과"] is not None else "  –  ") + f"({r[k]['날수']})"
    print(f"  {co:12} {f('전체')}  {f('시황')}  {f('기타')}")
json.dump(OUT, open(j.HOME / "실험" / "결과" / "섹터점수방식.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
