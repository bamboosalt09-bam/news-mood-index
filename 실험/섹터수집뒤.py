# 섹터 검색어 더한 뒤 섹터·기업별 하루 제목 수 다시 세기 (2026-10-31, docs/실험기록.md '섹터 검색어 수집'). 판정 점수 없이 사전으로만 센다.
# python 실험/섹터수집뒤.py   → 실험/결과/섹터_수집뒤.log
import sys, collections
sys.path.insert(0, r"C:\english_folder\somethings"); sys.path.insert(0, r"C:\english_folder\somethings\web"); sys.argv = sys.argv[:1]
import numpy as np, judge as j, index as ix, build as b
sys.stdout.reconfigure(encoding="utf-8")
src = open(j.HOME / "실험" / "섹터.py", encoding="utf-8").read(); ns = {}; exec(src[:src.index("OUT = {}")], ns)
print("daily+sector 파일 합침, 제목 중복 제거, 수준만 제목도 포함(점수 전이라). 최근 60거래일.")
for lang in ("ko", "en"):
    days, _, _ = b.prices(b.CONF[lang]["price"]); D = ns["load"](lang); pos = {d: i for i, d in enumerate(days)}
    sn = collections.defaultdict(lambda: np.zeros(len(days))); cn = collections.defaultdict(lambda: np.zeros(len(days))); seen = set()
    for f in (f"{lang}_daily.tsv", f"{lang}_sector.tsv"):
        for l in open(j.HOME / "headlines" / f, encoding="utf-8"):
            p = l.rstrip("\n").split("\t")
            if len(p) < 3 or p[0][:2] != "20" or p[2] in seen:
                continue
            seen.add(p[2]); d = ix.market_day(p[0][:16], days, lang)
            if d:
                secs, cos = ns["tag"](p[2], D, lang)
                for s in secs: sn[s][pos[d]] += 1
                for c in cos: cn[c][pos[d]] += 1
    last = np.arange(len(days))[-60:]
    print(f"\n== {lang}: 섹터 (최근60 하루 평균, 20개 넘는 날 비율)")
    for s, n in sorted(sn.items(), key=lambda x: -x[1][last].mean()):
        ok = (n[last] >= 20).mean(); print(f"  {s:14} {n[last].mean():6.1f}  {ok:4.0%}  {'지수 가능' if ok >= 0.8 else '모자람'}")
    print("  하루 5개 넘는 기업:", ", ".join(f"{c} {n[last].mean():.1f}" for c, n in sorted(cn.items(), key=lambda x: -x[1][last].mean()) if n[last].mean() >= 5))
