# 섹터·기업 사전으로 제목 나누기 (2026-10-31, docs/실험기록.md 같은 날 '섹터·기업 사전 초안'). 모델은 그대로, 점수는 data/cache/지수자료_<lang>.tsv.
# python 실험/섹터.py   → 실험/결과/섹터.json·.log
import sys, re, json, random, collections
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
sys.path.insert(0, r"C:\english_folder\somethings\web")
import judge as j
import index as ix
import build as b
sys.stdout.reconfigure(encoding="utf-8")


from sectors import load, tag, STRIP, BANKS  # 사전 함수는 sectors.py로 옮김


OUT = {}; random.seed(0)
for lang in ("ko", "en"):
    c = b.CONF[lang]
    days, C, chg = b.prices(c["price"])
    by = b.assign(lang, days, b.items(lang, c["col"]))
    D = load(lang)
    sec_n = collections.defaultdict(lambda: np.zeros(len(days)))
    co_n = collections.defaultdict(lambda: np.zeros(len(days)))
    sec_s = collections.defaultdict(lambda: collections.defaultdict(list))
    samples = collections.defaultdict(list)
    tot = multi = none = 0
    for i, d in enumerate(days):
        for title, it, _ in by.get(d, []):
            secs, cos = tag(title, D, lang)
            tot += 1; multi += len(secs) > 1; none += not secs
            for s in secs:
                sec_n[s][i] += 1; sec_s[s][i].append(it["s"]); samples[s].append(title)
            for co in cos:
                co_n[co][i] += 1
    live = np.array([len(by.get(d, [])) >= b.MIN_DAY for d in days])
    last60 = np.where(live)[0][-60:]
    first = np.where(live)[0][0]
    r = {"제목": tot, "섹터 둘 이상": round(multi / tot, 3), "섹터 없음": round(none / tot, 3), "섹터": {}, "기업": {}}
    for s, n in sorted(sec_n.items(), key=lambda x: -x[1][last60].mean()):
        mood = np.array([np.mean(sec_s[s][i]) if len(sec_s[s][i]) >= b.MIN_DAY else np.nan for i in range(len(days))])
        r["섹터"][s] = {"하루평균_전체": round(n[first:].mean(), 1), "하루평균_최근60": round(n[last60].mean(), 1),
                       "20개이상_최근60": round(float((n[last60] >= b.MIN_DAY).mean()), 2),
                       "지수가능": bool((n[last60] >= b.MIN_DAY).mean() >= 0.8),
                       "같은날_시장등락_상관": round(ix.corr(mood, chg)[0], 3) if np.isfinite(mood).sum() > 30 else None,
                       "표본": random.sample(samples[s], min(10, len(samples[s])))}
    for co, n in sorted(co_n.items(), key=lambda x: -x[1][last60].mean()):
        r["기업"][co] = {"하루평균_전체": round(n[first:].mean(), 1), "하루평균_최근60": round(n[last60].mean(), 1), "따로보여줄만": bool(n[last60].mean() >= 5)}
    OUT[lang] = r
    print(f"\n== {lang}: 제목 {tot:,}, 섹터 둘 이상 {r['섹터 둘 이상']:.1%}, 섹터 없음 {r['섹터 없음']:.1%}")
    for s, v in r["섹터"].items():
        print(f"  {s:14} 하루 {v['하루평균_전체']:6.1f} (최근60 {v['하루평균_최근60']:6.1f}, 20개↑ {v['20개이상_최근60']:.0%}) {'지수 가능' if v['지수가능'] else '모자람':6} 시장 상관 {v['같은날_시장등락_상관']}")
    print("  기업 (최근60 하루 평균):", ", ".join(f"{k} {v['하루평균_최근60']}" for k, v in r["기업"].items()))
json.dump(OUT, open(j.HOME / "실험" / "결과" / "섹터.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
