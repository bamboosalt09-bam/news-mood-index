# 지수 분석용 제목 자료 (2026-10-31): headlines/<lang>_daily.tsv·<lang>_sector.tsv의 모든 제목을 판정 창 주력으로 한 번 판정해 data/cache/지수자료_<lang>.tsv로.
# 열: 제목, 모델 점수(관점마다), 안전장치 뒤, 최종(전망 약화 뒤), 안전장치 바뀜, 전망 표지, 수준만, 시황, 원인 꼴, 주제
# python 실험/지수자료.py ko|en          # 전부 다시 판정 (모델·규칙을 바꿨을 때)
# python 실험/지수자료.py ko|en --new    # 캐시에 없는 새 제목만 판정해 덧붙임 (주기 갱신용, 2026-11-02)
import sys
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
import index as ix
sys.stdout.reconfigure(encoding="utf-8")
LANG, NEW = sys.argv[1], "--new" in sys.argv
MODEL, TOPIC = j.MAIN[f"{LANG}-감성"], f"{LANG}-주제"
out = j.HOME / "data" / "cache" / f"지수자료_{LANG}.tsv"
day = {}  # 제목 → 처음 나온 날 (환율 기억에 씀)
for f in ("daily", "sector"):
    for l in open(j.HOME / "headlines" / f"{LANG}_{f}.tsv", encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) > 2 and p[0][:2] == "20":  # sector: 섹터 지수용 (2026-10-31). 시장 지수는 web/build.py가 daily 제목만 씀
            day[p[2]] = min(day.get(p[2], p[0][:10]), p[0][:10])
ts = sorted(day)
if NEW and out.exists():
    have = {l.rstrip("\n").split("\t")[-1] for l in open(out, encoding="utf-8")}
    ts = [t for t in ts if t not in have]
    print(f"{LANG} 새 제목 {len(ts):,}개, 모델 {MODEL}")
    if not ts:
        sys.exit(0)
else:
    NEW = False
    print(f"{LANG} 제목 {len(ts):,}개, 모델 {MODEL}")
meta = j.load(MODEL)[2]
heads = range(len(meta["heads"])) if meta["kind"] == "cmsm_en" else [None]
lvl, opin = j.level_only(ts), j.soften(ts, np.zeros(len(ts)))[1]
mkt = np.array([bool(ix.MARKET[LANG].search(t)) for t in ts])
low = [j.clean(t).lower() for t in ts]
cause = np.array([bool(j.BG_CAUSE.search(t) or j.QUOTED_CAUSE.search(t)) for t in low]) if LANG == "ko" else np.zeros(len(ts), bool)
Pt, mt = j.predict(TOPIC, ts)
topic = [j.OLD_TOPIC.get(mt["labels"][k], mt["labels"][k]) for k in Pt.argmax(1)]
cols = []
for h in heads:
    raw = np.concatenate([j.score_of(*j.predict(MODEL, ts[i:i + 5000], *([h] if h is not None else []))) for i in range(0, len(ts), 5000)])
    g, changed = j.guard(ts, raw, meta.get("guard"), meta.get("guard_lexicon"))
    fin = j.soften(ts, g)[0]
    if LANG == "ko":  # 환율 기억: 애매한 환율 제목만, 발행일 전날까지의 원·달러와 견줌 (fxmemory.py, docs/실험기록.md 2026-11-02)
        import fxmemory
        n = 0
        for i, t in enumerate(ts):
            a = fxmemory.adjust(t, day[t], fin[i], lvl[i], opin[i])
            if a:
                fin[i], lvl[i] = a; n += 1
        print(f"  환율 기억으로 점수 준 제목 {n:,}개")
    cols.append((raw, g, fin, changed))
    print(f"  관점 {h}: 안전장치로 0이 된 제목 {changed.sum():,}개")
names = [f"{k}{'_' + meta['heads'][h] if h is not None else ''}" for h in heads for k in ("모델", "안전장치뒤", "최종", "안전장치바뀜")]
out.parent.mkdir(exist_ok=True)
with open(out, "a" if NEW else "w", encoding="utf-8") as f:
    if not NEW:
        f.write("\t".join(names + ["전망표지", "수준만", "시황", "원인꼴", "주제", "제목"]) + "\n")
    for i, t in enumerate(ts):
        vals = [f"{c[0][i]:.3f}\t{c[1][i]:.3f}\t{c[2][i]:.3f}\t{int(c[3][i])}" for c in cols]
        f.write("\t".join(vals + [str(int(opin[i])), str(int(lvl[i])), str(int(mkt[i])), str(int(cause[i])), topic[i], t]) + "\n")
print("저장", out, "(덧붙임)" if NEW else "")
