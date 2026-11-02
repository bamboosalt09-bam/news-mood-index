# 작은 모델 비교: 따로 둔 8월 이후 제목에서 선생과 같은 칸, 무작위700, 개발용622, 합 1,322, 기본 행동
# python cmp.py ko_distill_ai9k3-감성 ko_distill_big-감성 ...
import sys
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
import evaluate as e
sys.stdout.reconfigure(encoding="utf-8")
T = j.HOME / "labeled" / "시험"
read = lambda fs: [l.split("\t") for f in fs for l in (T / f).read_text(encoding="utf-8").splitlines()[1:]]
sets = {"무작위700": read(["ko_무작위300.tsv", "ko_무작위400.tsv"]), "개발용622": read(["ko_시험563.tsv", "ko_오늘60.tsv"])}
sets["합1322"] = sets["무작위700"] + sets["개발용622"]
tests = {r[-1] for f in T.glob("ko_*.tsv") for r in read([f.name])}
first = {}
for l in open(j.HOME / "headlines" / "ko_daily.tsv", encoding="utf-8"):
    p = l.split("\t")
    if p[0] and p[2] not in first:
        first[p[2]] = p[0][:10]
teacher = {t.rstrip("\n"): float(s) for s, t in (l.split("\t", 1) for l in open(j.HOME / "data" / "ko_scores_ko-감성-미세.tsv", encoding="utf-8"))}
held = [t for t, d in first.items() if d >= "2026-08-01" and t in teacher and t not in tests]
ht = np.array([teacher[t] for t in held])
cell = lambda s: np.where(s >= 0.9, 1, np.where(s <= -0.9, -1, 0))
for name in sys.argv[1:]:
    P, meta = j.predict(name, held)
    hs = j.score_of(P, meta)
    print(f"\n[{name}] 따로 둔 {len(held)}개: 선생과 같은 칸 {np.mean(cell(hs) == cell(ht)):.2%}, 평균 차이 {np.abs(hs - ht).mean():.3f}")
    for k, rows in sets.items():
        y = np.array([float(r[0]) for r in rows]); s = e.scores(name, [r[-1] for r in rows]); m = e.screen(s, y)
        print(f"  {k:9} 맞음 {m['맞음']:.1%}  헛경보 {m['헛경보']:.1%}  놓침 {m['놓침']:.1%}  방향 틀림 {m['방향 틀림']}  "
              f"(중립인데 반대 {m['중립인데 반대로']})  7단계 일치 {np.mean(np.abs(np.array(meta['labels'], float)[np.abs(s[:, None] - np.array(meta['labels'], float)).argmin(1)] - y) < 1e-9):.1%}")
    f = e.behavior(name)
    print(f"  기본 행동 {len(e.BEHAVIOR) - len(f)}/{len(e.BEHAVIOR)}" + "".join(f"\n    실패: {x}" for x in f))
