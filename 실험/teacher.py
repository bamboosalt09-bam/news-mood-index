# 2026-08-01 전 ko_daily 제목에 선생(원래 정밀도) 7단계 확률 → data/ko_probs_ko-감성-미세.tsv (이미 있는 제목은 건너뜀)
# 선생 점수 파일에 없는 제목은 같은 확률로 점수도 덧붙임 (기존 점수와 같은 계산)
import sys, time
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
j.QUANT = ()
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sp, pp = j.HOME / "data" / "ko_scores_ko-감성-미세.tsv", j.HOME / "data" / "ko_probs_ko-감성-미세.tsv"
scores = {t: float(s) for s, t in (l.split("\t", 1) for l in sp.read_text(encoding="utf-8").splitlines())}
have = {l.rsplit("\t", 1)[1] for l in pp.read_text(encoding="utf-8").splitlines()} if pp.exists() else set()
first = {}
for l in open(j.HOME / "headlines" / "ko_daily.tsv", encoding="utf-8"):
    p = l.split("\t")
    if p[0] and p[2] not in first:
        first[p[2]] = p[0][:10]
todo = sorted({t for t, d in first.items() if d < "2026-08-01"} - have, key=len)
print(f"확률 계산할 제목 {len(todo)}개 (그중 점수도 없는 것 {sum(t not in scores for t in todo)}개)")
t0, diff = time.time(), []
for i in range(0, len(todo), 2000):
    part = todo[i:i + 2000]
    P, meta = j.predict("ko-감성-미세", part)
    s = j.score_of(P, meta)
    diff += [abs(a - scores[t]) for t, a in zip(part, s) if t in scores]
    with open(pp, "a", encoding="utf-8") as f:
        f.writelines("\t".join(f"{x:.4f}" for x in row) + f"\t{t}\n" for t, row in zip(part, P))
    with open(sp, "a", encoding="utf-8") as f:
        f.writelines(f"{a:.3f}\t{t}\n" for t, a in zip(part, s) if t not in scores)
    print(f"  {i + len(part)}/{len(todo)} ({time.time() - t0:.0f}초), 기존 점수와 차이 최대 {max(diff, default=0):.3f}")
