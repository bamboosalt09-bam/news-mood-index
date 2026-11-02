# 본문 문장·제목 채점 속도 재기: seqexp 모델(eval)과 v1(evaluate.scores)
# 사용: python 실험/속도.py <한 줄짜리 제목 파일이 든 폴더(one.txt)> … 주의: seqexp를 show=로 불러서 실험/결과/seq_…_eval_show.json을 덮어쓴다. 끝나면 git checkout으로 되돌릴 것.
import sys, time, statistics, numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
S, tag, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
body = [l.split("\t")[-1] for l in open(r"C:\english_folder\somethings\labeled\시험\ko_본문시험1500.tsv", encoding="utf-8").read().splitlines()[1:]]
title = [l.split("\t")[-1] for l in open(r"C:\english_folder\somethings\labeled\시험\ko_새시험1500.tsv", encoding="utf-8").read().splitlines()[1:]]
if tag == "v1":
    import evaluate as e
    f = lambda ts: e.scores("ko_v1-감성", ts); f(title[:5])
else:
    sys.argv = [r"C:\english_folder\somethings\실험\seqexp.py", "cmsm", "d", "morph", "lexinit", "seqlr=0.001", "body", f"tag={tag}", "eval", f"show={S}/one.txt"] + extra
    ns = {"__name__": "__main__", "__file__": sys.argv[0]}
    try: exec(compile(open(sys.argv[0], encoding="utf-8").read(), sys.argv[0], "exec"), ns)
    except SystemExit: pass
    f = ns["scores"]
sys.stdout.reconfigure(encoding="utf-8")
for nm, ts in [("본문", body), ("제목", title)]:
    a = time.perf_counter(); f(ts); bt = (time.perf_counter() - a) / len(ts) * 1000
    one = []
    for t in ts[:200]:
        a = time.perf_counter(); f([t]); one.append((time.perf_counter() - a) * 1000)
    print(f"{tag} {nm}: 묶음 {bt:.2f}ms/문장 (초당 {1000/bt:,.0f}문장), 하나씩 중앙값 {statistics.median(one):.1f}ms")
