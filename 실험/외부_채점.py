# 외부 자료 채점: v1(evaluate.scores)과 seqexp 모델(eval), 7단계 → 3분류
# 사용: python 실험/외부_채점.py <한 줄짜리 제목 파일이 든 폴더(one.txt)> … 주의: seqexp를 show=로 불러서 실험/결과/seq_…_eval_show.json을 덮어쓴다. 끝나면 git checkout으로 되돌릴 것.
import sys, re, json, numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
H = r"C:\english_folder\somethings"
S = sys.argv[1]
V = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
sets = {}
for f in ["en-fpb-ko.tsv", "korfin-asc_문장.tsv"]:
    rows = [l.split("\t") for l in open(f"{H}/data/외부/{f}", encoding="utf-8").read().splitlines()[1:]]
    sets[f] = (np.array([int(r[0]) for r in rows]), [r[-1] for r in rows])
import evaluate as e
models = {"v1": lambda ts: e.scores("ko_v1-감성", ts)}
sys.argv = [f"{H}/실험/seqexp.py", "cmsm", "d", "morph", "lexinit", "seqlr=0.001", "body", "tag=v2LZ1", "len=128", "eval", f"show={S}/one.txt"]
ns = {"__name__": "__main__", "__file__": sys.argv[0]}
try: exec(compile(open(sys.argv[0], encoding="utf-8").read(), sys.argv[0], "exec"), ns)
except SystemExit: pass
models["본문 분석기 v2LZ1"] = ns["scores"]
sys.stdout.reconfigure(encoding="utf-8")
end = re.compile(r"(다|요|음|됨|함|임)[.”’\"')\]]*$|[?!][”’\"')\]]*$")
def report(nm, y, s):
    p = np.sign(V[np.abs(s[:, None] - V).argmin(1)]).astype(int)
    f1 = []
    for c in (-1, 0, 1):
        tp = np.sum((p == c) & (y == c)); pr = tp / max(1, np.sum(p == c)); rc = tp / max(1, np.sum(y == c))
        f1.append(2 * pr * rc / max(1e-9, pr + rc))
    rec = {c: np.mean(p[y == c] == c) for c in (-1, 0, 1)}
    flip = np.mean((p * y) < 0)
    return f"  {nm}: 정확도 {np.mean(p == y):.1%} macro-F1 {np.mean(f1):.3f} 방향 틀림 {flip:.1%} | 재현율 부정 {rec[-1]:.0%} 중립 {rec[0]:.0%} 긍정 {rec[1]:.0%} | 예측 분포 부정 {np.mean(p==-1):.0%} 중립 {np.mean(p==0):.0%} 긍정 {np.mean(p==1):.0%}"
out = {}
for f, (y, ts) in sets.items():
    print(f"{f} {len(y)}개 (정답 분포 부정 {np.mean(y==-1):.0%} 중립 {np.mean(y==0):.0%} 긍정 {np.mean(y==1):.0%}, 전부 중립이라 해도 정확도 {np.mean(y==0):.1%})")
    for nm, fn in models.items():
        s = np.asarray(fn(ts)); out[f"{f}|{nm}"] = dict(zip(ts, map(float, s)))
        print(report(nm, y, s))
        if f.startswith("korfin"):
            k = np.array([bool(end.search(t)) for t in ts])
            print("     문장꼴", report("", y[k], s[k]).strip())
            print("     그 밖", report("", y[~k], s[~k]).strip())
json.dump(out, open(f"{H}/실험/결과/외부_점수.json", "w", encoding="utf-8"), ensure_ascii=False)
