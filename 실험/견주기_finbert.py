# KR-FinBert-SC와 우리 모델 견주기: 3분류 정답률·속도. 실험기록 2026-10-28
# 사용: python 실험/견주기_finbert.py <한 줄짜리 제목 파일이 든 폴더(one.txt)> … 주의: seqexp를 show=로 불러서 실험/결과/seq_…_eval_show.json을 덮어쓴다. 끝나면 git checkout으로 되돌릴 것.
import sys, os, re, json, time, statistics, numpy as np, torch
H = "C:/english_folder/somethings"; S = sys.argv[1]
sys.path.insert(0, H)
torch.set_num_threads(max(1, os.cpu_count() // 2))
V = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
def load(path, ylab=lambda r: int(np.sign(float(r[0])))):
    rows = [l.split("\t") for l in open(path, encoding="utf-8").read().splitlines()[1:]]
    return np.array([ylab(r) for r in rows]), [r[-1] for r in rows]
sets = {"새 시험지(제목)": load(f"{H}/labeled/시험/ko_새시험1500.tsv"), "과거 시험지(제목)": load(f"{H}/labeled/시험/ko_과거시험1500.tsv"),
        "본문 시험지": load(f"{H}/labeled/시험/ko_본문시험1500.tsv"), "en-fpb-ko": load(f"{H}/data/외부/en-fpb-ko.tsv"), "KorFin-ASC": load(f"{H}/data/외부/korfin-asc_문장.tsv")}
from transformers import AutoTokenizer, AutoModelForSequenceClassification
P = f"{H}/models/외부/KR-FinBert-SC"
tok = AutoTokenizer.from_pretrained(P); fb = AutoModelForSequenceClassification.from_pretrained(P).eval()
def finbert(ts, bs=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(ts), bs):
            x = tok(ts[i:i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt")
            out.append(fb(**x).logits.argmax(1).numpy() - 1)  # 0 부정 1 중립 2 긍정 → -1 0 1
    return np.concatenate(out)
import evaluate as e
three = lambda s: np.sign(V[np.abs(np.asarray(s)[:, None] - V).argmin(1)]).astype(int)
models = {"KR-FinBert-SC": finbert, "v1": lambda ts: three(e.scores("ko_v1-감성", ts))}
sys.argv = [f"{H}/실험/seqexp.py", "cmsm", "d", "morph", "lexinit", "seqlr=0.001", "body", "tag=v2LZ1", "len=128", "eval", f"show={S}/one.txt"]
ns = {"__name__": "__main__", "__file__": sys.argv[0]}
try: exec(compile(open(sys.argv[0], encoding="utf-8").read(), sys.argv[0], "exec"), ns)
except SystemExit: pass
models["본문 분석기 v2LZ1"] = lambda ts: three(ns["scores"](ts))
sys.stdout.reconfigure(encoding="utf-8")
def rep(y, p):
    f1 = []
    for c in (-1, 0, 1):
        tp = np.sum((p == c) & (y == c)); pr = tp / max(1, np.sum(p == c)); rc = tp / max(1, np.sum(y == c)); f1.append(2 * pr * rc / max(1e-9, pr + rc))
    r = [np.mean(p[y == c] == c) for c in (-1, 0, 1)]
    return f"정확도 {np.mean(p == y):.1%} F1 {np.mean(f1):.3f} 방향 틀림 {np.mean(p * y < 0):.1%} 재현율 {r[0]:.0%}/{r[1]:.0%}/{r[2]:.0%} 예측 중립 {np.mean(p == 0):.0%}"
res = {}
for sn, (y, ts) in sets.items():
    print(f"{sn} {len(y)}개 (부정/중립/긍정 {np.mean(y==-1):.0%}/{np.mean(y==0):.0%}/{np.mean(y==1):.0%})")
    for mn, f in models.items():
        p = f(ts); res[f"{sn}|{mn}"] = p.tolist(); print(f"  {mn:14} {rep(y, p)}")
json.dump(res, open(f"{H}/실험/결과/외부_견주기.json", "w", encoding="utf-8"))
# 속도: 본문 시험지와 새 시험지
for sn in ["새 시험지(제목)", "본문 시험지"]:
    ts = sets[sn][1]
    for mn, f in models.items():
        a = time.perf_counter(); f(ts); bt = (time.perf_counter() - a) / len(ts) * 1000
        one = []
        for t in ts[:100]:
            a = time.perf_counter(); f([t]); one.append((time.perf_counter() - a) * 1000)
        print(f"속도 {sn} {mn}: 묶음 {bt:.2f}ms/문장 (초당 {1000/bt:,.0f}), 하나씩 중앙값 {statistics.median(one):.1f}ms")
import psutil; print(f"이 프로세스 최대 메모리(세 모델 함께) {psutil.Process().memory_info().peak_wset / 2**30:.2f}GB")
