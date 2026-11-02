# 영어 새 모델(미국 칸)과 외부 영어 금융 감성 모델 견주기: 3분류 정답률·속도. 실험기록 2026-10-28
# python 실험/견주기_영어.py <새 모델 점수 json(en_seqexp show=… out=…)>
import sys, os, time, json, statistics, numpy as np, torch
H = "C:/english_folder/somethings"; sys.path.insert(0, H)
torch.set_num_threads(max(1, os.cpu_count() // 2))
V = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
three = lambda s: np.sign(V[np.abs(np.asarray(s)[:, None] - V).argmin(1)]).astype(int)
def load(p, col=0):
    rows = [l.split("\t") for l in open(p, encoding="utf-8").read().splitlines()[1:]]
    return np.array([int(np.sign(float(r[col]))) for r in rows]), [r[-1] for r in rows]
sets = {"영어 새 시험지(미국 칸)": load(f"{H}/labeled/시험/en_새시험1500.tsv"), "en-fpb": load(f"{H}/data/외부/en-fpb.tsv")}
new = json.load(open(sys.argv[1], encoding="utf-8"))
import evaluate as e
from transformers import AutoTokenizer, AutoModelForSequenceClassification
def hf(path):
    tok = AutoTokenizer.from_pretrained(path); m = AutoModelForSequenceClassification.from_pretrained(path).eval()
    sgn = np.array([{"positive": 1, "neutral": 0, "negative": -1}[m.config.id2label[i].lower()] for i in range(m.config.num_labels)])
    def f(ts, bs=32):
        out = []
        with torch.no_grad():
            for i in range(0, len(ts), bs):
                out.append(sgn[m(**tok(ts[i:i + bs], padding=True, truncation=True, max_length=128, return_tensors="pt")).logits.argmax(1).numpy()])
        return np.concatenate(out)
    return f
models = {"새 모델 미국 칸": lambda ts: three([new[t]["미국"] for t in ts]),
          "en-감성-미세 (지금)": lambda ts: three(e.scores("en-감성-미세", ts)),
          "ProsusAI/finbert": hf(f"{H}/models/외부/finbert"),
          "distilroberta 금융": hf(f"{H}/models/외부/distilroberta-fin")}
sys.stdout.reconfigure(encoding="utf-8")
for sn, (y, ts) in sets.items():
    print(f"{sn} {len(y)}개 (부정/중립/긍정 {np.mean(y==-1):.0%}/{np.mean(y==0):.0%}/{np.mean(y==1):.0%})")
    for mn, f in models.items():
        p = f(ts); f1 = []
        for c in (-1, 0, 1):
            tp = np.sum((p == c) & (y == c)); pr = tp / max(1, np.sum(p == c)); rc = tp / max(1, np.sum(y == c)); f1.append(2 * pr * rc / max(1e-9, pr + rc))
        r = [np.mean(p[y == c] == c) for c in (-1, 0, 1)]
        print(f"  {mn:20} 정확도 {np.mean(p == y):.1%} F1 {np.mean(f1):.3f} 방향 틀림 {np.mean(p * y < 0):.1%} 재현율 {r[0]:.0%}/{r[1]:.0%}/{r[2]:.0%} 예측 중립 {np.mean(p == 0):.0%}")
ts = sets["영어 새 시험지(미국 칸)"][1]
for mn, f in list(models.items())[1:]:
    a = time.perf_counter(); f(ts); bt = (time.perf_counter() - a) / len(ts) * 1000
    one = []
    for t in ts[:100]:
        a = time.perf_counter(); f([t]); one.append((time.perf_counter() - a) * 1000)
    print(f"속도 {mn}: 묶음 {bt:.2f}ms/제목 (초당 {1000/bt:,.0f}), 하나씩 중앙값 {statistics.median(one):.1f}ms")
