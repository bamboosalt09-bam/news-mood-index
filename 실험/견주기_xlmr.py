# Hyeonseo/ko-finance_news_classifier (XLM-R) 5개 자료 3분류 + 속도
import sys, os, time, statistics, json, numpy as np, torch
H = "C:/english_folder/somethings"; sys.path.insert(0, H)
torch.set_num_threads(max(1, os.cpu_count() // 2))
def load(path):
    rows = [l.split("\t") for l in open(path, encoding="utf-8").read().splitlines()[1:]]
    return np.array([int(np.sign(float(r[0]))) for r in rows]), [r[-1] for r in rows]
sets = {"새 시험지(제목)": load(f"{H}/labeled/시험/ko_새시험1500.tsv"), "과거 시험지(제목)": load(f"{H}/labeled/시험/ko_과거시험1500.tsv"),
        "본문 시험지": load(f"{H}/labeled/시험/ko_본문시험1500.tsv"), "en-fpb-ko": load(f"{H}/data/외부/en-fpb-ko.tsv"), "KorFin-ASC": load(f"{H}/data/외부/korfin-asc_문장.tsv")}
from transformers import AutoTokenizer, AutoModelForSequenceClassification
P = f"{H}/models/외부/ko-finance_news_classifier"
tok = AutoTokenizer.from_pretrained(P); m = AutoModelForSequenceClassification.from_pretrained(P).eval()
print(m.config.id2label)
def f(ts, bs=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(ts), bs):
            out.append(m(**tok(ts[i:i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt")).logits.argmax(1).numpy() - 1)
    return np.concatenate(out)
sys.stdout.reconfigure(encoding="utf-8")
res = {}
for sn, (y, ts) in sets.items():
    p = f(ts); res[sn] = p.tolist()
    f1 = []
    for c in (-1, 0, 1):
        tp = np.sum((p == c) & (y == c)); pr = tp / max(1, np.sum(p == c)); rc = tp / max(1, np.sum(y == c)); f1.append(2 * pr * rc / max(1e-9, pr + rc))
    r = [np.mean(p[y == c] == c) for c in (-1, 0, 1)]
    print(f"{sn} {len(y)}: 정확도 {np.mean(p == y):.1%} F1 {np.mean(f1):.3f} 방향 틀림 {np.mean(p * y < 0):.1%} 재현율 {r[0]:.0%}/{r[1]:.0%}/{r[2]:.0%} 예측 중립 {np.mean(p == 0):.0%}")
json.dump(res, open(f"{H}/실험/결과/외부_견주기_xlmr.json", "w", encoding="utf-8"))
for sn in ["새 시험지(제목)", "본문 시험지"]:
    ts = sets[sn][1]; a = time.perf_counter(); f(ts); bt = (time.perf_counter() - a) / len(ts) * 1000
    one = []
    for t in ts[:100]:
        a = time.perf_counter(); f([t]); one.append((time.perf_counter() - a) * 1000)
    print(f"속도 {sn}: 묶음 {bt:.2f}ms/문장 (초당 {1000/bt:,.0f}), 하나씩 중앙값 {statistics.median(one):.1f}ms")
