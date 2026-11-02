# KR-FinBert-SC로 ko_daily 제목 전부 판정 → index.py와 같은 일별 지수 → v1 지수와 코스피 상관 견주기
import sys, os, numpy as np, torch
H = "C:/english_folder/somethings"; sys.path.insert(0, H); os.chdir(H)
torch.set_num_threads(max(1, os.cpu_count() // 2))
import index as ix, judge as j
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
market = ix.kospi(); trading = sorted(market)
rows = [l.split("\t") for l in open(f"{H}/headlines/ko_daily.tsv", encoding="utf-8")]
rows = [(ix.market_day(r[0], trading, "ko"), r[2]) for r in rows if r[0]]
rows = [(d, t) for d, t in rows if d]
titles = sorted({t for _, t in rows}, key=len)
cp = f"{H}/data/ko_scores_KR-FinBert-SC.tsv"
cache = {}
if os.path.exists(cp):
    cache = {t: float(s) for s, t in (l.rstrip("\n").split("\t", 1) for l in open(cp, encoding="utf-8"))}
todo = [t for t in titles if t not in cache]
print(f"제목 {len(titles):,}개, 새로 판정 {len(todo):,}개")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
P = f"{H}/models/외부/KR-FinBert-SC"
tok = AutoTokenizer.from_pretrained(P); fb = AutoModelForSequenceClassification.from_pretrained(P).eval()
with torch.no_grad(), open(cp, "a", encoding="utf-8") as f:
    for i in range(0, len(todo), 64):
        part = todo[i:i + 64]
        pr = torch.softmax(fb(**tok(part, padding=True, truncation=True, max_length=96, return_tensors="pt")).logits, 1).numpy()
        for t, p in zip(part, pr[:, 2] - pr[:, 0]):
            cache[t] = float(p); f.write(f"{p:.3f}\t{t}\n")
        if i % 6400 == 0: print(f"  {i:,}/{len(todo):,}")
days = [d for d in trading if d >= min(d for d, _ in rows)]
M = np.array([market[d] for d in days])
def build(sc):
    by = {g: {} for g in ("전체", "시황", "재료")}
    for d, t in rows:
        v = sc.get(t, np.nan)
        if np.isnan(v): continue
        for g in by:
            if g == "전체" or (g == "시황") == bool(ix.MARKET["ko"].search(t)):
                by[g].setdefault(d, []).append(v)
    return {g: np.array([np.mean(by[g][d]) if len(by[g].get(d, [])) >= ix.MIN_PER_DAY else np.nan for d in days]) for g in by}
v1 = ix.scores("ko", "ko_v1-감성", titles)
I = {"v1": build(v1), "KR-FinBert-SC": build(cache)}
print(f"\n{'비교':44}" + "".join(f"{m + ' ' + g:>26}" for m in I for g in ("전체", "시황")))
for name, f in ix.tests_for("코스피", M):
    print(f"{name:44}" + "".join(f"{'%+.2f (p=%.3f, %d일)' % f(I[m][g]):>26}" for m in I for g in ("전체", "시황")))
a, b = I["v1"]["전체"], I["KR-FinBert-SC"]["전체"]; ok = ~(np.isnan(a) | np.isnan(b))
print(f"\n두 지수끼리 상관 (전체): {np.corrcoef(a[ok], b[ok])[0, 1]:+.2f}, {ok.sum()}일")
