"""작은 모델 증류: 지수 산출 때 쌓인 선생 점수(data/ko_scores_ko-감성-미세.tsv)를 7단계로 반올림해 라벨로 쓰고,
글자 조각 + 사전 특징 모델(약 105만 매개변수)을 torch로 학습해 models/ko_distill-감성.npz(judge의 numpy 형식)로 저장.

python distill.py   # 2026-08-01 전 제목으로 학습 (이후 3개월은 비교용, docs/실험기록.md)
python distill.py ko_distill_ai-감성 labeled/ko_ai_opus.tsv 3  # AI 라벨(감성<TAB>제목)을 더함: 같은 제목은 AI 라벨로, 손실 가중치 3배(기본)
python distill.py ko_distill_b-감성 labeled/ko_ai_opus.tsv 3 labeled/ko_more8.tsv 5 soft
    # 사람 라벨도 더함(우선순위 사람 > AI > 선생, 가중치 5배). soft: 선생 |점수| 0.4~0.9는 선생 7단계 확률을 그대로 배움
    # (확률은 data/ko_probs_ko-감성-미세.tsv, docs/실험기록.md 2026-10-26 변형 B)
python distill.py ko_distill_c-감성 labeled/ko_ai_opus.tsv 3 labeled/ko_more8.tsv 5 mid=labeled/ko_ai_haiku_mid.tsv
    # 변형 C: 선생 |점수| 0.4~0.9 제목은 이 파일(감성<TAB>제목)의 라벨로
"""
import os, sys, json, time, zlib, unicodedata, collections
import numpy as np
import torch
import judge as j
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
torch.manual_seed(0)
torch.set_num_threads(max(1, os.cpu_count() // 2))  # CPU 절반은 사용자 몫으로 남김
BUCKETS, NGRAM, H, CUTDAY = 16384, 3, 64, "2026-08-01"
AI_W = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0  # AI 라벨 손실 가중치
NAME = sys.argv[1] if len(sys.argv) > 1 else "ko_distill-감성"
labels = lambda k: {p[-1]: float(p[0]) for p in (l.split("\t") for l in open(sys.argv[k], encoding="utf-8").read().splitlines()[1:])} if len(sys.argv) > k else {}
AI, HUM = labels(2), labels(4)  # 라벨 파일: 첫 열 감성, 마지막 열 제목
HUM_W, SOFT = (float(sys.argv[5]) if len(sys.argv) > 5 else 5.0), "soft" in sys.argv[6:]
MID = next((a[4:] for a in sys.argv[6:] if a.startswith("mid=")), None)  # 중간 구간을 이 파일 라벨로 (감성<TAB>제목, 변형 C)
AI = {t: v for t, v in AI.items() if t not in HUM}
LABELS = ["-2", "-1", "-0.5", "0", "0.5", "1", "2"]
V = np.array(LABELS, float)

first = {}
for l in open(j.HOME / "headlines" / "ko_daily.tsv", encoding="utf-8"):
    p = l.split("\t")
    if p[0] and p[2] not in first:
        first[p[2]] = p[0][:10]
teacher = {t.rstrip("\n"): float(s) for s, t in (l.split("\t", 1) for l in open(j.HOME / "data" / "ko_scores_ko-감성-미세.tsv", encoding="utf-8"))}
titles = [t for t, d in first.items() if d < CUTDAY and t in teacher and t not in AI and t not in HUM] + list(AI) + list(HUM)
# 시험지 제목과 거의 같은 제목(3글자 조각 자카드 ≥ 0.5)은 학습에서 뺌 (2026-10-26, docs/실험기록.md)
tri = lambda t: {t[i:i + 3] for i in range(len(t) - 2)} if len(t := "".join(t.lower().split())) > 2 else {t}
TEST = [tri(l.split("\t")[-1]) for f in (j.HOME / "labeled" / "시험").glob("ko_*.tsv") for l in f.read_text(encoding="utf-8").splitlines()[1:]]
inv = collections.defaultdict(list)
for k, s in enumerate(TEST):
    for g in s:
        inv[g].append(k)
def near_test(t):
    a = tri(t)
    return any(n / (len(a) + len(TEST[k]) - n) >= 0.5 for k, n in collections.Counter(k for g in a for k in inv.get(g, ())).items())
n0 = len(titles); titles = [t for t in titles if not near_test(t)]
print(f"시험지와 같거나 거의 같은 제목 {n0 - len(titles)}개 뺌")
target = lambda t: HUM.get(t, AI.get(t, teacher.get(t)))
Y = np.abs(np.array([target(t) for t in titles])[:, None] - V).argmin(1)
from_ai, from_hum = np.array([t in AI for t in titles]), np.array([t in HUM for t in titles])
print(f"AI 라벨 {from_ai.sum()}개 (가중치 {AI_W}배), 사람 라벨 {from_hum.sum()}개 (가중치 {HUM_W}배)")
Q = np.eye(7, dtype=np.float32)[Y]  # 정답 분포: 기본은 반올림한 한 등급
mid = [i for i, t in enumerate(titles) if not from_ai[i] and not from_hum[i] and 0.4 <= abs(teacher[t]) < 0.9]
if MID:
    ml = {p[-1]: float(p[0]) for p in (l.rstrip("\n").split("\t") for l in open(MID, encoding="utf-8").read().splitlines()[1:])}
    miss = [i for i in mid if titles[i] not in ml]
    assert not miss, f"중간 구간인데 {MID} 라벨 없는 제목 {len(miss)}개"
    Y[mid] = np.abs(np.array([ml[titles[i]] for i in mid])[:, None] - V).argmin(1)
    Q = np.eye(7, dtype=np.float32)[Y]
    print(f"중간 구간 {MID} 라벨로 배우는 제목 {len(mid)}개")
elif SOFT:
    probs = {p[-1]: [float(x) for x in p[:-1]] for p in (l.rstrip("\n").split("\t") for l in open(j.HOME / "data" / "ko_probs_ko-감성-미세.tsv", encoding="utf-8"))}
    miss = [i for i in mid if titles[i] not in probs]
    assert not miss, f"선생 확률 없는 중간 구간 제목 {len(miss)}개"
    Q[mid] = [probs[titles[i]] for i in mid]
    print(f"중간 구간 선생 확률로 배우는 제목 {len(mid)}개")
print(f"학습용 제목 {len(titles)}개, 라벨 분포", dict(zip(LABELS, np.bincount(Y, minlength=7))))

states = [j.clean(t) for t in titles]  # predict()와 같게
lex = j.load_lexicon(j.LEXICON)
t0 = time.time()
PH = torch.tensor(j.phrase_features(states, lex, j.HINT))
idx, wts, offs = [], [], []
for t in states:  # judge.text_features와 같은 조각·같은 정규화
    t = f" {unicodedata.normalize('NFC', t.lower())} "
    c = {}
    for n in range(1, NGRAM + 1):
        for i in range(len(t) - n + 1):
            k = zlib.crc32(t[i:i + n].encode()) % BUCKETS
            c[k] = c.get(k, 0) + 1
    norm = np.sqrt(sum(v * v for v in c.values()))
    offs.append(len(idx)); idx += list(c); wts += [v / norm for v in c.values()]
print(f"특징 {time.time() - t0:.0f}초")
idx, wts, offs = torch.tensor(idx), torch.tensor(wts, dtype=torch.float32), torch.tensor(offs)
ends = torch.cat([offs[1:], torch.tensor([len(idx)])])

F = BUCKETS + PH.shape[1]
emb = torch.nn.EmbeddingBag(BUCKETS, H, mode="sum")
lin = torch.nn.Linear(PH.shape[1], H)
out = torch.nn.Linear(H, len(LABELS))
with torch.no_grad():
    emb.weight.normal_(0, 1 / np.sqrt(F)); lin.weight.normal_(0, 1 / np.sqrt(F)); lin.bias.zero_()
params = [*emb.parameters(), *lin.parameters(), *out.parameters()]
print(f"매개변수 {sum(p.numel() for p in params):,}개")

def batch(rows):
    rows = torch.as_tensor(rows)
    s, e = offs[rows], ends[rows]
    ii = torch.cat([torch.arange(a, b) for a, b in zip(s.tolist(), e.tolist())])
    o = torch.cat([torch.tensor([0]), torch.cumsum(e - s, 0)[:-1]])
    return torch.tanh(emb(idx[ii], o, per_sample_weights=wts[ii]) + lin(PH[rows]))

def logits(rows):
    return out(batch(rows))

rng = np.random.default_rng(0)
perm = rng.permutation(len(Y)); va, tr = perm[:len(Y) // 10], perm[len(Y) // 10:]
Qt = torch.tensor(Q)
W = (torch.where(torch.tensor(np.abs(V[Y]) >= 1), j.NONZERO, 1.0) * torch.where(torch.tensor(from_ai), AI_W, 1.0)
     * torch.where(torch.tensor(from_hum), HUM_W, 1.0))
opt = torch.optim.Adam(params, lr=3e-3)
lossf = lambda r: (-(Qt[r] * torch.log_softmax(logits(r), 1)).sum(1) * W[r]).sum() / W[r].sum()  # 한 등급이면 교차 엔트로피와 같음
best, best_state, bad = np.inf, None, 0
for ep in range(30):
    rng.shuffle(tr)
    for k in range(0, len(tr), 256):
        opt.zero_grad(); lossf(tr[k:k + 256]).backward(); opt.step()
    with torch.no_grad():
        vl = float(lossf(va))
    print(f"  에폭 {ep + 1}: 검증 손실 {vl:.4f}")
    if vl < best:
        best, bad, best_state = vl, 0, [p.detach().clone() for p in params]
    else:
        bad += 1
        if bad >= 2:
            break
with torch.no_grad():
    for p, b in zip(params, best_state):
        p.copy_(b)
    zva = logits(va).numpy()
nll = lambda z: -np.log(j.softmax(z)[np.arange(len(va)), Y[va]] + 1e-12).mean()
T = float(min(np.exp(np.linspace(-2, 2, 81)), key=lambda T: nll(zva / T)))
ev, yt = j.softmax(zva / T) @ V, np.array([target(titles[i]) for i in va])
stretch = float(np.clip((ev @ yt) / max(ev @ ev, 1e-9), 1.0, 4.0))
print(f"온도 {T:.2f}, 배율 {stretch:.2f}, 검증 점수 오차(선생 점수와) {np.abs(np.clip(stretch * ev, -2, 2) - yt).mean():.3f}")
meta = {"kind": "text", "buckets": BUCKETS, "ngram": NGRAM, "lexicon": lex, "hint": j.HINT, "phrase": True,
        "stretch": stretch, "labels": LABELS, "numeric": True, "guard": 1.0, "neutral": [0.9, 0.85]}  # 안전장치·중립 경계는 선생과 같게
W1 = torch.cat([emb.weight, lin.weight.T]).detach().numpy().astype(np.float64)
np.savez(j.HOME / "models" / f"{NAME}.npz", W1=W1, b1=lin.bias.detach().numpy().astype(np.float64),
         W2=out.weight.T.detach().numpy().astype(np.float64), b2=out.bias.detach().numpy().astype(np.float64),
         T=T, meta=json.dumps(meta, ensure_ascii=False))
# numpy 판정 경로와 torch 학습 경로가 같은 답을 내는지 검산
P = j.predict(NAME, titles[:200])[0]
with torch.no_grad():
    P2 = j.softmax(logits(np.arange(200)).numpy() / T)
assert np.abs(P - P2).max() < 1e-4, np.abs(P - P2).max()
print(f"저장: models/{NAME}.npz (numpy 판정과 일치 확인)")
