# 데이터 양 곡선: distill.py와 같은 구조·설정, 선생 점수만, 학습 제목 1/8·1/4·1/2·전부 (docs/실험기록.md 2026-10-26)
import sys, json, zlib, unicodedata, os
import numpy as np
import torch
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
import evaluate as e
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
BUCKETS, NGRAM, H, CUTDAY = 16384, 3, 64, "2026-08-01"
LABELS = ["-2", "-1", "-0.5", "0", "0.5", "1", "2"]
V = np.array(LABELS, float)
T_DIR = j.HOME / "labeled" / "시험"
TESTF = ["ko_무작위300.tsv", "ko_무작위400.tsv", "ko_시험563.tsv", "ko_오늘60.tsv", "ko_최종500.tsv"]
tests = {l.split("\t")[-1] for f in TESTF for l in (T_DIR / f).read_text(encoding="utf-8").splitlines()[1:]}
dev = [l.split("\t") for f in ["ko_시험563.tsv", "ko_오늘60.tsv"] for l in (T_DIR / f).read_text(encoding="utf-8").splitlines()[1:]]

first = {}
for l in open(j.HOME / "headlines" / "ko_daily.tsv", encoding="utf-8"):
    p = l.split("\t")
    if p[0] and p[2] not in first:
        first[p[2]] = p[0][:10]
teacher = {t.rstrip("\n"): float(s) for s, t in (l.split("\t", 1) for l in open(j.HOME / "data" / "ko_scores_ko-감성-미세.tsv", encoding="utf-8"))}
pool = [t for t, d in first.items() if d < CUTDAY and t in teacher and t not in tests]
held = [t for t, d in first.items() if d >= CUTDAY and t in teacher and t not in tests]
rng0 = np.random.default_rng(0)
pool = [pool[i] for i in rng0.permutation(len(pool))]
print(f"풀 {len(pool)}개, 따로 둔 제목 {len(held)}개")

states = [j.clean(t) for t in pool]
lex = j.load_lexicon(j.LEXICON)
PH_all = torch.tensor(j.phrase_features(states, lex, j.HINT))
feats = []
for t in states:
    t = f" {unicodedata.normalize('NFC', t.lower())} "
    c = {}
    for n in range(1, NGRAM + 1):
        for i in range(len(t) - n + 1):
            k = zlib.crc32(t[i:i + n].encode()) % BUCKETS
            c[k] = c.get(k, 0) + 1
    norm = np.sqrt(sum(v * v for v in c.values()))
    feats.append((list(c), [v / norm for v in c.values()]))
Y_all = np.abs(np.array([teacher[t] for t in pool])[:, None] - V).argmin(1)


def train(n, name):
    torch.manual_seed(0)
    idx, wts, offs = [], [], []
    for a, b in feats[:n]:
        offs.append(len(idx)); idx += a; wts += b
    idx, wts, offs = torch.tensor(idx), torch.tensor(wts, dtype=torch.float32), torch.tensor(offs)
    ends = torch.cat([offs[1:], torch.tensor([len(idx)])])
    PH, Y = PH_all[:n], Y_all[:n]
    F = BUCKETS + PH.shape[1]
    emb = torch.nn.EmbeddingBag(BUCKETS, H, mode="sum"); lin = torch.nn.Linear(PH.shape[1], H); out = torch.nn.Linear(H, 7)
    with torch.no_grad():
        emb.weight.normal_(0, 1 / np.sqrt(F)); lin.weight.normal_(0, 1 / np.sqrt(F)); lin.bias.zero_()
    params = [*emb.parameters(), *lin.parameters(), *out.parameters()]

    def logits(rows):
        rows = torch.as_tensor(rows)
        s, e_ = offs[rows], ends[rows]
        ii = torch.cat([torch.arange(a, b) for a, b in zip(s.tolist(), e_.tolist())])
        o = torch.cat([torch.tensor([0]), torch.cumsum(e_ - s, 0)[:-1]])
        return out(torch.tanh(emb(idx[ii], o, per_sample_weights=wts[ii]) + lin(PH[rows])))

    rng = np.random.default_rng(0)
    perm = rng.permutation(n); va, tr = perm[:n // 10], perm[n // 10:]
    Yt = torch.tensor(Y); W = torch.where(torch.tensor(np.abs(V[Y]) >= 1), j.NONZERO, 1.0)
    opt = torch.optim.Adam(params, lr=3e-3)
    lossf = lambda r: (torch.nn.functional.cross_entropy(logits(r), Yt[r], reduction="none") * W[r]).sum() / W[r].sum()
    best, best_state, bad = np.inf, None, 0
    for ep in range(30):
        rng.shuffle(tr)
        for k in range(0, len(tr), 256):
            opt.zero_grad(); lossf(tr[k:k + 256]).backward(); opt.step()
        with torch.no_grad():
            vl = float(lossf(va))
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
    ev, yt = j.softmax(zva / T) @ V, np.array([teacher[pool[i]] for i in va])
    stretch = float(np.clip((ev @ yt) / max(ev @ ev, 1e-9), 1.0, 4.0))
    meta = {"kind": "text", "buckets": BUCKETS, "ngram": NGRAM, "lexicon": lex, "hint": j.HINT, "phrase": True,
            "stretch": stretch, "labels": LABELS, "numeric": True, "guard": 1.0}
    W1 = torch.cat([emb.weight, lin.weight.T]).detach().numpy().astype(np.float64)
    np.savez(j.HOME / "models" / f"{name}.npz", W1=W1, b1=lin.bias.detach().numpy().astype(np.float64),
             W2=out.weight.T.detach().numpy().astype(np.float64), b2=out.bias.detach().numpy().astype(np.float64),
             T=T, meta=json.dumps(meta, ensure_ascii=False))
    return ep + 1


cell = lambda s: np.where(s >= 0.9, 1, np.where(s <= -0.9, -1, 0))
ht = np.array([teacher[t] for t in held])
yd = np.array([float(r[0]) for r in dev])
for frac in (8, 4, 2, 1):
    n, name = len(pool) // frac, f"_curve{frac}-감성"
    eps = train(n, name)
    P, meta = j.predict(name, held)
    hs = j.score_of(P, meta)
    m = e.screen(e.scores(name, [r[-1] for r in dev]), yd)
    print(f"1/{frac}: 학습 {n}개 ({eps}에폭) | 선생과 같은 칸 {np.mean(cell(hs) == cell(ht)):.2%}, 평균 차이 {np.abs(hs - ht).mean():.3f}"
          f" | 개발용622 맞음 {m['맞음']:.1%} 헛경보 {m['헛경보']:.1%} 놓침 {m['놓침']:.1%} 방향 틀림 {m['방향 틀림']}")
    os.remove(j.HOME / "models" / f"{name}.npz")
