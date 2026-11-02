# 영어 작은 모델 (2026-10-28): v1(M2)과 같은 구조 = 글자 조각 + 사전 특징 + 낱말 행렬 곱(CMSM), 판정층 둘(미국·한국 관점, 한 몸 두 머리).
# 정답은 Haiku 두 칸 라벨(labeled/en_ai_haiku.tsv). 학습 뒤 검증 몫(10%)으로 재고, test를 주면 labeled/시험/en_새시험1500.tsv와 옛 영어 303도. 뒤처리는 evaluate.scores와 같게.
# python 실험/en_seqexp.py [tag=이름] [seed=0] [h=64] [d=16] [deep=0] [bi] [att] [zerow=1] [ow=2] [hw=1] [fast] [patience=2] [seqlr=0.001] [smooth=0.1] [pool=제목파일 [pre=에폭] [kd=확률.npy]] [test]
# python 실험/en_seqexp.py tag=이름 eval [test] [check=라벨파일] [edit=고친사전] [speed] [old] | [show=제목파일 [out=점수json]] | [probs=제목파일 out=확률.npy]
#   opus2: llm에 Opus 2차도. opus3: Opus 3차(과거 8,000)도. haiku3: llm에 과거 Haiku 라벨도. llm: 모음 Haiku 라벨 + Opus 라벨(0.75 Opus + 0.25 Haiku, 가중치 2배)도 학습, Opus 검증 1,000은 뺌. check: 그 라벨 파일(미국 한국 … 제목)로도 잼.
#   smooth: 정답 옆 칸에 나눠 주는 몫. pool: 라벨 없는 제목(낱말 사전에도 넣음). pre: pool로 낱말 행렬 먼저 학습(가린 낱말 맞히기).
#   kd: pool 줄마다 선생 확률 (n, 2, 7) → pool도 같이 학습 (증류).
#   → 모델 models/en_seq_<tag>.pt, 결과 실험/결과/en_seq_<tag>.json
import os, sys, json, time, zlib, re, unicodedata, collections
import numpy as np
import torch
from torch import nn
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
import evaluate as e
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
torch.set_num_threads(max(1, os.cpu_count() // 2))
A = sys.argv[1:]
arg = lambda k, d=None: next((a[len(k) + 1:] for a in A if a.startswith(k + "=")), d)
TAG, EVAL, TEST = arg("tag", "base"), "eval" in A, "test" in A
torch.manual_seed(int(arg("seed", 0)))
SEQLR, SMOOTH, PRE = float(arg("seqlr", 1e-3)), float(arg("smooth", 0)), int(arg("pre", 0))
SHOW, PROBS, POOL, KD, CHECK, LLM = arg("show"), arg("probs"), arg("pool"), arg("kd"), arg("check"), "llm" in A
EDIT = arg("edit")
LAB = j.HOME / "labeled"
tsv = lambda f: [l.split("	") for l in open(f, encoding="utf-8").read().splitlines()[1:] if l.strip()]
BUCKETS, NGRAM, L, CAP = 16384, 3, 40, 4.0
H, D, DEEP = int(arg("h", 64)), int(arg("d", 16)), int(arg("deep", 0))  # 은닉 폭, 낱말 행렬 크기, 은닉층 더 쌓을 수
BI, ATT = "bi" in A, "att" in A  # 합치기 전 깊이: 거꾸로도 읽는 행렬 곱, 트랜스포머 한 층
V = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
HEADS = ("미국", "한국")
NAME = j.HOME / "models" / f"en_seq_{TAG}.pt"
OUT = j.HOME / "실험" / "결과" / f"en_seq_{TAG}.json"
norm = lambda t: unicodedata.normalize("NFC", t.lower())
toks = lambda sts: [re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)?%?|[$%]", norm(t)) for t in sts]
lines = lambda f: [l.rstrip("\n").split("\t")[-1] for l in open(f, encoding="utf-8") if l.strip()]
j.LEXICON = arg("lex", j.LEXICON)  # lex=파일: 다른 사전판으로 (특징·안전장치 모두)
lex = j.load_lexicon(j.LEXICON)

if not EVAL:
    rows = [l.split("\t") for l in (j.HOME / "labeled" / "en_ai_haiku.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    titles = [r[-1] for r in rows]
    Y = np.stack([np.abs(np.array([float(r[k]) for r in rows])[:, None] - V).argmin(1) for k in (0, 1)], 1)  # (n, 2)
    pool = [j.clean(t) for t in lines(POOL)] if POOL else []
    states = [j.clean(t) for t in titles]
    HPOOL = ["en_ai_haiku_pool.tsv"] + (["en_ai_haiku_과거.tsv"] if "haiku3" in A else [])  # haiku3: 과거(2023~25) Haiku 라벨도 (2026-10-29)
    extra = [j.clean(r[-1]) for f in HPOOL for r in tsv(LAB / f)] if LLM else []  # llm: 낱말 사전에 모음 전부
    cnt = collections.Counter(w for ts in toks(states + pool + extra) for w in ts)
    VOCAB = {w: i + 2 for i, w in enumerate(sorted(w for w, n in cnt.items() if n >= 2))}  # 0 빈칸, 1 모름
    print(f"학습 제목 {len(titles):,}개, 라벨 없는 제목 {len(pool):,}개, 낱말 사전 {len(VOCAB):,}개")


def ids(sts):
    C = np.zeros((len(sts), L), np.int64)
    for r, ts in enumerate(toks(sts)):
        x = [VOCAB.get(w, 1) for w in ts][:L]
        C[r, :len(x)] = x
    return torch.tensor(C)


def featurize(sts):
    """글자 조각(희소), 사전 특징, 낱말 번호. seqexp.featurize와 같은 조각·정규화"""
    idx, wts, offs = [], [], []
    for t in sts:
        t2 = f" {norm(t)} "
        c = {}
        for n in range(1, NGRAM + 1):
            for i in range(len(t2) - n + 1):
                k = zlib.crc32(t2[i:i + n].encode()) % BUCKETS
                c[k] = c.get(k, 0) + 1
        nn_ = np.sqrt(sum(v * v for v in c.values()))
        offs.append(len(idx)); idx += list(c); wts += [v / nn_ for v in c.values()]
    PH = np.vstack([j.phrase_features(sts[i:i + 2000], lex, j.HINT) for i in range(0, len(sts), 2000)])
    return (torch.tensor(idx), torch.tensor(wts, dtype=torch.float32), torch.tensor(offs)), torch.tensor(PH), ids(sts)


class Net(nn.Module):
    def __init__(s, F, nv):
        super().__init__()
        s.emb, s.lin = nn.EmbeddingBag(BUCKETS, H, mode="sum"), nn.Linear(F, H)
        s.out = nn.ModuleList(nn.Linear(H, 7) for _ in HEADS)
        with torch.no_grad():
            s.emb.weight.normal_(0, 1 / np.sqrt(BUCKETS + F)); s.lin.weight.normal_(0, 1 / np.sqrt(BUCKETS + F)); s.lin.bias.zero_()
            s.A = nn.Embedding(nv, D * D); s.A.weight.normal_(0, 0.02); s.A.weight[0].zero_()  # 빈칸 = 단위 행렬
            s.v0 = nn.Parameter(torch.zeros(D)); s.v0[D - 1] = 1.0  # 마지막 칸 = 상수 1, 0번 칸 = 감성
        s.seq = nn.Linear(D, H)
        s.mid = nn.ModuleList(nn.Linear(H, H) for _ in range(DEEP))
        if BI:
            with torch.no_grad():
                s.A2 = nn.Embedding(nv, D * D); s.A2.weight.normal_(0, 0.02); s.A2.weight[0].zero_()
            s.seq2 = nn.Linear(D, H)
        if ATT:
            s.E, s.P = nn.Embedding(nv, H, padding_idx=0), nn.Embedding(L, H)
            with torch.no_grad():
                s.E.weight.normal_(0, 0.02); s.E.weight[0].zero_(); s.P.weight.normal_(0, 0.02)
            s.enc = nn.TransformerEncoderLayer(H, 4, 2 * H, batch_first=True)
            s.att = nn.Linear(H, H)

    def lexinit(s, vocab):
        """사전 낱말과 같은 낱말은 역할대로 시작 (seqexp lexinit과 같음)"""
        step = {w.strip(): v for k, v in j.STEP.items() for w in lex.get(k, [])}
        neg, past = {w.strip() for w in lex.get("부정", [])}, {w.strip() for w in lex.get("지난", [])}
        n = 0
        with torch.no_grad():
            for Aw in [s.A] + ([s.A2] if BI else []):
                for w, i in vocab.items():
                    Ai = Aw.weight[i].view(D, D)
                    if w in step: Ai[0, D - 1] += step[w]; n += 1
                    if w in neg: Ai[0, 0] += -2.0; n += 1
                    if w in past: Ai[0, 0] += -0.8; n += 1
        print(f"사전 출발점 낱말 {n}개")

    def state(s, C, rev=False):
        mask = C > 0
        C = C[:, :max(1, int(mask.sum(1).max()))]
        if rev:  # 거꾸로 읽기: 앞쪽 빈칸은 건너뛴다
            C = C.flip(1)
        mask = C > 0
        M = (s.A2 if rev else s.A)(C).view(len(C), C.shape[1], D, D) + torch.eye(D)
        st = s.v0.expand(len(C), D)
        for t in range(C.shape[1]):  # 읽는 순서대로: 상태 ← M(낱말) · 상태, 크기는 CAP을 넘지 않게
            new = torch.bmm(M[:, t], st.unsqueeze(2)).squeeze(2)
            new = new / (new.norm(dim=1, keepdim=True) / CAP).clamp(min=1.0)
            st = torch.where(mask[:, t:t + 1], new, st)
        return st

    def forward(s, bag, PH, C):
        (idx, wts, offs) = bag
        h = s.emb(idx, offs, per_sample_weights=wts) + s.lin(PH) + s.seq(s.state(C))
        if BI:
            h = h + s.seq2(s.state(C, rev=True))
        if ATT:
            C2 = C[:, :max(1, int((C > 0).sum(1).max()))]; m = C2 > 0
            x = s.enc(s.E(C2) + s.P.weight[:C2.shape[1]], src_key_padding_mask=~m)
            h = h + s.att((x * m.unsqueeze(2)).sum(1) / m.sum(1, keepdim=True).clamp(min=1))
        h = torch.tanh(h)
        for m in s.mid:
            h = torch.tanh(m(h))
        return torch.stack([o(h) for o in s.out], 1)  # (n, 2, 7)


def pretrain(net, sts, epochs):
    """낱말 행렬 사전학습: 제목마다 아는 낱말 하나를 '모름'(1번)으로 가리고, 행렬 곱 상태로 그 낱말을 맞힌다 (판정층·글자 조각은 안 씀)"""
    C = ids(sts)
    C = C[((C > 1).sum(1) >= 3)]
    head = nn.Sequential(nn.Linear(D, H), nn.Tanh(), nn.Linear(H, len(VOCAB) + 2))
    opt = torch.optim.Adam([net.A.weight, net.v0, *head.parameters()], lr=1e-3)
    g = torch.Generator().manual_seed(0)
    for ep in range(epochs):
        perm, tot, t1 = torch.randperm(len(C), generator=g), 0.0, time.time()
        for k in range(0, len(C), 256):
            c = C[perm[k:k + 256]].clone()
            known = (c > 1).float()
            pos = torch.multinomial(known, 1, generator=g).squeeze(1)
            r = torch.arange(len(c))
            tgt = c[r, pos].clone(); c[r, pos] = 1
            opt.zero_grad(); loss = nn.functional.cross_entropy(head(net.state(c)), tgt); loss.backward(); opt.step()
            tot += float(loss) * len(c)
        print(f"  사전학습 에폭 {ep + 1}: 가린 낱말 손실 {tot / len(C):.3f} ({time.time() - t1:.0f}초, 제목 {len(C):,}개)")


def take(feat, rows):
    (idx, wts, offs), PH, C = feat
    rows = torch.as_tensor(rows)
    ends = torch.cat([offs[1:], torch.tensor([len(idx)])])
    s_, e_ = offs[rows], ends[rows]
    ii = torch.cat([torch.arange(a, b) for a, b in zip(s_.tolist(), e_.tolist())])
    o = torch.cat([torch.tensor([0]), torch.cumsum(e_ - s_, 0)[:-1]])
    return (idx[ii], wts[ii], o), PH[rows], C[rows]


F = j.phrase_features(["x"], lex, j.HINT).shape[1]
if EVAL:
    saved = torch.load(NAME, weights_only=False)
    VOCAB, T, stretch = saved["vocab"], saved["T"], saved["stretch"]
    H, D, DEEP, BI, ATT = (list(saved.get("cfg", (64, 16, 0))) + [False, False])[:5]
    net = Net(F, len(VOCAB) + 2); net.load_state_dict(saved["state"]); net.eval()
else:
    net = Net(F, len(VOCAB) + 2)
    if PRE:
        pretrain(net, pool, PRE)
    net.lexinit(VOCAB)
    print(f"매개변수 {sum(p.numel() for p in net.parameters()):,}개")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Y)); va, tr = perm[:len(Y) // 10], perm[len(Y) // 10:]
    P = np.eye(7)[Y]  # (n, 2, 7) 정답 확률
    if SMOOTH:
        P = (1 - 2 * SMOOTH) * P + SMOOTH * np.roll(P, 1, 2) * (Y[..., None] < 6) + SMOOTH * np.roll(P, -1, 2) * (Y[..., None] > 0)
        P = P / P.sum(2, keepdims=True)  # 끝 칸은 한쪽만 나눠 줌
    W = np.where(np.abs(V[Y]) >= 1, j.NONZERO, 1.0)  # (n, 2)
    trs = states
    if KD:
        Pk = np.load(KD); assert Pk.shape == (len(pool), len(HEADS), 7)
        trs, P, W = states + pool, np.concatenate([P, Pk]), np.concatenate([W, np.ones((len(pool), len(HEADS)))])
        tr = np.concatenate([tr, np.arange(len(states), len(trs))])
    if LLM:  # 두 선생: 모음 Haiku 전부, Opus 붙은 제목은 0.75 Opus + 0.25 Haiku에 가중치 2배. Opus 검증 제목은 통째로 뺌
        hold = {r[-1] for r in tsv(LAB / "en_opus검증1000.tsv")}
        op = {r[-1]: r for r in tsv(LAB / "en_ai_opus_pool.tsv")}
        if "opus2" in A:  # Opus 2차 (모음 나머지 21,327개, 2026-10-29)
            op |= {r[-1]: r for r in tsv(LAB / "en_ai_opus_pool2.tsv")}
        if "opus3" in A:  # Opus 3차 (과거 Haiku 중 무작위 8,000개, 2026-10-31)
            op |= {r[-1]: r for r in tsv(LAB / "en_ai_opus_과거.tsv")}
        hr = [r for f in HPOOL for r in tsv(LAB / f) if r[-1] not in hold]
        lv = lambda r: np.abs(np.array([float(r[0]), float(r[1])])[:, None] - V).argmin(1)  # (2,) 칸 번호
        P2, W2 = [], []
        for r in hr:
            ph = np.eye(7)[lv(r)]
            if r[-1] in op:
                yo = lv(op[r[-1]]); P2.append(0.75 * np.eye(7)[yo] + 0.25 * ph); W2.append(float(arg("ow", 2)) * np.where(np.abs(V[yo]) >= 1, j.NONZERO, 1.0))
            else:  # ow=·hw=: Opus 섞은 제목·Haiku만 있는 제목 무게 (2026-10-31)
                P2.append(ph); W2.append(float(arg("hw", 1)) * np.where(np.abs(V[lv(r)]) >= 1, j.NONZERO, 1.0))
        n0 = len(trs)
        trs, P, W = trs + [j.clean(r[-1]) for r in hr], np.concatenate([P, P2]), np.concatenate([W * float(arg("hw", 1)), W2])
        tr = np.concatenate([tr, np.arange(n0, len(trs))])
        print(f"LLM 라벨 더함 {len(hr):,}개 (Opus 섞음 {sum(r[-1] in op for r in hr):,}개, 뺀 Opus 검증 {len(hold):,}개)")
    t0 = time.time()
    if "fast" in A:  # fast: 같은 제목·사전·낱말 사전이면 특징을 저장해 다시 씀 (2026-10-31)
        import hashlib
        key = hashlib.sha1((chr(10).join(trs) + open(j.HOME / j.LEXICON, encoding="utf-8").read() + str(sorted(VOCAB.items()))).encode()).hexdigest()[:16]
        cp = j.HOME / "data" / "cache" / f"en_feat_{key}.pt"
        if cp.exists():
            feat = torch.load(cp, weights_only=False)
        else:
            feat = featurize(trs); cp.parent.mkdir(exist_ok=True); torch.save(feat, cp)
    else:
        feat = featurize(trs)
    print(f"특징 {time.time() - t0:.0f}초")
    if arg("zerow"):  # zerow=K: 정답이 0(중립)인 칸의 손실 가중치를 K배 (2026-10-31 확인용)
        W = np.where(np.asarray(P).argmax(-1) == 3, float(arg("zerow")) * np.asarray(W), W)
    Pt, W = torch.tensor(P, dtype=torch.float32), torch.tensor(W, dtype=torch.float32)
    seqp = [p for n_, p in net.named_parameters() if n_.split(".")[0] in ("A", "v0", "seq", "A2", "seq2")]
    opt = torch.optim.Adam([{"params": [p for p in net.parameters() if all(p is not q for q in seqp)]}, {"params": seqp, "lr": SEQLR}], lr=3e-3)

    def lossf(r):
        z = net(*take(feat, r)); w = W[r]
        ce = -(Pt[r] * nn.functional.log_softmax(z, -1)).sum(-1)  # (b, 2)
        return (ce * w).sum() / w.sum()
    best, best_state, bad = np.inf, None, 0
    for ep in range(30):
        net.train(); rng.shuffle(tr); t1 = time.time()
        for k in range(0, len(tr), 256):
            opt.zero_grad(); lossf(tr[k:k + 256]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(sum(lossf(va[k:k + 2000]) * W[va[k:k + 2000]].sum() for k in range(0, len(va), 2000)) / W[va].sum())
        print(f"  에폭 {ep + 1}: 검증 손실 {vl:.4f} ({time.time() - t1:.0f}초)")
        if vl < best:
            best, bad, best_state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= int(arg("patience", 2)):  # patience=1: 한 번 나빠지면 멈춤
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        zva = torch.cat([net(*take(feat, va[k:k + 2000])) for k in range(0, len(va), 2000)]).numpy()
    T, stretch = [], []
    for k in range(len(HEADS)):  # 판정층마다 온도·배율
        nll = lambda z: -np.log(j.softmax(z)[np.arange(len(va)), Y[va, k]] + 1e-12).mean()
        T.append(float(min(np.exp(np.linspace(-2, 2, 81)), key=lambda t: nll(zva[:, k] / t))))
        ev, yt = j.softmax(zva[:, k] / T[k]) @ V, V[Y[va, k]]
        stretch.append(float(np.clip((ev @ yt) / max(ev @ ev, 1e-9), 1.0, 4.0)))
    print(f"온도 {T}, 배율 {stretch}")
    torch.save({"state": net.state_dict(), "vocab": VOCAB, "T": T, "stretch": stretch, "cfg": (H, D, DEEP, BI, ATT)}, NAME)


def logits(texts, noph=False):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 2000):
            f = featurize([j.clean(t) for t in texts[i:i + 2000]])
            if noph:
                f = (f[0], torch.zeros_like(f[1]), f[2])
            out.append(net(*take(f, np.arange(len(f[2])))).numpy())
    return np.concatenate(out) if out else np.zeros((0, len(HEADS), 7))


def raw(texts, noph=False):
    z = logits(texts, noph)
    return np.stack([np.clip(stretch[k] * (j.softmax(z[:, k] / T[k]) @ V), -2, 2) for k in range(len(HEADS))], 1)


def scores(texts):  # evaluate.scores와 같은 뒤처리, 판정층마다
    r = raw(texts)
    return np.stack([np.where(j.level_only(texts), 0.0, j.soften(texts, j.guard(texts, r[:, k], 1.0)[0])[0]) for k in range(len(HEADS))], 1)


def flips(s, y):  # 3분류 부호 반대: |점수| ≥ 0.25와 0 아닌 정답의 부호가 반대
    return int(((np.abs(s) >= 0.25) & (y != 0) & (np.sign(s) * np.sign(y) < 0)).sum())


def metrics(s, Ym):
    res = {}
    for k, h in enumerate(HEADS):
        y = Ym[:, k]; m = e.screen(s[:, k], y); s7 = V[np.abs(s[:, k][:, None] - V).argmin(1)]; nz = y != 0
        res[h] = {"7단계": float(np.mean(s7 == y)), "맞음": m["맞음"], "헛경보": m["헛경보"], "놓침": m["놓침"], "방향틀림": m["방향 틀림"],
                  "방향틀림3": flips(s[:, k], y), "평균차이": float(np.abs(s[:, k] - y).mean()), "0아닌_7단계": float(np.mean(s7[nz] == y[nz])),
                  "0아닌_부호": float(np.mean(np.sign(s[nz, k]) == np.sign(y[nz]))), "0아닌_0으로": float(np.mean(s7[nz] == 0))}
    return res


if SHOW:
    dst = arg("out", str(OUT).replace(".json", "_show.json"))
    texts = lines(SHOW)
    s = raw(texts) if "raw" in A else scores(texts)  # raw: 뒤처리(안전장치·0으로) 전 모델 점수 (진단용)
    json.dump({t: dict(zip(HEADS, map(float, v))) for t, v in zip(texts, s)}, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
    sys.exit()
if PROBS:  # 선생 확률 (온도 반영)
    z = logits(lines(PROBS))
    np.save(arg("out"), np.stack([j.softmax(z[:, k] / T[k]) for k in range(len(HEADS))], 1).astype(np.float32))
    sys.exit()

res = json.load(open(OUT, encoding="utf-8")) if EVAL and OUT.exists() else {}
res.update({"tag": TAG, "args": A, "params": sum(p.numel() for p in net.parameters()), "T": T, "stretch": stretch})
if not EVAL:
    vt = [titles[i] for i in va]
    res["검증"] = metrics(scores(vt), V[Y[va]])
    strong = [t for t, r in zip(vt, j.rule_signal(vt, lex)) if abs(r) >= 1]
    res["검증"]["사전_지우면_평균변화"] = float(np.abs(raw(strong) - raw(strong, noph=True)).mean())
    res["검증"]["사전_강한제목"] = len(strong)
def speed(ts):  # 제목 하나씩 101번(첫 번은 뺌)의 중앙값, 묶음 채점의 제목당 시간
    one = []
    for t in ts[:101]:
        a = time.perf_counter(); scores([t]); one.append(time.perf_counter() - a)
    a = time.perf_counter(); scores(ts); return {"묶음_제목당ms": 1000 * (time.perf_counter() - a) / len(ts), "하나_ms중앙값": 1000 * sorted(one[1:])[50]}


if "speed" in A:  # 시험지 없이 Opus 검증 제목으로 속도만
    res["속도_Opus검증"] = speed([r[-1] for r in tsv(LAB / "en_opus검증1000.tsv")])
if CHECK:
    cr = tsv(CHECK)
    res[os.path.basename(CHECK)[:-4]] = metrics(scores([r[-1] for r in cr]), np.array([[float(r[k]) for k in (0, 1)] for r in cr]))
if EDIT:  # 사전 고침 따라가기: 고친 사전으로 채점만 바꿀 때 모델 점수만(뒤처리 없이)으로 센 방향 틀림, Haiku 검증 + Opus 검증
    hr = tsv(LAB / "en_ai_haiku.tsv")
    rr = [hr[i] for i in np.random.default_rng(0).permutation(len(hr))[:len(hr) // 10]] + tsv(LAB / "en_opus검증1000.tsv")
    ts_, ym = [r[-1] for r in rr], np.array([[float(r[k]) for k in (0, 1)] for r in rr])
    a = raw(ts_); lex = j.load_lexicon(EDIT); b = raw(ts_); lex = j.load_lexicon(j.LEXICON)
    res["사전고침"] = {h: {"전": flips(a[:, k], ym[:, k]), "후": flips(b[:, k], ym[:, k])} for k, h in enumerate(HEADS)}
if TEST or "old" in A:  # 옛 영어 시험지 303 (5단계, 미국 칸 부호만) — old: 새 시험지 없이 이것만
    old = [l.split("\t") for f in ("en_test.tsv", "en_newtest.tsv") for l in (j.HOME / "labeled" / f).read_text(encoding="utf-8").splitlines()[1:]]
    res["옛영어303_방향틀림3"] = flips(scores([r[-1] for r in old])[:, 0], np.array([float(r[0]) for r in old]))
if TEST:
    rows = [l.split("\t") for l in (j.HOME / "labeled" / "시험" / "en_새시험1500.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    ts = [r[-1] for r in rows]
    s = scores(ts)
    res.update(metrics(s, np.array([[float(r[k]) for k in (0, 1)] for r in rows])))
    res.update(speed(ts))
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
