# 순서 갈래 실험: B0(글자 조각+사전, A'와 같은 구조) / CNN(+음절 합성곱) / CMSM(+음절 행렬 곱). docs/실험기록.md 2026-10-27
# python 실험/seqexp.py bag|cnn|cmsm [d] [seqlr=0.001] [표본 수] [eval] [show=제목파일] [seed=K] [opus2] [mix] [bi] [check=라벨파일] [featlex=사전] [haiku2] [opus3] [fast] [patience=2] [h=64] [d=16] [ow=3] [hw=1] [tag=이름]
#   d: 변형 D 정답(사람>Opus>Haiku), eval: 저장된 모델로 채점만, show=: 그 제목들 점수만. 결과 json은 실험/결과/, 모델은 models/seq_*.pt
import os, sys, json, time, zlib, unicodedata, collections
import numpy as np
import torch
from torch import nn
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
import evaluate as e
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
torch.set_num_threads(max(1, os.cpu_count() // 2))
KIND = sys.argv[1]
arg = lambda k, d=None: next((a[len(k) + 1:] for a in sys.argv[2:] if a.startswith(k + "=")), d)
torch.manual_seed(int(arg("seed", 0)))  # seed=K: 씨앗만 바꿈 (자료 나누기는 그대로, 2026-10-29)
OPUS2, MIX, BI, CHECK = "opus2" in sys.argv[2:], "mix" in sys.argv[2:], "bi" in sys.argv[2:], arg("check")
# opus2: 과거 Opus 2차 2만(ko_ai_opus_과거2, 한국어 Opus 검증 1,000 뺌)도 Opus 무게로. mix: Opus·Haiku가 다 있는 제목은 0.75 Opus + 0.25 Haiku.
# bi: 행렬 곱을 거꾸로도 읽음. check=파일: 그 라벨 파일(감성 … 제목)로도 잼. 한국어 Opus 검증 제목은 늘 학습에서 뺌 (2026-10-29)
LIMIT = next((int(a) for a in sys.argv[2:] if a.isdigit()), 0)
DLAB, EVAL = "d" in sys.argv[2:], "eval" in sys.argv[2:]  # d: 변형 D 정답(사람>Opus>Haiku), eval: 저장된 모델로 채점만
SL = next((a for a in sys.argv[2:] if a.startswith("seqlr=")), "")
TAG = next((a[4:] for a in sys.argv[2:] if a.startswith("tag=")), "")  # tag=이름: 같은 설정의 다른 사전판을 따로 저장
MORPH, LEXINIT = "morph" in sys.argv[2:], "lexinit" in sys.argv[2:]  # 형태소 단위 행렬, 사전 출발점 (2026-10-27 M1·M2)
OUT = os.path.join(j.HOME, "실험", "결과", f"seq_{KIND}{'_d' if DLAB else ''}{'_morph' if MORPH else ''}{'_lexinit' if LEXINIT else ''}{'_' + SL if SL else ''}{'_past' if 'past' in sys.argv[2:] else ''}{'_' + TAG if TAG else ''}{'_test' if LIMIT else ''}.json")  # len=·bodyzero=는 tag=로 이름을 나눈다
LEN = next((int(a[4:]) for a in sys.argv[2:] if a.startswith("len=")), 0)  # len=N: 순서 갈래가 읽는 최대 길이 (본문은 형태소 120개 넘음, 2026-10-28)
BZ = next((float(a[9:]) for a in sys.argv[2:] if a.startswith("bodyzero=")), 0)  # bodyzero=K: 본문 0 문장을 0 아닌 문장의 K배까지만 (무작위)
BUCKETS, NGRAM, H, CUTDAY, L, D = 16384, 3, int(arg("h", 64)), "2026-08-01", LEN or (40 if MORPH else 64), int(arg("d", 16))  # h=·d=: 은닉·행렬 크기 (2026-10-31)
CAP = 4.0 if MORPH else 1.0  # 상태 크기 상한 (형태소판은 상수 칸 1 + 감성 칸이 있어 넉넉히)
LABELS = ["-2", "-1", "-0.5", "0", "0.5", "1", "2"]
V = np.array(LABELS, float)

# ── 데이터: distill.py와 같게 (선생 점수 + Opus 3배, 시험지·거의 같은 제목 제외) ──
AI = {p[-1]: float(p[0]) for p in (l.split("\t") for l in open(j.HOME / "labeled" / "ko_ai_opus.tsv", encoding="utf-8").read().splitlines()[1:])}
first = {}
for l in open(j.HOME / "headlines" / "ko_daily.tsv", encoding="utf-8"):
    p = l.split("\t")
    if p[0] and p[2] not in first:
        first[p[2]] = p[0][:10]
teacher = {t.rstrip("\n"): float(s) for s, t in (l.split("\t", 1) for l in open(j.HOME / "data" / "ko_scores_ko-감성-미세.tsv", encoding="utf-8"))}
labf = lambda f: {p[-1]: float(p[0]) for p in (l.split("\t") for l in open(j.HOME / "labeled" / f, encoding="utf-8").read().splitlines()[1:])}
HUM, HAIKU = (labf("ko_more8.tsv"), labf("ko_ai_haiku_mid.tsv") | labf("ko_ai_haiku_rest.tsv")) if DLAB else ({}, {})
if "haiku2" in sys.argv[2:]:  # haiku2: 과거(2023~24) 새 Haiku 라벨 9,929개도 (가벼운 방식, 2026-10-31)
    HAIKU |= labf("ko_ai_haiku_과거2.tsv")
if "past" in sys.argv[2:]:  # past: 과거 기간(2023-10~2025) Opus 라벨도 Opus와 같은 무게로 (2026-10-27)
    AI |= {p[-1]: float(p[0]) for p in (l.split("	") for l in open(j.HOME / "labeled" / "ko_ai_opus_과거.tsv", encoding="utf-8").read().splitlines()[1:])}
HOLDF = j.HOME / "labeled" / "ko_opus검증1000.tsv"
HOLD = {l.split("	")[-1] for l in open(HOLDF, encoding="utf-8").read().splitlines()[1:]} if HOLDF.exists() else set()
if OPUS2:
    AI |= {t: v for t, v in labf("ko_ai_opus_과거2.tsv").items() if t not in HOLD}
if "opus3" in sys.argv[2:]:  # opus3: 과거 새 Haiku 9,929개에 붙인 Opus 3차 (2026-10-31)
    AI |= {t: v for t, v in labf("ko_ai_opus_과거3.tsv").items() if t not in HOLD}
AI = {t: v for t, v in AI.items() if t not in HUM}
BODY = labf("ko_ai_opus_본문.tsv") if {"body", "onlybody"} & set(sys.argv[2:]) else {}  # body: 본문 문장 Opus 라벨도 (Opus 무게), onlybody: 본문 문장만 (2026-10-28)
if BZ and BODY:
    bz = sorted(t for t, v in BODY.items() if v == 0); n_nz = len(BODY) - len(bz)
    drop = set(bz) - {bz[i] for i in np.random.default_rng(0).permutation(len(bz))[:int(BZ * n_nz)]}
    BODY = {t: v for t, v in BODY.items() if t not in drop}
    print(f"본문 0 문장 {len(bz):,}개 중 {len(bz) - len(drop):,}개만 (0 아닌 {n_nz:,}개의 {BZ:g}배)")
AI |= BODY
titles = [t for t, d in first.items() if d < CUTDAY and (t in teacher or t in HAIKU) and t not in AI and t not in HUM] + list(AI) + list(HUM)
tri = lambda t: {t[i:i + 3] for i in range(len(t) - 2)} if len(t := "".join(t.lower().split())) > 2 else {t}
TEST = [tri(l.split("\t")[-1]) for f in (j.HOME / "labeled" / "시험").glob("ko_*.tsv") for l in f.read_text(encoding="utf-8").splitlines()[1:]]
inv = collections.defaultdict(list)
for k, s in enumerate(TEST):
    for g in s:
        inv[g].append(k)
near = lambda t: any(n / (len(a) + len(TEST[k]) - n) >= 0.5 for a in [tri(t)] for k, n in collections.Counter(k for g in a for k in inv.get(g, ())).items())
if "onlybody" in sys.argv[2:]:
    titles = list(BODY)
FAST = "fast" in sys.argv[2:]  # fast: 시험지 겹침 판정을 저장해 다시 쓰고, 학습 뒤 채점은 check=만 (2026-10-31)
if FAST:
    import hashlib
    _cp = j.HOME / "data" / "cache" / f"near_{hashlib.sha1(chr(10).join(sorted(''.join(sorted(x)) for x in TEST)).encode()).hexdigest()[:12]}.json"
    _cache = json.load(open(_cp, encoding="utf-8")) if _cp.exists() else {}
    _n0 = len(_cache)
    for t in titles:
        if t not in _cache:
            _cache[t] = bool(near(t))
    if len(_cache) > _n0:
        _cp.parent.mkdir(exist_ok=True); json.dump(_cache, open(_cp, "w", encoding="utf-8"), ensure_ascii=False)
    near = lambda t: _cache[t]
titles = [t for t in titles if not near(t) and t not in HOLD]
if LIMIT:
    titles = [titles[i] for i in np.random.default_rng(0).permutation(len(titles))[:LIMIT]]
target = lambda t: HUM.get(t, AI.get(t, HAIKU.get(t, teacher.get(t))))
Y = np.abs(np.array([target(t) for t in titles])[:, None] - V).argmin(1)
from_ai, from_hum = np.array([t in AI for t in titles]), np.array([t in HUM for t in titles])
P = np.eye(7)[Y]  # 정답 확률 (mix면 Opus·Haiku 섞음)
if MIX:
    mixed = [i for i, t in enumerate(titles) if t in AI and t not in HUM and t in HAIKU]
    P[mixed] = 0.75 * P[mixed] + 0.25 * np.eye(7)[np.abs(np.array([HAIKU[titles[i]] for i in mixed])[:, None] - V).argmin(1)]
    print(f"두 선생 섞은 제목 {len(mixed):,}개")
print(f"[{KIND}{' D' if DLAB else ''}] 학습 제목 {len(titles):,}개 (사람 {from_hum.sum():,}, Opus {from_ai.sum():,}, "
      f"Haiku {sum(t in HAIKU and t not in AI and t not in HUM for t in titles):,})")
j.LEXICON = next((a[4:] for a in sys.argv[2:] if a.startswith("lex=")), j.LEXICON)  # lex=파일: 다른 사전판으로 학습·채점 (안전장치도 같은 사전)
lex = j.load_lexicon(j.LEXICON)
if arg("featlex"):  # featlex=파일: 특징·사전 출발점만 이 사전으로 (안전장치는 j.LEXICON 그대로). v1 고정판과 같은 조건 (2026-10-29)
    lex = j.load_lexicon(arg("featlex"))
states = [j.clean(t) for t in titles]
norm = lambda t: unicodedata.normalize("NFC", t.lower())
if MORPH:
    from kiwipiepy import Kiwi
    KIWI = Kiwi(num_workers=max(1, os.cpu_count() // 2))
    toks = lambda sts: [[f"{norm(x.form)}/{x.tag}" for x in r] for r in KIWI.tokenize(list(sts))]
else:
    toks = lambda sts: [list(norm(t)) for t in sts]
cnt = collections.Counter(c for ts in toks(states) for c in ts)
VOCAB = {c: i + 2 for i, c in enumerate(sorted(c for c, n in cnt.items() if n >= 2))}  # 0 빈칸, 1 모름


def featurize(sts):
    """글자 조각(희소), 사전 특징, 음절 번호. distill.py·judge.text_features와 같은 조각·정규화"""
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
    C = np.zeros((len(sts), L), np.int64)
    for r, ts in enumerate(toks(sts)):
        ids = [VOCAB.get(ch, 1) for ch in ts][:L]
        C[r, :len(ids)] = ids
    return (torch.tensor(idx), torch.tensor(wts, dtype=torch.float32), torch.tensor(offs)), torch.tensor(PH), torch.tensor(C)


class Net(nn.Module):
    def __init__(s, F):
        super().__init__()
        s.emb, s.lin, s.out = nn.EmbeddingBag(BUCKETS, H, mode="sum"), nn.Linear(F, H), nn.Linear(H, 7)
        with torch.no_grad():
            s.emb.weight.normal_(0, 1 / np.sqrt(BUCKETS + F)); s.lin.weight.normal_(0, 1 / np.sqrt(BUCKETS + F)); s.lin.bias.zero_()
        if KIND == "cnn":
            s.ce = nn.Embedding(len(VOCAB) + 2, 64, padding_idx=0)
            s.convs = nn.ModuleList(nn.Conv1d(64, 64, k, padding="same") for k in (2, 3, 4, 5))
            s.seq = nn.Linear(256, H)
        if KIND == "cmsm":
            s.A = nn.Embedding(len(VOCAB) + 2, D * D)
            with torch.no_grad():
                s.A.weight.normal_(0, 0.02); s.A.weight[0].zero_()  # 빈칸 = 단위 행렬
            s.v0 = nn.Parameter(torch.randn(D) / np.sqrt(D))
            if LEXINIT:  # 마지막 칸 = 상수 1, 0번 칸 = 감성. 사전 낱말과 형태가 같은 형태소는 역할대로 시작
                with torch.no_grad():
                    s.v0.zero_(); s.v0[D - 1] = 1.0
                    step = {w.strip(): v for k, v in j.STEP.items() for w in lex.get(k, [])}
                    neg, past = {w.strip() for w in lex.get("부정", [])}, {w.strip() for w in lex.get("지난", [])}
                    n_init = 0
                    for tok, i in VOCAB.items():
                        f = tok.rsplit("/", 1)[0]; Ai = s.A.weight[i].view(D, D)
                        if f in step: Ai[0, D - 1] += step[f]; n_init += 1
                        if f in neg: Ai[0, 0] += -2.0; n_init += 1
                        if f in past: Ai[0, 0] += -0.8; n_init += 1
                    print(f"사전 출발점 형태소 {n_init}개")
            s.seq = nn.Linear(D, H)
            if BI:  # 거꾸로 읽기용 행렬 (사전 출발점도 같게)
                s.A2 = nn.Embedding(len(VOCAB) + 2, D * D)
                with torch.no_grad():
                    s.A2.weight.copy_(s.A.weight)
                    if not LEXINIT:
                        s.A2.weight.normal_(0, 0.02); s.A2.weight[0].zero_()
                s.seq2 = nn.Linear(D, H)

    def forward(s, bag, PH, C):
        (idx, wts, offs) = bag
        h = s.emb(idx, offs, per_sample_weights=wts) + s.lin(PH)
        mask = C > 0
        if KIND == "cnn":
            x = s.ce(C).transpose(1, 2)
            f = [torch.relu(cv(x)).masked_fill(~mask[:, None, :], -1e4).amax(2).clamp(min=0) for cv in s.convs]
            h = h + s.seq(torch.cat(f, 1))
        if KIND == "cmsm":
            C = C[:, :max(1, int(mask.sum(1).max()))]; mask = C > 0  # 긴 L에서도 묶음의 가장 긴 문장까지만
            for Aw, sq, Cx in [(s.A, s.seq, C)] + ([(s.A2, s.seq2, C.flip(1))] if BI else []):  # 거꾸로: 앞쪽 빈칸은 건너뜀
                mk = Cx > 0
                M = Aw(Cx).view(len(Cx), Cx.shape[1], D, D) + torch.eye(D)
                st = s.v0.expand(len(Cx), D)
                for t in range(Cx.shape[1]):  # 읽는 순서대로: 상태 ← M(글자) · 상태, 크기는 CAP을 넘지 않게
                    new = torch.bmm(M[:, t], st.unsqueeze(2)).squeeze(2)
                    new = new / (new.norm(dim=1, keepdim=True) / CAP).clamp(min=1.0)
                    st = torch.where(mk[:, t:t + 1], new, st)
                h = h + sq(st)
        return s.out(torch.tanh(h))


def take(feat, rows):
    (idx, wts, offs), PH, C = feat
    rows = torch.as_tensor(rows)
    ends = torch.cat([offs[1:], torch.tensor([len(idx)])])
    s_, e_ = offs[rows], ends[rows]
    ii = torch.cat([torch.arange(a, b) for a, b in zip(s_.tolist(), e_.tolist())])
    o = torch.cat([torch.tensor([0]), torch.cumsum(e_ - s_, 0)[:-1]])
    return (idx[ii], wts[ii], o), PH[rows], C[rows]


if EVAL:  # 저장된 모델로 채점만 (학습 안 함)
    saved = torch.load(os.path.join(j.HOME, "models", os.path.basename(OUT).replace("_eval", "").replace(".json", ".pt")), weights_only=False)
    VOCAB, T, stretch, ep = saved["vocab"], saved["T"], saved["stretch"], -1
    H, D = saved.get("cfg", (H, D))
    net = Net(j.phrase_features(["x"], lex, j.HINT).shape[1]); net.load_state_dict(saved["state"]); net.eval()
    OUT = OUT.replace(".json", "_eval.json")
else:
    t0 = time.time()
    feat = featurize(states)
    print(f"특징 {time.time() - t0:.0f}초, 음절 사전 {len(VOCAB):,}개")
    net = Net(feat[1].shape[1])
    print(f"매개변수 {sum(p.numel() for p in net.parameters()):,}개")
rng = np.random.default_rng(0)
if not EVAL:
    perm = rng.permutation(len(Y)); va, tr = perm[:len(Y) // 10], perm[len(Y) // 10:]
    Yt = torch.tensor(Y)
    W = (torch.where(torch.tensor(np.abs(V[Y]) >= 1), j.NONZERO, 1.0) * torch.where(torch.tensor(from_ai), float(arg("ow", 3)), 1.0)
         * torch.where(torch.tensor(from_hum), 5.0, 1.0) * torch.where(torch.tensor(~from_ai & ~from_hum), float(arg("hw", 1)), 1.0))  # ow=·hw=: Opus·Haiku만 있는 제목 무게 (2026-10-31)
    SEQLR = next((float(a[6:]) for a in sys.argv[2:] if a.startswith("seqlr=")), 3e-3)  # 순서 갈래만 따로 (CMSM 과적합 검토)
    seqp = [p for n_, p in net.named_parameters() if n_.split(".")[0] in ("A", "v0", "seq", "ce", "convs", "A2", "seq2")]
    Pt = torch.tensor(P, dtype=torch.float32)
    opt = torch.optim.Adam([{"params": [p for p in net.parameters() if all(p is not q for q in seqp)]}] + ([{"params": seqp, "lr": SEQLR}] if seqp else []), lr=3e-3)
    lossf = lambda r: (-(Pt[r] * nn.functional.log_softmax(net(*take(feat, r)), -1)).sum(-1) * W[r]).sum() / W[r].sum()  # 한 칸 정답이면 교차 엔트로피와 같음
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
    nll = lambda z: -np.log(j.softmax(z)[np.arange(len(va)), Y[va]] + 1e-12).mean()
    T = float(min(np.exp(np.linspace(-2, 2, 81)), key=lambda T: nll(zva / T)))
    ev, yt = j.softmax(zva / T) @ V, np.array([target(titles[i]) for i in va])
    stretch = float(np.clip((ev @ yt) / max(ev @ ev, 1e-9), 1.0, 4.0))
    print(f"온도 {T:.2f}, 배율 {stretch:.2f}")


def raw(texts):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 2000):
            f = featurize([j.clean(t) for t in texts[i:i + 2000]])
            out.append(np.clip(stretch * (j.softmax(net(*take(f, np.arange(len(f[2])))).numpy() / T) @ V), -2, 2))
    return np.concatenate(out) if out else np.zeros(0)


def scores(texts):  # evaluate.scores와 같은 뒤처리
    s = j.soften(texts, j.guard(texts, raw(texts), 1.0)[0])[0]
    return np.where(j.level_only(texts), 0.0, s)


SHOW = next((a[5:] for a in sys.argv[2:] if a.startswith("show=")), None)  # show=파일: 그 제목들 점수만 내고 끝
if SHOW:
    texts = [l for l in open(SHOW, encoding="utf-8").read().splitlines() if l.strip()]
    json.dump(dict(zip(texts, map(float, scores(texts)))), open(OUT.replace(".json", "_show.json"), "w", encoding="utf-8"), ensure_ascii=False)
    sys.exit()


if CHECK:  # 그 라벨 파일로: 7단계, 화면 맞음·헛경보, 방향 틀림(3분류: |점수| ≥ 0.25와 0 아닌 정답의 부호 반대)
    cr = [l.split("	") for l in open(CHECK, encoding="utf-8").read().splitlines()[1:] if l.strip()]
    cy, cs = np.array([float(r[0]) for r in cr]), scores([r[-1] for r in cr])
    cm = e.screen(cs, cy); c7 = V[np.abs(cs[:, None] - V).argmin(1)]
    CHK = {"7단계": float(np.mean(c7 == cy)), "맞음": cm["맞음"], "헛경보": cm["헛경보"], "놓침": cm["놓침"],
           "방향틀림3": int(((np.abs(cs) >= 0.25) & (cy != 0) & (np.sign(cs) * np.sign(cy) < 0)).sum()), "평균차이": float(np.mean(np.abs(cs - cy))),
           "0아닌_7단계": float(np.mean(c7[cy != 0] == cy[cy != 0])), "0아닌_0으로": float(np.mean(c7[cy != 0] == 0))}
    print("check", json.dumps(CHK, ensure_ascii=False))
if FAST and not EVAL:  # 학습 뒤 채점은 check=만 하고 모델 저장
    torch.save({"state": net.state_dict(), "vocab": VOCAB, "T": T, "stretch": stretch, "cfg": (H, D)}, os.path.join(j.HOME, "models", os.path.basename(OUT).replace(".json", ".pt")))
    json.dump({"kind": KIND, "검증손실": float(best), "epochs": ep + 1, "T": T, "stretch": stretch, "params": sum(p.numel() for p in net.parameters()),
               **({os.path.basename(CHECK)[:-4]: CHK} if CHECK else {})}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit()
res = {"kind": KIND, "검증손실": float(best) if not EVAL else None, "seqlr": SL, "params": sum(p.numel() for p in net.parameters()), "epochs": ep + 1, "T": T, "stretch": stretch, "cfg": (H, D)}
read = lambda fs: [l.split("\t") for f in fs for l in (j.HOME / "labeled" / "시험" / f).read_text(encoding="utf-8").splitlines()[1:]]
rows = read(["ko_무작위300.tsv", "ko_무작위400.tsv", "ko_시험563.tsv", "ko_오늘60.tsv"])
y = np.array([float(r[0]) for r in rows]); s = scores([r[-1] for r in rows])
m = e.screen(s, y)
s7 = V[np.abs(s[:, None] - V).argmin(1)]
res |= {"합1322_맞음": m["맞음"], "헛경보": m["헛경보"], "놓침": m["놓침"], "방향틀림": m["방향 틀림"], "중립인데반대": m["중립인데 반대로"],
        "7단계일치": float(np.mean(s7 == y)), "7단계평균차이": float(np.mean(np.abs(s - y))),
        "무작위700_7단계": float(np.mean(s7[:700] == y[:700]))}  # 앞 700개 = 무작위300+400 (보완 기준)
fails = []
for desc, a, b, rel in e.BEHAVIOR:
    sa = scores([a])[0]; sb = scores([b])[0] if b else None
    ok = {"<": lambda: sa < sb, "=": lambda: abs(sa - sb) < 0.05, ">0": lambda: sa > 0, "<0": lambda: sa < 0,
          ">=0": lambda: sa >= -0.1, "<=0": lambda: sa <= 0.1}[rel]()
    if not ok:
        fails.append(f"{desc} ({sa:+.2f}" + (f" / {sb:+.2f})" if b else ")"))
res |= {"기본행동": f"{len(e.BEHAVIOR) - len(fails)}/{len(e.BEHAVIOR)}", "기본행동실패": fails}
pairs = read(["ko_순서짝.tsv"])  # 짝, 정답 부호, 제목: 정답 +인 제목 점수 > 정답 −인 제목 점수면 맞음
ps = scores([r[-1] for r in pairs])
by = collections.defaultdict(dict)
for r, v in zip(pairs, ps):
    by[r[0]][int(r[1])] = v
res["순서짝정확도"] = float(np.mean([d[1] > d[-1] for d in by.values()]))
res["순서짝_부호맞음"] = float(np.mean([(v > 0) == (int(r[1]) > 0) for r, v in zip(pairs, ps) if abs(v) >= 0.1]))
tests = {r[-1] for f in (j.HOME / "labeled" / "시험").glob("ko_*.tsv") for r in read([f.name])}
held = [t for t, d in first.items() if d >= CUTDAY and t in teacher and t not in tests]
hs, ht = raw(held), np.array([teacher[t] for t in held])
cell = lambda x: np.where(x >= 0.9, 1, np.where(x <= -0.9, -1, 0))
res["선생같은칸"] = float(np.mean(cell(hs) == cell(ht)))
scores(held[:2000])  # 데우기 (측정 순서 영향 줄이기)
one = []
for t in held[:101]:
    a = time.perf_counter(); scores([t]); one.append(time.perf_counter() - a)
a = time.perf_counter(); scores(held[:2000]); res["묶음_제목당ms"] = 1000 * (time.perf_counter() - a) / 2000
res["하나_ms중앙값"] = 1000 * sorted(one[1:])[50]
if CHECK:
    res[os.path.basename(CHECK)[:-4]] = CHK
if not EVAL:
    torch.save({"state": net.state_dict(), "vocab": VOCAB, "T": T, "stretch": stretch, "cfg": (H, D)}, os.path.join(j.HOME, "models", os.path.basename(OUT).replace("_eval", "").replace(".json", ".pt")))
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
