# 섹터·업체 관점 작은 모델, 주가로 만든 라벨 (2026-10-31, docs/실험기록.md 같은 날). 한국어.
# python 실험/업체모델.py   → 실험/결과/업체모델.json·.log
import sys, json, zlib, collections
import numpy as np, torch
sys.path.insert(0, r"C:\english_folder\somethings")
sys.path.insert(0, r"C:\english_folder\somethings\web")
sys.argv = sys.argv[:1]
import judge as j
import index as ix
import build as b
import sectors as S
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
torch.set_num_threads(9); torch.manual_seed(0); rng = np.random.default_rng(0)
START, TR_END, VA_END, END = "2025-11-03", "2026-06-30", "2026-07-31", "2026-10-30"

# ── 주가: 업체·섹터 바구니 초과 등락 ──
days, _, kchg = b.prices("코스피")
ev = [i for i, d in enumerate(days) if START <= d <= END]
E = [days[i] for i in ev]; kos = dict(zip(E, kchg[ev]))
px = collections.defaultdict(dict)
for l in open(j.HOME / "data" / "섹터업체주가.tsv", encoding="utf-8").read().splitlines()[1:]:
    p = l.split("\t"); px[p[1]][p[0]] = float(p[3])
CAP = {l.split("\t")[0]: float(l.split("\t")[2]) for l in open(j.HOME / "data" / "섹터업체시총.tsv", encoding="utf-8").read().splitlines()[1:]}
EXC = {n: {x: 100 * (d[x] / d[days[i - 1]] - 1) - kos[x] for i, x in zip(ev, E) if d.get(x) and d.get(days[i - 1])} for n, d in px.items()}
D = S.load("ko")
home = {co: s for _, s, co in D if co}
SECS = sorted(set(home.values()))
SEX = {}
for s in SECS:
    mem = [n for n in home if home[n] == s and n in EXC]
    SEX[s] = {x: float(np.average([EXC[n][x] for n in mem if x in EXC[n]], weights=[CAP[n] for n in mem if x in EXC[n]])) for x in E if any(x in EXC[n] for n in mem)}
sd = lambda m: np.std([v for x, v in m.items() if x <= TR_END])
ZC = {n: {x: v / sd(m) for x, v in m.items()} for n, m in EXC.items()}
ZS = {s: {x: v / sd(m) for x, v in m.items()} for s, m in SEX.items()}

# ── 제목과 라벨 ──
IT = b.items("ko", "최종")
by = b.assign("ko", days, IT, ("daily", "sector"), since="2025-10-25")
rows = []  # (날, 제목, 섹터들, 업체들, 라벨, 지금 점수)
for d in E:
    for t, it, _ in by.get(d, []):
        ss, cos = S.tag(t, D, "ko")
        if not ss:
            continue
        cs = [c for c in cos if c in ZC and d in ZC[c]]
        ys = [ZC[c][d] for c in cs] if cs else [ZS[s][d] for s in ss if d in ZS[s]]
        if ys:
            rows.append((d, t, sorted(ss), cs, float(np.clip(np.mean(ys), -3, 3)), it["s"]))
day = np.array([r[0] for r in rows]); T = [r[1] for r in rows]; Y = np.array([r[4] for r in rows], np.float32); NOW = np.array([r[5] for r in rows], np.float32)
tr, va, te = day <= TR_END, (day > TR_END) & (day <= VA_END), day > VA_END
print(f"제목 {len(rows):,} (학습 {tr.sum():,} · 고르기 {va.sum():,} · 시험 {te.sum():,}), 업체 이름 있는 제목 {sum(1 for r in rows if r[3]):,}")

# ── 특징: 주력 은닉 활성 (뉴런) + 사전 + 글자 조각 ──
p, Tt, meta = j.load(j.MAIN["ko-감성"])
H = p["out_w"].shape[-1]
q = dict(p, out_w=np.eye(H, dtype=np.float32), out_b=np.zeros(H, np.float32))
CT = [j.clean(t) for t in T]
ACT = np.vstack([j.cmsm_logits(q, CT[i:i + 2000], meta) for i in range(0, len(CT), 2000)]).astype(np.float32)
# 검산: 활성 → 원래 판정층 → 점수가 주력 점수와 같아야 함
z =ACT[:200] @ p["out_w"].T + p["out_b"]
raw = j.score_of(j.softmax(z / Tt), meta)
ref = j.score_of(*j.predict(j.MAIN["ko-감성"], T[:200]))
print(f"검산: 활성으로 다시 낸 점수와 주력 점수 최대 차 {np.abs(raw - ref).max():.2e}")
LEX = j.phrase_features(CT, meta["lexicon"], meta["hint"]).astype(np.float32)
NB = 1 << 18
def grams(t):
    t = t.replace(" ", "_")
    return [zlib.crc32(t[i:i + n].encode()) % NB for n in (1, 2, 3) for i in range(len(t) - n + 1)]
G = [grams(t) for t in CT]
DENSE = np.hstack([ACT, NOW[:, None], LEX])
mu, sg = DENSE[tr].mean(0), DENSE[tr].std(0) + 1e-6
DENSE = (DENSE - mu) / sg
corr = lambda a, b_: float(np.corrcoef(a, b_)[0, 1]) if len(a) > 2 and np.std(a) > 0 else float("nan")


def ridge(X, y, lam):  # 닫힌 식 (절편 포함)
    X1 = np.hstack([X, np.ones((len(X), 1), np.float32)])
    A = X1.T @ X1 + lam * np.eye(X1.shape[1]); A[-1, -1] -= lam
    return np.linalg.solve(A, X1.T @ y)


predA = lambda w, X: np.hstack([X, np.ones((len(X), 1), np.float32)]) @ w
XA = DENSE[:, :H + 1]  # A: 활성 64 + 지금 점수


class Lin(torch.nn.Module):  # B: 글자 조각 해시 + 조밀 특징 → 하나
    def __init__(s, dd):
        super().__init__(); s.e = torch.nn.EmbeddingBag(NB, 1, mode="sum"); torch.nn.init.zeros_(s.e.weight); s.l = torch.nn.Linear(dd, 1)
    def forward(s, idx, off, w, x):
        return (s.e(idx, off, per_sample_weights=w) + s.l(x)).squeeze(1)


def batch(ii):
    idx = [torch.tensor(G[i]) for i in ii]; n = torch.tensor([len(g) for g in idx])
    w = torch.cat([torch.full((len(g),), 1 / max(1, len(g)) ** 0.5) for g in idx])
    return torch.cat(idx), torch.cat([torch.zeros(1, dtype=torch.long), n.cumsum(0)[:-1]]), w, torch.tensor(DENSE[ii])


def predB(m, ii):
    out = []
    with torch.no_grad():
        for k in range(0, len(ii), 4096):
            out.append(m(*batch(ii[k:k + 4096])).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def trainB(itr, iva, wd):
    m = Lin(DENSE.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=0)
    best, state = -9, None
    for ep in range(4):
        perm = rng.permutation(itr)
        for k in range(0, len(perm), 512):
            ii = perm[k:k + 512]; out = m(*batch(ii))
            loss = ((out - torch.tensor(Y[ii])) ** 2).mean() + wd * (m.e.weight ** 2).sum() + 1e-4 * (m.l.weight ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        c = corr(predB(m, iva), Y[iva])
        if c > best:
            best, state = c, {k_: v.clone() for k_, v in m.state_dict().items()}
    m.load_state_dict(state)
    return m, best


IDX = np.arange(len(rows))
member = {s: np.array([s in r[2] for r in rows]) for s in SECS}
cands = {}
# A 공유
best = max(((corr(predA(ridge(XA[tr], Y[tr], lam), XA[va]), Y[va]), lam) for lam in (0.1, 1, 10, 100, 1000)))
wA = ridge(XA[tr], Y[tr], best[1]); cands["A 공유"] = {"고르기상관": best[0], "pred": predA(wA, XA), "설정": f"L2 {best[1]}"}
# A 섹터별: 제목이 여러 섹터면 그 섹터 모델들의 평균
PA = np.zeros(len(rows)); cnt = np.zeros(len(rows)); setting = {}
for s in SECS:
    m_ = member[s]
    bs = max(((corr(predA(ridge(XA[tr & m_], Y[tr & m_], lam), XA[va & m_]), Y[va & m_]), lam) for lam in (1, 10, 100, 1000)), key=lambda x: (np.nan_to_num(x[0], nan=-9), x[1]))
    w_ = ridge(XA[tr & m_], Y[tr & m_], bs[1]); PA[m_] += predA(w_, XA[m_]); cnt[m_] += 1; setting[s] = bs[1]
PA /= np.maximum(cnt, 1)
cands["A 섹터별"] = {"고르기상관": corr(PA[va], Y[va]), "pred": PA, "설정": setting}
print("A 끝", {k: round(v["고르기상관"], 3) for k, v in cands.items()})
# B 공유
resB = []
for wd in (1e-6, 1e-5, 1e-4):
    m, c = trainB(IDX[tr], IDX[va], wd); resB.append((c, wd, m)); print(f"  B 공유 L2 {wd}: 고르기 상관 {c:.3f}")
c, wd, mB = max(resB, key=lambda x: x[0])
cands["B 공유"] = {"고르기상관": c, "pred": predB(mB, IDX), "설정": f"L2 {wd}"}
# B 섹터별 (L2는 공유에서 고른 값)
PB = np.zeros(len(rows)); cnt = np.zeros(len(rows))
for s in SECS:
    m_ = member[s]
    ms, _ = trainB(IDX[tr & m_], IDX[va & m_], wd); PB[m_] += predB(ms, IDX[m_]); cnt[m_] += 1
PB /= np.maximum(cnt, 1)
cands["B 섹터별"] = {"고르기상관": corr(PB[va], Y[va]), "pred": PB, "설정": f"L2 {wd}"}
pick = max(cands, key=lambda k: cands[k]["고르기상관"])
print("고르기 상관:", {k: round(v["고르기상관"], 3) for k, v in cands.items()}, "→ 고름:", pick)

# 기준선: 이름 외우기, 지금 점수
cm = collections.defaultdict(list); sm = collections.defaultdict(list)
for r, y, t_ in zip(rows, Y, tr):
    if t_:
        (cm[tuple(r[3])] if r[3] else sm[tuple(r[2])]).append(y)
NAME = np.array([np.mean(cm.get(tuple(r[3]), [0])) if r[3] else np.mean(sm.get(tuple(r[2]), [0])) for r in rows])
ALLP = {k: v["pred"] for k, v in cands.items()} | {"이름 외우기": NAME, "지금 점수 V0": NOW}


def sector_eval(P):  # 시험 기간 섹터 하루·7일 값 ↔ 시총 바구니 초과 등락
    TE = [x for x in E if x > VA_END]; res = {}
    for s in SECS:
        v = collections.defaultdict(list)
        for r, pr, t_ in zip(rows, P, te):
            if t_ and s in r[2]:
                v[r[0]].append(pr)
        x1 = np.array([np.mean(v[x]) if len(v[x]) >= 5 else np.nan for x in TE]); y1 = np.array([SEX[s].get(x, np.nan) for x in TE])
        x7 = np.array([np.mean([q_ for xx in TE[max(0, k - 4):k + 1] for q_ in v[xx]]) if sum(len(v[xx]) for xx in TE[max(0, k - 4):k + 1]) >= 10 else np.nan for k in range(len(TE))])
        y7 = np.array([np.sum(y1[max(0, k - 4):k + 1]) if k >= 4 else np.nan for k in range(len(TE))])
        res[s] = {"하루": round(ix.corr(x1, y1)[0], 3), "7일": round(ix.corr(x7, y7)[0], 3)}
    return res


OUT = {"고름": pick, "고르기상관": {k: round(v["고르기상관"], 3) for k, v in cands.items()}, "시험": {}}
for k, P in ALLP.items():
    se = sector_eval(P)
    OUT["시험"][k] = {"제목상관": round(corr(P[te], Y[te]), 3), "섹터하루_중앙값": round(float(np.nanmedian([v["하루"] for v in se.values()])), 3),
                     "섹터7일_중앙값": round(float(np.nanmedian([v["7일"] for v in se.values()])), 3), "섹터별": se}
print("\n시험 기간 (2026-08-01 ~ 10-30):")
for k, v in OUT["시험"].items():
    print(f"  {k:10} 제목 상관 {v['제목상관']:+.3f} | 섹터 하루 중앙값 {v['섹터하루_중앙값']:+.3f} | 7일 {v['섹터7일_중앙값']:+.3f}")
print("\n섹터별 하루 상관 (고른 후보 / 지금 / 이름):")
for s in SECS:
    print(f"  {s:14} {OUT['시험'][pick]['섹터별'][s]['하루']:+.2f} / {OUT['시험']['지금 점수 V0']['섹터별'][s]['하루']:+.2f} / {OUT['시험']['이름 외우기']['섹터별'][s]['하루']:+.2f}")
a, v0, nm = OUT["시험"][pick]["섹터하루_중앙값"], OUT["시험"]["지금 점수 V0"]["섹터하루_중앙값"], OUT["시험"]["이름 외우기"]["섹터하루_중앙값"]
OUT["채택"] = bool(a - v0 >= 0.05 and a > nm)
print(f"\n채택: {'예' if OUT['채택'] else '아니오'} ({pick} {a:+.3f} vs 지금 {v0:+.3f}, 이름 {nm:+.3f})")
print("\n10/30 기아 실적 제목 (정답 z / 고른 후보 / 지금 점수):")
for i, r in enumerate(rows):
    if r[0] == "2026-10-30" and "기아" in r[1] and any(w in r[1] for w in ("영업익", "실적", "쇼크", "급락")):
        print(f"  {Y[i]:+.2f} / {ALLP[pick][i]:+.2f} / {NOW[i]:+.2f}  {r[1][:60]}")
json.dump(OUT, open(j.HOME / "실험" / "결과" / "업체모델.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
