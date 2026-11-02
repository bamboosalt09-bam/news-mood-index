# 업체·섹터 기억 (2026-11-01, docs/실험기록.md 같은 날 '업체·섹터 기억'). 주의 기억 2(M4b)에 업체·섹터 고유 벡터, 긴 기억(20거래일 뉴스), 누구 입장인가를 더함.
# 새 시험 구간 2026-03-01 ~ 04-30, 고르기 2026-08, 학습 나머지. python 실험/주의기억3.py → 실험/결과/주의기억3.json·.log
import sys, re, json, collections
import numpy as np, torch
sys.path.insert(0, r"C:\english_folder\somethings")
sys.path.insert(0, r"C:\english_folder\somethings\web")
sys.argv = sys.argv[:1]
import judge as j
import index as ix
import build as b
import sectors as S
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
torch.set_num_threads(9)
TE0, TE1, VA0, VA1, KMAX = "2026-03-01", "2026-04-30", "2026-08-01", "2026-08-31", 100
V = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
NEG = re.compile(r"어닝\s?쇼크|실적\s?쇼크|실적\s?충격|(기대|예상|추정|컨센서스)\s?치?\s?하회|못\s?미쳐|못\s?미친|기대\s?못\s?미")
POS = re.compile(r"어닝\s?서프라이즈|깜짝\s?실적|(기대|예상|추정|컨센서스)\s?치?\s?상회|웃돌")
EARN = re.compile(r"영업익|영업이익|순이익|순익|매출|실적|어닝|분기")
CMP = re.compile(r"추월|제치|제쳤|역전|앞질|밀려|뒤처|뒤져|따돌")
NUM = re.compile(r"\d[\d,.]{2,}")

L = [l.rstrip("\n").split("\t") for l in open(j.HOME / "labeled" / "ko_기억라벨.tsv", encoding="utf-8").read().splitlines()[1:]]
IT = b.items("ko", "최종")
L = [r for r in L if r[3] in IT]
day = np.array([r[0] for r in L]); CO = [r[1] for r in L]; T = [r[3] for r in L]; Y = np.array([float(r[2]) for r in L], np.float32)
NOW = np.array([IT[t]["s"] for t in T], np.float32); MKT = np.array([IT[t]["mkt"] for t in T], np.float32)
NG = np.array([bool(NEG.search(t)) for t in T], np.float32); PS = np.array([bool(POS.search(t)) for t in T], np.float32)
ER = np.array([bool(EARN.search(t)) for t in T], np.float32)
TOP = sorted({IT[t]["topic"] for t in T}); TOH = np.eye(len(TOP), dtype=np.float32)[[TOP.index(IT[t]["topic"]) for t in T]]
D = S.load("ko"); home = {co: s for _, s, co in D if co}
COS = sorted(home); SECS = sorted(set(home.values()))
CID = np.array([COS.index(c) for c in CO]); SID = np.array([SECS.index(home[c]) for c in CO])

# 주력 은닉 활성
p, Tt, meta = j.load(j.MAIN["ko-감성"])
H = p["out_w"].shape[-1]
q_ = dict(p, out_w=np.eye(H, dtype=np.float32), out_b=np.zeros(H, np.float32))
uniq = sorted(set(T)); pos = {t: i for i, t in enumerate(uniq)}
CU = [j.clean(t) for t in uniq]
AU = np.vstack([j.cmsm_logits(q_, CU[i:i + 2000], meta) for i in range(0, len(CU), 2000)]).astype(np.float32)
ACT = AU[[pos[t] for t in T]]

# 같은 날·전날 기억 칸 (주의 기억 2와 같음)
grp = collections.defaultdict(list)
for i, k in enumerate(zip(day, CO)):
    grp[k].append(i)
days, _, _ = b.prices("코스피")
ds = [d for d in days if min(day) <= d <= max(day)]; di = {d: k for k, d in enumerate(ds)}
prevd = {d: ds[k - 1] if k else None for k, d in enumerate(ds)}
rng = np.random.default_rng(0)
MEM = np.full((len(L), KMAX), -1, np.int64); PREV = np.zeros((len(L), KMAX), np.float32)
for (d, c), ii in grp.items():
    pv = grp.get((prevd[d], c), [])
    for i in ii:
        cand = [(x, 0.0) for x in ii if x != i] + [(x, 1.0) for x in pv]
        if len(cand) > KMAX:
            cand.sort(key=lambda z: -(NG[z[0]] + PS[z[0]] + MKT[z[0]] + rng.random() * 0.5)); cand = cand[:KMAX]
        for k, (x, f) in enumerate(cand):
            MEM[i, k] = x; PREV[i, k] = f
TRI = [{t[i:i + 3] for i in range(len(t) - 2)} for t in (j.clean(x).replace(" ", "") for x in T)]
NUMS = [{n.replace(",", "").rstrip(".") for n in NUM.findall(t)} for t in T]
PAIR = np.zeros((len(L), KMAX, 3), np.float32)
for i in range(len(L)):
    for k in range(KMAX):
        x = MEM[i, k]
        if x < 0: break
        PAIR[i, k] = (len(TRI[i] & TRI[x]) / max(1, len(TRI[i] | TRI[x])), float(bool(NUMS[i] & NUMS[x])), ER[i] * ER[x])

# 긴 기억: 업체 20거래일, 섹터 5·20거래일 (전 거래일까지, 뉴스 지금 점수만)
cm = np.full((len(COS), len(ds)), np.nan); cn = np.zeros((len(COS), len(ds)))
for (d, c), ii in grp.items():
    cm[COS.index(c), di[d]] = NOW[ii].mean(); cn[COS.index(c), di[d]] = len(ii)
by = b.assign("ko", days, IT, ("daily", "sector"), since=ds[0])
sm = np.full((len(SECS), len(ds)), np.nan)
for d in ds:
    acc = collections.defaultdict(list)
    for t, it, _ in by.get(d, []):
        for s in S.tag(t, D, "ko")[0]: acc[s].append(it["s"])
    for s, v in acc.items():
        if len(v) >= 3: sm[SECS.index(s), di[d]] = np.mean(v)
win = lambda A, r, k, n: np.nanmean(A[r, max(0, k - n):k]) if k and np.isfinite(A[r, max(0, k - n):k]).any() else 0.0
LONG = np.array([[win(cm, CID[i], di[day[i]], 20), np.mean(cn[CID[i], max(0, di[day[i]] - 20):di[day[i]]]) if di[day[i]] else 0.0,
                  win(sm, SID[i], di[day[i]], 5), win(sm, SID[i], di[day[i]], 20)] for i in range(len(L))], np.float32)


# 누구 입장인가: 대상 업체가 처음 나오는지, 다른 사전 업체 수, 이름 위치, 비교 낱말, 비교 낱말 + 대상이 앞
def where(t):
    t = S.STRIP["ko"].sub(" ", t); at = {}
    for pat, sec, co in D:
        mm = pat.search(t)
        if mm and co:
            at.setdefault(co, mm.start()); t = pat.sub(lambda z: " " * len(z.group()), t)
    return at


PERS = np.zeros((len(L), 5), np.float32)
for i, (t, c) in enumerate(zip(T, CO)):
    at = where(t); me = at.get(c, 0); first = float(me == min(at.values())) if at else 1.0; cmpw = float(bool(CMP.search(t)))
    PERS[i] = (first, len(at) - 1 if c in at else len(at), me / max(1, len(t)), cmpw, cmpw * first)
print(f"기억 라벨 {len(L):,}. 여러 업체 제목 {np.mean(PERS[:, 1] > 0):.0%}, 비교 낱말 {PERS[:, 3].mean():.1%}")

KEY = np.hstack([ACT, TOH, ER[:, None]])
VAL = np.hstack([ACT, NOW[:, None], NG[:, None], PS[:, None], MKT[:, None]])
OWN = np.hstack([ACT, NOW[:, None], NG[:, None], PS[:, None], MKT[:, None]])
te = (day >= TE0) & (day <= TE1); va = (day >= VA0) & (day <= VA1); tr = ~te & ~va
nz = lambda X: (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
KEYt, VALt, OWNt, LONGt, PERSt = (torch.tensor(nz(X).astype(np.float32)) for X in (KEY, VAL, OWN, LONG, PERS))
MEMt, PREVt, Yt, PAIRt = torch.tensor(MEM), torch.tensor(PREV), torch.tensor(Y), torch.tensor(PAIR)
CIDt, SIDt = torch.tensor(CID), torch.tensor(SID)
FLIPt = torch.tensor(((np.abs(Y) >= 0.5) & (np.sign(Y) * np.sign(NOW) < 0)).astype(np.float32))
print(f"학습 {tr.sum():,} · 고르기 {va.sum():,} · 시험 {te.sum():,} (시험 {TE0} ~ {TE1})")


class Att(torch.nn.Module):  # use: 'emb' 고유 벡터, 'long' 긴 기억, 'pers' 입장
    def __init__(s, use, d=16):
        super().__init__(); s.use = use
        s.q = torch.nn.Linear(KEY.shape[1], d); s.k = torch.nn.Linear(KEY.shape[1] + 1, d); s.v = torch.nn.Linear(VAL.shape[1] + 4, d); s.pb = torch.nn.Linear(3, 1, bias=False)
        s.null = torch.nn.Parameter(torch.zeros(d)); s.d = d
        s.ce = torch.nn.Embedding(len(COS), 8); s.se = torch.nn.Embedding(len(SECS), 4)
        extra = (12 if "emb" in use else 0) + (4 if "long" in use else 0) + (5 if "pers" in use else 0)
        s.out = torch.nn.Sequential(torch.nn.Linear(OWN.shape[1] + d + extra, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))

    def forward(s, ii):
        m = MEMt[ii]; ok = m >= 0; mm = m.clamp(min=0); pv = PREVt[ii][..., None]; pr = PAIRt[ii]
        k = s.k(torch.cat([KEYt[mm], pv], -1)); v = s.v(torch.cat([VALt[mm], pv, pr], -1)); q = s.q(KEYt[ii])[:, None]
        sc = ((q * k).sum(-1) / s.d ** 0.5 + s.pb(pr).squeeze(-1)).masked_fill(~ok, -1e9)
        sc = torch.cat([sc, (q.squeeze(1) * s.null).sum(-1, keepdim=True) / s.d ** 0.5], 1)
        w = torch.softmax(sc, 1); mem = (w[:, :-1, None] * v).sum(1)
        xs = [OWNt[ii], mem] + ([s.ce(CIDt[ii]), s.se(SIDt[ii])] if "emb" in s.use else []) + ([LONGt[ii]] if "long" in s.use else []) + ([PERSt[ii]] if "pers" in s.use else [])
        return s.out(torch.cat(xs, 1)).squeeze(1)


def dirs(pr, m_):
    y, x = Y[m_], pr[m_]
    return int(((np.abs(y) >= 0.5) & (np.abs(x) >= 0.25) & (np.sign(x) != np.sign(y))).sum())


def fit(use, name):  # 고르기 기간 방향 틀림으로 L2·횟수 (같으면 손실)
    best = (9e9, 9e9, None)
    itr = np.where(tr)[0]; iva = torch.tensor(np.where(va)[0])
    for wd in (1e-4, 1e-3):
        torch.manual_seed(0); m = Att(use); opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=wd)
        for ep in range(30):
            perm = np.random.default_rng(ep).permutation(itr)
            for k in range(0, len(perm), 256):
                ii = torch.tensor(perm[k:k + 256]); w = 1 + 2 * FLIPt[ii]
                loss = (w * (m(ii) - Yt[ii]) ** 2).sum() / w.sum()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pv_ = m(iva).numpy()
            key = (dirs(_full(pv_), va), float(((pv_ - Y[va]) ** 2).mean()))
            if key < best[:2]:
                best = (*key, {a: b_.clone() for a, b_ in m.state_dict().items()}, wd, ep + 1)
    m = Att(use); m.load_state_dict(best[2]); m.eval()
    with torch.no_grad():
        pr = torch.cat([m(torch.tensor(np.arange(k, min(k + 2048, len(L))))) for k in range(0, len(L), 2048)]).numpy()
    print(f"  {name}: L2 {best[3]}, {best[4]}회, 고르기 방향 틀림 {best[0]}, 손실 {best[1]:.4f}")
    return pr


def _full(pv_):  # 고르기 기간 예측을 전체 길이 배열에 놓기
    out = np.zeros(len(L), np.float32); out[va] = pv_; return out


P = {"M0 지금 점수": NOW}
for use, name in ((set(), "M4b 주의 기억 2"), ({"emb", "long", "pers"}, "M5 업체·섹터 기억"), ({"emb", "pers"}, "M5 − 긴 기억"), ({"emb", "long"}, "M5 − 입장")):
    P[name] = fit(use, name)


def metr(pr, m_):
    y, x = Y[m_], pr[m_]; nzz = np.abs(y) >= 0.5
    return {"상관": round(float(np.corrcoef(x, y)[0, 1]), 3), "7단계": round(float(np.mean(V[np.abs(x[:, None] - V).argmin(1)] == y)), 3),
            "방향틀림": dirs(pr, m_), "라벨0아닌": int(nzz.sum())}


multi = PERS[:, 1] > 0
OUT = {"시험_라벨": {k: metr(v, te) | {"방향틀림_여러업체": dirs(v, te & multi), "방향틀림_한업체": dirs(v, te & ~multi)} for k, v in P.items()}}
print(f"\n새 시험 구간 {TE0} ~ {TE1} 라벨 기준 (방향 틀림: 전체 / 한 업체 제목 / 여러 업체 제목):")
for k, v in OUT["시험_라벨"].items():
    print(f"  {k:14} 상관 {v['상관']:+.3f}  7단계 {v['7단계']:.1%}  방향 틀림 {v['방향틀림']} / {v['방향틀림_한업체']} / {v['방향틀림_여러업체']}  (라벨 0 아닌 {v['라벨0아닌']})")

# 섹터·업체 주가 (시험 구간)
E = [d for d in days if TE0 <= d <= TE1]; ei = [days.index(d) for d in E]; kos = b.prices("코스피")[2][ei]
px = collections.defaultdict(dict)
for l in open(j.HOME / "data" / "섹터업체주가.tsv", encoding="utf-8").read().splitlines()[1:]:
    pp = l.split("\t"); px[pp[1]][pp[0]] = float(pp[3])
RET = {n: np.array([100 * (dd[x] / dd[days[i - 1]] - 1) if dd.get(x) and dd.get(days[i - 1]) else np.nan for i, x in zip(ei, E)]) for n, dd in px.items()}
CAP = {l.split("\t")[0]: float(l.split("\t")[2]) for l in open(j.HOME / "data" / "섹터업체시총.tsv", encoding="utf-8").read().splitlines()[1:]}
idx = {(d, c, t): i for i, (d, c, t) in enumerate(zip(day, CO, T))}


def down(pr):
    sec = collections.defaultdict(lambda: collections.defaultdict(list)); com = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in E:
        for t, it, _ in by.get(d, []):
            ss, cos = S.tag(t, D, "ko")
            for s in ss:
                mine = [c for c in cos if home[c] == s and (d, c, t) in idx]
                sec[s][d].append(np.mean([pr[idx[d, c, t]] for c in mine]) if mine else it["s"])
            for c in cos:
                if (d, c, t) in idx: com[c][d].append(pr[idx[d, c, t]])
    rs = {}
    for s in sec:
        mem = [n for n in home if home[n] == s and n in CAP and n in RET]
        R = np.vstack([RET[n] for n in mem]); w = np.array([CAP[n] for n in mem])
        ex = np.array([np.average(R[np.isfinite(R[:, k]), k], weights=w[np.isfinite(R[:, k])]) if np.isfinite(R[:, k]).any() else np.nan for k in range(len(E))]) - kos
        rs[s] = round(ix.corr(np.array([np.mean(sec[s][d]) if len(sec[s][d]) >= 5 else np.nan for d in E]), ex)[0], 3)
    rc = {c: round(ix.corr(x, RET[c] - kos)[0], 3) for c in com if c in RET
          for x in [np.array([np.mean(com[c][d]) if len(com[c][d]) >= 3 else np.nan for d in E])] if np.isfinite(x).sum() >= 15}
    return rs, rc


OUT["시험_주가"] = {}
for k, v in list(P.items()) + [("Haiku 라벨 자체", Y)]:
    rs, rc = down(v)
    OUT["시험_주가"][k] = {"섹터중앙값": round(float(np.nanmedian(list(rs.values()))), 3), "섹터": rs, "업체": rc}
print("\n새 시험 구간 주가 기준 (섹터 하루 상관 중앙값 · 업체 초과 상관):")
for k, v in OUT["시험_주가"].items():
    print(f"  {k:14} 섹터 {v['섹터중앙값']:+.3f} | " + ", ".join(f"{c} {r:+.2f}" for c, r in sorted(v["업체"].items())[:10]))
a, b4 = OUT["시험_라벨"]["M5 업체·섹터 기억"], OUT["시험_라벨"]["M4b 주의 기억 2"]
OUT["채택_M5"] = bool(a["방향틀림"] <= 0.85 * b4["방향틀림"] and a["상관"] >= b4["상관"]
                    and OUT["시험_주가"]["M5 업체·섹터 기억"]["섹터중앙값"] >= OUT["시험_주가"]["M4b 주의 기억 2"]["섹터중앙값"] - 0.01)
print(f"\nM5로 바꿈: {'예' if OUT['채택_M5'] else '아니오'}")
json.dump(OUT, open(j.HOME / "실험" / "결과" / "주의기억3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
