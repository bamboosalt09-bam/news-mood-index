# 주의(attention) 기억 머리 (2026-11-01, docs/실험기록.md 같은 날 '주의 기억'·'주의 기억 2'). 같은 업체의 같은 날·전날 제목 중 닮은(같은 사건) 것만 골라 기억으로 쓴다.
# 2판: 같은 사건 특징(글자 3조각 자카드·공유 숫자·둘 다 실적)을 주의 점수와 값에 더하고, M4a/M4b(뒤집을 제목 무게 3)를 고르기 기간 방향 틀림으로 고름
# python 실험/주의기억.py   → 실험/결과/주의기억2.json·.log
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
TR_END, VA_END, KMAX = "2026-07-31", "2026-08-31", 100
V = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
NEG = re.compile(r"어닝\s?쇼크|실적\s?쇼크|실적\s?충격|(기대|예상|추정|컨센서스)\s?치?\s?하회|못\s?미쳐|못\s?미친|기대\s?못\s?미")
POS = re.compile(r"어닝\s?서프라이즈|깜짝\s?실적|(기대|예상|추정|컨센서스)\s?치?\s?상회|웃돌")
EARN = re.compile(r"영업익|영업이익|순이익|순익|매출|실적|어닝|분기")

L = [l.rstrip("\n").split("\t") for l in open(j.HOME / "labeled" / "ko_기억라벨.tsv", encoding="utf-8").read().splitlines()[1:]]
IT = b.items("ko", "최종")
L = [r for r in L if r[3] in IT]
day = np.array([r[0] for r in L]); CO = [r[1] for r in L]; T = [r[3] for r in L]; Y = np.array([float(r[2]) for r in L], np.float32)
NOW = np.array([IT[t]["s"] for t in T], np.float32); MKT = np.array([IT[t]["mkt"] for t in T], np.float32)
NG = np.array([bool(NEG.search(t)) for t in T], np.float32); PS = np.array([bool(POS.search(t)) for t in T], np.float32)
ER = np.array([bool(EARN.search(t)) for t in T], np.float32)
TOP = sorted({IT[t]["topic"] for t in T}); TOH = np.eye(len(TOP), dtype=np.float32)[[TOP.index(IT[t]["topic"]) for t in T]]

p, Tt, meta = j.load(j.MAIN["ko-감성"])
H = p["out_w"].shape[-1]
q_ = dict(p, out_w=np.eye(H, dtype=np.float32), out_b=np.zeros(H, np.float32))
uniq = sorted(set(T)); pos = {t: i for i, t in enumerate(uniq)}
CU = [j.clean(t) for t in uniq]
AU = np.vstack([j.cmsm_logits(q_, CU[i:i + 2000], meta) for i in range(0, len(CU), 2000)]).astype(np.float32)
ACT = AU[[pos[t] for t in T]]

grp = collections.defaultdict(list)
for i, k in enumerate(zip(day, CO)):
    grp[k].append(i)
ds = sorted(set(day)); prevd = {d: ds[k - 1] if k else None for k, d in enumerate(ds)}
rng = np.random.default_rng(0)
MEM = np.full((len(L), KMAX), -1, np.int64); PREV = np.zeros((len(L), KMAX), np.float32)
for (d, c), ii in grp.items():
    pv = grp.get((prevd[d], c), [])
    for i in ii:
        same = [x for x in ii if x != i]
        cand = [(x, 0.0) for x in same] + [(x, 1.0) for x in pv]
        if len(cand) > KMAX:  # 넘치면 쇼크·서프라이즈·시황 제목 먼저, 나머지는 무작위
            cand.sort(key=lambda z: -(NG[z[0]] + PS[z[0]] + MKT[z[0]] + rng.random() * 0.5)); cand = cand[:KMAX]
        for k, (x, f) in enumerate(cand):
            MEM[i, k] = x; PREV[i, k] = f
NUM = re.compile(r"\d[\d,.]{2,}")
TRI = [{t[i:i + 3] for i in range(len(t) - 2)} for t in (j.clean(x).replace(" ", "") for x in T)]
NUMS = [{n.replace(",", "").rstrip(".") for n in NUM.findall(t)} for t in T]
PAIR = np.zeros((len(L), KMAX, 3), np.float32)
for i in range(len(L)):
    for k in range(KMAX):
        x = MEM[i, k]
        if x < 0: break
        a, c_ = TRI[i], TRI[x]
        PAIR[i, k] = (len(a & c_) / max(1, len(a | c_)), float(bool(NUMS[i] & NUMS[x])), ER[i] * ER[x])
print(f"같은 사건 특징: 자카드 평균 {PAIR[..., 0][MEM >= 0].mean():.3f}, 숫자 공유 비율 {PAIR[..., 1][MEM >= 0].mean():.3f}")
KEY = np.hstack([ACT, TOH, ER[:, None]])
VAL = np.hstack([ACT, NOW[:, None], NG[:, None], PS[:, None], MKT[:, None]])
OWN = np.hstack([ACT, NOW[:, None], NG[:, None], PS[:, None], MKT[:, None]])
tr, va, te = day <= TR_END, (day > TR_END) & (day <= VA_END), day > VA_END
nz = lambda X: (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
KEYt, VALt, OWNt = (torch.tensor(nz(X).astype(np.float32)) for X in (KEY, VAL, OWN))
MEMt, PREVt, Yt, PAIRt = torch.tensor(MEM), torch.tensor(PREV), torch.tensor(Y), torch.tensor(PAIR)
FLIP = ((np.abs(Y) >= 0.5) & (np.sign(Y) * np.sign(NOW) < 0)).astype(np.float32); FLIPt = torch.tensor(FLIP)
print(f"기억 라벨 {len(L):,}, 기억 칸 평균 {(MEM >= 0).sum(1).mean():.1f}개 (최대 {KMAX}), 학습 {tr.sum():,} · 고르기 {va.sum():,} · 시험 {te.sum():,}")


class Att(torch.nn.Module):
    def __init__(s, d=16):
        super().__init__()
        s.q = torch.nn.Linear(KEY.shape[1], d); s.k = torch.nn.Linear(KEY.shape[1] + 1, d); s.v = torch.nn.Linear(VAL.shape[1] + 1 + 3, d); s.pb = torch.nn.Linear(3, 1, bias=False)
        s.null = torch.nn.Parameter(torch.zeros(d)); s.d = d
        s.out = torch.nn.Sequential(torch.nn.Linear(OWN.shape[1] + d, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))

    def forward(s, ii, give_w=False):
        m = MEMt[ii]; ok = m >= 0; mm = m.clamp(min=0); pv = PREVt[ii][..., None]
        pr = PAIRt[ii]; k = s.k(torch.cat([KEYt[mm], pv], -1)); v = s.v(torch.cat([VALt[mm], pv, pr], -1))
        q = s.q(KEYt[ii])[:, None]
        sc = (q * k).sum(-1) / s.d ** 0.5 + s.pb(pr).squeeze(-1)
        sc = sc.masked_fill(~ok, -1e9)
        sc = torch.cat([sc, (q.squeeze(1) * s.null).sum(-1, keepdim=True) / s.d ** 0.5], 1)  # 마지막 칸 = 안 가져옴
        w = torch.softmax(sc, 1)
        mem = (w[:, :-1, None] * v).sum(1)
        y = s.out(torch.cat([OWNt[ii], mem], 1)).squeeze(1)
        return (y, w) if give_w else y


def vdir(pr, m_):
    y, x = Y[m_], pr[m_]
    return int(((np.abs(y) >= 0.5) & (np.abs(x) >= 0.25) & (np.sign(x) != np.sign(y))).sum())


def fit(flipw):
    best = (9e9, None)
    itr = np.where(tr)[0]; iva = torch.tensor(np.where(va)[0])
    for wd in (1e-4, 1e-3):
        torch.manual_seed(0); m = Att(); opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=wd)
        for ep in range(30):
            perm = np.random.default_rng(ep).permutation(itr)
            for k in range(0, len(perm), 256):
                ii = torch.tensor(perm[k:k + 256]); w = 1 + (flipw - 1) * FLIPt[ii]
                loss = (w * (m(ii) - Yt[ii]) ** 2).sum() / w.sum()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                vl = float(((m(iva) - Yt[iva]) ** 2).mean())
            if vl < best[0]:
                best = (vl, {a: b_.clone() for a, b_ in m.state_dict().items()}, wd, ep + 1)
    m = Att(); m.load_state_dict(best[1]); m.eval()
    with torch.no_grad():
        out = [m(torch.tensor(np.arange(k, min(k + 2048, len(L)))), True) for k in range(0, len(L), 2048)]
    pr = torch.cat([o[0] for o in out]).numpy()
    print(f"  무게 {flipw}: L2 {best[2]}, {best[3]}회, 고르기 손실 {best[0]:.4f}, 고르기 방향 틀림 {vdir(pr, va)}")
    return pr, torch.cat([o[1][:, -1] for o in out]).numpy(), f"뒤집기 무게 {flipw}, L2 {best[2]}, {best[3]}회", best[0]


fa, fb = fit(1), fit(3)
pick = min((fa, "M4a"), (fb, "M4b"), key=lambda z: (vdir(z[0][0], va), z[0][3]))
print("고름:", pick[1])
PA, NULLW, setting, _ = pick[0]; NAME = f"M4 주의 기억 2 ({pick[1]})"
print("고른 설정:", setting)


def metr(pr, m_):
    y, x = Y[m_], pr[m_]; nzz = np.abs(y) >= 0.5
    return {"상관": round(float(np.corrcoef(x, y)[0, 1]), 3), "7단계": round(float(np.mean(V[np.abs(x[:, None] - V).argmin(1)] == y)), 3),
            "방향틀림": int((nzz & (np.abs(x) >= 0.25) & (np.sign(x) != np.sign(y))).sum()), "라벨0아닌": int(nzz.sum())}


prev = json.load(open(j.HOME / "실험" / "결과" / "기억모델.json", encoding="utf-8"))
OUT = {"시험_라벨": json.load(open(j.HOME / "실험" / "결과" / "주의기억.json", encoding="utf-8"))["시험_라벨"] | {NAME: metr(PA, te)}, "설정": setting}
print("\n시험 기간 라벨 기준:")
for k, v in OUT["시험_라벨"].items():
    print(f"  {k:10} 상관 {v['상관']:+.3f}  7단계 {v['7단계']:.1%}  방향 틀림 {v['방향틀림']} / {v['라벨0아닌']}")

# 섹터·업체 주가 (실험/기억모델.py와 같은 계산)
days, _, kchg = b.prices("코스피")
E = [d for d in days if VA_END < d <= "2026-10-30"]; ei = [days.index(d) for d in E]; kos = kchg[ei]
px = collections.defaultdict(dict)
for l in open(j.HOME / "data" / "섹터업체주가.tsv", encoding="utf-8").read().splitlines()[1:]:
    pp = l.split("\t"); px[pp[1]][pp[0]] = float(pp[3])
RET = {n: np.array([100 * (dd[x] / dd[days[i - 1]] - 1) if dd.get(x) and dd.get(days[i - 1]) else np.nan for i, x in zip(ei, E)]) for n, dd in px.items()}
CAP = {l.split("\t")[0]: float(l.split("\t")[2]) for l in open(j.HOME / "data" / "섹터업체시총.tsv", encoding="utf-8").read().splitlines()[1:]}
D = S.load("ko"); home = {co: s for _, s, co in D if co}
by = b.assign("ko", days, IT, ("daily", "sector"), since=E[0])
idx = {(d, c, t): i for i, (d, c, t) in enumerate(zip(day, CO, T))}
sec = collections.defaultdict(lambda: collections.defaultdict(list)); com = collections.defaultdict(lambda: collections.defaultdict(list))
for d in E:
    for t, it, _ in by.get(d, []):
        ss, cos = S.tag(t, D, "ko")
        for s in ss:
            mine = [c for c in cos if home[c] == s and (d, c, t) in idx]
            sec[s][d].append(np.mean([PA[idx[d, c, t]] for c in mine]) if mine else it["s"])
        for c in cos:
            if (d, c, t) in idx: com[c][d].append(PA[idx[d, c, t]])
rs = {}
for s in sec:
    mem = [n for n in home if home[n] == s and n in CAP and n in RET]
    R = np.vstack([RET[n] for n in mem]); w = np.array([CAP[n] for n in mem])
    ex = np.array([np.average(R[np.isfinite(R[:, k]), k], weights=w[np.isfinite(R[:, k])]) if np.isfinite(R[:, k]).any() else np.nan for k in range(len(E))]) - kos
    rs[s] = round(ix.corr(np.array([np.mean(sec[s][d]) if len(sec[s][d]) >= 5 else np.nan for d in E]), ex)[0], 3)
rc = {c: round(ix.corr(x, RET[c] - kos)[0], 3) for c in com if c in RET
      for x in [np.array([np.mean(com[c][d]) if len(com[c][d]) >= 3 else np.nan for d in E])] if np.isfinite(x).sum() >= 20}
OUT["시험_주가"] = json.load(open(j.HOME / "실험" / "결과" / "주의기억.json", encoding="utf-8"))["시험_주가"] | {NAME: {"섹터중앙값": round(float(np.nanmedian(list(rs.values()))), 3), "섹터": rs, "업체": rc}}
print("\n시험 기간 주가 기준 (섹터 하루 상관 중앙값 · 업체 초과 상관):")
for k, v in OUT["시험_주가"].items():
    print(f"  {k:12} 섹터 {v['섹터중앙값']:+.3f} | " + ", ".join(f"{c} {r:+.2f}" for c, r in sorted(v["업체"].items())[:12]))
m0, m3 = OUT["시험_라벨"]["M0 지금 점수"], OUT["시험_라벨"][NAME]
OUT["채택"] = bool(m3["방향틀림"] <= 0.7 * m0["방향틀림"] and m3["상관"] > max(m0["상관"], OUT["시험_라벨"]["M1 제목만 머리"]["상관"])
                 and OUT["시험_주가"][NAME]["섹터중앙값"] >= OUT["시험_주가"]["M0 지금 점수"]["섹터중앙값"] - 0.01)
print(f"\n채택: {'예' if OUT['채택'] else '아니오'}  ('안 가져옴' 칸 무게 평균 {NULLW[te].mean():.2f})")
print("\n10/30 기아 (라벨 / 주의 기억 / 지금 / 안 가져옴 무게):")
for i in range(len(L)):
    if (day[i], CO[i]) == ("2026-10-30", "기아") and i % 2 == 0:
        print(f"  {Y[i]:+.1f} / {PA[i]:+.2f} / {NOW[i]:+.2f} / {NULLW[i]:.2f}  {T[i][:55]}")
json.dump(OUT, open(j.HOME / "실험" / "결과" / "주의기억2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
