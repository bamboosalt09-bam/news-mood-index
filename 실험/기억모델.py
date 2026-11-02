# 기억 판정층: Haiku 기억 라벨을 주력 활성 위 작은 MLP로 옮기기 (2026-10-31, docs/실험기록.md 같은 날 '기억 판정층').
# python 실험/기억모델.py   → 실험/결과/기억모델.json·.log
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
torch.set_num_threads(9); torch.manual_seed(0)
TR_END, VA_END = "2026-07-31", "2026-08-31"
V = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
NEG = re.compile(r"어닝\s?쇼크|실적\s?쇼크|실적\s?충격|(기대|예상|추정|컨센서스)\s?치?\s?하회|못\s?미쳐|못\s?미친|기대\s?못\s?미")
POS = re.compile(r"어닝\s?서프라이즈|깜짝\s?실적|(기대|예상|추정|컨센서스)\s?치?\s?상회|웃돌")

L = [l.rstrip("\n").split("\t") for l in open(j.HOME / "labeled" / "ko_기억라벨.tsv", encoding="utf-8").read().splitlines()[1:]]
IT = b.items("ko", "최종")
L = [r for r in L if r[3] in IT]
day = np.array([r[0] for r in L]); CO = [r[1] for r in L]; T = [r[3] for r in L]; Y = np.array([float(r[2]) for r in L], np.float32)
NOW = np.array([IT[t]["s"] for t in T], np.float32); MKT = np.array([IT[t]["mkt"] for t in T])
print(f"기억 라벨 {len(L):,}, 업체·날 묶음 {len(set(zip(day, CO))):,}")

# 주력 은닉 활성 (실험/업체모델.py와 같은 방법, 검산 포함)
p, Tt, meta = j.load(j.MAIN["ko-감성"])
H = p["out_w"].shape[-1]
q = dict(p, out_w=np.eye(H, dtype=np.float32), out_b=np.zeros(H, np.float32))
uniq = sorted(set(T)); pos = {t: i for i, t in enumerate(uniq)}
CU = [j.clean(t) for t in uniq]
AU = np.vstack([j.cmsm_logits(q, CU[i:i + 2000], meta) for i in range(0, len(CU), 2000)]).astype(np.float32)
ref = j.score_of(*j.predict(j.MAIN["ko-감성"], uniq[:200]))
print(f"검산: 활성 → 판정층 → 점수 최대 차 {np.abs(j.score_of(j.softmax((AU[:200] @ p['out_w'].T + p['out_b']) / Tt), meta) - ref).max():.1e}")
ACT = AU[[pos[t] for t in T]]

# 묶음 기억 특징
grp = collections.defaultdict(list)
for i, k in enumerate(zip(day, CO)):
    grp[k].append(i)
days_sorted = sorted(set(day)); prevd = {d: days_sorted[k - 1] if k else None for k, d in enumerate(days_sorted)}
G = np.zeros((len(L), 3 * H + 9), np.float32)
for (d, c), ii in grp.items():
    a, s = ACT[ii], NOW[ii]
    pv = grp.get((prevd[d], c), [])
    f = np.concatenate([a.mean(0), a.min(0), a.max(0),
                        [s.mean(), s.min(), s.max(), np.log1p(len(ii)),
                         sum(bool(NEG.search(T[x])) for x in ii), sum(bool(POS.search(T[x])) for x in ii),
                         np.mean([NOW[x] for x in ii if MKT[x]]) if MKT[ii].any() else 0.0,
                         np.mean(NOW[pv]) if pv else 0.0, float(bool(pv))]])
    G[ii] = f
X1 = np.hstack([ACT, NOW[:, None]]); X2 = np.hstack([X1, G])
tr, va, te = day <= TR_END, (day > TR_END) & (day <= VA_END), day > VA_END
print(f"학습 {tr.sum():,} · 고르기 {va.sum():,} · 시험 {te.sum():,}")


def fit(X):
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
    Z = torch.tensor((X - mu) / sg); y = torch.tensor(Y)
    best = (9e9, None)
    for wd in (1e-4, 1e-3, 1e-2):
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
        opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=wd)
        itr = np.where(tr)[0]
        for ep in range(30):
            perm = np.random.default_rng(ep).permutation(itr)
            for k in range(0, len(perm), 256):
                ii = perm[k:k + 256]; loss = ((m(Z[ii]).squeeze(1) - y[ii]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                vl = float(((m(Z[va]).squeeze(1) - y[va]) ** 2).mean())
            if vl < best[0]:
                best = (vl, {k_: v.clone() for k_, v in m.state_dict().items()}, wd, ep)
    m.load_state_dict(best[1])
    with torch.no_grad():
        return m(Z).squeeze(1).numpy(), f"L2 {best[2]}, {best[3] + 1}회"


P = {"M0 지금 점수": NOW}
P["M1 제목만 머리"], s1 = fit(X1)
P["M2 기억 머리"], s2 = fit(X2)
print("고른 설정:", s1, "/", s2)


def metr(pr, m_):
    y, x = Y[m_], pr[m_]; nz = np.abs(y) >= 0.5
    return {"상관": round(float(np.corrcoef(x, y)[0, 1]), 3), "7단계": round(float(np.mean(V[np.abs(x[:, None] - V).argmin(1)] == y)), 3),
            "방향틀림": int((nz & (np.abs(x) >= 0.25) & (np.sign(x) != np.sign(y))).sum()), "라벨0아닌": int(nz.sum())}


OUT = {"시험_라벨": {k: metr(v, te) for k, v in P.items()}, "설정": [s1, s2]}
print("\n시험 기간 라벨 기준:")
for k, v in OUT["시험_라벨"].items():
    print(f"  {k:10} 상관 {v['상관']:+.3f}  7단계 {v['7단계']:.1%}  방향 틀림 {v['방향틀림']} / {v['라벨0아닌']}")

# 섹터·업체 주가와 (시험 기간)
days, _, kchg = b.prices("코스피")
E = [d for d in days if d > VA_END and d <= "2026-10-30"]; ei = [days.index(d) for d in E]; kos = kchg[ei]
px = collections.defaultdict(dict)
for l in open(j.HOME / "data" / "섹터업체주가.tsv", encoding="utf-8").read().splitlines()[1:]:
    pp = l.split("\t"); px[pp[1]][pp[0]] = float(pp[3])
RET = {n: np.array([100 * (dd[x] / dd[days[i - 1]] - 1) if dd.get(x) and dd.get(days[i - 1]) else np.nan for i, x in zip(ei, E)]) for n, dd in px.items()}
CAP = {l.split("\t")[0]: float(l.split("\t")[2]) for l in open(j.HOME / "data" / "섹터업체시총.tsv", encoding="utf-8").read().splitlines()[1:]}
D = S.load("ko"); home = {co: s for _, s, co in D if co}
by = b.assign("ko", days, IT, ("daily", "sector"), since=E[0])
idx = {(d, c, t): i for i, (d, c, t) in enumerate(zip(day, CO, T))}


def down(pr, label=False):
    sec = collections.defaultdict(lambda: collections.defaultdict(list)); com = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in E:
        for t, it, _ in by.get(d, []):
            ss, cos = S.tag(t, D, "ko")
            for s in ss:
                mine = [c for c in cos if home[c] == s and (d, c, t) in idx]
                sec[s][d].append(np.mean([(Y if label else pr)[idx[d, c, t]] for c in mine]) if mine else it["s"])
            for c in cos:
                if (d, c, t) in idx: com[c][d].append((Y if label else pr)[idx[d, c, t]])
    rs = {}
    for s in sec:
        mem = [n for n in home if home[n] == s and n in CAP and n in RET]
        R = np.vstack([RET[n] for n in mem]); w = np.array([CAP[n] for n in mem])
        ex = np.array([np.average(R[np.isfinite(R[:, k]), k], weights=w[np.isfinite(R[:, k])]) if np.isfinite(R[:, k]).any() else np.nan for k in range(len(E))]) - kos
        x = np.array([np.mean(sec[s][d]) if len(sec[s][d]) >= 5 else np.nan for d in E])
        rs[s] = round(ix.corr(x, ex)[0], 3)
    rc = {}
    for c in com:
        if c in RET:
            x = np.array([np.mean(com[c][d]) if len(com[c][d]) >= 3 else np.nan for d in E])
            if np.isfinite(x).sum() >= 20:
                rc[c] = round(ix.corr(x, RET[c] - kos)[0], 3)
    return rs, rc


OUT["시험_주가"] = {}
for k, v in list(P.items()) + [("Haiku 라벨 자체", None)]:
    rs, rc = down(v, label=v is None)
    OUT["시험_주가"][k] = {"섹터중앙값": round(float(np.nanmedian(list(rs.values()))), 3), "섹터": rs, "업체": rc}
print("\n시험 기간 주가 기준 (섹터 하루 상관 중앙값 · 업체 초과 상관):")
for k, v in OUT["시험_주가"].items():
    print(f"  {k:12} 섹터 {v['섹터중앙값']:+.3f} | " + ", ".join(f"{c} {r:+.2f}" for c, r in sorted(v["업체"].items(), key=lambda x: x[0])[:12]))
m0, m2 = OUT["시험_라벨"]["M0 지금 점수"], OUT["시험_라벨"]["M2 기억 머리"]
OUT["채택"] = bool(m2["방향틀림"] <= 0.7 * m0["방향틀림"] and m2["상관"] > max(m0["상관"], OUT["시험_라벨"]["M1 제목만 머리"]["상관"])
                 and OUT["시험_주가"]["M2 기억 머리"]["섹터중앙값"] >= OUT["시험_주가"]["M0 지금 점수"]["섹터중앙값"] - 0.01)
print(f"\n채택: {'예' if OUT['채택'] else '아니오'}")
print("\n10/30 기아·10/29 삼성전자 예 (라벨 / M2 / 지금):")
for i in range(len(L)):
    if (day[i], CO[i]) in (("2026-10-30", "기아"), ("2026-10-29", "삼성전자")) and (i % 3 == 0):
        print(f"  {CO[i]} {Y[i]:+.1f} / {P['M2 기억 머리'][i]:+.2f} / {NOW[i]:+.2f}  {T[i][:55]}")
json.dump(OUT, open(j.HOME / "실험" / "결과" / "기억모델.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
