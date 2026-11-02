# 판정층 은닉 채널 실험 (2026-10-31, 사용자 요청): CNN 채널 가지치기·증폭 방식을 지금 CMSM 주력에 그대로 적용.
# 채널 = tanh 뒤 은닉 칸 (한국어 64, 영어 512). 1) 활성 양상 2) 채널 하나씩 0으로 → 손실 증가 3) 중요 채널 상위 10%
# 4) 다른 가중치는 고정, 고른 채널에만 학습 가능한 배율 alpha (활성 × alpha) 5) Opus 검증 1,000에서 원래 모델과 비교.
# 대조군: 같은 수의 무작위 채널에 alpha (씨앗 3개). 상관: CorrSteer(ICML 2026)처럼 활성-정답 상관으로 고른 채널. 고르기·alpha 학습은 Opus 3차 라벨(주력 학습에 Opus 정답으로 안 쓴 것)을 반씩.
# python 실험/채널.py ko|en
import sys, json
import numpy as np, torch
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j, evaluate as e
sys.stdout.reconfigure(encoding="utf-8")
LANG = sys.argv[1]
NAME = {"ko": "ko_v2-감성", "en": "en_v2-감성"}[LANG]
V = np.array([-2, -1, -0.5, 0, 0.5, 1, 2], float)
p, T, meta = j.load(NAME)
EN = meta["kind"] == "cmsm_en"
tsv = lambda f: [l.rstrip("\n").split("\t") for l in open(j.HOME / "labeled" / f, encoding="utf-8").read().splitlines()[1:] if l.strip()]
ncol = 2 if EN else 1
TRAIN = tsv({"ko": "ko_ai_opus_과거3.tsv", "en": "en_ai_opus_과거.tsv"}[LANG])
CHECK = tsv({"ko": "ko_opus검증1000.tsv", "en": "en_opus검증1000.tsv"}[LANG])


def hidden(texts):  # tanh(h): 판정층을 단위 행렬로 바꿔 끼우면 logits 함수가 은닉 활성을 그대로 낸다
    H = p["out_w"].shape[-1]
    q = dict(p, out_w=np.eye(H, dtype=np.float32)[None] if EN else np.eye(H, dtype=np.float32), out_b=np.zeros((1, H) if EN else H, np.float32))
    f = j.cmsm_en_logits if EN else j.cmsm_logits
    texts = [j.clean(t) for t in texts]
    return np.vstack([f(q, texts[i:i + 2000], meta).reshape(-1, H) for i in range(0, len(texts), 2000)])


heads = range(len(meta["T_heads"])) if EN else [0]
Tk = meta["T_heads"] if EN else [T]
Sk = meta["stretch_heads"] if EN else [meta["stretch"]]
W = [torch.tensor(p["out_w"][k] if EN else p["out_w"]) for k in heads]
B = [torch.tensor(p["out_b"][k] if EN else p["out_b"]) for k in heads]


def logits(a, alpha):  # a: (n, H) 활성, alpha: (H,) → 판정층마다 (n, 7)
    return [(a * alpha) @ W[k].T + B[k] for k in heads]


def loss(a, y, alpha):  # 판정층마다 온도로 나눈 교차 엔트로피 합
    return sum(torch.nn.functional.cross_entropy(z / Tk[k], y[:, k]) for k, z in zip(heads, logits(a, alpha)))


def scores(texts, a, alpha):  # evaluate.scores와 같은 뒤처리
    with torch.no_grad():
        zs = logits(torch.tensor(a), torch.tensor(alpha))
    out = []
    for k, z in zip(heads, zs):
        s = np.clip(Sk[k] * (j.softmax(z.numpy() / Tk[k]) @ V), -2, 2)
        s = j.soften(texts, j.guard(texts, s, e.GUARD, meta.get("guard_lexicon"))[0])[0]
        out.append(np.where(j.level_only(texts), 0.0, s))
    return np.stack(out, 1)


def metrics(s, y):
    res = {}
    for k in heads:
        m = e.screen(s[:, k], y[:, k]); s7 = V[np.abs(s[:, k][:, None] - V).argmin(1)]; nz = y[:, k] != 0
        res[["미국", "한국"][k] if EN else "감성"] = {"7단계": float(np.mean(s7 == y[:, k])), "맞음": m["맞음"], "헛경보": m["헛경보"], "놓침": m["놓침"],
            "방향틀림3": int(((np.abs(s[:, k]) >= 0.25) & nz & (np.sign(s[:, k]) * np.sign(y[:, k]) < 0)).sum()), "0아닌_0으로": float(np.mean(s7[nz] == 0))}
    return res


tr_t = [r[-1] for r in TRAIN]
tr_y = np.abs(np.array([[float(x) for x in r[:ncol]] for r in TRAIN])[:, :, None] - V).argmin(-1)
perm = np.random.default_rng(2038).permutation(len(tr_t)); A_, B_ = perm[: len(perm) // 2], perm[len(perm) // 2:]
a_tr = hidden(tr_t); ck_t = [r[-1] for r in CHECK]; a_ck = hidden(ck_t)
ck_y = np.array([[float(x) for x in r[:ncol]] for r in CHECK])
H = a_tr.shape[1]
print(f"{NAME}: 채널 {H}개, 고르기·학습 {len(A_):,} / 멈춤 판단 {len(B_):,} / Opus 검증 {len(ck_t):,}")

# 1) 활성 양상
std, mabs, sat = a_tr.std(0), np.abs(a_tr).mean(0), (np.abs(a_tr) > 0.95).mean(0)
print(f"활성: 평균 |a| {mabs.mean():.3f}, 포화(|a|>0.95) 비율 평균 {sat.mean():.1%}, 거의 안 움직이는 채널(표준편차<0.02) {int((std < 0.02).sum())}개, "
      f"표준편차 상위 5 {np.round(np.sort(std)[::-1][:5], 3).tolist()}")

# 2) 채널 하나씩 0으로 → 손실 증가 (고르기 몫 A)
aA, yA = torch.tensor(a_tr[A_]), torch.tensor(tr_y[A_])
one = torch.ones(H)
with torch.no_grad():
    base = float(loss(aA, yA, one))
    inc = np.array([float(loss(aA, yA, torch.where(torch.arange(H) == c, 0.0, 1.0))) - base for c in range(H)])
order = np.argsort(-inc)
K = max(1, H // 10)
print(f"0으로 지웠을 때 손실 증가: 원래 {base:.4f}, 상위 5 {np.round(inc[order[:5]], 4).tolist()}, 중앙값 {np.median(inc):.5f}, "
      f"오히려 줄어든 채널 {int((inc < 0).sum())}개 (가장 많이 준 것 {inc.min():.4f})")
print(f"상위 {K}개 채널이 전체 손실 증가 합의 {inc[order[:K]].sum() / inc[inc > 0].sum():.0%}")
# 3') CorrSteer식 고르기 (2026-10-31 추가): 활성이 정답 감성값과 가장 강하게 상관된 채널 (판정층마다 |상관|의 최댓값)
yv = V[tr_y[A_]]  # (n, 판정층 수)
cor = np.nan_to_num(np.array([[abs(np.corrcoef(a_tr[A_, c], yv[:, k])[0, 1]) for k in range(yv.shape[1])] for c in range(H)]).max(1))
corder = np.argsort(-cor)
print(f"상관 상위 {K}개와 지우기 상위 {K}개 겹침 {len(set(corder[:K]) & set(order[:K]))}개, 상관 상위 5 {np.round(cor[corder[:5]], 3).tolist()}")


# 4) 고른 채널에만 alpha (나머지 가중치 고정)
def fit(sel):
    mask = torch.zeros(H); mask[sel] = 1
    al = torch.zeros(H, requires_grad=True)  # alpha = 1 + mask·al
    opt = torch.optim.Adam([al], lr=0.01)
    aB, yB = torch.tensor(a_tr[B_]), torch.tensor(tr_y[B_])
    best, keep, bad = float("inf"), None, 0
    for it in range(500):
        opt.zero_grad(); l = loss(aA, yA, 1 + mask * al); l.backward(); opt.step()
        with torch.no_grad():
            vl = float(loss(aB, yB, 1 + mask * al))
        if vl < best - 1e-5:
            best, keep, bad = vl, (1 + mask * al).detach().clone(), 0
        else:
            bad += 1
            if bad >= 20:
                break
    return keep.numpy(), best


res = {"모델": NAME, "채널": H, "K": K, "중요채널": order[:K].tolist(), "손실증가": inc.tolist()}
with torch.no_grad():
    res["멈춤몫손실_원래"] = float(loss(torch.tensor(a_tr[B_]), torch.tensor(tr_y[B_]), one))
res["원래"] = metrics(scores(ck_t, a_ck, np.ones(H, np.float32)), ck_y)
al, vl = fit(order[:K]); res["중요"] = metrics(scores(ck_t, a_ck, al.astype(np.float32)), ck_y)
res["중요_alpha"] = np.round(al[order[:K]], 3).tolist(); res["중요_멈춤몫손실"] = vl
al, vl = fit(corder[:K]); res["상관"] = metrics(scores(ck_t, a_ck, al.astype(np.float32)), ck_y); res["상관_멈춤몫손실"] = vl; res["상관채널"] = corder[:K].tolist()
for s in range(3):
    sel = np.random.default_rng(s).permutation(H)[:K]
    al, vl = fit(sel); res[f"무작위{s}"] = metrics(scores(ck_t, a_ck, al.astype(np.float32)), ck_y); res[f"무작위{s}_멈춤몫손실"] = vl
al, vl = fit(np.arange(H)); res["전부"] = metrics(scores(ck_t, a_ck, al.astype(np.float32)), ck_y); res["전부_멈춤몫손실"] = vl
for k in ("원래", "중요", "상관", "무작위0", "무작위1", "무작위2", "전부"):
    print(k, res.get(f"{k}_멈춤몫손실", res["멈춤몫손실_원래"] if k == "원래" else None), json.dumps(res[k], ensure_ascii=False))
print("중요 채널 alpha:", res["중요_alpha"])
json.dump(res, open(j.HOME / "실험" / "결과" / f"채널_{LANG}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
