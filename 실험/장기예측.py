# 감성 지수의 장거리 예측력 (2026-10-29): k일 평균 지수 → 앞으로 k일 누적 등락, 지난 k일 등락 통제.
# 겹치는 창은 Newey-West 표준오차, 겹치지 않는 창은 k거래일마다 하나씩. docs/실험기록.md 같은 날.
# python 실험/장기예측.py 한국어_K2v 영어_F4_미국_SP500 …   (data/지수_<이름>.tsv, 실험/지수_점수.py가 만든 것)
import sys, math
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
sys.stdout.reconfigure(encoding="utf-8")
p_of = lambda t: math.erfc(abs(t) / math.sqrt(2))


def ols(y, X, lag=None):
    """표준화한 y·X(상수 칸 포함)로 OLS → (x 계수, t). lag가 있으면 Newey-West 표준오차."""
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    if lag is None:
        V = XtXi * (e @ e) / (len(y) - X.shape[1])
    else:
        Xe = X * e[:, None]
        S = Xe.T @ Xe
        for l in range(1, lag + 1):
            G = Xe[l:].T @ Xe[:-l]
            S += (1 - l / (lag + 1)) * (G + G.T)
        V = XtXi @ S @ XtXi
    return b[1], b[1] / math.sqrt(V[1, 1])


def roll(v, k, f):
    out = np.full(len(v), np.nan)
    for i in range(k - 1, len(v)):
        w = v[i - k + 1:i + 1]
        w = w[~np.isnan(w)]
        if len(w) > k // 2:
            out[i] = f(w)
    return out


z = lambda v: (v - np.nanmean(v)) / np.nanstd(v)
print("| 지수 | k일 | 겹침: 계수 (t, p, 표본) | 겹치지 않음: 계수 (t, p, 표본) | 판단 |")
print("|---|---|---|---|---|")
for name in sys.argv[1:]:
    rows = [l.split("\t") for l in (j.HOME / "data" / f"지수_{name}.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    idx = np.array([float(r[3]) if r[3] != "nan" else np.nan for r in rows])
    R = np.array([float(r[2]) if r[2] != "nan" else np.nan for r in rows])
    for k in (5, 10, 20, 60):
        x = roll(idx, k, np.mean)
        past = roll(R, k, np.sum)
        fut = np.r_[past[k:], [np.nan] * k]  # 내일부터 k일 누적 = k일 뒤의 '지난 k일 누적'
        ok = ~(np.isnan(x) | np.isnan(past) | np.isnan(fut))
        Y, X = z(fut[ok]), np.c_[np.ones(ok.sum()), z(x[ok]), z(past[ok])]
        b1, t1 = ols(Y, X, lag=k)
        sel = np.flatnonzero(ok)[::k]  # 겹치지 않게
        Y2, X2 = z(fut[sel]), np.c_[np.ones(len(sel)), z(x[sel]), z(past[sel])]
        b2, t2 = ols(Y2, X2)
        both = p_of(t1) < 0.05 and p_of(t2) < 0.05 and np.sign(b1) == np.sign(b2)
        print(f"| {name} | {k} | {b1:+.2f} ({t1:+.1f}, {p_of(t1):.3f}, {ok.sum()}) | {b2:+.2f} ({t2:+.1f}, {p_of(t2):.3f}, {len(sel)}) | "
              f"{'후보' + (' (역방향: 나쁜 뉴스 뒤 오름)' if b1 < 0 else ' (같은 방향)') if both else '없음'} |")
