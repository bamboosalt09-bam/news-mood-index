"""작은 AI 실험실: 줄 단위 텍스트를 배워서 비슷한 줄을 지어냅니다.

python train.py          # 학습 + 생성
python train.py check    # 역전파 기울기 검산
"""
import sys
import unicodedata
from pathlib import Path

import numpy as np

# ── 내가 정하는 것들 ─────────────────────────────────────
DATA    = "data/places.txt"  # 한 줄에 예시 하나
CONTEXT = 6       # 다음 글자를 맞힐 때 볼 앞 글자 수 (자모 단위)
EMB     = 8       # 글자 하나를 나타내는 숫자 개수
HIDDEN  = 64      # 은닉층 뉴런 수
STEPS   = 20000
BATCH   = 64
LR      = 0.1     # 절반 지나면 1/10로 줄임
SEED    = 42
SAMPLES = 20
# ─────────────────────────────────────────────────────────

END = "."  # 줄의 시작/끝 표시, 어휘 0번


def load(path):
    with open(Path(__file__).parent / path, encoding="utf-8") as f:
        # 한글 음절을 자모로 쪼개 어휘를 작게 유지 (강 → ㄱ ㅏ ㅇ)
        return [unicodedata.normalize("NFD", l.strip()) for l in f if l.strip()]


def make_xy(lines, stoi):
    X, Y = [], []
    for line in lines:
        ctx = [0] * CONTEXT
        for ch in line + END:
            X.append(ctx)
            Y.append(stoi[ch])
            ctx = ctx[1:] + [stoi[ch]]
    return np.array(X), np.array(Y)


def init(V, rng):
    return {
        "C":  rng.normal(0, 1, (V, EMB)),
        "W1": rng.normal(0, 1, (CONTEXT * EMB, HIDDEN)) / np.sqrt(CONTEXT * EMB),
        "b1": np.zeros(HIDDEN),
        "W2": rng.normal(0, 1, (HIDDEN, V)) * 0.01,
        "b2": np.zeros(V),
    }


def forward(p, X, Y):
    """평균 교차엔트로피 손실과 모든 매개변수의 기울기."""
    n = len(X)
    e = p["C"][X].reshape(n, -1)
    h = np.tanh(e @ p["W1"] + p["b1"])
    logits = h @ p["W2"] + p["b2"]
    prob = np.exp(logits - logits.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    loss = -np.log(prob[np.arange(n), Y]).mean()

    d = prob
    d[np.arange(n), Y] -= 1
    d /= n
    g = {"W2": h.T @ d, "b2": d.sum(0)}
    dh = d @ p["W2"].T * (1 - h**2)
    g["W1"] = e.T @ dh
    g["b1"] = dh.sum(0)
    g["C"] = np.zeros_like(p["C"])
    np.add.at(g["C"], X, (dh @ p["W1"].T).reshape(n, CONTEXT, EMB))
    return loss, g


def sample(p, itos, rng):
    ctx, out = [0] * CONTEXT, []
    while len(out) < 40:
        h = np.tanh(p["C"][ctx].reshape(-1) @ p["W1"] + p["b1"])
        logits = h @ p["W2"] + p["b2"]
        prob = np.exp(logits - logits.max())
        i = rng.choice(len(prob), p=prob / prob.sum())
        if i == 0:
            break
        out.append(itos[i])
        ctx = ctx[1:] + [i]
    return unicodedata.normalize("NFC", "".join(out))


def check():
    rng = np.random.default_rng(0)
    p = init(5, rng)
    X, Y = rng.integers(0, 5, (7, CONTEXT)), rng.integers(0, 5, 7)
    _, g = forward(p, X, Y)
    for k in p:
        for _ in range(5):
            idx = tuple(rng.integers(0, s) for s in p[k].shape)
            old = p[k][idx]
            p[k][idx] = old + 1e-5
            lp, _ = forward(p, X, Y)
            p[k][idx] = old - 1e-5
            lm, _ = forward(p, X, Y)
            p[k][idx] = old
            num = (lp - lm) / 2e-5
            assert abs(num - g[k][idx]) < 1e-6, (k, idx, num, g[k][idx])
    print("기울기 검산 통과")


def main():
    rng = np.random.default_rng(SEED)
    lines = load(DATA)
    itos = [END] + sorted(set("".join(lines)))
    stoi = {c: i for i, c in enumerate(itos)}
    rng.shuffle(lines)
    cut = int(len(lines) * 0.9)
    Xtr, Ytr = make_xy(lines[:cut], stoi)
    Xva, Yva = make_xy(lines[cut:], stoi)
    p = init(len(itos), rng)

    print(f"데이터 {len(lines)}줄, 어휘 {len(itos)}개, 학습 예시 {len(Xtr)}개")
    for k, v in p.items():
        print(f"  {k:3} {str(v.shape):10} {v.size:6}")
    print(f"  매개변수 합계 {sum(v.size for v in p.values())}")
    print(f"찍기 수준 손실 ln({len(itos)}) = {np.log(len(itos)):.3f}\n")

    for step in range(STEPS + 1):
        b = rng.integers(0, len(Xtr), BATCH)
        _, g = forward(p, Xtr[b], Ytr[b])
        lr = LR if step < STEPS // 2 else LR / 10
        for k in p:
            p[k] -= lr * g[k]
        if step % (STEPS // 10) == 0:
            tr, va = forward(p, Xtr, Ytr)[0], forward(p, Xva, Yva)[0]
            print(f"step {step:6}  train {tr:.3f}  val {va:.3f}")

    seen = {unicodedata.normalize("NFC", l) for l in lines}
    print("\n지어낸 것 (* = 데이터에 실제로 있음):")
    for _ in range(SAMPLES):
        s = sample(p, itos, rng)
        print(f"  {s}{' *' if s in seen else ''}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    check() if sys.argv[1:] == ["check"] else main()
