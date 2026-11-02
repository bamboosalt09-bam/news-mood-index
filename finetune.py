"""문장 모델 미세 조정: 사전 학습된 문장 모델 전체 + 해석 사전 특징 → 판정층(은닉 HIDDEN) → 숫자 등급.
judge.py의 고정 모델과 달리 문장 모델 안의 매개변수(약 1억 개)까지 우리 라벨로 조금씩 고친다.

python finetune.py ko 감성                     # labeled/ko.tsv의 감성 → models/ko-감성-미세.pt
python finetune.py ko_more6 감성 --test held  # 학습 끝에 시험 파일로 화면 기준 성능 출력 (아래 HELD 참고)
python finetune.py ko_more7 감성 --base klue/roberta-base --out ko_klue-감성-미세 --test held  # 바탕 모델 바꿔 비교
python finetune.py ko_more7 감성 --base klue/roberta-small --lr 5e-5 --out ko_rsmall-감성-미세  # 문장 모델 학습률 바꾸기

만든 모델은 judge.py가 알아서 쓴다: `python judge.py ask "제목" ko-감성-미세`, 판정하기.bat(감성 자동).
필요: torch, transformers (CPU로 충분. 한국어 3~4천 개 기준 에폭당 5~6분)
"""
import copy
import sys
import time
from pathlib import Path

import numpy as np

import judge as j

# ── 내가 정하는 것들 ─────────────────────────────────────
BASE     = {"ko": "jhgan/ko-sroberta-multitask",   # 언어별 바탕 문장 모델
            "en": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}
EPOCHS   = 3       # 데이터 전체를 몇 바퀴 돌지 (감성은 지금까지 매번 3에폭이 최고). 검증 손실이 가장 낮았던 에폭을 남김
PATIENCE = 1       # 검증 손실이 이만큼 연달아 나빠지면 남은 에폭을 건너뜀 (감성은 보통 3에폭이 최고)
BATCH    = 32      # 한 걸음에 보는 제목 수
LR_ENC   = 2e-5    # 문장 모델 학습률: 작게 (세게 밀면 원래 알던 언어 지식을 잊음)
LR_HEAD  = 1e-3    # 판정층 학습률: 새로 만든 층이라 크게
WD       = 0.01    # 가중치 감쇠
HIDDEN   = 16      # 판정층 은닉 크기
MAX_LEN  = 64      # 제목을 자를 최대 조각 수
VAL      = 0.15    # 검증(조기 종료·온도·배율)으로 떼어 둘 비율
NEUTRAL  = 0.8     # 화면에서 이 안쪽(±)이면 중립. 넓힐수록 헛경보가 줄고 놓침이 는다
SEED     = 0
# ─────────────────────────────────────────────────────────


def read(data, col):
    path = Path(data) if Path(data).exists() else j.HOME / "labeled" / f"{data}.tsv"
    header, *rows = [l.rstrip("\n").split("\t") for l in open(path, encoding="utf-8") if "\t" in l]
    c = header.index(col)
    return path, [(r[c], r[-1]) for r in rows if r[c]]


def main(data, col, base=None, out=None, lr=None):
    import torch
    torch.manual_seed(SEED)
    path, rows = read(data, col)
    lang = "ko" if path.stem.startswith("ko") else "en"
    numeric = all(j.is_number(y) for y, _ in rows)  # 감성처럼 숫자면 점수, 범위·주제처럼 이름이면 분류만
    labels = sorted({y for y, _ in rows}, key=float if numeric else None)
    v = np.array(labels, float) if numeric else None
    lex = j.load_lexicon(j.LEXICON)
    F = j.phrase_features(["x"], lex, j.HINT).shape[1]
    base = base or BASE[lang]
    tok, enc, head = j.ft_build(base, F, HIDDEN, len(labels))

    idx = np.random.default_rng(SEED).permutation(len(rows))
    cut = int(len(rows) * (1 - VAL))
    train, val = [rows[i] for i in idx[:cut]], [rows[i] for i in idx[cut:]]
    Y = lambda rs: torch.tensor([labels.index(y) for y, _ in rs])
    # 숫자 라벨: |라벨| ≥ 1(호재·악재)을 틀리면 j.NONZERO배 벌점. 0과 ±0.5(소폭)는 1배 (judge.py와 같은 설정)
    cw = torch.tensor([j.NONZERO if numeric and abs(float(l)) >= 1 else 1.0 for l in labels])

    def loss(z, y):
        w = cw[y]
        return (torch.nn.functional.cross_entropy(z, y, reduction="none") * w).sum() / w.sum()

    def logits_of(rs):
        enc.eval(); head.eval()
        with torch.no_grad():
            return torch.cat([j.ft_logits(tok, enc, head, [t for _, t in rs[i:i + 64]], lex, j.HINT, MAX_LEN)
                              for i in range(0, len(rs), 64)])

    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": lr or LR_ENC}, {"params": head.parameters(), "lr": LR_HEAD}],
                            weight_decay=WD)
    print(f"{path.name} '{col}': 학습 {len(train)} / 검증 {len(val)}, 바탕 문장 모델 {base}")
    best, state = np.inf, None
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        enc.train(); head.train()
        order = np.random.default_rng(SEED + ep).permutation(len(train))
        for i in range(0, len(order), BATCH):
            b = [train[k] for k in order[i:i + BATCH]]
            l = loss(j.ft_logits(tok, enc, head, [t for _, t in b], lex, j.HINT, MAX_LEN), Y(b))
            opt.zero_grad(); l.backward(); opt.step()
        vl = float(loss(logits_of(val), Y(val)))
        print(f"  에폭 {ep}: 검증 손실 {vl:.3f} ({time.time() - t0:.0f}초)")
        if vl < best:
            best, state, bad = vl, (copy.deepcopy(enc.state_dict()), copy.deepcopy(head.state_dict())), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  검증 손실이 나빠져 여기서 멈춤 (가장 좋았던 에폭 상태를 씀)")
                break
    enc.load_state_dict(state[0]); head.load_state_dict(state[1])

    # judge.py와 같은 보정: 온도(검증 NLL 최소), 점수 배율(기대값이 0으로 쪼그라드는 것을 폄)
    zv, yv = logits_of(val).numpy(), Y(val).numpy()
    T = float(min(np.exp(np.linspace(-2, 2, 81)), key=lambda T: -np.log(j.softmax(zv / T)[np.arange(len(yv)), yv] + 1e-12).mean()))
    meta = {"base": base, "features": F, "hidden": HIDDEN, "labels": labels, "numeric": numeric, "T": T,
            "lexicon": lex, "hint": j.HINT, "phrase": True}
    if numeric:
        ev = j.softmax(zv / T) @ v
        meta.update(stretch=float(np.clip((ev @ v[yv]) / max(ev @ ev, 1e-9), 1.0, 4.0)), neutral=NEUTRAL)
    out = j.HOME / "models" / f"{out or f'{path.stem}-{col}-미세'}.pt"
    torch.save({"enc": enc.state_dict(), "head": head.state_dict(), "meta": meta}, out)
    extra = f", 점수 배율 {meta['stretch']:.2f}, 중립 경계 ±{NEUTRAL}" if numeric else ""
    print(f"온도 T={T:.2f}{extra}\n저장: 모델 이름 {out.stem}")
    return out.stem


def screen_report(name, data, col):
    """시험 파일로 화면 기준(중립 경계 안쪽 = 중립) 맞음·헛경보·놓침·방향 틀림. 이름 라벨이면 정확도만."""
    _, rows = read(data, col)
    P, meta = j.predict(name, [t for _, t in rows])
    if not meta["numeric"]:
        y = np.array([meta["labels"].index(a) if a in meta["labels"] else -1 for a, _ in rows])
        print(f"  정확도 {(P.argmax(1) == y).mean():.1%} ({len(rows)}개)")
        return
    s, y = j.score_of(P, meta), np.array([float(a) for a, _ in rows])
    nz = np.abs(y) >= 1  # ±0.5(소폭)는 화면에선 중립 쪽이 정답
    for th in sorted({0.5, 0.6, 0.7, 0.8, 0.9} | set(np.atleast_1d(meta.get("neutral", 0.5)).tolist())):
        sh = np.abs(s) >= th
        print(f"  경계 ±{th}: 맞음 {((nz & sh & (np.sign(s) == np.sign(y))) | (~nz & ~sh)).mean():5.1%}  "
              f"헛경보 {(~nz & sh).mean():4.1%}  놓침 {(nz & ~sh).mean():5.1%}  "
              f"방향 틀림 {(nz & sh & (np.sign(s) != np.sign(y))).mean():3.1%}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        sys.exit()
    opt = lambda k: args[args.index(k) + 1] if k in args else None
    name = main(args[0], args[1], opt("--base"), opt("--out"), float(opt("--lr")) if opt("--lr") else None)
    if "--test" in args:
        screen_report(name, args[args.index("--test") + 1], args[1])
