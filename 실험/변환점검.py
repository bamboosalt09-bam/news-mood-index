# 변환기 사전 실험 (2026-11-02): 모델이 못 알아듣는 말을 쉬운 말로 바꿔 넣으면 나아지는지.
# python 실험/변환점검.py errors  → 발견용 자료(Opus 검증 1,000 두 언어)의 방향 틀림 제목 목록 (규칙 만들기용)
# python 실험/변환점검.py judge [사전]  → 판단용 자료에서 변환 전·후 (한국어 evaluate 무작위 700, 영어 Haiku 검증 2,010 + 옛 영어 303 + 새 시험지)
import sys, re
from pathlib import Path
HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))
import numpy as np
import judge as j, evaluate as e, rewrite
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
LAB = HOME / "labeled"
tsv = lambda f: [l.split("\t") for l in open(f, encoding="utf-8").read().splitlines()[1:] if l.strip()]
V = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])


def scores(lang, texts, head=None, rw=False):  # evaluate.scores와 같은 뒤처리 (안전장치 ±1, 전망 약화, 수준만은 0). rw: 모델 입력만 변환
    name = j.MAIN[f"{lang}-감성"]
    inp = rewrite.apply(texts, lang) if rw else texts
    P, meta = j.predict(name, inp, head=head) if lang == "en" else j.predict(name, inp)
    s = j.soften(texts, j.guard(texts, j.score_of(P, meta), e.GUARD, meta.get("guard_lexicon"))[0])[0]
    return np.where(j.level_only(texts), 0.0, s)


def flips(s, y):  # 3분류 부호 반대
    return (np.abs(s) >= 0.25) & (y != 0) & (np.sign(s) * np.sign(y) < 0)


def errors():
    for lang, f, col in (("ko", "ko_opus검증1000.tsv", 0), ("en", "en_opus검증1000.tsv", 0)):
        rows = tsv(LAB / f)
        t, y = [r[-1] for r in rows], np.array([float(r[col]) for r in rows])
        s = scores(lang, t, head=0)
        bad = np.where(flips(s, y))[0]
        print(f"\n== {lang} 방향 틀림 {len(bad)} / {len(rows)}")
        for i in bad:
            print(f"{y[i]:+.1f} {s[i]:+.2f}  {t[i]}")


def judge():  # 해 없음 판단 (사전 등록 기준)
    seven = lambda s, y: float(np.mean(V[np.abs(s[:, None] - V).argmin(1)] == y))
    rows = [r for f in e.TESTS["무작위700 (최종)"] for r in tsv(LAB / "시험" / f)]
    t, y = [r[-1] for r in rows], np.array([float(r[0]) for r in rows])
    print("[한국어 무작위 700]  바뀐 제목", sum(a != b for a, b in zip(t, rewrite.apply(t, "ko"))))
    for rw in (False, True):
        m = e.screen(scores("ko", t, rw=rw), y)
        print(f"  {'변환' if rw else '지금'}: 맞음 {m['맞음']:.1%} 헛경보 {m['헛경보']:.1%} 놓침 {m['놓침']:.1%} 방향 틀림 {m['방향 틀림']}")
    fails = {rw: [d for d, a, b, rel in e.BEHAVIOR if not {"<": lambda sa, sb: sa < sb, "=": lambda sa, sb: abs(sa - sb) < 0.05, ">0": lambda sa, sb: sa > 0,
             "<0": lambda sa, sb: sa < 0, ">=0": lambda sa, sb: sa >= -0.1, "<=0": lambda sa, sb: sa <= 0.1}[rel](scores("ko", [a], rw=rw)[0], scores("ko", [b], rw=rw)[0] if b else None)]
             for rw in (False, True)}
    print(f"  기본 행동: 지금 {16 - len(fails[False])}/16, 변환 {16 - len(fails[True])}/16 {fails[True]}")
    hr = tsv(LAB / "en_ai_haiku.tsv")
    hv = [hr[i] for i in np.random.default_rng(0).permutation(len(hr))[:len(hr) // 10]]
    old = [r for f in ("en_test.tsv", "en_newtest.tsv") for r in tsv(LAB / f)]
    new = tsv(LAB / "시험" / "en_새시험1500.tsv")
    for name, rr in (("영어 Haiku 검증 2,010", hv), ("옛 영어 303", old), ("영어 새 시험지 1,500 (참고)", new)):
        t, y = [r[-1] for r in rr], np.array([float(r[0]) for r in rr])
        print(f"[{name}]  바뀐 제목", sum(a != b for a, b in zip(t, rewrite.apply(t, "en"))))
        for rw in (False, True):
            s = scores("en", t, head=0, rw=rw); m = e.screen(s, y)
            print(f"  {'변환' if rw else '지금'}: 7단계 {seven(s, y):.1%} 맞음 {m['맞음']:.1%} 헛경보 {m['헛경보']:.1%} 방향틀림3 {int(flips(s, y).sum())}")


GAIN = HOME / "실험" / "결과" / "변환" / "이득"


def gain_make():  # 이득 판단 표본: 지수 캐시 제목 중 변환으로 바뀌는 것, 라벨·시험 자료 제목은 뺌, 무작위 최대 300 (시드 2041)
    import random
    sys.path.insert(0, str(HOME / "web")); import repcheck as rc
    seen = {r[-1] for f in LAB.rglob("*.tsv") for r in tsv(f)}
    GAIN.mkdir(parents=True, exist_ok=True)
    for lang in ("ko", "en"):
        t = [l.rstrip("\n").split("\t")[-1] for l in open(HOME / "data" / "cache" / f"지수자료_{lang}.tsv", encoding="utf-8")][1:]
        t = sorted({x for x in t if x not in seen})
        hit = [x for x, y in zip(t, rewrite.apply(t, lang)) if x != y]
        pick = random.Random(2041).sample(hit, min(300, len(hit)))
        (GAIN / f"{lang}.tsv").write_text("".join(x + "\n" for x in pick), encoding="utf-8")
        mix = list(enumerate(pick)); random.Random(lang).shuffle(mix)
        (GAIN / f"묶음_{lang}.txt").write_text(rc.GUIDE.format(view=rc.VIEW[lang]) + "".join(f"{k}\t{x}\n" for k, x in mix), encoding="utf-8")
        print(lang, "바뀌는 제목", len(hit), "/", len(t), "→ 표본", len(pick))


def gain_score():
    import json
    for lang in ("ko", "en"):
        t = (GAIN / f"{lang}.tsv").read_text(encoding="utf-8").splitlines()
        hk = json.loads((GAIN / f"haiku_{lang}.json").read_text(encoding="utf-8"))
        y = np.array([{"호재": 1, "악재": -1, "중립": 0}[hk[str(k)]] if str(k) in hk else 0 for k in range(len(t))], float)
        a, b = scores(lang, t, head=0), scores(lang, t, head=0, rw=True)
        fa, fb = flips(a, y), flips(b, y)
        print(f"[{lang}] 표본 {len(t)}, Haiku 호재 {int((y > 0).sum())} 악재 {int((y < 0).sum())} 중립 {int((y == 0).sum())}")
        print(f"  방향 틀림: 지금 {int(fa.sum())} → 변환 {int(fb.sum())}, 고쳐짐 {int((fa & ~fb).sum())}, 새로 틀림 {int((~fa & fb).sum())}")
        print(f"  부호 맞음(Haiku 호재·악재 중 |점수|≥0.25로 같은 쪽): 지금 {int(((np.sign(a) == y) & (np.abs(a) >= 0.25) & (y != 0)).sum())} → 변환 {int(((np.sign(b) == y) & (np.abs(b) >= 0.25) & (y != 0)).sum())}")
        for i in np.where(~fa & fb)[0]:
            print(f"  새로 틀림 {y[i]:+.0f} {a[i]:+.2f}→{b[i]:+.2f} {t[i]}")


if __name__ == "__main__":
    {"errors": errors, "judge": judge, "gain_make": gain_make, "gain_score": gain_score}[sys.argv[1]]()
