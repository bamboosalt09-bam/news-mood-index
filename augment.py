"""감성 학습용 데이터 만들기: 방향만 뒤집은 반대 사례를 더하고 0점은 일부만 남긴다.

python augment.py ko      (또는 en)
  → labeled/ko_test.tsv  시험용 실제 제목 (손대지 않음, 두 모델을 같은 문제로 비교)
    labeled/ko_base.tsv  나머지 실제 제목 그대로
    labeled/ko_aug.tsv   나머지 + 반대 사례, 0점은 줄임
python judge.py train ko_base 감성   /   python judge.py train ko_aug 감성
python judge.py eval ko_test 감성 ko_base-감성 ko_aug-감성
"""
import random
import re
import sys
from pathlib import Path

HOME = Path(__file__).parent
TEST = 0.15       # 시험용으로 떼어 둘 실제 제목 비율
ZERO_KEEP = 0.5   # 0점 사례를 '0이 아닌 사례 수 × 이 값'만큼만 남김
SEED = 0

# 한쪽을 다른 쪽으로 바꾸면 의미 방향이 뒤집히는 말. 한 제목 안의 모든 짝을 동시에 바꾼다
# (예: "급락 딛고 반등" → "급등 딛고 반락"). 사전에 없는 방향어가 섞인 제목은 틀릴 수 있다.
PAIRS = {
    "ko": [("급등", "급락"), ("폭등", "폭락"), ("상승", "하락"), ("반등", "반락"), ("강세", "약세"),
           ("인상", "인하"), ("증가", "감소"), ("흑자", "적자"), ("개선", "악화"), ("확대", "축소"),
           ("상회", "하회"), ("상향", "하향"), ("호재", "악재"), ("최고", "최저"), ("돌파", "붕괴"),
           ("오른", "내린"), ("낙폭", "상승폭"), ("↑", "↓"), ("호실적", "실적 부진"), ("훈풍", "한파")],
    "en": [("surge", "plunge"), ("surges", "plunges"), ("surged", "plunged"), ("soar", "sink"),
           ("soars", "sinks"), ("rise", "fall"), ("rises", "falls"), ("rising", "falling"), ("rose", "fell"),
           ("jump", "drop"), ("jumps", "drops"), ("jumped", "dropped"), ("gains", "losses"),
           ("beat", "miss"), ("beats", "misses"), ("higher", "lower"), ("high", "low"),
           ("rally", "selloff"), ("rallies", "slides"), ("climbs", "slides"), ("strong", "weak"),
           ("strength", "weakness"), ("bullish", "bearish"), ("upgrade", "downgrade"),
           ("raises", "cuts"), ("rate hike", "rate cut"), ("rate hikes", "rate cuts"),
           ("boost", "drag"), ("eases", "worsens"), ("rebound", "slump"), ("rebounds", "slumps")],
}


def flipper(lang):
    m = {}
    for a, b in PAIRS[lang]:
        m[a], m[b] = b, a
        if lang == "en":
            m[a.capitalize()], m[b.capitalize()] = b.capitalize(), a.capitalize()
    alt = "|".join(map(re.escape, sorted(m, key=len, reverse=True)))
    rx = re.compile(rf"\b(?:{alt})\b" if lang == "en" else alt)
    return lambda s: rx.sub(lambda x: m[x.group()], s)


def write(path, rows):
    path.write_text("".join("\t".join(r) + "\n" for r in [["감성", "범위", "주제", "제목"]] + rows), encoding="utf-8")


def main(lang):
    header, *rows = [l.rstrip("\n").split("\t") for l in open(HOME / "labeled" / f"{lang}.tsv", encoding="utf-8")]
    rows = [r for r in rows if r[0] != ""]
    rng = random.Random(SEED)
    rng.shuffle(rows)
    cut = int(len(rows) * TEST)
    test, rest = rows[:cut], rows[cut:]

    flip = flipper(lang)
    seen = {r[3] for r in rows}
    nonzero = [r for r in rest if r[0] != "0"]
    flips = []
    for s, scope, topic, title in nonzero:
        t = flip(title)
        if t != title and t not in seen:
            seen.add(t)
            flips.append([str(-int(s)), scope, topic, t])
    zeros = [r for r in rest if r[0] == "0"]
    keep = zeros[:round(ZERO_KEEP * (len(nonzero) + len(flips)))]
    aug = nonzero + flips + keep
    rng.shuffle(aug)

    write(HOME / "labeled" / f"{lang}_test.tsv", test)
    write(HOME / "labeled" / f"{lang}_base.tsv", rest)
    write(HOME / "labeled" / f"{lang}_aug.tsv", aug)
    print(f"{lang}: 시험 {len(test)} / 원본 학습 {len(rest)} → 증강 학습 {len(aug)} "
          f"(0이 아닌 원본 {len(nonzero)} + 반대 사례 {len(flips)} + 0점 {len(keep)}/{len(zeros)})")
    for s, _, _, t in flips[:6]:
        print(f"  {int(s):+d}  {t}")


def demo():
    assert flipper("ko")("코스피 4% 급락 딛고 반등") == "코스피 4% 급등 딛고 반락"
    assert flipper("en")("Stocks rise as Fed signals rate cut") == "Stocks fall as Fed signals rate hike"
    assert flipper("en")("Highway") == "Highway"  # 단어 경계: high가 들어 있어도 안 바꿈
    print("뒤집기 검산 통과")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1:] in (["ko"], ["en"]):
        main(sys.argv[1])
    elif sys.argv[1:] == ["check"]:
        demo()
    else:
        print(__doc__)
