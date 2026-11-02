"""정해진 방법으로 감성 모델 채점 (모델을 비교할 때는 항상 이것으로).

python evaluate.py ko-감성-미세 ko_v3-감성-미세        # 모델 여러 개를 같은 조건으로
python evaluate.py --최종 후보모델                      # 동결 시험지 ko_최종500까지 (후보를 정한 뒤 한 번만)

- 시험지: labeled/시험/ko_무작위300.tsv + ko_무작위400.tsv (최종 확인용, 실제 흐름 무작위 700개), ko_시험563.tsv·ko_오늘60.tsv (개발용)
- 화면 기준: 경계 ±0.9, 안전장치 ±1 (사전 규칙과 방향이 엇갈리면 중립), 전망·의견 제목은 점수 절반 (judge.soften), 수준만 말한 제목은 중립 (judge.level_only). 모든 모델에 같은 값
- 맞음: 정답 |1| 이상은 같은 방향으로 경계를 넘으면, 정답 0·±0.5는 경계 안쪽이면 맞음
- 기본 행동 시험: 말·숫자만 바꾼 문장 짝에서 점수가 당연한 순서인지 (전부 통과해야 '보장')
"""
import sys

import numpy as np

import judge as j

TESTS = {"무작위700 (최종)": ("ko_무작위300.tsv", "ko_무작위400.tsv"), "시험지562 (개발)": ("ko_시험563.tsv",),
         "오늘60 (개발)": ("ko_오늘60.tsv",)}
FINAL = {"최종500 (동결)": ("ko_최종500.tsv",)}  # 라벨 고치지 않음, 설정 고르기에 쓰지 않음 (docs/실험기록.md 2026-10-26)
CUT, GUARD = 0.9, 1.0

# (설명, 앞 문장, 뒤 문장, 관계): 관계 '<'는 앞 점수 < 뒤 점수, '>0'·'<0'·'>=0'은 앞 문장 하나만, '='는 거의 같음
BEHAVIOR = [
    ("하락 < 상승", "코스피 하락 마감", "코스피 상승 마감", "<"),
    ("급락 < 하락", "코스피 급락 마감", "코스피 하락 마감", "<"),
    ("8% 하락 < 0.1% 하락", "코스피 8% 하락 마감", "코스피 0.1% 하락 마감", "<"),
    ("0.1% 상승 < 8% 상승", "코스피 0.1% 상승 마감", "코스피 8% 상승 마감", "<"),
    ("환율 30원 오름 < 환율 3원 오름", "원·달러 환율 30원 오른 1,410원 마감", "원·달러 환율 3원 오른 1,383원 마감", "<"),
    ("환율 급등은 악재", "원·달러 환율 급등…1,500원 돌파", None, "<0"),
    ("환율 급락은 호재", "원·달러 환율 급락…1,300원대 진입", None, ">0"),
    ("유가 급등은 악재", "국제유가 급등…배럴당 100달러 돌파", None, "<0"),
    ("유가 급락은 호재", "국제유가 급락…공급 과잉 우려", None, ">0"),
    ("금리 인상은 악재", "한국은행, 기준금리 0.25%p 인상", None, "<0"),
    ("실업률 오르지 않음은 악재 아님", "실업률, 우려와 달리 오르지 않아", None, ">=0"),
    ("위기설 부인은 악재 아님", "D그룹, 유동성 위기설 공식 부인", None, ">=0"),
    ("급락서 강보합 마감은 악재 아님", "코스피 4% 급락서 강보합 마감", None, ">=0"),
    ("원인 '~에'는 결과가 아님", "뉴욕증시, 연준·유가에 급락", None, "<=0"),
    ("파업은 악재", "G전자 노조 전면 파업 돌입", None, "<0"),
    ("[속보]·번호는 결과를 안 바꿈", "[속보] 코스피 2% 상승 마감", "코스피 2% 상승 마감", "="),
]


def screen(s, y):
    up, dn = s >= CUT, s <= -CUT
    nz, sh = np.abs(y) >= 1, up | dn
    right = (up & (y > 0)) | (dn & (y < 0))
    return {"맞음": ((nz & right) | (~nz & ~sh)).mean(), "헛경보": (~nz & sh).mean(), "놓침": (nz & ~sh).mean(),
            "방향 틀림": int((nz & sh & ~right).sum()), "중립인데 반대로": int((~nz & sh & (np.sign(s) * np.sign(y) < 0)).sum())}


def scores(name, texts):
    P, meta = j.predict(name, texts)
    s = j.soften(texts, j.guard(texts, j.score_of(P, meta), GUARD, meta.get("guard_lexicon"))[0])[0]
    return np.where(j.level_only(texts), 0.0, s)


def behavior(name):
    fails = []
    for desc, a, b, rel in BEHAVIOR:
        sa = scores(name, [a])[0]
        sb = scores(name, [b])[0] if b else None
        ok = {"<": lambda: sa < sb, "=": lambda: abs(sa - sb) < 0.05, ">0": lambda: sa > 0, "<0": lambda: sa < 0,
              ">=0": lambda: sa >= -0.1, "<=0": lambda: sa <= 0.1}[rel]()
        if not ok:
            fails.append(f"{desc} ({sa:+.2f}" + (f" / {sb:+.2f})" if b else ")"))
    return fails


def main(names, final=False):
    sets = {k: [l.split("\t") for f in fs for l in (j.HOME / "labeled" / "시험" / f).read_text(encoding="utf-8").splitlines()[1:]]
            for k, fs in (TESTS | FINAL if final else TESTS).items()}
    print(f"경계 ±{CUT}, 안전장치 ±{GUARD}")
    for name in names:
        print(f"\n[{name}]")
        for k, rows in sets.items():
            y = np.array([float(r[0]) for r in rows])
            m = screen(scores(name, [r[-1] for r in rows]), y)
            print(f"  {k:16} 맞음 {m['맞음']:5.1%}  헛경보 {m['헛경보']:4.1%}  놓침 {m['놓침']:5.1%}  "
                  f"방향 틀림 {m['방향 틀림']}개  (중립인데 반대 신호 {m['중립인데 반대로']}개)  {len(rows)}개")
        fails = behavior(name)
        print(f"  기본 행동 시험 {len(BEHAVIOR) - len(fails)}/{len(BEHAVIOR)} 통과" + "".join(f"\n    실패: {x}" for x in fails))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    args = [a for a in sys.argv[1:] if a != "--최종"]
    main(args or ["ko-감성-미세"], final="--최종" in sys.argv)
