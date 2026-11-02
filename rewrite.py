# 변환기 사전 (2026-11-02, docs/실험기록.md 같은 날 '변환기 사전'): 주력 모델이 못 알아듣는 관용·비유 표현을 모델이 잘 아는 쉬운 말로 바꿔 넣는다.
# 판정 입력에만 쓰고 화면에 보이는 제목은 그대로. 위에서부터 차례로 (문맥 예외를 먼저). 후보는 실험/변환탐색.py로 찾음.
import re

I = re.I
EN = [
    # wipe out: 손실·낙폭을 지우면 회복, 나머지는 잃음
    (r"\bwip(?:e|es|ed|ing) (?:out|off) ((?:\S+ ){0,4}?(?:losses|declines|drops?|slump|selloff|sell-off)\b)", r"rebounds from \1"),
    (r"\bwipes (?:out|off)\b", "loses"), (r"\bwipe (?:out|off)\b", "lose"), (r"\bwiped (?:out|off)\b", "lost"), (r"\bwiping (?:out|off)\b", "losing"),
    (r"\bshav(?:e|es|ed|ing) (.{1,40}?) off\b", r"cuts \1 from"),
    (r"\bsheds? (?!light)", "loses "), (r"\bshedding (?!light)", "losing "),
    (r"\b(stocks?|shares?) tank(?:s|ed)?\b", r"\1 plunge"), (r"\btank(?:ed|ing)\b", "plunged"),
    (r"\bpar(?:e|es|ed|ing) (?:early |some |most |all )?(?:its |their )?gains\b", "falls from highs"),
    (r"\bpar(?:e|es|ed|ing) (?:early |some |most |all )?(?:its |their )?losses\b", "rises from lows"),
    (r"\b(?:give|gives|gave|giving) up (?:early |some |most |all )?(?:its |their )?gains\b", "falls after gains"),
    (r"\b(?:give|gives|gave|giving) up (?:early |some |most |all )?(?:its |their )?losses\b", "rises after losses"),
    (r"\berases? (?:early |some |most |all )?(?:its |their )?(?:\S+ )?gains\b", "falls and loses gains"),
    (r"\bsnap(?:s|ped|ping)? (?:a |an |its |their )?(\S+ )?winning streak\b", r"falls after \1winning streak"),
    (r"\bsnap(?:s|ped|ping)? (?:a |an |its |their )?(\S+ )?losing streak\b", r"rises after \1losing streak"),
    (r"\bclaw(?:s|ed|ing)? back\b", "rebounds from"), (r"\brecoup(?:s|ed|ing)?\b", "rebounds from"),
    (r"\bnotch(?:es|ed|ing)? (?:a )?record\b", "rises to record"),
    (r"\brouted\b", "plunge"), (r"\bin (?:a )?free ?fall\b", "plunging"),
    (r"\b(?:hammered|pummeled|pummelled|clobbered|battered|walloped) by\b", "plunge on"),
    (r"\b(?:hammered|pummeled|pummelled|clobbered|battered|walloped)\b", "plunge"),
    (r"\b(?:roiled|rattled|whipsawed) by\b", "fall on"),
    (r"\bskid(?:s|ded|ding)?\b", "falls"), (r"\bsag(?:s|ged|ging)?\b", "falls"),
    (r"\bretreat(?:s|ed|ing)? from\b", "falls from"),
    (r"\btak(?:e|es|ing) a beating\b", "falls sharply"), (r"\btook a beating\b", "fell sharply"),
    (r"\bextend(?:s|ed|ing)? (?:its |their )?losses\b", "falls further"),
    (r"\bdef(?:y|ies|ied|ying) the (?:down|weak|falling) market\b", "rises despite the down market"),
]
KO = [
    (r"^\s*['‘\"“]전날[^'’\"”]{1,20}['’\"”]\s*", ""),  # 따옴표 안 전날 배경 ('전날 7%대 급락' 코스피, 상승 출발)
    (r"상승분(?:을)? (?:모두 |다 |일부 )?(?:반납|토해내\S*)", "상승분 잃고 하락"),
    (r"반납", "하락"),
    (r"뒷걸음(?:질)?(?:쳐|치는|친|치며|쳤다|)", "하락"),
    (r"고꾸라\S*", "급락"), (r"주르륵\s?(?:밀려|빠져|내려)\S*", "하락"), (r"반토막", "절반으로 급락"),
    (r"날개\s?꺾\S*", "하락"), (r"날개\s?달\S*", "상승"), (r"훨훨", "급등"),
    (r"(?<![가-힣])뚝(?![심섬])", "하락"),
    (r"화색", "상승"), (r"(?:빨간불|적신호)", "악화"), (r"(?:청신호|파란불)", "개선"),
    (r"맥\s?못\s?(?:춰|추\S*)", "약세"), (r"기지개", "상승"), (r"꽁꽁", "악화"),
    (r"(심리|투심)(?:가|이)? 녹\S*", r"\1 개선"), (r"낙폭(?:을)? 만회", "반등"),
]
RULES = {"en": [(re.compile(p, I), r) for p, r in EN], "ko": [(re.compile(p), r) for p, r in KO]}


def apply(texts, lang):
    out = []
    for t in texts:
        for p, r in RULES[lang]:
            t = p.sub(r, t)
        out.append(t)
    return out


if __name__ == "__main__":  # 규칙 동작 확인
    cases = [("en", "S&P 500 Wipes Out Iran War Losses", "S&P 500 rebounds from Iran War Losses"),
             ("en", "Tech stocks set to wipe out more than $1 trillion", "Tech stocks set to lose more than $1 trillion"),
             ("en", "Nvidia shaves 400 points off the Dow", "Nvidia cuts 400 points from the Dow"),
             ("en", "Fed report sheds light on rates", "Fed report sheds light on rates"),
             ("en", "Poland buys K2 tanks", "Poland buys K2 tanks"),
             ("en", "Dow snaps five-day winning streak", "Dow falls after five-day winning streak"),
             ("ko", "'전날 7%대 급락' 코스피, 1%대 상승 출발", "코스피, 1%대 상승 출발"),
             ("ko", "코스피 상승분 모두 반납", "코스피 상승분 잃고 하락"),
             ("ko", "뚝심의 경영", "뚝심의 경영"), ("ko", "코스피 뚝", "코스피 하락")]
    for lang, a, b in cases:
        got = apply([a], lang)[0]
        assert got == b, (a, got, b)
    print("변환 규칙 확인 통과")
