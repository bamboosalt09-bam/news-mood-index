# 변환기 사전 후보 찾기 (2026-11-02): 경제 제목에 흔한 관용·비유 표현을 넣은 문장과, 같은 문장을 쉬운 말로 바꾼 문장을 주력 모델에 넣어
# 쉬운 말은 방향이 맞는데 원래 표현은 틀리거나(부호 반대) 약하면(|점수| < 쉬운 말의 절반) 후보.
# python 실험/변환탐색.py → 실험/결과/변환/탐색.tsv
import sys
from pathlib import Path
HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))
import numpy as np
import judge as j
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# (언어, 기대 부호, 원래 문장, 쉬운 말 문장)
PROBE = [
    ("en", -1, "Tech stocks wipe out $1 trillion in market value", "Tech stocks lose $1 trillion in market value"),
    ("en", -1, "Nvidia shaves 400 points off the Dow", "Nvidia cuts 400 points from the Dow"),
    ("en", -1, "Stocks shed $500 billion as AI fears grow", "Stocks lose $500 billion as AI fears grow"),
    ("en", -1, "Shares of Intel tank after earnings", "Shares of Intel plunge after earnings"),
    ("en", -1, "Bank stocks crater as credit fears spread", "Bank stocks plunge as credit fears spread"),
    ("en", -1, "Tesla shares nosedive on weak deliveries", "Tesla shares plunge on weak deliveries"),
    ("en", -1, "Stocks pare gains as yields rise", "Stocks fall from highs as yields rise"),
    ("en", -1, "Nasdaq gives up early gains", "Nasdaq falls after early gains"),
    ("en", -1, "S&P 500 erases gains for the year", "S&P 500 falls below last year's level"),
    ("en", -1, "Dow snaps five-day winning streak", "Dow falls after five days of gains"),
    ("en", 1, "Dow snaps five-day losing streak", "Dow rises after five days of losses"),
    ("en", 1, "Stocks claw back losses after selloff", "Stocks rise after selloff"),
    ("en", 1, "Stocks recoup losses as oil eases", "Stocks rise as oil eases"),
    ("en", 1, "S&P 500 wipes out Iran war losses", "S&P 500 recovers Iran war losses"),
    ("en", 1, "Stocks rip higher on Fed hopes", "Stocks surge on Fed hopes"),
    ("en", 1, "Shares notch record close", "Shares rise to record close"),
    ("en", -1, "Bloodbath on Wall Street as tech stocks sink", "Selloff on Wall Street as tech stocks fall"),
    ("en", -1, "Chip stocks routed as export curbs widen", "Chip stocks plunge as export curbs widen"),
    ("en", -1, "Market meltdown deepens", "Market crash deepens"),
    ("en", -1, "Stocks in freefall after jobs report", "Stocks plunge after jobs report"),
    ("en", -1, "Stocks battered by rate fears", "Stocks fall on rate fears"),
    ("en", -1, "Tech shares hammered after Alphabet results", "Tech shares plunge after Alphabet results"),
    ("en", -1, "Retail stocks pummeled by tariff news", "Retail stocks plunge on tariff news"),
    ("en", -1, "Oil shock rattles markets", "Oil shock hurts markets"),
    ("en", -1, "Bond rout spooks investors", "Bond selloff worries investors"),
    ("en", -1, "Stocks roiled by trade war", "Stocks fall on trade war"),
    ("en", -1, "Nvidia deals devastating blow to Intel", "Nvidia hurts Intel badly"),
    ("en", -1, "Tariffs weigh on automakers", "Tariffs hurt automakers"),
    ("en", -1, "Weak guidance drags Apple lower", "Weak guidance pushes Apple lower"),
    ("en", 1, "Strong earnings buoy Wall Street", "Strong earnings lift Wall Street"),
    ("en", 1, "Rate cut hopes prop up stocks", "Rate cut hopes lift stocks"),
    ("en", 1, "Plug Power stock defies the down market", "Plug Power stock rises despite the down market"),
    ("en", 1, "Eaton cuts its post-earnings losses", "Eaton recovers its post-earnings losses"),
    ("en", -1, "Earnings fall short of expectations", "Earnings miss expectations"),
    ("en", 1, "Earnings top estimates", "Earnings beat estimates"),
    ("en", -1, "Stocks slide into correction territory", "Stocks fall into correction territory"),
    ("en", -1, "Stocks skid as dollar jumps", "Stocks fall as dollar jumps"),
    ("en", -1, "Shares sag after downgrade", "Shares fall after downgrade"),
    ("en", -1, "Stocks retreat from record highs", "Stocks fall from record highs"),
    ("en", -1, "Investors dump tech shares", "Investors sell tech shares"),
    ("en", -1, "Selloff hits chip stocks", "Selloff hurts chip stocks"),
    ("en", 1, "Stocks bounce back after rout", "Stocks rise after rout"),
    ("en", -1, "Crypto stocks bleed as bitcoin slumps", "Crypto stocks fall as bitcoin slumps"),
    ("en", -1, "Stocks whipsawed by tariff headlines", "Stocks fall on tariff headlines"),
    ("en", -1, "Stocks take a beating", "Stocks fall sharply"),
    ("en", 1, "Stocks extend rally", "Stocks rise further"),
    ("en", -1, "Stocks extend losses", "Stocks fall further"),
    ("en", -1, "Stocks tumble to a three-month low", "Stocks fall to a three-month low"),
    ("en", 1, "Stocks hit an all-time high", "Stocks rise to a record high"),
    ("ko", -1, "코스피, 하루 만에 7천선 반납", "코스피, 하루 만에 7천선 아래로 하락"),
    ("ko", -1, "코스피 상승분 모두 반납", "코스피 상승분 모두 잃고 하락"),
    ("ko", -1, "삼성전자 주가 뒷걸음", "삼성전자 주가 하락"),
    ("ko", -1, "5만원대에 갇힌 삼성전자", "5만원대에서 못 오르는 삼성전자"),
    ("ko", -1, "코스피 곤두박질…2500선 붕괴", "코스피 급락…2500선 아래로"),
    ("ko", -1, "코스닥 고꾸라져", "코스닥 급락"),
    ("ko", -1, "코스피 미끄러져 2600선 내줘", "코스피 하락해 2600선 아래로"),
    ("ko", -1, "반도체주 휘청", "반도체주 하락"),
    ("ko", -1, "코스피 털썩", "코스피 하락"),
    ("ko", -1, "코스피 주르륵 밀려", "코스피 하락"),
    ("ko", -1, "바이오주 와르르", "바이오주 급락"),
    ("ko", -1, "카카오 주가 반토막", "카카오 주가 절반으로 급락"),
    ("ko", -1, "2차전지주 날개 꺾였다", "2차전지주 하락"),
    ("ko", 1, "2차전지주 날개 달았다", "2차전지주 상승"),
    ("ko", 1, "조선주 훨훨", "조선주 급등"),
    ("ko", 1, "방산주 껑충", "방산주 급등"),
    ("ko", 1, "원전주 불기둥", "원전주 급등"),
    ("ko", 1, "코스피 쑥", "코스피 상승"),
    ("ko", -1, "코스피 뚝", "코스피 하락"),
    ("ko", -1, "코스피 주춤", "코스피 약보합"),
    ("ko", 1, "증시 화색", "증시 상승"),
    ("ko", -1, "개미들 울상", "개인 투자자 손실"),
    ("ko", -1, "증시 비명", "증시 급락"),
    ("ko", -1, "수출 빨간불", "수출 악화"),
    ("ko", 1, "수출 청신호", "수출 개선"),
    ("ko", -1, "실적 적신호", "실적 악화"),
    ("ko", 1, "반도체 훈풍", "반도체 호재"),
    ("ko", -1, "증시에 찬물", "증시에 악재"),
    ("ko", -1, "관세 직격탄 맞은 자동차", "관세로 큰 피해 입은 자동차"),
    ("ko", -1, "환율이 발목 잡은 코스피", "환율 때문에 하락한 코스피"),
    ("ko", -1, "코스피 숨고르기", "코스피 소폭 하락"),
    ("ko", 1, "코스피 반등 숨통", "코스피 반등"),
    ("ko", -1, "외국인 팔아치워 코스피 하락", "외국인 순매도에 코스피 하락"),
    ("ko", 1, "외국인 쓸어담아 코스피 상승", "외국인 순매수에 코스피 상승"),
    ("ko", -1, "급등 하루 만에 상승분 토해내", "급등 하루 만에 하락"),
    ("ko", 1, "급락 하루 만에 낙폭 만회", "급락 하루 만에 반등"),
    ("ko", 1, "코스피, 장중 상승 전환", "코스피, 장중 상승"),
    ("ko", -1, "코스피, 장중 하락 전환", "코스피, 장중 하락"),
    ("ko", 1, "'전날 7%대 급락' 코스피, 1%대 상승 출발", "코스피, 1%대 상승 출발"),
    ("ko", -1, "'전날 7%대 급등' 코스피, 1%대 하락 출발", "코스피, 1%대 하락 출발"),
    ("ko", -1, "삼성전자, 8만전자 뒷걸음질", "삼성전자, 8만원 아래로 하락"),
    ("ko", -1, "코스피 6천피 무너져", "코스피 6천 아래로 급락"),
    ("ko", 1, "코스피 7천피 탈환", "코스피 7천 위로 상승"),
    ("ko", -1, "코스피 맥 못 춰", "코스피 약세"),
    ("ko", 1, "코스피 기지개", "코스피 상승"),
    ("ko", -1, "증시 먹구름", "증시 악재"),
    ("ko", -1, "투자심리 꽁꽁", "투자심리 악화"),
    ("ko", 1, "투자심리 녹았다", "투자심리 개선"),
]


def score(lang, texts):
    name = j.MAIN[f"{lang}-감성"]
    P, meta = j.predict(name, texts, head=0) if lang == "en" else j.predict(name, texts)
    return j.score_of(P, meta)


rows = ["언어\t기대\t원래\t쉬운말\t판정\t원래문장\t쉬운말문장"]
for lang in ("en", "ko"):
    P = [p for p in PROBE if p[0] == lang]
    a, b = score(lang, [p[2] for p in P]), score(lang, [p[3] for p in P])
    for (l, sg, x, y), sa, sb in zip(P, a, b):
        ok_easy = sg * sb >= 0.25
        bad = "반대" if sg * sa < -0.1 else "약함" if sg * sa < max(0.25, 0.5 * sg * sb) else "됨"
        tag = bad if ok_easy else "쉬운말도 안 됨"
        rows.append(f"{l}\t{sg}\t{sa:+.2f}\t{sb:+.2f}\t{tag}\t{x}\t{y}")
        if tag != "됨":
            print(f"{l} {tag:8} {sa:+.2f} → {sb:+.2f}  {x}  →  {y}")
(HOME / "실험" / "결과" / "변환" / "탐색.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")

if "apply" in sys.argv:  # 변환 규칙을 원래 문장에 적용한 점수 (규칙 만들 때 본 자료라 판단에는 쓰지 않음)
    import rewrite
    for lang in ("en", "ko"):
        P = [p for p in PROBE if p[0] == lang]
        x = [p[2] for p in P]; xr = rewrite.apply(x, lang)
        a, c = score(lang, x), score(lang, xr)
        for (l, sg, *_), t, tr, sa, sc in zip(P, x, xr, a, c):
            if t != tr:
                print(f"{l} {'됨' if sg * sc >= 0.25 else '안됨':3} {sa:+.2f} → {sc:+.2f}  {t}  ⇒  {tr}")
