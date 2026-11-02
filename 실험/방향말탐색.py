# 방향 말 미세 조정 후보 (2026-11-02): 변환기를 거친 뒤에도 모델이 방향을 못 내는 방향 말. 같은 문장의 '쉬운 말'(하락/급락/상승, falls/plunges/rises)과 견줌.
# python 실험/방향말탐색.py → 후보와 캐시 제목 빈도
import sys, re, collections
from pathlib import Path
HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))
import numpy as np
import judge as j
sys.stdout.reconfigure(encoding="utf-8")
# (언어, 낱말, 기대 부호, 쉬운 말, 문장 틀 여러 개: {w} 자리에 낱말)
KO = [("깨져", -1, "하락해", ["코스피 2500선 {w}", "삼성전자 6만원선 {w}"]), ("주춤", -1, "하락", ["반도체주 {w}", "수출 증가세 {w}"]),
      ("휘청", -1, "하락", ["코스피 {w}", "2차전지주 {w}"]), ("곤두박질", -1, "급락", ["코스닥 {w}", "카카오 주가 {w}"]),
      ("털썩", -1, "하락", ["코스피 {w}", "반도체주 {w}"]), ("미끄러져", -1, "하락해", ["코스피 2600선 아래로 {w}", "삼성전자 {w} 6만원대"]),
      ("무너져", -1, "급락해", ["코스피 2500선 {w}", "바이오주 {w}"]), ("꺾여", -1, "하락해", ["상승세 {w} 코스피", "반도체 랠리 {w}"]),
      ("밀려", -1, "하락해", ["코스피 2600선으로 {w}", "외국인 매도에 {w}"]), ("내줘", -1, "하락해", ["코스피 2600선 {w}", "상승분 {w}"]),
      ("주저앉아", -1, "급락해", ["코스피 {w}", "카카오 {w}"]), ("비틀", -1, "하락", ["반도체주 {w}", "증시 {w}"]),
      ("치솟아", 1, "급등해", ["조선주 {w}", "방산주 {w}"]), ("뛰어", 1, "상승해", ["코스피 2% {w}", "삼성전자 5% {w}"]),
      ("날아올라", 1, "급등해", ["원전주 {w}", "조선주 {w}"]), ("반색", 1, "상승", ["증시 {w}", "반도체 업계 {w}"]), ("들썩", 1, "상승", ["건설주 {w}", "원전주 {w}"])]
EN = [("crater", -1, "plunge", ["Bank stocks {w} as credit fears spread", "Shares {w} after weak guidance"]),
      ("nosedive", -1, "plunge", ["Tesla shares {w} on weak deliveries", "Stocks {w} after jobs report"]),
      ("slip", -1, "fall", ["Stocks {w} as yields rise", "Shares {w} after downgrade"]),
      ("dent", -1, "hurt", ["Tariffs {w} automaker profits", "Weak sales {w} Apple outlook"]),
      ("erode", -1, "cut", ["Rising costs {w} margins", "Tariffs {w} profits"]),
      ("falter", -1, "fall", ["Tech stocks {w} after rally", "Markets {w} as oil jumps"]),
      ("stumble", -1, "fall", ["Stocks {w} as rate fears return", "Nvidia shares {w} after earnings"]),
      ("buckle", -1, "fall", ["Bank shares {w} under pressure", "Stocks {w} on tariff news"]),
      ("sour", -1, "worsen", ["Investor sentiment {w}s", "Markets {w} on China data"]),
      ("wobble", -1, "fall", ["Stocks {w} as bond yields climb", "Chip stocks {w} after warning"]),
      ("tumble", -1, "plunge", ["Stocks {w} as AI fears grow", "Oil prices {w} on supply glut"]),
      ("slump", -1, "fall", ["Shares {w} after profit warning", "Stocks {w} on recession fears"]),
      ("dive", -1, "plunge", ["Shares {w} after guidance cut", "Stocks {w} as Fed signals hikes"]),
      ("sink", -1, "fall", ["Stocks {w} as oil spikes", "Shares {w} on weak outlook"]),
      ("skyrocket", 1, "surge", ["Shares {w} after FDA approval", "Stocks {w} on deal news"]),
      ("rally", 1, "rise", ["Stocks {w} on Fed hopes", "Chip stocks {w} after earnings"]),
      ("rebound", 1, "rise", ["Stocks {w} after selloff", "Shares {w} on upgrade"]),
      ("leap", 1, "surge", ["Shares {w} on takeover bid", "Stocks {w} after CPI"]),
      ("pop", 1, "surge", ["Shares {w} after earnings beat", "Stocks {w} on rate cut"]),
      ("perk up", 1, "rise", ["Stocks {w} as yields ease", "Markets {w} on trade deal"])]


def sc(lang, t):
    P, m = j.predict(j.MAIN[f"{lang}-감성"], t, **({"head": 0} if lang == "en" else {}))
    return j.score_of(P, m)


for lang, L in (("ko", KO), ("en", EN)):
    print(f"== {lang}")
    for w, sg, easy, fr in L:
        a, b = sc(lang, [f.format(w=w) for f in fr]), sc(lang, [f.format(w=easy) for f in fr])
        bad = np.mean(sg * a) < max(0.25, 0.5 * np.mean(sg * b))
        print(f"{'후보' if bad else '    '} {w:10} 원래 {np.mean(a):+.2f} 쉬운말 {np.mean(b):+.2f}  " + " / ".join(f"{x:+.2f}" for x in a))
