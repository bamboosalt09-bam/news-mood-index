# 섹터·기업 사전(sectors.tsv)으로 제목에 섹터·기업 붙이기 (2026-10-31). 실험/섹터.py에서 옮김, web/build.py가 씀.
import re
from pathlib import Path
HOME = Path(__file__).resolve().parent


def load(lang):  # [(이름 패턴, 섹터, 기업 대표 이름 또는 None)], 긴 것부터
    out = []
    for l in open(HOME / "sectors.tsv", encoding="utf-8").read().splitlines()[1:]:
        lg, sec, kind, names = l.split("\t")
        if lg != lang:
            continue
        names = names.split(",")
        for n in names:
            if lang == "ko":
                pat = re.compile(re.escape(n))
            else:  # 기업은 대소문자 그대로, 낱말은 무시
                pat = re.compile(r"(?<![\w-])" + re.escape(n) + r"(?![\w-])", 0 if kind == "기업" else re.I)
            out.append((len(n), pat, sec, names[0] if kind == "기업" else None))
    return [x[1:] for x in sorted(out, key=lambda x: -x[0])]


BANKS = r"(Goldman Sachs|Goldman|JPMorgan|JP Morgan|Morgan Stanley|Citigroup|Citi|Bank of America|BofA|Wells Fargo|BlackRock)"
STRIP = {"ko": re.compile(r"네이버 블로그|건설사업|바이오디젤|하이브리드|카카오값|카카오 가격|\S*(硏|연구소|연구원)|"  # 표본 검토로 더함 (실험기록)
                          r"카카오(페이증권|페이|게임즈|모빌리티|엔터테인먼트|엔터|헬스케어|T)|현대차증권|기아\s?타이거즈|네이버(페이|웹툰|파이낸셜|클라우드)"),  # 계열사는 그 업체가 아님 (2026-10-31 기억 라벨 시범)
         "en": re.compile(r"(?i)blue[- ]chips?|" + BANKS + r"(['’]s)?(?= (says|warns|expects|sees|predicts|forecasts|chief|strategists?|analysts?|economists?|research|delivers|makes|backs|gives|reinstates|upgrades|downgrades|rebalances|calls|names|picks)\b)|"
                       r"(?<= as )" + BANKS + r"|[:|–-] " + BANKS + r"\s*$")}


def tag(t, D, lang):
    t = STRIP[lang].sub(" ", t)
    secs, cos = set(), set()
    for pat, sec, co in D:
        t2 = pat.sub(" ", t)
        if t2 != t:
            secs.add(sec)
            if co:
                cos.add(co)
            t = t2
    return secs, cos
