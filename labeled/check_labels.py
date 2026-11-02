"""라벨 기준(라벨기준.md)에 어긋날 수 있는 라벨을 규칙으로 찾는다. 고치지는 않고 후보 목록만 낸다.

python labeled/check_labels.py             # labeled/ko.tsv → labeled/점검/기준점검_<날짜>.tsv, 규칙별 개수 출력
python labeled/check_labels.py --all       # 시험지(오늘60·무작위700)까지

열: 규칙, 처리(자동: 기계적으로 고쳐도 되는 것 / 검토: 사람이 봐야 하는 것), 현재·제안 감성, 현재·제안 주제, 제목
"""
import datetime
import re
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parent
COMPANY = {"국내 기업·실적", "해외 기업·실적", "국내 업종·테마"}  # 감성 최대 ±1
INDEX = re.compile(r"코스피|코스닥|증시|지수|나스닥|다우|s&p|뉴욕장|증권시장|kospi")
STRONG = re.compile(r"급락|폭락|급등|폭등|패닉|사이드카|서킷|검은|블랙|최고치|사상 최고|사상 첫|역대 최고|전쟁|확전|공습|휴전|종전|"
                    r"봉쇄|관세|부도|법정관리|디폴트|깜짝|빅컷|금융위기|외환위기|쇼크|대폭|최악|대란|붕괴|와르르|털썩|추락|"
                    r"돌파|\d{1,3}(\.\d+)?\s?%")
# 앞으로의 일·의견만 (예상치를 밑돈 같은 '이미 나온 사실'은 빼고, 수사적 물음표만 있는 제목도 뺌)
OPINION = re.compile(r"할까|일까|을까|날까|갈까|볼까|나올까|전망|가능성|관측|할 듯|될 듯|칼럼|사설|기고|시론|인터뷰|\[.{0,8}(시론|칼럼|기고|진단)\]"
                     # 2026-10-26 원인 분석에서 놓친 꼴: 말끝 추측·질문, 연재 칼럼·분석 표지, 말한 사람이 있는 따옴표 의견
                     r"|[가-힣]\s?것[\"'”’…]|지도[\"'”’…]|유력|되나|하나[\s…?\"'”’]*$|[가-힣]나\?|\?"
                     r"|\[[^\]]{1,12}의 [^\]]{2,15}\]|market comment|리포트\]|\[기획\]|\[분석\]|why&next"
                     r"|(?:증권|證|센터장|연구원|애널리스트|ceo|coo|회장|총재|장관|의장|이사|ib|모간|골드만|ubs|블랙록|아문디|노무라|oecd|imf)\S{0,6}\s?[\"“'‘]"
                     r"|^[가-힣]{2,4}\s?[\"“]|[가-힣]나[\s…\"'”’]*(?:-.*)?$|우려|변수|갑론을박|가성비|사라[\"”,]|매수금지|!")
SMALL = re.compile(r"소폭|강보합|약보합|(?<![\d.])0\.[0-4]\d?\s?%")  # 지수 0.5% 미만 = ±0.5 (6번 표)
OWN = re.compile(r"상한가|하한가|급등|급락|폭등|폭락|최대 실적|어닝 ?서프라이즈|깜짝 실적|어닝 ?쇼크|적자 전환|흑자 전환|수주|질주|곤두박질")
SECTOR = re.compile(r"[가-힣a-z]{1,8}주(?:가)?[\s'\"‘’]*(?:급|상승|하락|강세|약세|활짝|웃|울|반등|질주|들썩|날개|휘청|폭|껑충|뚝|훨훨|불기둥|비명)")
SECTOR2 = re.compile(r"업종|테마|관련주|수혜주|[가-힣]{1,6}株|[가-힣a-z]{1,8}주(?:들)?[\s'\"‘’…·,]|etf")  # 업종·테마 묶음 후보 (넓게, 사람이 봄)
IPO = re.compile(r"상장 첫날|따상|따따블|공모가")
DEPOSIT = re.compile(r"예금|적금|대출금리|주담대|예대|코픽스|금리 상품")
FOREIGN_RATE = re.compile(r"美|미국|미 재무|미 국채|일본|日|연준|fomc|fed|ecb|boj|유럽|중국|中|인민은행|호주|캐나다|영국|boe")
GEO = re.compile(r"이란|이스라엘|전쟁|전면전|확전|휴전|종전|공습|폭격|호르무즈|우크라|러시아|하마스|헤즈볼라|후티|홍해|중동")
MARKET = re.compile(r"유가|원유|wti|브렌트|코스피|코스닥|증시|뉴욕|다우|나스닥|s&p|환율|원·달러|금리|국채|금값|주가|비트코인|[가-힣]+주\b")
FOREIGN_INDEX = re.compile(r"지수|증시|닛케이|상하이|항셍|니케이|유럽증시|아시아증시|토픽스|stoxx|dax")
FX = re.compile(r"(\d+(?:\.\d+)?)\s?원\s?(오른|내린|상승|하락|올라|내려|급등|급락|↑|↓)")
LANDMARK = re.compile(r"돌파|최고|최저|연고점|연저점|경신|위기|마지노선|넘|깨|고착")
PERIOD = re.compile(r"(\d+|두|세|한)\s?(달|주|개월|년|일)\s?(새|간|만에|째)")  # "두 달 새 200원": 하루 움직임이 아님
STEP = [-2, -1, -0.5, 0, 0.5, 1, 2]


def weaker(v):
    """한 단계 약하게: ±2→±1, ±1→±0.5, ±0.5→0."""
    return {2: 1, -2: -1, 1: 0.5, -1: -0.5, 0.5: 0, -0.5: 0}.get(v, v)


def check(rows):
    out = []
    add = lambda rule, how, r, s=None, t=None: out.append((rule, how, r[0], "" if s is None else f"{s:g}", r[2], t or "", r[3]))
    for r in rows:
        v, topic, t = float(r[0]), r[2], r[3].lower()
        # 1. ±2는 시장 전체만: 개별 기업 주제인데 지수 말이 없으면 ±1로
        sector_move = topic == "국내 증시" and SECTOR.search(t) and not INDEX.search(t)  # 5번에서 업종·테마로 옮겨질 것
        if abs(v) == 2 and (topic in COMPANY or sector_move) and not INDEX.search(t):
            add("±2 개별 기업", "자동", r, v / 2)
        elif abs(v) == 2 and not STRONG.search(t):
            add("±2 근거 약함", "검토", r, v / 2)
        # 2. 전망·의견·질문형인데 ±1 이상 (목표가 같은 공식 수치는 제외). ±0.5는 이미 약하게 붙인 값일 수 있어 다시 잡지 않음
        #    (2026-10-26: ±0.5→0으로 내린 203개가 두 번 약하게 한 실수였음, docs/실험기록.md)
        if abs(v) >= 1 and OPINION.search(t) and "목표가" not in t:
            add("전망·의견", "검토", r, weaker(v))
        # 6번 표: 지수 소폭 움직임은 ±0.5 (급락 딛고 강보합 마감처럼 마지막 결과가 소폭이면)
        if abs(v) >= 1 and INDEX.search(t) and SMALL.search(t) and (re.search(r"딛고|급락서|하락서|급등서|낙폭", t) or not re.search(r"(?<![\d.])(?:[1-9]\d*(?:\.\d+)?)\s?%\s?(?:대\s?)?(?:상승|하락|급|올|내|↑|↓)", t)):
            add("지수 소폭", "검토", r, 0.5 if v > 0 else -0.5)
        # 3. 개별 기업인데 0: 그 종목 기준으로 분명한 소식이면 ±1
        if v == 0 and topic in COMPANY and OWN.search(t):
            add("개별 종목 0", "검토", r)
        # 5. 주제 경계 (2026-10-26 17칸·결과 기준)
        if topic in ("국내 증시", "국내 기업·실적") and SECTOR2.search(t + " ") and not (topic == "국내 증시" and INDEX.search(t)):
            add("업종·테마 후보", "검토", r, None, "국내 업종·테마")
        if topic == "국내 증시" and IPO.search(t):
            add("상장 첫날 → 기업", "검토", r, None, "국내 기업·실적")
        if topic == "국내 경기·물가" and DEPOSIT.search(t):
            add("은행 상품 금리 → 국내 금리", "검토", r, None, "국내 금리·통화정책")
        if topic == "국내 금리·통화정책" and FOREIGN_RATE.search(t) and not re.search(r"한은|한국은행|금통위|국고채|국내|韓|한국", t):
            add("해외 금리 → 해외 금리·통화", "검토", r, None, "해외 금리·통화")
        if topic == "국제정치·지정학" and MARKET.search(t):
            add("지정학인데 시장 결과", "검토", r, None, "(결과 칸)")
        if topic == "경제와 무관" and GEO.search(t):
            add("무관인데 지정학 사건", "검토", r, None, "국제정치·지정학")
        if topic == "기타 해외 증시" and not FOREIGN_INDEX.search(t):
            add("해외 업종·기업 → 해외 기업", "검토", r, None, "해외 기업·실적")
        # 6. 환율 숫자: 하루 N원 움직임과 등급이 맞는지 (상징적인 선이면 건너뜀)
        m = FX.search(t)
        if m and topic == "원화·환율" and not LANDMARK.search(t) and not PERIOD.search(t):
            n, up = float(m.group(1)), m.group(2) in ("오른", "상승", "올라", "급등", "↑")
            want = (0.5 if n < 10 else 1 if n < 30 else 2) * (-1 if up else 1)
            if v != want:
                add("환율 숫자", "검토", r, want)
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    files = ["ko.tsv"] + (["시험/ko_오늘60.tsv", "시험/ko_무작위300.tsv", "시험/ko_무작위400.tsv"] if "--all" in sys.argv else [])
    rows = [l.split("\t") for f in files for l in (HOME / f).read_text(encoding="utf-8").splitlines()[1:]]
    # 이미 사람이 보고 그대로 두기로 한 제목은 다시 내지 않음 (감성 규칙과 주제 규칙은 따로 기록)
    read = lambda p: {l for l in p.read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")} if p.exists() else set()
    done, done_topic = read(HOME / "점검" / "검토완료.txt"), read(HOME / "점검" / "검토완료_주제.txt")
    out = [x for x in check(rows) if x[6] not in (done_topic if x[5] else done)]
    path = HOME / "점검" / f"기준점검_{datetime.date.today()}.tsv"
    path.write_text("규칙\t처리\t현재 감성\t제안 감성\t현재 주제\t제안 주제\t제목\n" + "".join("\t".join(x) + "\n" for x in out), encoding="utf-8")
    from collections import Counter
    c = Counter((x[0], x[1]) for x in out)
    print(f"라벨 {len(rows)}개 중 후보 {len(out)}건 (제목 {len({x[6] for x in out})}개) → {path}")
    for (rule, how), n in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"  {rule:20} {how}  {n}건")
