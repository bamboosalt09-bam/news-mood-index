"""경제 뉴스 제목 수집기: RSS에서 새 제목만 모아 쌓는다. 하루 몇 번씩 돌리면 늘어난다.

python collect.py              # → headlines/ko.tsv, en.tsv
                               #   열: 수집일(과거 수집은 그 달 1일), 출처, 제목, 링크, 요약
python collect.py backfill 24  # 지난 24개월을 한 달씩 구글 뉴스로 검색 (상승장·하락장이 고루 섞이게)
python collect.py target 6     # 모델이 약한 유형(TARGET)만 지난 6개월 검색 → 출처 열 "목표:유형"
python collect.py target 6 정유 airlines  # 그중 이 유형들만
python collect.py daily 365 en  # 지난 365일을 하루씩 검색 → headlines/en_daily.tsv, 날짜 열은 기사 발행 시각(UTC)
python collect.py daily 1097 ko 367  # 367일 전보다 앞의 730일만 (최근은 이미 모았을 때)
"""
import datetime
import email.utils
import html
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOME = Path(__file__).parent


def gnews(q, lang):
    hl, gl = ("ko", "KR") if lang == "ko" else ("en-US", "US")
    return (f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}"
            f"&hl={hl}&gl={gl}&ceid={gl}:{hl.split('-')[0]}")


# 국제 쪽에 무게를 둔 검색어. 원하는 대로 더하고 빼면 된다.
FEEDS = {
    "ko": [
        "https://www.yna.co.kr/rss/economy.xml",
        "https://www.yna.co.kr/rss/international.xml",
        "https://www.hankyung.com/feed/economy",
        "https://www.hankyung.com/feed/international",
        "https://www.hankyung.com/feed/finance",
        "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=ko&gl=KR&ceid=KR:ko",
    ] + [gnews(q, "ko") for q in
         ["뉴욕증시", "연준 금리", "관세", "유가", "환율 달러", "중국 경제", "지정학 리스크", "코스피", "한국은행"]],
    "en": [
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://www.cnbc.com/id/100727362/device/rss/rss.html",
        "https://www.cnbc.com/id/15839069/device/rss/rss.html",
        "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "https://feeds.content.dowjones.io/public/rss/mw_marketpulse",
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
        "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",
    ] + [gnews(q, "en") for q in
         ["stock market", "Federal Reserve rates", "tariffs trade", "oil prices", "dollar currency",
          "China economy", "geopolitical risk markets", "earnings"]],
}


# 과거 수집용 검색어. 방향이 들어간 말은 일부러 빼서 그 달의 실제 분위기를 그대로 받는다.
BACKFILL = {
    "ko": ["코스피", "코스닥", "뉴욕증시", "연준 금리", "한국은행 기준금리", "국제유가", "원달러 환율",
           "관세", "반도체 주가", "기업 실적"],
    "en": ["stock market", "Wall Street", "Fed interest rates", "oil prices", "dollar", "tariffs",
           "earnings", "Treasury yields", "China economy"],
}


# 섹터 지수용 검색어 (`python collect.py sector 1 ko -1`, 2026-10-31). 섹터마다 방향 없는 말을 OR로 묶어 하나씩, sectors.tsv의 섹터와 짝.
# 지수 검색어(BACKFILL)와 따로 headlines/{lang}_sector.tsv에 쌓아 기존 지수의 기사 구성은 그대로 둔다.
SECTOR = {
    "ko": ["2차전지 OR 배터리주", "현대차 OR 기아 OR 자동차주", "바이오 OR 제약사", "네이버 OR 카카오 OR 게임주",
           "은행주 OR 증권주 OR 금융지주", "조선업 OR 조선사 OR 방산", "원전 OR 전력기기", "석유화학 OR 철강 OR 정유주",
           "건설주 OR 건설사", "유통업계 OR 화장품 OR 식품업계"],
    "en": ["chip stocks", "tech stocks", "automakers shares", "bank stocks", "energy stocks", "drugmakers shares",
           "retail stocks", "airline stocks OR defense stocks", "crypto stocks", "Nvidia shares"],
}


# 모델이 약한 유형을 골라 모으는 검색어 (`python collect.py target 6`). 방향이 한쪽으로 쏠리지 않게
# 오름·내림 검색어를 짝으로 넣는다. 출처 열에 "목표:유형"으로 남아 유형별 시험에 쓴다.
TARGET = {
    "ko": {
        "유가": ["국제유가 급등", "국제유가 급락", "유가 상승 증시", "유가 하락 증시", "WTI 상승", "WTI 하락",
               "유가 100달러", "유가 배럴당 내려"],
        "금리": ["기준금리 인상", "기준금리 인하", "국채금리 급등", "국채금리 하락", "금리 인하 기대 증시",
               "금리 인상 우려 증시", "금리 동결"],
        "환율": ["원달러 환율 급등", "원달러 환율 하락", "원화 강세", "원화 약세", "달러 강세", "달러 약세",
               "환율 1500원"],
        "관세": ["관세 인상", "관세 인하", "관세 유예", "관세 철회", "관세 부과 증시", "관세 합의 증시", "상호관세"],
        "방산·지정학": ["방산주 급등", "방산주 하락", "지정학 리스크 증시", "휴전 증시", "중동 긴장 증시",
                   "전쟁 우려 주가"],
        "섹터 예외": ["정유주 유가", "항공주 유가", "항공주 급등", "은행주 금리 인상", "수출주 환율 수혜",
                  "환율 상승 수출주"],
        "반어": ["반등 무색", "상승 무색", "랠리 끝나나", "착시", "찬물 증시", "반등 발목", "간다더니"],
        "정유": ["정유주 상승", "정유주 하락", "정유주 약세", "에쓰오일 주가", "SK이노베이션 주가 유가", "정제마진 정유사"],
        "항공": ["항공주 상승", "항공주 하락", "대한항공 주가", "제주항공 주가 유가", "항공주 환율", "항공유 가격 항공사"],
        "은행": ["은행주 상승", "은행주 하락", "금융지주 주가", "KB금융 주가", "은행주 금리 인하", "은행 순이자마진"],
        "방산": ["방산주 상승", "방산주 하락", "한화에어로스페이스 주가", "LIG넥스원 주가", "현대로템 주가", "방산주 휴전"],
        # 역접: 악재에도 오른 쪽 / 호재에도 내린 쪽을 따로 모아 한쪽으로 쏠리지 않게
        "역접오름": ["악재에도 상승", "우려에도 강세", "부진에도 주가 상승", "적자에도 주가 급등", "딛고 반등",
                  "에도 불구하고 상승", "오히려 상승", "악재 뚫고"],
        "역접내림": ["호재에도 하락", "실적에도 주가 하락", "최대 실적에도 주가", "간다더니", "무색", "착시",
                  "에도 불구하고 하락", "반등 무색"],
    },
    "en": {
        "oil": ["oil prices surge stocks", "oil prices fall", "crude tumbles", "oil jumps", "Brent crude rises",
                "oil slides"],
        "rates": ["Fed raises rates", "Fed cuts rates", "Treasury yields jump", "Treasury yields fall",
                  "rate hike fears stocks", "rate cut hopes stocks"],
        "dollar": ["dollar strengthens", "dollar weakens", "dollar slides", "dollar rallies"],
        "tariffs": ["tariffs raised", "tariff cut", "tariff pause stocks", "tariffs stocks fall", "trade deal tariffs"],
        "defense·geopolitics": ["defense stocks surge", "defense stocks fall", "ceasefire stocks",
                                "Middle East tensions stocks"],
        "sector": ["airline stocks oil prices", "energy stocks oil", "refiners shares", "bank stocks rates rise"],
        "irony": ["rally fades", "erases gains", "despite strong earnings shares fall", "fails to lift stocks"],
        "refiners": ["refiner stocks", "Valero shares", "Marathon Petroleum shares", "energy stocks fall",
                     "energy stocks rise", "Exxon shares oil"],
        "airlines": ["airline stocks fall", "airline stocks rise", "Delta shares fuel", "United Airlines stock",
                     "jet fuel airline shares"],
        "banks": ["bank stocks rally", "bank stocks fall", "JPMorgan shares", "bank shares rate cut",
                  "net interest margin banks"],
        "defense": ["defense stocks rise", "defense stocks fall", "Lockheed Martin shares", "Rheinmetall shares",
                    "defense stocks ceasefire"],
        "despite_up": ["shares rise despite", "stock gains despite", "rallies despite", "shrugs off",
                       "stock jumps despite loss", "climbs despite", "rebounds despite"],
        "despite_down": ["shares fall despite", "stock drops despite strong", "fails to lift shares", "erases gains",
                         "slides despite beat", "rally fades", "despite record profit shares"],
    },
}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    root = ET.fromstring(urllib.request.urlopen(req, timeout=20).read())
    for item in root.iter("item"):
        title = " ".join(item.findtext("title", "").split())
        if "news.google.com" in url:
            title = re.sub(r" - [^-]+$", "", title)  # 끝의 " - 언론사" 제거
        summary = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", item.findtext("description", ""))).split())
        if summary.startswith(title):  # 구글뉴스 요약은 제목 반복이라 버림
            summary = ""
        if title:
            pub = item.findtext("pubDate")
            pub = email.utils.parsedate_to_datetime(pub).astimezone(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M") if pub else ""
            yield title, item.findtext("link", "").strip(), summary[:200], pub


def collect(sources):
    """sources: {파일 이름: [(url, 날짜[, 출처]), ...]} → 새 제목만 headlines/{파일 이름}.tsv에 덧붙임.
    출처를 안 주면 사이트 주소. 날짜가 "발행"이면 기사 발행 시각(UTC)을 쓴다."""
    (HOME / "headlines").mkdir(parents=True, exist_ok=True)
    for lang, pairs in sources.items():
        path = HOME / "headlines" / f"{lang}.tsv"
        seen = set()
        if path.exists():
            seen = {l.split("\t")[2] for l in open(path, encoding="utf-8")}
        n = 0
        with ThreadPoolExecutor(WORKERS) as pool:  # 받는 건 동시에, 쓰는 건 순서대로
            for k, ((url, day, *src), got) in enumerate(zip(pairs, pool.map(fetch_all, [p[0] for p in pairs])), 1):
                src = src[0] if src else urllib.parse.urlparse(url).netloc
                new = []
                for t, link, summary, pub in got:
                    if t not in seen:
                        seen.add(t)
                        new.append(f"{pub if day == '발행' else day}\t{src}\t{t}\t{link}\t{summary}\n")
                with open(path, "a", encoding="utf-8") as f:  # 주소마다 바로 써서 중간에 끊겨도 남게
                    f.writelines(new)
                n += len(new)
                if k % 100 == 0:
                    print(f"  {k}/{len(pairs)} 주소, 새 제목 {n}개", flush=True)
        print(f"{lang}: 새 제목 {n}개, 누적 {len(seen)}개 → {path}")


WORKERS = int(os.environ.get("WORKERS", 4))  # 동시에 받는 주소 수. 구글이 막기 시작하면(실패가 쏟아지면) 줄인다


def fetch_all(url):
    try:
        return list(fetch(url))
    except Exception as e:  # 피드 하나가 죽어도 나머지는 계속
        print(f"  실패 {url}: {e}", flush=True)
        return []
    finally:
        if "news.google.com/rss/search" in url:
            time.sleep(0.5)  # 검색은 연달아 많이 부르므로 천천히


def months_back(n):
    """이번 달 바로 전부터 n개월: (그 달 1일, 다음 달 1일) 문자열 쌍."""
    y, m = time.localtime().tm_year, time.localtime().tm_mon
    for _ in range(n):
        ny, nm = y, m
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
        yield f"{y}-{m:02d}-01", f"{ny}-{nm:02d}-01"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1:2] == ["backfill"]:
        windows = list(months_back(int(sys.argv[2])))
        collect({lang: [(gnews(f"{q} after:{a} before:{b}", lang), a) for a, b in windows for q in qs]
                 for lang, qs in BACKFILL.items()})
    elif sys.argv[1:2] == ["target"]:
        windows, only = list(months_back(int(sys.argv[2]))), sys.argv[3:]  # 유형 이름을 주면 그 유형만
        collect({lang: [(gnews(f"{q} after:{a} before:{b}", lang), a, f"목표:{kind}")
                        for a, b in windows for kind, qs in kinds.items() if not only or kind in only for q in qs]
                 for lang, kinds in TARGET.items()})
    elif sys.argv[1:2] in (["daily"], ["sector"]):  # 지수용: 검색어 묶음을 고정하고 하루씩 → 날마다 같은 기준으로 모임
        lang, today = sys.argv[3], datetime.date.today()
        skip = int(sys.argv[4]) if len(sys.argv) > 4 else 0  # 최근 며칠은 이미 모았으면 건너뜀
        days = [today - datetime.timedelta(d) for d in range(int(sys.argv[2]), skip, -1)]
        qs, out = (BACKFILL, "daily") if sys.argv[1] == "daily" else (SECTOR, "sector")
        collect({f"{lang}_{out}": [(gnews(f"{q} after:{d} before:{d + datetime.timedelta(1)}", lang), "발행", f"검색:{q}")
                                   for d in days for q in qs[lang]]})
    else:
        today = time.strftime("%Y-%m-%d")
        collect({lang: [(u, today) for u in urls] for lang, urls in FEEDS.items()})
