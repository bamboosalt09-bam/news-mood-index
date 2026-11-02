"""언론 지수: 하루 단위로 모은 제목(collect.py daily)을 판정해 시황·재료 지수를 만들고 시장 지표와 비교한다.

python index.py en en-감성-미세  # 미국: headlines/en_daily.tsv ↔ CNN 공포탐욕지수 → data/en_index.tsv
python index.py ko ko-감성-미세  # 한국: headlines/ko_daily.tsv ↔ 코스피 등락·등락 크기(변동성) → data/ko_index.tsv
python index.py ko ko_v1-감성  # 같은 지수를 고정판 주력(v1 = M2 + 묶은 안전장치 사전)으로 (docs/실험기록.md 2026-10-27)
python index.py en en-감성-미세 sox  # 시장을 골라서: spx(S&P 500)·nasdaq·sox(필라델피아 반도체)·kospi·fear_greed → data/en_index_sox.tsv

시황 = 증시 움직임을 전하는 제목 (MARKET 단어가 들어간 것), 재료 = 나머지 (금리·유가·관세·실적 등 원인 쪽).
날짜: 과거 기사는 구글 뉴스가 날짜만 주므로(시각이 07:00·08:00 UTC로 찍힘) 그 날짜 그대로 쓴다.
시각이 진짜인 기사는 장 마감(미국 동부 16시, 한국 15:30) 뒤면 다음 거래일로. 휴장일 기사는 그다음 거래일로.
"""
import bisect
import datetime as dt
import json
import math
import re
import sys
import urllib.request

import numpy as np

import judge as j

MARKET = {
    "en": re.compile(r"\b(stocks?|shares|equit(y|ies)|wall street|dow|s&p|nasdaq|futures|rall(y|ies|ied)|"
                     r"sell-?offs?|bull|bear|market (rout|slump|crash)|record (high|close))\b", re.I),
    "ko": re.compile(r"코스피|코스닥|증시|주가|주식|나스닥|다우|S&P|종목|시총|시가총액|상한가|하한가|순매수|순매도|"
                     r"랠리|투매|강세장|약세장"),
}
CLOSE = {"en": (16, 0), "ko": (15, 30)}   # 장 마감 (현지 시각)
MIN_PER_DAY = 20  # 제목이 이보다 적은 날은 지수에서 뺌
LONG = 20         # 긴 기간 비교에 쓰는 거래일 수
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"}


def cached(name, fetch):
    """data/{name}.tsv에 {날짜: 값}을 저장해 두고 하루 한 번만 새로 받는다."""
    path = j.HOME / "data" / f"{name}.tsv"
    if not path.exists() or dt.date.fromtimestamp(path.stat().st_mtime) < dt.date.today():
        days = fetch()
        path.write_text(f"날짜\t{name}\n" + "".join(f"{d}\t{v:.2f}\n" for d, v in sorted(days.items())), encoding="utf-8")
    return {d: float(v) for d, v in (l.split("\t") for l in path.read_text(encoding="utf-8").splitlines()[1:])}


def fear_greed():
    """CNN 공포탐욕지수 일별 값."""
    def fetch():
        req = urllib.request.Request("https://production.dataviz.cnn.io/index/fearandgreed/graphdata/2025-01-01",
                                     headers=UA | {"Accept": "application/json", "Origin": "https://edition.cnn.com",
                                                   "Referer": "https://edition.cnn.com/"})
        h = json.loads(urllib.request.urlopen(req, timeout=20).read())["fear_and_greed_historical"]["data"]
        return {dt.datetime.fromtimestamp(p["x"] / 1000, dt.timezone.utc).strftime("%Y-%m-%d"): p["y"] for p in h}
    return cached("fear_greed", fetch)


def kospi():
    """코스피 일별 종가 (네이버 금융 차트, 최근 1000거래일 ≈ 2022년 가을부터). 줄 모양: data="20261023|시가|고가|저가|종가|거래량"."""
    def fetch():
        req = urllib.request.Request("https://fchart.stock.naver.com/sise.nhn?symbol=KOSPI&timeframe=day&count=1000"
                                     "&requestType=0", headers=UA)
        text = urllib.request.urlopen(req, timeout=20).read().decode("euc-kr", "replace")
        rows = [r.split("|") for r in re.findall(r'data="([^"]+)"', text)]
        return {f"{r[0][:4]}-{r[0][4:6]}-{r[0][6:]}": float(r[4]) for r in rows}
    return cached("kospi", fetch)


def us_index(code):
    """미국 지수 일별 종가 (네이버 해외지수). code: .INX(S&P 500), .IXIC(나스닥), .SOX(필라델피아 반도체)."""
    def fetch():
        end = dt.date.today().strftime("%Y%m%d")
        req = urllib.request.Request(f"https://api.stock.naver.com/chart/foreign/index/{code}/day"
                                     f"?startDateTime=202301010000&endDateTime={end}2359", headers=UA)
        return {f"{r['localDate'][:4]}-{r['localDate'][4:6]}-{r['localDate'][6:]}": float(r["closePrice"])
                for r in json.loads(urllib.request.urlopen(req, timeout=20).read())}
    return cached(f"us{code}", fetch)


MARKETS = {"fear_greed": ("공포탐욕", fear_greed), "kospi": ("코스피", kospi), "spx": ("S&P500", lambda: us_index(".INX")),
           "nasdaq": ("나스닥", lambda: us_index(".IXIC")), "sox": ("반도체", lambda: us_index(".SOX"))}


def local(t, lang):
    """UTC → 현지 시각. 미국 동부는 서머타임(3월 둘째 일요일 ~ 11월 첫째 일요일) 반영."""
    if lang == "ko":
        return t + dt.timedelta(hours=9)
    sunday = lambda d: d + dt.timedelta((6 - d.weekday()) % 7)
    start = dt.datetime.combine(sunday(dt.date(t.year, 3, 8)), dt.time(7))
    end = dt.datetime.combine(sunday(dt.date(t.year, 11, 1)), dt.time(6))
    return t - dt.timedelta(hours=4 if start <= t < end else 5)


def raw_day(stamp, lang):
    """발행 시각(UTC 문자열) → 현지 기준 그 기사의 날 (장 마감 뒤면 다음 날). 휴장일일 수 있음."""
    if stamp[11:] in ("07:00", "08:00"):  # 날짜만 있는 기사 (미국 서부 자정으로 찍힘)
        return stamp[:10]
    t = local(dt.datetime.strptime(stamp, "%Y-%m-%d %H:%M"), lang)
    return (t.date() + dt.timedelta(1 if (t.hour, t.minute) >= CLOSE[lang] else 0)).isoformat()


def market_day(stamp, trading, lang):
    """발행 시각(UTC 문자열) → 그 기사가 반영될 거래일. trading은 정렬된 거래일 목록."""
    i = bisect.bisect_left(trading, raw_day(stamp, lang))
    return trading[i] if i < len(trading) else None


def is_off(stamp, trading, lang):
    """주말·휴장일 몫 기사인가 (장 마감 뒤 넘어간 날이 거래일이 아님). 2026-11-02부터 웹 지수는 이 기사를 다음 거래일에 넣지 않고 따로 셈."""
    d = raw_day(stamp, lang)
    i = bisect.bisect_left(trading, d)
    return i < len(trading) and trading[i] != d


def scores(lang, model, titles):
    """제목별 감성 점수 (판정 창과 같게 안전장치·전망 약화까지). 모델 점수는 느리므로 data/{lang}_scores_{model}.tsv에
    쌓아 두고 새 제목만 판정. 모델 파일이 캐시보다 새로우면 캐시를 버리고 다시 판정."""
    path = j.HOME / "data" / f"{lang}_scores_{model}{'_q8' if model in j.QUANT else ''}.tsv"  # 8비트 점수는 따로 쌓음
    mfile = next(p for p in (j.HOME / "models" / f"{model}{e}" for e in (".pt", ".npz")) if p.exists())
    if path.exists() and path.stat().st_mtime < mfile.stat().st_mtime:
        path.unlink()
    cache = {}
    if path.exists():
        cache = {t: float(s) for s, t in (l.split("\t", 1) for l in path.read_text(encoding="utf-8").splitlines())}
    todo = sorted({t for t in titles if t not in cache}, key=len)  # 길이 비슷한 것끼리 묶어 빈칸 계산을 줄임
    for i in range(0, len(todo), 2000):
        part = todo[i:i + 2000]
        P, meta = j.predict(model, part)
        new = dict(zip(part, j.score_of(P, meta)))
        cache.update(new)
        with open(path, "a", encoding="utf-8") as f:
            f.writelines(f"{s:.3f}\t{t}\n" for t, s in new.items())
        print(f"  판정 {min(i + 2000, len(todo))}/{len(todo)}", flush=True)
    ts = list(cache)
    meta = j.predict(model, ts[:1])[1]
    s = j.soften(ts, j.guard(ts, list(cache.values()), meta.get("guard"), meta.get("guard_lexicon"))[0])[0]
    return dict(zip(ts, np.where(j.level_only(ts), np.nan, s)))  # 수준만 말한 제목은 지수에서 뺌 (nan)


def corr(x, y):
    """상관계수, 표본 수, 양쪽 p값 (t 근사)."""
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y, n = x[ok], y[ok], int(ok.sum())
    r = float(np.corrcoef(x, y)[0, 1])
    t = r * math.sqrt((n - 2) / max(1 - r * r, 1e-12))
    return r, n, math.erfc(abs(t) / math.sqrt(2))


def partial(x, y, z):
    """z의 영향을 x·y 양쪽에서 빼고 남은 상관."""
    ok = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    A = np.c_[np.ones(ok.sum()), z[ok]]
    res = lambda v: v[ok] - A @ np.linalg.lstsq(A, v[ok], rcond=None)[0]
    return corr(res(x), res(y))


shift = lambda v, k: np.r_[[np.nan] * k, v[:-k]] if k > 0 else np.r_[v[-k:], [np.nan] * -k]  # +k: k일 전 값, -k: k일 뒤 값


def rolling(v, k, f=np.mean):
    """오늘까지 k거래일 값으로 f (빈 날은 빼고, 절반 넘게 비면 nan)."""
    out = np.full(len(v), np.nan)
    for i in range(k - 1, len(v)):
        w = v[i - k + 1:i + 1]
        w = w[~np.isnan(w)]
        if len(w) > k // 2:
            out[i] = f(w)
    return out


def tests_for(name, M):
    """비교 목록: (이름, 지수 → (상관, 표본, p)). M은 거래일별 시장 값, name은 그 이름 (공포탐욕이면 심리 지표용, 나머지는 종가용)."""
    k = LONG
    if name == "공포탐욕":
        dF = np.r_[np.nan, np.diff(M)]
        fut, past = shift(M, -k) - M, M - shift(M, k)
        return [("수준: 오늘 뉴스 ↔ 오늘 공포탐욕", lambda x: corr(x, M)),
                ("같은 날: 오늘 뉴스 ↔ 오늘 공포탐욕 변화", lambda x: corr(x, dF)),
                ("다음 날: 오늘 뉴스 → 다음 날 공포탐욕 변화", lambda x: corr(x, shift(dF, -1))),
                ("다음 날 (오늘 변화 효과 뺀 뒤)", lambda x: partial(x, shift(dF, -1), dF)),
                ("역방향: 전날 공포탐욕 변화 → 오늘 뉴스", lambda x: corr(x, shift(dF, 1))),
                (f"{k}일 평균 뉴스 ↔ 공포탐욕 수준", lambda x: corr(rolling(x, k), M)),
                (f"{k}일 평균 → 앞으로 {k}일 변화 (수준 효과 뺀 뒤)", lambda x: partial(rolling(x, k), fut, M))]
    R = np.r_[np.nan, np.diff(np.log(M))] * 100       # 오늘 등락률 (%)
    A = np.abs(R)                                      # 등락 크기 = 그날 변동성
    vol = rolling(R, k, np.std)                        # 지난 k일 변동성
    fut_r = rolling(R, k, np.sum)                      # 지난 k일 누적 등락 → 앞으로 k일은 shift(-k)
    return [(f"같은 날: 오늘 뉴스 ↔ 오늘 {name} 등락", lambda x: corr(x, R)),
            ("다음 날: 오늘 뉴스 → 다음 날 등락", lambda x: corr(x, shift(R, -1))),
            ("다음 날 (오늘 등락 효과 뺀 뒤)", lambda x: partial(x, shift(R, -1), R)),
            ("역방향: 전날 등락 → 오늘 뉴스", lambda x: corr(x, shift(R, 1))),
            ("변동성 같은 날: 오늘 뉴스 ↔ 오늘 등락 크기", lambda x: corr(x, A)),
            ("변동성 다음 날: 오늘 뉴스 → 다음 날 등락 크기", lambda x: corr(x, shift(A, -1))),
            ("변동성 다음 날 (오늘 크기 효과 뺀 뒤)", lambda x: partial(x, shift(A, -1), A)),
            (f"{k}일 평균 뉴스 ↔ 지난 {k}일 변동성", lambda x: corr(rolling(x, k), vol)),
            (f"{k}일 평균 → 앞으로 {k}일 변동성 (지금 변동성 뺀 뒤)", lambda x: partial(rolling(x, k), shift(vol, -k), vol)),
            (f"{k}일 평균 → 앞으로 {k}일 등락 (지난 {k}일 등락 뺀 뒤)", lambda x: partial(rolling(x, k), shift(fut_r, -k), fut_r))]


def main(lang, model, which=None):
    which = which or ("fear_greed" if lang == "en" else "kospi")
    mname, market = MARKETS[which][0], MARKETS[which][1]()
    trading = sorted(market)
    rows = [l.split("\t") for l in open(j.HOME / "headlines" / f"{lang}_daily.tsv", encoding="utf-8")]
    rows = [(market_day(r[0], trading, lang), r[2]) for r in rows if r[0]]
    rows = [(d, t) for d, t in rows if d]
    print(f"제목 {len(rows)}개, 거래일 {len({d for d, _ in rows})}일, 감성 모델 {model}")
    sc = scores(lang, model, [t for _, t in rows])

    days = [d for d in trading if d >= min(d for d, _ in rows)]
    is_market = MARKET[lang].search
    groups = {"전체": lambda t: True, "시황": lambda t: bool(is_market(t)), "재료": lambda t: not is_market(t)}
    print(f"시황 제목 {sum(bool(is_market(t)) for _, t in rows)}개 / 재료 {sum(not is_market(t) for _, t in rows)}개")
    by = {g: {} for g in groups}
    for d, t in rows:
        if np.isnan(sc[t]):
            continue
        for g, keep in groups.items():
            if keep(t):
                by[g].setdefault(d, []).append(sc[t])
    idx = {g: np.array([np.mean(by[g][d]) if len(by[g].get(d, [])) >= MIN_PER_DAY else np.nan for d in days])
           for g in groups}
    M = np.array([market[d] for d in days])

    out = j.HOME / "data" / (f"{lang}_index.tsv" if which in ("fear_greed", "kospi") else f"{lang}_index_{which}.tsv")
    out.write_text(f"날짜\t{mname}\t" + "\t".join(f"{g}\t{g}_개수" for g in groups) + "\n"
                   + "".join(f"{d}\t{M[i]:.2f}\t" + "\t".join(f"{idx[g][i]:.3f}\t{len(by[g].get(d, []))}" for g in groups)
                             + "\n" for i, d in enumerate(days)), encoding="utf-8")
    print(f"일별 지수 저장: {out}\n")

    print(f"{'비교':40}" + "".join(f"{g:>20}" for g in groups))
    for name, f in tests_for(mname, M):
        cells = [(lambda r, n, p: f"{r:+.2f} (p={p:.3f}, {n}일)")(*f(idx[g])) for g in groups]
        print(f"{name:40}" + "".join(f"{c:>20}" for c in cells))
    print(f"\n상관은 -1~+1. p가 0.05보다 작으면 우연으로 보기 어렵다는 뜻. {LONG}일 평균끼리는 겹쳐서 p가 실제보다 작게 나옴.")


def check():
    trading = ["2026-03-06", "2026-03-09", "2026-03-10"]
    assert market_day("2026-03-07 07:00", trading, "ko") == "2026-03-09"   # 날짜만 있는 토요일 기사 → 월요일
    assert market_day("2026-03-09 05:00", trading, "ko") == "2026-03-09"   # 한국 14시 → 그날
    assert market_day("2026-03-09 07:30", trading, "ko") == "2026-03-10"   # 한국 16:30 → 다음 거래일
    assert market_day("2026-03-09 21:30", trading, "en") == "2026-03-10"   # 동부 17:30 (서머타임) → 다음 거래일
    assert shift(np.array([1., 2, 3]), 1)[1:].tolist() == [1, 2] and shift(np.array([1., 2, 3]), -1)[:2].tolist() == [2, 3]
    print("검산 통과")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if sys.argv[1:] == ["check"]:
        check()
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "en", sys.argv[2] if len(sys.argv) > 2 else "en-감성",
             sys.argv[3] if len(sys.argv) > 3 else None)
