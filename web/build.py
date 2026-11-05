# 웹 자료 만들기 (2026-10-31, web/DATA.md 형식): 판정 캐시 + 시세 → web/public/data/latest.json, archive/<날짜>.json.
# 거래일 배정·갈래·중복 묶기는 실험/지수분석.py와 같은 방식. AI 해설은 web/summarize.py.
# python web/build.py              # 제목이 캐시보다 새로우면 점수부터 (실험/지수자료.py, 한국어 ~1분·영어 ~6분)
# python web/build.py --skip-score # 있는 캐시로만
# python web/build.py --check      # latest.json 형식 검사
import sys, os, re, json, math, time, collections, subprocess, urllib.request, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge as j
import index as ix
import summarize
import sectors as S
import memjudge
import sectorjudge
import repcheck
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
OUT = j.HOME / "web" / "public" / "data"
PRICE = j.HOME / "data" / "일별_시고저종.tsv"
KST = dt.timezone(dt.timedelta(hours=9))
CONF = {"ko": {"name": "코스피", "price": "코스피", "news": "한국어 경제 뉴스", "col": "최종", "kinds": ("시장반응", "시장해석", "경제전망", "경제사건")},
        "en": {"name": "S&P 500", "price": "S&P500", "news": "영어 경제 뉴스 (미국 투자자 관점)", "col": "최종_미국", "kinds": ("시황", "재료")}}
MIN_DAY, MIN_KIND, REL, SERIES = 20, 5, 60, 120


# ── 시세: 네이버에서 마감된 봉만 받아 data/일별_시고저종.tsv의 마지막 날부터 덮어씀 (실패하면 기존 파일) ──
def fetch_ohlc(lang, code):
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    loc = ix.local(now, lang)
    h, m = ix.CLOSE[lang]
    closed = lambda d: d < loc.date().isoformat() or (d == loc.date().isoformat() and loc.hour * 60 + loc.minute >= h * 60 + m + 30)
    if lang == "ko":
        req = urllib.request.Request(f"https://fchart.stock.naver.com/sise.nhn?symbol={code}&timeframe=day&count=30&requestType=0", headers=ix.UA)
        text = urllib.request.urlopen(req, timeout=20).read().decode("euc-kr", "replace")
        rows = [r.split("|")[:5] for r in re.findall(r'data="([^"]+)"', text)]
    else:
        start = (now - dt.timedelta(40)).strftime("%Y%m%d")
        req = urllib.request.Request(f"https://api.stock.naver.com/chart/foreign/index/{code}/day?startDateTime={start}0000"
                                     f"&endDateTime={now:%Y%m%d}2359", headers=ix.UA)
        rows = [[r["localDate"], r["openPrice"], r["highPrice"], r["lowPrice"], r["closePrice"]] for r in json.loads(urllib.request.urlopen(req, timeout=20).read())]
    out = {}
    for r in rows:
        d = f"{r[0][:4]}-{r[0][4:6]}-{r[0][6:8]}"
        if closed(d):
            out[d] = [str(x) for x in r[1:5]]
    return out


def update_prices():
    lines = PRICE.read_text(encoding="utf-8").splitlines()
    rows = {(r[0], r[1]): r[2:] for r in (l.split("\t") for l in lines[1:])}
    order = list(dict.fromkeys(m for m, _ in rows))
    for mk, lang, code in (("코스피", "ko", "KOSPI"), ("코스닥", "ko", "KOSDAQ"), ("S&P500", "en", ".INX"), ("나스닥", "en", ".IXIC")):
        last = max(d for m, d in rows if m == mk)
        try:
            new = {d: v for d, v in fetch_ohlc(lang, code).items() if d >= last}
        except Exception as e:
            print(f"  시세 {mk} 받기 실패 ({type(e).__name__}), 기존 파일로 계속")
            continue
        rows.update({(mk, d): v for d, v in new.items()})
        print(f"  시세 {mk}: {last} 이후 {len(new)}일 갱신, 마지막 {max(new, default=last)}")
    PRICE.write_text("\n".join([lines[0]] + ["\t".join([m, d, *rows[m, d]]) for m in order for d in sorted(d for mm, d in rows if mm == m)]) + "\n", encoding="utf-8")


def prices(name):
    d = {r[1]: float(r[5]) for r in (l.split("\t") for l in PRICE.read_text(encoding="utf-8").splitlines()[1:]) if r[0] == name}
    days = sorted(d)
    C = np.array([d[x] for x in days])
    return days, C, np.r_[np.nan, 100 * (C[1:] / C[:-1] - 1)]


# ── 업체 규모 무게 (2026-11-02, docs/실험기록.md 같은 날): 시총 상위 10개 업체 제목은 지수에서 ×2 (한국어로 찾고 영어로 확인),
# 대표 제목 순위에선 작은 종목 꼴(특징주·상장 첫날 등) ×0.25 (화면만)
BIG = {"ko": re.compile("삼성전자|SK하이닉스|LG에너지솔루션|삼성바이오로직스|현대차|KB금융|삼성생명|셀트리온|삼성SDI|신한지주"),
       "en": re.compile(r"\b(nvidia|apple|microsoft|alphabet|google|amazon|meta|broadcom|tesla|berkshire|jpmorgan)\b", re.I)}
SMALL = re.compile(r"특징주|핫종목|상장 첫날|공모가|상한가|하한가|급등주|테마주|따상|신고가 경신|\bpenny stock|\bmeme stock|\bIPO debut", re.I)
BIG_W, SMALL_RANK = 2.0, 0.25
rank_w = lambda lang, t: BIG_W if BIG[lang].search(t) else SMALL_RANK if SMALL.search(t) else 1.0
wmean = lambda xs: float(np.average([it["s"] for it in xs], weights=[it["w"] for it in xs]))


# ── 제목 (실험/지수분석.py items·kind·assign 그대로, 링크만 더 가져감) ──
def items(lang, col):
    rows = [l.rstrip("\n").split("\t") for l in open(j.HOME / "data" / "cache" / f"지수자료_{lang}.tsv", encoding="utf-8")]
    k = {c: i for i, c in enumerate(rows[0])}
    return {r[-1]: {"s": float(r[k[col]]), "w": BIG_W if BIG[lang].search(r[-1]) else 1.0,
                    "mkt": r[k["시황"]] == "1", "cause": r[k["원인꼴"]] == "1", "opin": r[k["전망표지"]] == "1", "topic": r[k["주제"]]}
            for r in rows[1:] if r[k["수준만"]] != "1"}


def kind(it, lang):
    if lang == "en":
        return "시황" if it["mkt"] else "재료"
    if it["mkt"]:
        return "시장해석" if it["cause"] else "시장반응"
    return "경제전망" if it["opin"] else "경제사건"


def assign(lang, days, IT, files=("daily",), since="", off=None, pending=None):
    """거래일 → [(제목, 판정, 링크)]. 주말·휴장일 몫(금요일 장 마감 뒤 ~ 일요일 등)은 다음 거래일에 넣지 않고 off[다음 거래일]에 (제목, 판정, 링크, 그 날)로 (2026-11-02)."""
    by, seen = collections.defaultdict(list), set()
    for f in files:
        for l in open(j.HOME / "headlines" / f"{lang}_{f}.tsv", encoding="utf-8"):
            p = l.rstrip("\n").split("\t")
            if len(p) < 3 or p[0][:2] != "20" or p[0] < since or p[2] not in IT or p[2] in seen:
                continue
            seen.add(p[2])
            d = ix.market_day(p[0][:16], days, lang)
            if d is None and pending is not None:  # 아직 마감 안 된 거래일(또는 진행 중인 주말) 몫: (제목, 판정, 링크, 그 날)
                pending.append((p[2], IT[p[2]], p[3] if len(p) > 3 else "", ix.raw_day(p[0][:16], lang)))
            elif d and ix.is_off(p[0][:16], days, lang):
                if off is not None:
                    off.setdefault(d, []).append((p[2], IT[p[2]], p[3] if len(p) > 3 else "", ix.raw_day(p[0][:16], lang)))
            elif d:
                by[d].append((p[2], IT[p[2]], p[3] if len(p) > 3 else ""))
    return by


# ── 하루 값 ──
def stats(s, w=None):  # mood만 업체 규모 무게 평균, 나머지(충격·엇갈림·비율)는 제목마다 같게
    s = np.asarray(s, float)
    a = np.abs(s).sum()
    return {"mood": float(np.average(s, weights=w)) if w is not None else s.mean(), "shock": np.abs(s).mean(), "spread": s.std(), "cancel": 1 - abs(s.sum()) / a if a > 0 else 0.0,
            "neg_share": (s <= -0.25).mean(), "pos_share": (s >= 0.25).mean(), "titles": len(s)}


def rel(v, i):  # 앞 REL개 지수일 대비 z
    w = np.array(v[max(0, i - REL):i])
    return (v[i] - w.mean()) / w.std() if len(w) >= 20 and w.std() > 0 else None


def top(prev, m):  # 상위 몇 %: 앞 날들 가운데 오늘만큼 이상 치우친 날의 비율 (오늘 포함하지 않음, 1% 아래는 1%)
    prev = [x for x in prev if x is not None]
    if len(prev) < 20 or m is None:
        return None
    pos = m >= float(np.median(prev))
    k = sum(1 for x in prev if (x >= m if pos else x <= m))
    return {"side": "긍정" if pos else "부정", "pct": max(1, round(100 * k / len(prev))), "n": len(prev), "days": k}


def mood_label(z):
    return "중립" if z is None else "매우 부정" if z < -1.5 else "부정" if z < -0.5 else "중립" if z <= 0.5 else "긍정" if z <= 1.5 else "매우 긍정"


def shock_label(z):
    return "보통" if z is None else "평온" if z < 0 else "보통" if z < 1 else "높음" if z < 2 else "매우 높음"


def spread_label(z):  # 호재·악재 엇갈림 = 그날 점수 표준편차의 앞 60일 대비 z (충격도와 따로, docs/실험기록.md 2026-11-04)
    return "보통" if z is None else "적음" if z < 0 else "보통" if z < 1 else "큼" if z < 2 else "매우 큼"


def clusters(lst):  # 13번 묶기: 글자 3-gram Jaccard ≥ 0.5
    tri = lambda t: {t[i:i + 3] for i in range(len(t) - 2)} or {t}
    reps, mem, inv = [], [], collections.defaultdict(list)
    for t, it, link in lst:
        g = tri("".join(j.clean(t).lower().split()))
        cand = collections.Counter(c for x in g for c in inv.get(x, ()))
        best = next((c for c, n in cand.most_common(5) if n / (len(g) + len(reps[c]) - n) >= 0.5), None)
        if best is None:
            best = len(reps); reps.append(g); mem.append([])
            for x in g:
                inv[x].append(best)
        mem[best].append((t, it, link))
    out = []
    for m in mem:
        sc = [it["s"] for _, it, _ in m]
        out.append({"title": m[0][0], "score": round(float(np.mean(sc)), 2), "count": len(m), "contribution": round(sum(sc) / len(lst), 4),
                    "topic": collections.Counter(it["topic"] for _, it, _ in m).most_common(1)[0][0], "link": m[0][2]})
    return out


r2 = lambda x, k=3: None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), k)


def market(lang):
    c = CONF[lang]
    days, C, chg = prices(c["price"])
    IT = items(lang, c["col"])
    off, pend = {}, []
    by = assign(lang, days, IT, off=off, pending=pend)
    idx = [i for i, d in enumerate(days) if len(by.get(d, [])) >= MIN_DAY]
    D = [days[i] for i in idx]
    S = [stats([it["s"] for _, it, _ in by[d]], [it["w"] for _, it, _ in by[d]]) for d in D]
    mood, shock = [s["mood"] for s in S], [s["shock"] for s in S]
    kinds_of = lambda d: {k: [it["s"] for _, it, _ in by[d] if kind(it, lang) == k] for k in c["kinds"]}
    series = []
    for n in range(max(0, len(D) - SERIES), len(D)):
        i, s = idx[n], S[n]
        series.append({"date": D[n], "mood": r2(s["mood"]), "mood_rel": r2(rel(mood, n), 2), "shock": r2(s["shock"]), "spread": r2(s["spread"]),
                       "cancel": r2(s["cancel"]), "titles": s["titles"], "close": r2(C[i], 2), "change_pct": r2(chg[i], 2),
                       "kinds": {k: r2(np.mean(v)) if len(v) >= MIN_KIND else None for k, v in kinds_of(D[n]).items()}})
    n, i, d, s = len(D) - 1, idx[-1], D[-1], S[-1]
    mr, sr, spr = rel(mood, n), rel(shock, n), rel([s["spread"] for s in S], n)
    today = {"mood": r2(s["mood"]), "mood_rel": r2(mr, 2), "mood_label": mood_label(mr), "shock": r2(s["shock"]), "spread": r2(s["spread"]),
             "shock_rel": r2(sr, 2), "shock_label": shock_label(sr), "spread_rel": r2(spr, 2), "spread_label": spread_label(spr),
             "cancel": r2(s["cancel"]), "neg_share": r2(s["neg_share"]),
             "pos_share": r2(s["pos_share"]), "titles": s["titles"], "change_pct": r2(chg[i], 2), "close": r2(C[i], 2),
             "usual_mood": r2(float(np.mean(mood[max(0, n - REL):n]))), "usual_sd": r2(float(np.std(mood[max(0, n - REL):n]))),
             "rank": top(mood[max(0, n - REL):n], s["mood"])}
    # 상관: 지수 전체 기간, 거래일 배열 위에서 (같은 날 = mood↔등락률, 다음 날 = 오늘 등락 뺀 부분상관)
    x = np.full(len(days), np.nan); x[idx] = mood
    x, y = x[idx[0]:], chg[idx[0]:]
    method = {"model": f"{j.MAIN[lang + '-감성']} (판정 창 주력, 8비트{', 미국 투자자 관점' if lang == 'en' else ''})",
              "titles_total": sum(len(by[d]) for d in D), "days_total": len(D), "since": D[0],
              "same_day_corr": r2(ix.corr(x, y)[0], 2), "next_day_corr": r2(ix.partial(x, ix.shift(y, -1), y)[0], 2)}
    # 주제: 그날 비중 상위 6개, 앞 7개 지수일 비중과 함께
    cnt = collections.Counter(it["topic"] for _, it, _ in by[d])
    prior = collections.Counter(it["topic"] for dd in D[-8:-1] for _, it, _ in by[dd])
    ptot = max(sum(prior.values()), 1)
    topics = [{"topic": t, "share": r2(k / len(by[d])), "share_prior7": r2(prior[t] / ptot), "mood": r2(np.mean([it["s"] for _, it, _ in by[d] if it["topic"] == t]))}
              for t, k in cnt.most_common() if t != "경제와 무관"][:6]
    cl = clusters(by[d])
    # 대표 후보 10개씩 → Haiku 방향 확인에서 반대로 나온 것은 빼고 5개 (docs/실험기록.md 2026-11-02, 40일 800개 중 방향 틀림 1.5%)
    neg = [x for x in sorted(cl, key=lambda x: x["contribution"] * rank_w(lang, x["title"])) if x["contribution"] < 0][:10]
    pos = [x for x in sorted(cl, key=lambda x: -x["contribution"] * rank_w(lang, x["title"])) if x["contribution"] > 0][:10]
    ok = repcheck.check(lang, [x["title"] for x in neg + pos])
    neg, pos = repcheck.keep(neg, ok, -1, 5), repcheck.keep(pos, ok, 1, 5)
    for tag, lst in (("n", neg), ("p", pos)):
        for k, x in enumerate(lst, 1):
            x["id"] = f"{lang}-{tag}{k}"
    return {"id": lang, "name": c["name"], "news": c["news"], "date": d, "today": today, "weekend": weekend(lang, off, d), "live": live(lang, pend, mood, C[-1], by, D), "series": series, "method": method,
            "kinds": {k: {"mood": r2(np.mean(v)) if len(v) >= MIN_KIND else None, "titles": len(v)} for k, v in kinds_of(d).items()},
            "top_negative": [{"id": x.pop("id"), **x} for x in neg], "top_positive": [{"id": x.pop("id"), **x} for x in pos],
            "topics": topics, "sectors": sectors(lang, days, IT, d, chg[i]), "summary": None}


OPEN = {"ko": (9, 0), "en": (9, 30)}


def intraday(lang, day):  # 그 날 장중 마지막 지수 값 (주기 수집기가 모은 것: 코스피 1분, S&P 500 2분), 없으면 None
    if lang == "ko":
        rows = [l.split("	") for l in (j.HOME / "data" / "kospi_minute.tsv").read_text(encoding="utf-8").splitlines()[1:]]
        rows = [r for r in rows if r[0][:10] == day]
        return (rows[-1][0][11:16], float(rows[-1][1])) if rows else None
    rows = [l.split("	") for l in (j.HOME / "data" / "us_realtime.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    rows = [r for r in rows if r[1] == ".INX" and r[0][:10] == day]
    return (rows[-1][0][11:16], float(rows[-1][2])) if rows else None


def stamps(lang):  # 제목 → 처음 나온 시각 (현지), 실시간 칸의 '같은 시각까지' 비교용
    out = {}
    for l in open(j.HOME / "headlines" / f"{lang}_daily.tsv", encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) > 2 and p[0][:2] == "20" and p[2] not in out:
            out[p[2]] = p[0][:16]
    return out


def same_time(lang, by, D, live_day, now):  # 지난 거래일마다, 실시간 칸과 같은 시각(그 날 0시 기준 같은 거리)까지 나온 제목만의 평균
    st = stamps(lang)
    gap = dt.datetime.combine(dt.date.fromisoformat(live_day), dt.time()) - now
    out = []
    for d in D[-REL:]:
        cut = dt.datetime.combine(dt.date.fromisoformat(d), dt.time()) - gap
        s = [it for t, it, _ in by[d] if t in st and ix.local(dt.datetime.strptime(st[t], "%Y-%m-%d %H:%M"), lang) <= cut]
        if len(s) >= MIN_DAY:
            out.append(wmean(s))
    return out


def live(lang, pend, moods, last_close, by=None, D=None):  # 실시간(잠정): 아직 마감 안 된 다음 거래일 몫, 주기 갱신마다 바뀜 (2026-11-02)
    wd = lambda x: dt.date.fromisoformat(x).weekday() < 5
    days_ = sorted({r for *_, r in pend})
    nxt = next((x for x in days_ if wd(x)), None)
    now = ix.local(dt.datetime.now(dt.timezone.utc).replace(tzinfo=None), lang)
    if nxt is None:  # 진행 중인 주말만 있음 → 다음 평일
        nxt = (now.date() + dt.timedelta(1)).isoformat()
        while not wd(nxt):
            nxt = (dt.date.fromisoformat(nxt) + dt.timedelta(1)).isoformat()
    lst = [x for x in pend if x[3] == nxt]
    wk = [x for x in pend if not wd(x[3]) and x[3] < nxt]
    hm = (now.hour, now.minute)
    status = ("장 시작 전" if now.date().isoformat() < nxt or hm < OPEN[lang] else "장중" if hm < ix.CLOSE[lang] else "마감 집계 중")
    out = {"date": nxt, "status": status, "updated": now.strftime("%H:%M"), "titles": len(lst), "mood": None, "mood_rel": None, "mood_label": None,
           "change_pct": None, "price_time": None, "top_negative": [], "top_positive": [],
           "weekend": {"titles": len(wk), "mood": r2(wmean([it for _, it, _, _ in wk]))} if len(wk) >= MIN_DAY else None}
    p = intraday(lang, nxt)
    if p:
        out["price_time"], out["change_pct"] = p[0], r2(100 * (p[1] / last_close - 1), 2)
    if len(lst) >= MIN_DAY:
        m = wmean([it for _, it, _, _ in lst])
        # 평소 = 지난 거래일들의 '같은 시각까지' 몫 (장 시작 전 몫을 하루 전체와 견주면 시황 기사가 없어 덜 부정적으로 나옴 — 2026-11-02)
        usual = same_time(lang, by, D, nxt, now) if by is not None else []
        base = usual if len(usual) >= 20 else list(moods)[-REL:]
        z = rel(base + [m], len(base))
        out |= {"usual_mood": r2(float(np.mean(base))), "usual_sd": r2(float(np.std(base))), "usual_kind": "같은 시각까지" if base is usual else "하루 전체",
                "rank": top(base, m)}
        cl = clusters([(t, it, link) for t, it, link, _ in lst])
        cand = {sg: [x for x in sorted(cl, key=lambda x: -sg * x["contribution"] * rank_w(lang, x["title"])) if sg * x["contribution"] > 0][:6] for sg in (-1, 1)}
        ok = repcheck.check(lang, [x["title"] for v in cand.values() for x in v])
        pick = lambda sg: [{k: x[k] for k in ("title", "score", "count", "link", "check")} for x in repcheck.keep(cand[sg], ok, sg, 2)]
        out |= {"mood": r2(m), "mood_rel": r2(z, 2), "mood_label": mood_label(z), "top_negative": pick(-1), "top_positive": pick(1)}
    return out


def weekend(lang, off, d):  # 그 거래일 바로 앞 주말·휴장 몫 (금요일 장 마감 뒤 ~ 일요일 장 마감 시각 전): 거래일 지수에 넣지 않고 따로 (2026-11-02)
    lst = off.get(d, [])
    if len(lst) < MIN_DAY:
        return None
    hist = [k for k in sorted(off) if k <= d and len(off[k]) >= MIN_DAY]
    moods = [wmean([it for _, it, _, _ in off[k]]) for k in hist]
    z = rel(moods, len(moods) - 1)
    cl = clusters([(t, it, link) for t, it, link, _ in lst])
    cand = {sg: [x for x in sorted(cl, key=lambda x: -sg * x["contribution"] * rank_w(lang, x["title"])) if sg * x["contribution"] > 0][:6] for sg in (-1, 1)}
    ok = repcheck.check(lang, [x["title"] for v in cand.values() for x in v])
    pick = lambda sg: [{k: x[k] for k in ("title", "score", "count", "link", "check")} for x in repcheck.keep(cand[sg], ok, sg, 2)]
    raw = sorted({r for _, _, _, r in lst})
    return {"from": raw[0], "to": raw[-1], "titles": len(lst), "mood": r2(moods[-1]), "mood_rel": r2(z, 2), "mood_label": mood_label(z),
            "rank": top(moods[-REL - 1:-1], moods[-1]),
            "top_negative": pick(-1), "top_positive": pick(1)}


# 화면에 보일 섹터 (2026-11-01 사용자 결정): 늘 보이는 섹터 + 그날의 특종 섹터 하나 (없으면 안 띄움)
FIXED = {"ko": ("반도체",), "en": ("반도체", "빅테크·소프트웨어")}
SCOOP_Z = 2.0  # 특종 = 그날 전체 제목 중 그 섹터 비중이 앞 REL개 시장일 대비 z ≥ 이 값 (건수는 월요일·연휴 뒤에 다 같이 뛰어서 비중으로)


def scoop(lang, per, by, days, d):  # 늘 보이는 섹터 밖에서 비중 z가 가장 큰 섹터 (오늘 제목 MIN_DAY개 이상, z ≥ SCOOP_Z), 없으면 None
    ds = [x for x in days if x <= d and x in by]
    best = None
    for name, p in per.items():
        if name in FIXED[lang] or len(p.get(d, [])) < MIN_DAY:
            continue
        v = [len(p.get(x, [])) / len(by[x]) for x in ds]
        z = rel(v, len(ds) - 1)
        if z is not None and z >= SCOOP_Z and (best is None or z > best[1]):
            best = (name, z, v[-1], float(np.mean(v[-REL - 1:-1])))
    return best
COMP = j.HOME / "data" / "섹터업체주가.tsv"  # 날짜, 업체, 코드, 종가 (한국어 사전 업체)
CAPS = j.HOME / "data" / "섹터업체시총.tsv"


def update_company_prices():  # 사전 업체 종가: 네이버 일봉 중 마감된 봉만 덧붙임 (실패하면 기존 파일)
    rows = {(r[0], r[1]): r for r in (l.split("\t") for l in COMP.read_text(encoding="utf-8").splitlines()[1:])}
    codes = {r[1]: r[2] for r in rows.values()}
    n = 0
    for name, code in codes.items():
        try:
            for dd, v in fetch_ohlc("ko", code).items():
                rows[dd, name] = [dd, name, code, v[3]]; n += 1
        except Exception as e:
            print(f"  업체 시세 {name} 받기 실패 ({type(e).__name__}), 기존 값으로")
    COMP.write_text("날짜\t업체\t코드\t종가\n" + "".join("\t".join(r) + "\n" for r in sorted(rows.values())), encoding="utf-8")
    print(f"  업체 시세: {len(codes)}개 업체, 마감 봉 {n}개 갱신")


def update_fx():  # 원·달러 일별 (환율 기억 fxmemory.py가 씀): 네이버 최근 60일을 data/원달러.tsv에 합침 (실패하면 기존 파일)
    path = j.HOME / "data" / "원달러.tsv"
    rows = dict(l.split("\t") for l in path.read_text(encoding="utf-8").splitlines()[1:])
    try:
        req = urllib.request.Request("https://api.stock.naver.com/marketindex/exchange/FX_USDKRW/prices?page=1&pageSize=60", headers=ix.UA)
        for x in json.loads(urllib.request.urlopen(req, timeout=20).read()):
            rows[x["localTradedAt"]] = x["closePrice"].replace(",", "")
    except Exception as e:
        print(f"  원·달러 받기 실패 ({type(e).__name__}), 기존 값으로")
    path.write_text("날짜\t원달러\n" + "".join(f"{k}\t{rows[k]}\n" for k in sorted(rows)), encoding="utf-8")


def basket_move(sector, home, d, days, index_chg):  # 그 섹터 사전 업체들의 그날 등락 (시가총액 무게)과 코스피 대비
    px = collections.defaultdict(dict)
    for l in COMP.read_text(encoding="utf-8").splitlines()[1:]:
        r = l.split("\t"); px[r[1]][r[0]] = float(r[3])
    cap = {l.split("\t")[0]: float(l.split("\t")[2]) for l in CAPS.read_text(encoding="utf-8").splitlines()[1:]}
    p = days[days.index(d) - 1]
    got = [(100 * (px[n][d] / px[n][p] - 1), cap[n]) for n in home if home[n] == sector and n in cap and d in px[n] and p in px[n]]
    if not got:
        return None
    pct = sum(r * w for r, w in got) / sum(w for _, w in got)
    return {"pct": r2(pct, 2), "excess": r2(pct - index_chg, 2) if index_chg is not None and np.isfinite(index_chg) else None, "companies": len(got)}


def sectors(lang, days, IT, d, index_chg=None):  # 섹터별 분위기 (2026-10-31): 지수용+섹터 검색어 제목, sectors.tsv 사전, 시장 지수와 같은 문턱·z
    D = S.load(lang)
    home = {co: s for _, s, co in D if co}  # 기업 → 그 기업의 섹터 (다른 섹터 카드에 안 나오게)
    by = assign(lang, days, IT, ("daily", "sector"), since=days[-200])
    # 업체 이름이 나온 한국어 제목은 업체 기억 판정(그 업체 주가 관점, 같은 날·전날 묶음을 본 Haiku)으로, 없으면 지금 점수 (docs/실험기록.md 2026-11-01)
    mem = {}
    if lang == "ko":
        memjudge.update(by, days, D, since=days[-200])
        mem = memjudge.load()
    sec = collections.defaultdict(lambda: collections.defaultdict(list))  # 섹터 → 날짜 → [(제목, it, 링크, 기업들, 기억 판정 여부)]
    for day, lst in by.items():
        for t, it, link in lst:
            ss, cos = S.tag(t, D, lang)
            for s in ss:
                mine = [c for c in cos if home[c] == s]
                got = [mem[day, c, t] for c in mine if (day, c, t) in mem]
                judged = bool(mine) and len(got) == len(mine)
                sec[s][day].append((t, dict(it, s=float(np.mean(got))) if judged else it, link, cos, judged))
    out, short, judge_in = [], [], {}
    sc = scoop(lang, sec, by, days, d)
    for name, per in sec.items():
        if name not in FIXED[lang] and (sc is None or name != sc[0]):
            continue
        if len(per.get(d, [])) < MIN_DAY:  # 오늘 제목 모자람
            short.append({"name": name, "titles": len(per.get(d, []))})
            continue
        sd = [x for x in sorted(per) if (len(per[x]) >= MIN_KIND and x < d) or x == d]  # 지난 날은 5제목 이상 (비주류 특종 섹터도 견줄 날이 있게), 오늘은 위에서 MIN_DAY 확인
        mood = [float(np.mean([it["s"] for _, it, _, _, _ in per[x]])) for x in sd]
        z = [rel(mood, i) for i in range(len(sd))]
        lst = per[d]
        co = collections.defaultdict(list)
        for t, it, _, cos, _ in lst:
            for c in cos:
                if home[c] == name:
                    co[c].append(mem.get((d, c, t), it["s"]))
        withco = [x[4] for x in lst if any(home[c] == name for c in x[3])]
        cl = clusters([(t, it, link) for t, it, link, _, _ in lst])
        cand = {sign: [x for x in sorted(cl, key=lambda x: -sign * x["contribution"]) if sign * x["contribution"] > 0][:4] for sign in (-1, 1)}
        ok = repcheck.check(lang, [x["title"] for v in cand.values() for x in v])
        pick = lambda sign: [{k: x[k] for k in ("title", "score", "count", "link", "check")} for x in repcheck.keep(cand[sign], ok, sign, 2)]
        judge_in[name] = {"stories": sectorjudge.stories(cl), "z": z[-1], "mood": mood[-1]}
        out.append({"name": name, "titles": len(lst), "mood": r2(mood[-1]), "mood_rel": r2(z[-1], 2),
                    "company_titles": len(withco), "memory_share": r2(np.mean(withco), 2) if withco else None,
                    "scoop": {"share": r2(sc[2], 4), "share_usual": r2(sc[3], 4), "share_rel": r2(sc[1], 2)} if sc and name == sc[0] else None,
                    "move": basket_move(name, home, d, days, index_chg) if lang == "ko" else None,
                    "spark": [{"date": x, "mood_rel": r2(v, 2)} for x, v in zip(sd[-30:], z[-30:]) if v is not None],
                    "companies": [{"name": c, "titles": len(v), "mood": r2(np.mean(v))} for c, v in sorted(co.items(), key=lambda x: -len(x[1])) if len(v) >= 5],
                    "top_negative": pick(-1), "top_positive": pick(1)})
    # 섹터 동향 라벨·근거: Haiku가 서로 다른 소식만 보고 판정 (주가는 안 줌), 자격 없으면 규칙 라벨 (docs/실험기록.md 2026-11-01)
    jd = sectorjudge.judge(lang, CONF[lang]["name"], d, judge_in)
    for x in out:
        x |= {"label": jd[x["name"]]["label"], "line": jd[x["name"]]["line"], "judge_model": jd[x["name"]]["model"]}
    keep = [x for x in out if x["judge_model"] != "dry-run" or x["mood_rel"] is not None]  # 규칙 라벨은 견줄 지난 날이 있어야
    short += [{"name": x["name"], "titles": x["titles"]} for x in out if x not in keep]
    out = sorted(keep, key=lambda x: FIXED[lang].index(x["name"]) if x["name"] in FIXED[lang] else len(FIXED[lang]))  # 늘 보이는 섹터, 그다음 특종
    return {"date": d, "list": out, "short": sorted(short, key=lambda x: -x["titles"])}


def old_summary(m):  # archive에 같은 시장·같은 날 실제 모델 해설이 있으면 그것 (비용 방지)
    for p in sorted((OUT / "archive").glob("*.json"), reverse=True):
        o = json.loads(p.read_text(encoding="utf-8"))["markets"].get(m["id"], {})
        if o.get("date") == m["date"] and (o.get("summary") or {}).get("model", "dry-run") != "dry-run":
            return o["summary"]


def check(path=OUT / "latest.json"):
    o = json.loads(Path(path).read_text(encoding="utf-8"))
    assert set(o) == {"updated", "markets"} and set(o["markets"]) == {"ko", "en"}
    keys = {"id", "name", "news", "date", "today", "series", "method", "kinds", "top_negative", "top_positive", "topics", "summary"}
    for lang, m in o["markets"].items():
        assert keys <= set(m), keys - set(m)
        assert m["id"] == lang and m["kinds"].keys() == set(CONF[lang]["kinds"])
        t = m["today"]
        assert {"mood", "mood_rel", "mood_label", "shock", "spread", "shock_rel", "shock_label", "cancel", "neg_share", "pos_share", "titles", "change_pct", "close"} <= set(t)
        assert t["mood_label"] in ("매우 부정", "부정", "중립", "긍정", "매우 긍정") and t["shock_label"] in ("평온", "보통", "높음", "매우 높음")
        assert -2 <= t["mood"] <= 2 and 0 <= t["cancel"] <= 1 and 0 <= t["neg_share"] + t["pos_share"] <= 1 and t["titles"] >= MIN_DAY
        assert 1 <= len(m["series"]) <= SERIES and m["series"][-1]["date"] == m["date"]
        assert [x["date"] for x in m["series"]] == sorted(x["date"] for x in m["series"])
        for x in m["series"]:
            assert {"date", "mood", "mood_rel", "shock", "spread", "cancel", "titles", "close", "change_pct", "kinds"} <= set(x) and x["kinds"].keys() == set(CONF[lang]["kinds"])
        me = m["method"]
        assert {"model", "titles_total", "days_total", "since", "same_day_corr", "next_day_corr"} <= set(me) and -1 <= me["same_day_corr"] <= 1 and -1 <= me["next_day_corr"] <= 1
        for tag, lst in (("n", m["top_negative"]), ("p", m["top_positive"])):
            assert len(lst) <= 5 and [x["id"] for x in lst] == [f"{lang}-{tag}{k}" for k in range(1, len(lst) + 1)]
            assert all({"title", "score", "count", "contribution", "topic", "link", "check"} <= set(x) and (x["contribution"] < 0) == (tag == "n") for x in lst)
            assert all(x["check"] != ("호재" if tag == "n" else "악재") for x in lst)  # 방향 반대로 확인된 대표는 없어야
        assert len(m["topics"]) <= 6 and all(x["topic"] != "경제와 무관" for x in m["topics"])
        lv = m["live"]
        assert lv["date"] > m["date"] and lv["status"] in ("장 시작 전", "장중", "마감 집계 중") and (lv["mood"] is None) == (lv["titles"] < MIN_DAY)
        w = m.get("weekend")
        if w is not None:
            assert w["from"] <= w["to"] < m["date"] and w["titles"] >= MIN_DAY and w["mood_label"] in ("매우 부정", "부정", "중립", "긍정", "매우 긍정")
        sc = m["sectors"]
        assert sc["date"] == m["date"]
        for x in sc["list"]:
            assert x["titles"] >= MIN_DAY and x["label"] in sectorjudge.LABELS and len(x["spark"]) <= 30
            assert x["judge_model"] != "dry-run" or (x["mood_rel"] is not None and x["line"] is None)
            assert x["name"] in FIXED[lang] or (x["scoop"] or {}).get("share_rel", 0) >= SCOOP_Z
            assert all(c["titles"] >= 5 for c in x["companies"]) and len(x["top_negative"]) <= 2 and len(x["top_positive"]) <= 2
            assert all(y["check"] != "호재" for y in x["top_negative"]) and all(y["check"] != "악재" for y in x["top_positive"])
        assert sum(x["scoop"] is not None for x in sc["list"]) <= 1
        s = m["summary"]
        if s is not None:
            ids = {x["id"] for x in m["top_negative"] + m["top_positive"]}
            assert {"headline", "bullets", "watch", "sources", "model", "generated"} <= set(s) and 2 <= len(s["bullets"]) <= 4 and set(s["sources"]) <= ids
    print(f"검사 통과: {path}")


def main():
    t0 = time.time()
    if "--skip-score" not in sys.argv:
        update_fx()
        for lang in CONF:
            cache, head = j.HOME / "data" / "cache" / f"지수자료_{lang}.tsv", j.HOME / "headlines" / f"{lang}_daily.tsv"
            sec = j.HOME / "headlines" / f"{lang}_sector.tsv"
            if not cache.exists() or max(head.stat().st_mtime, sec.stat().st_mtime) > cache.stat().st_mtime:
                print(f"{lang}: 제목이 캐시보다 새로움 → 실험/지수자료.py {lang}")
                env = os.environ | {"OMP_NUM_THREADS": str(max(1, os.cpu_count() // 2))}  # CPU 절반
                subprocess.run([sys.executable, str(j.HOME / "실험" / "지수자료.py"), lang, "--new"], check=True, env=env)  # 새 제목만 (전부 다시는 손으로)
    print("시세 갱신")
    update_prices()
    update_company_prices()
    update_fx()
    now = dt.datetime.now(KST).replace(microsecond=0).isoformat()
    out = {"updated": now, "markets": {}}
    for lang in CONF:
        m = market(lang)
        m["summary"] = old_summary(m) or summarize.summarize(m)
        out["markets"][lang] = m
        t = m["today"]
        print(f"{lang} {m['date']}: 분위기 {t['mood']:+.3f} (z {t['mood_rel']}, {t['mood_label']}), 충격 {t['shock_label']}, 제목 {t['titles']}, 등락 {t['change_pct']}%, 해설 {m['summary']['model']}")
    (OUT / "archive").mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, ensure_ascii=False, indent=1)
    (OUT / "latest.json").write_text(text, encoding="utf-8")
    (OUT / "archive" / f"{max(m['date'] for m in out['markets'].values())}.json").write_text(text, encoding="utf-8")
    check()
    print(f"저장 {OUT / 'latest.json'} ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    check() if "--check" in sys.argv else main()
