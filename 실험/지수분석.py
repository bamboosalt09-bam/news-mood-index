# 지수 실험 묶음 (2026-10-31, docs/실험기록.md 같은 날 '지수·모델 점검 묶음'). 자료는 실험/지수자료.py가 만든 data/cache/지수자료_<lang>.tsv.
# 주 시험 P1(표본외 RV 예측)·P2(고르는 기간 추가 설명력) + 서술 9~14·17·18·21~24. 결과는 실험/결과/지수분석.json, 화면에 표.
# python 실험/지수분석.py
import sys, math, json, bisect, collections, datetime as dt
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
import index as ix
sys.stdout.reconfigure(encoding="utf-8")
CUT, END = "2025-12-31", "2026-10-30"  # 고르는 기간 끝, 확인 기간 끝
OUT = {}


# ── 통계 도구 (statsmodels 없음): OLS + Newey-West ──
def pnorm2(t):
    return math.erfc(abs(t) / math.sqrt(2))


def ols(y, X, lag=5):
    X = np.column_stack([np.ones(len(y)), X])
    ok = ~np.isnan(y) & ~np.isnan(X).any(1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtXi = np.linalg.pinv(X.T @ X)
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, lag + 1):
        G = Xe[l:].T @ Xe[:-l]
        S += (1 - l / (lag + 1)) * (G + G.T)
    se = np.sqrt(np.maximum(np.diag(XtXi @ S @ XtXi), 0))
    t = b / np.where(se > 0, se, np.nan)
    r2 = 1 - (e @ e) / (((y - y.mean()) ** 2).sum())
    return {"b": b[1:], "t": t[1:], "p": [pnorm2(x) for x in t[1:]], "n": int(n), "R2": float(r2)}


def z(v):
    return (v - np.nanmean(v)) / np.nanstd(v)


def corr(x, y, lag=5):  # 표준화 회귀 = 상관, NW p (둘 다 있는 날만으로 표준화)
    ok = ~np.isnan(x) & ~np.isnan(y)
    x, y = np.where(ok, x, np.nan), np.where(ok, y, np.nan)
    r = ols(z(y), z(x)[:, None], lag)
    return round(float(r["b"][0]), 3), round(r["p"][0], 4), r["n"]


def resid(v, C):
    X = np.column_stack([np.ones(len(v)), C])
    ok = ~np.isnan(v) & ~np.isnan(X).any(1)
    out = np.full(len(v), np.nan)
    b = np.linalg.lstsq(X[ok], v[ok], rcond=None)[0]
    out[ok] = v[ok] - X[ok] @ b
    return out


def pcorr(x, y, C):
    ok = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(C).any(1)
    xx, yy = np.full(len(x), np.nan), np.full(len(y), np.nan)
    xx[ok], yy[ok] = resid(x[ok], C[ok]), resid(y[ok], C[ok])
    return corr(xx, yy)


# ── 시세 ──
def market(name):
    rows = [l.rstrip("\n").split("\t") for l in open(j.HOME / "data" / "일별_시고저종.tsv", encoding="utf-8")][1:]
    d = {r[1]: tuple(map(float, r[2:6])) for r in rows if r[0] == name}
    days = sorted(d)
    O, H, L, C = (np.array([d[x][k] for x in days]) for k in range(4))
    r = np.r_[np.nan, 100 * np.diff(np.log(C))]
    rv = 100 * np.sqrt(np.log(H / L) ** 2 / (4 * math.log(2)))
    gap = np.r_[np.nan, 100 * np.log(O[1:] / C[:-1])]
    return days, {"C": C, "O": O, "r": r, "absr": np.abs(r), "RV": rv, "gap": gap}


def lagmean(v, k):
    return np.array([np.nanmean(v[max(0, i - k + 1):i + 1]) for i in range(len(v))])


def lead(v, k=1):
    return np.r_[v[k:], [np.nan] * k]


# ── 제목 자료 ──
def items(lang, col):
    rows = [l.rstrip("\n").split("\t") for l in open(j.HOME / "data" / "cache" / f"지수자료_{lang}.tsv", encoding="utf-8")]
    h = rows[0]
    k = {c: i for i, c in enumerate(h)}
    out = {}
    for r in rows[1:]:
        if r[k["수준만"]] == "1":
            continue
        raw = float(r[k[col.replace("최종", "모델")]])
        out[r[-1]] = {"s": float(r[k[col]]), "s_noguard": raw / 2 if r[k["전망표지"]] == "1" else raw, "guarded": r[k[col.replace("최종", "안전장치바뀜")]] == "1",
                      "mkt": r[k["시황"]] == "1", "cause": r[k["원인꼴"]] == "1", "opin": r[k["전망표지"]] == "1", "topic": r[k["주제"]]}
    return out


def kind(it, lang):
    if lang == "en":
        return "시황" if it["mkt"] else "재료"
    if it["mkt"]:
        return "시장해석" if it["cause"] else "시장반응"
    return "경제전망" if it["opin"] else "경제사건"


def assign(lang, days, IT, how="close"):
    by = collections.defaultdict(list)
    timed = total = 0
    for l in open(j.HOME / "headlines" / f"{lang}_daily.tsv", encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) < 3 or p[0][:2] != "20" or p[2] not in IT:
            continue
        total += 1
        dateonly = p[0][11:16] in ("07:00", "08:00")
        timed += not dateonly
        if how == "close":
            d = ix.market_day(p[0][:16], days, lang)
        else:  # 달력: 현지 날짜 0~24시 (그날이 거래일일 때만)
            d = p[0][:10] if dateonly else ix.local(dt.datetime.strptime(p[0][:16], "%Y-%m-%d %H:%M"), lang).date().isoformat()
            d = d if d in DAYSET[lang] else None
        if d:
            by[d].append((p[2], IT[p[2]]))
    return by, timed / max(total, 1)


def feats(sc):
    s = np.asarray(sc, float)
    a = np.abs(s).sum()
    return {"S": s.mean(), "I": np.abs(s).mean(), "D": s.std(), "C": 1 - abs(s.sum()) / a if a > 0 else 0.0, "N": math.log(len(s)), "NEG": np.minimum(s, 0).mean()}


def daily(days, by, f=lambda it: True, key="s", mn=20):
    out = {k: np.full(len(days), np.nan) for k in ("S", "I", "D", "C", "N", "NEG")}
    for i, d in enumerate(days):
        sc = [it[key] for _, it in by.get(d, []) if f(it)]
        if len(sc) >= mn:
            for k, v in feats(sc).items():
                out[k][i] = v
    return out


def zprior(v, k=20):
    o = np.full(len(v), np.nan)
    for i in range(k, len(v)):
        w = v[i - k:i]; w = w[~np.isnan(w)]
        if len(w) >= 10 and w.std() > 0 and not np.isnan(v[i]):
            o[i] = (v[i] - w.mean()) / w.std()
    return o


# ── 시장마다 ──
CONF = {"ko": ("코스피", "최종"), "en": ("S&P500", "최종_미국")}
DAYSET = {}
for lang, (mname, col) in CONF.items():
    days, M = market(mname)
    DAYSET[lang] = set(days)
    IT = items(lang, col)
    by, timed = assign(lang, days, IT)
    F = daily(days, by)
    F["U"] = -zprior(F["NEG"])  # 평소보다 더 부정적이면 +
    sub = {k: daily(days, by, lambda it, k=k: kind(it, lang) == k, mn=5)["S"] for k in (("시황", "재료") if lang == "en" else ("시장반응", "시장해석", "경제전망", "경제사건"))}
    EV = sub["재료" if lang == "en" else "경제사건"]
    dd = np.array(days)
    ins, oos = (dd >= "2023-01-01") & (dd <= CUT), (dd > CUT) & (dd <= END)
    y1 = lead(M["RV"]); A = np.column_stack([M["RV"], lagmean(M["RV"], 5), lagmean(M["RV"], 22), M["absr"]])
    R = OUT.setdefault(lang, {"시장": mname, "실제시각비율": round(timed, 3), "지수일수": int((~np.isnan(F["S"])).sum())})
    print(f"\n===== {lang} ↔ {mname}: 지수 있는 거래일 {R['지수일수']}일, 실제 발행 시각이 있는 제목 {timed:.1%}")

    # 9·10 갈래
    cnt = collections.Counter(kind(it, lang) for it in IT.values())
    R["갈래"] = {k: {"제목수": cnt[k], "같은날_r": corr(v, M["r"]), "같은날_RV": corr(v, M["RV"]), "지수일수": int((~np.isnan(v)).sum())} for k, v in sub.items()}
    base = "시황" if lang == "en" else "시장반응"
    R["사건_부분상관_r"] = pcorr(EV, M["r"], sub[base][:, None])
    print("9·10 갈래 (제목 수, 같은 날 r 상관·p, 같은 날 RV 상관):")
    for k, v in R["갈래"].items():
        print(f"  {k:6} {v['제목수']:>7,}  r {v['같은날_r'][0]:+.3f} (p {v['같은날_r'][1]:.3f}, {v['같은날_r'][2]}일)  RV {v['같은날_RV'][0]:+.3f}")
    print(f"  {'재료' if lang == 'en' else '경제사건'} 지수의 r 부분상관 ({base} 지수 뺀 뒤): {R['사건_부분상관_r']}")
    rng = np.random.default_rng(0)
    exs = collections.defaultdict(list)
    for t, it in IT.items():
        exs[kind(it, lang)].append(t)
    R["갈래_예시"] = {k: [v[i] for i in rng.permutation(len(v))[:4]] for k, v in exs.items()}

    # 11 하루 특징
    R["특징"] = {k: {"r": corr(F[k], M["r"]), "absr": corr(F[k], M["absr"]), "RV": corr(F[k], M["RV"]), "다음날RV_부분": pcorr(F[k], y1, A)} for k in ("S", "I", "D", "C", "N", "U")}
    print("11 하루 특징 (같은 날 r / |r| / RV 상관, 다음 날 RV 부분상관(HAR·|r| 뺀 뒤)·p):")
    for k, v in R["특징"].items():
        print(f"  {k}: r {v['r'][0]:+.3f}  |r| {v['absr'][0]:+.3f}  RV {v['RV'][0]:+.3f}  다음날RV {v['다음날RV_부분'][0]:+.3f} (p {v['다음날RV_부분'][1]:.3f})")

    # P2 고르는 기간 안 추가 설명력
    X = np.column_stack([A, F["S"], F["D"], F["U"]])
    for tgt, yy in (("RV", y1), ("absr", lead(M["absr"]))):
        o = ols(np.where(ins, yy, np.nan), X)
        R[f"P2_{tgt}"] = {nm: {"계수": round(float(o["b"][4 + i]), 4), "t": round(float(o["t"][4 + i]), 2), "p": round(o["p"][4 + i], 4)} for i, nm in enumerate(("S", "D", "U"))} | {"n": o["n"], "R2": round(o["R2"], 3)}
    print(f"P2 (2023~2025, 다음 날 RV ~ HAR + |r| + S + D + U, NW): {R['P2_RV']}")
    print(f"   (참고: 다음 날 |r|) {R['P2_absr']}")

    # P1 표본외 2026
    def oos_pred(Xm, start="2023-01-01"):  # i날 예측에는 y(= 다음 날 RV)가 i날까지 알려진 줄(i−1까지)만 쓴다
        pr = np.full(len(days), np.nan)
        Xt = np.column_stack([np.ones(len(days)), Xm])
        base_ok = (dd >= start) & ~np.isnan(y1) & ~np.isnan(Xt).any(1)
        for i in np.where(oos)[0]:
            ok = base_ok.copy(); ok[i:] = False
            if ok.sum() >= 100 and not np.isnan(Xt[i]).any():
                pr[i] = Xt[i] @ np.linalg.lstsq(Xt[ok], y1[ok], rcond=None)[0]
        return pr
    models = {"A": A, "B (+S)": np.column_stack([A, F["S"]]), "C (+S·D)": np.column_stack([A, F["S"], F["D"]]),
              "D (+S·D·U)": np.column_stack([A, F["S"], F["D"], F["U"]]), "E (+사건 지수)": np.column_stack([A, EV])}
    P = {k: oos_pred(v) for k, v in models.items()}
    ok = oos & ~np.isnan(y1) & np.all([~np.isnan(v) for v in P.values()], 0)
    eA = (y1 - P["A"])[ok]
    R["P1"] = {"일수": int(ok.sum()), "A_MSE": round(float((eA ** 2).mean()), 4)}
    for k in list(models)[1:]:
        e = (y1 - P[k])[ok]
        d = eA ** 2 - e ** 2  # DM: d의 평균이 0보다 큰가 (NW 표준오차)
        n = len(d); dm = d - d.mean(); s = (dm @ dm) / n
        for l in range(1, 6):
            s += 2 * (1 - l / 6) * (dm[l:] @ dm[:-l]) / n
        t = d.mean() / math.sqrt(s / n) if s > 0 else 0.0
        R["P1"][k] = {"MSE": round(float((e ** 2).mean()), 4), "A대비%": round(100 * (1 - (e ** 2).mean() / (eA ** 2).mean()), 2), "DM_t": round(t, 2), "p_한쪽": round(0.5 * math.erfc(t / math.sqrt(2)), 4)}
    print(f"P1 (2026 표본외 다음 날 RV, {R['P1']['일수']}일): A MSE {R['P1']['A_MSE']}")
    for k in list(models)[1:]:
        print(f"  {k}: {R['P1'][k]}")
    # 사후 점검 (결과 본 뒤 추가, 결론에 안 씀): 2026년 수집량·지수 수준이 바뀌어서, 앞 60거래일 평균을 뺀 특징으로 같은 비교
    dm60 = lambda v: v - np.array([np.nanmean(v[max(0, i - 60):i]) if i >= 20 else np.nan for i in range(len(v))])
    Sd, Dd, EVd = dm60(F["S"]), dm60(F["D"]), dm60(EV)
    Ph = {"A": P["A"], "B (+S)": oos_pred(np.column_stack([A, Sd])), "C (+S·D)": oos_pred(np.column_stack([A, Sd, Dd])),
          "D (+S·D·U)": oos_pred(np.column_stack([A, Sd, Dd, F["U"]])), "E (+사건 지수)": oos_pred(np.column_stack([A, EVd]))}
    okh = oos & ~np.isnan(y1) & np.all([~np.isnan(v) for v in Ph.values()], 0)
    eAh = (y1 - Ph["A"])[okh]
    R["P1_사후_60일평균뺌"] = {"일수": int(okh.sum())}
    for k in list(Ph)[1:]:
        e_ = (y1 - Ph[k])[okh]; d = eAh ** 2 - e_ ** 2; n = len(d); dm_ = d - d.mean(); s = (dm_ @ dm_) / n
        for l in range(1, 6):
            s += 2 * (1 - l / 6) * (dm_[l:] @ dm_[:-l]) / n
        t = d.mean() / math.sqrt(s / n) if s > 0 else 0.0
        R["P1_사후_60일평균뺌"][k] = {"A대비%": round(100 * (1 - (e_ ** 2).mean() / (eAh ** 2).mean()), 2), "DM_t": round(t, 2), "p_한쪽": round(0.5 * math.erfc(t / math.sqrt(2)), 4)}
    print(f"  (사후, 앞 60일 평균 뺀 특징) {R['P1_사후_60일평균뺌']}")

    # 12 주제별 보도량·감성
    tc = collections.Counter(it["topic"] for it in IT.values() if it["topic"] != "경제와 무관")
    R["주제"] = {}
    for tp, _ in tc.most_common(5):
        n_ = np.array([sum(it["topic"] == tp for _, it in by.get(d, [])) for d in days], float)
        tot = np.array([len(by.get(d, [])) for d in days], float)
        mu = np.array([np.mean([it["s"] for _, it in by.get(d, []) if it["topic"] == tp]) if n_[i] >= 3 else np.nan for i, d in enumerate(days)])
        sm = np.array([sum(it["s"] for _, it in by.get(d, []) if it["topic"] == tp) / tot[i] if tot[i] >= 20 else np.nan for i, d in enumerate(days)])
        zc = zprior(np.where(tot >= 20, n_ / np.maximum(tot, 1), np.nan))
        R["주제"][tp] = {"제목수": tc[tp], "평균_r": corr(mu, M["r"]), "합_r": corr(sm, M["r"]), "보도량z_r": corr(zc, M["r"]),
                       "보도량z_다음날RV": pcorr(zc, y1, A), "합_다음날RV": pcorr(sm, y1, A)}
    print("12 주제 (같은 날 r: 주제 평균·주제 몫 합·보도량 z / 다음 날 RV 부분: 보도량 z·합):")
    for tp, v in R["주제"].items():
        print(f"  {tp:10} {v['제목수']:>6,}  평균 {v['평균_r'][0]:+.3f}  합 {v['합_r'][0]:+.3f}  보도량z {v['보도량z_r'][0]:+.3f} | 다음날RV: 보도량z {v['보도량z_다음날RV'][0]:+.3f} (p {v['보도량z_다음날RV'][1]:.3f})  합 {v['합_다음날RV'][0]:+.3f} (p {v['합_다음날RV'][1]:.3f})")

    # 13 중복 묶기
    tri = lambda t: {t[i:i + 3] for i in range(len(t) - 2)} or {t}
    dedup, conc = np.full(len(days), np.nan), np.full(len(days), np.nan)
    for i, d in enumerate(days):
        lst = by.get(d, [])
        if len(lst) < 20:
            continue
        reps, sizes, sums, inv = [], [], [], collections.defaultdict(list)
        for t, it in lst:
            g = tri("".join(j.clean(t).lower().split()))
            cand = collections.Counter(c for x in g for c in inv.get(x, ()))
            best = next((c for c, n in cand.most_common(5) if n / (len(g) + len(reps[c]) - n) >= 0.5), None)
            if best is None:
                best = len(reps); reps.append(g); sizes.append(0); sums.append(0.0)
                for x in g:
                    inv[x].append(best)
            sizes[best] += 1; sums[best] += it["s"]
        dedup[i] = np.mean([s / n for s, n in zip(sums, sizes)])
        conc[i] = sum(sorted(sizes)[-5:]) / len(lst)
    R["중복"] = {"원래_r": corr(F["S"], M["r"]), "중복제거_r": corr(dedup, M["r"]), "집중도_RV": corr(conc, M["RV"]), "집중도_다음날RV": pcorr(conc, y1, A),
                 "집중도_평균": round(float(np.nanmean(conc)), 3)}
    print(f"13 중복: {R['중복']}")

    # 14 주제 비중 효과 vs 주제 안 감성 효과 (앞 20일 기준)
    tps = sorted({it["topic"] for it in IT.values()})
    share = np.zeros((len(days), len(tps))); mean = np.full((len(days), len(tps)), np.nan)
    for i, d in enumerate(days):
        lst = by.get(d, [])
        if len(lst) < 20:
            share[i] = np.nan; continue
        for k_, tp in enumerate(tps):
            sc = [it["s"] for _, it in lst if it["topic"] == tp]
            share[i, k_] = len(sc) / len(lst)
            if sc:
                mean[i, k_] = np.mean(sc)
    comp, within, dS = (np.full(len(days), np.nan) for _ in range(3))
    for i in range(20, len(days)):
        if np.isnan(share[i]).any():
            continue
        ps, ms = np.nanmean(share[i - 20:i], 0), np.nan_to_num(np.nanmean(mean[i - 20:i], 0))
        base_ = (ps * ms).sum()
        mi = np.where(np.isnan(mean[i]), ms, mean[i])
        comp[i], within[i], dS[i] = ((share[i] - ps) * ms).sum(), (ps * (mi - ms)).sum(), F["S"][i] - base_
    ok = ~np.isnan(dS) & ~np.isnan(comp)
    R["주제분해"] = {"비중효과_분산몫": round(float(np.var(comp[ok]) / np.var(dS[ok])), 3), "주제안효과_분산몫": round(float(np.var(within[ok]) / np.var(dS[ok])), 3),
                   "비중효과_r": corr(comp, M["r"]), "주제안효과_r": corr(within, M["r"]), "일수": int(ok.sum())}
    print(f"14 주제 분해: {R['주제분해']}")

    # 17 국면별
    rv20 = lagmean(M["RV"], 20)
    q1, q2 = np.nanpercentile(rv20[ins], [100 / 3, 200 / 3])
    hi120 = np.array([M["C"][max(0, i - 119):i + 1].max() for i in range(len(days))])
    dd_state = M["C"] / hi120 - 1 <= -0.10
    R["국면"] = {}
    for nm, sel in (("변동성 낮음", rv20 <= q1), ("변동성 중간", (rv20 > q1) & (rv20 <= q2)), ("변동성 높음", rv20 > q2), ("고점 −10% 아래", dd_state), ("그 밖", ~dd_state)):
        o = ols(np.where(sel, y1, np.nan), np.column_stack([A, F["S"], F["U"]]))
        R["국면"][nm] = {"n": o["n"], "S_계수": round(float(o["b"][4]), 3), "S_p": round(o["p"][4], 4), "U_계수": round(float(o["b"][5]), 3), "U_p": round(o["p"][5], 4)}
    print(f"17 국면 (다음 날 RV ~ HAR + |r| + S + U): {R['국면']}")

    # 18 극단 구간
    R["극단"] = {}
    for nm, v in (("S", F["S"]), ("사건", EV)):
        cuts = np.nanpercentile(v[ins], [5, 10, 20, 90])
        bins = [("하위 5%", v <= cuts[0]), ("5~10%", (v > cuts[0]) & (v <= cuts[1])), ("10~20%", (v > cuts[1]) & (v <= cuts[2])),
                ("20~90%", (v > cuts[2]) & (v <= cuts[3])), ("상위 10%", v > cuts[3])]
        r1, a1 = lead(M["r"]), lead(M["absr"])
        R["극단"][nm] = {b: {"n": int((s & ~np.isnan(y1)).sum()), "다음날r": round(float(np.nanmean(r1[s])), 3), "다음날|r|": round(float(np.nanmean(a1[s])), 3),
                            "다음날RV": round(float(np.nanmean(y1[s])), 3), "|r|>2%": round(float(np.nanmean(a1[s] > 2)), 3)} for b, s in bins}
    print("18 극단 구간 (다음 날):")
    for nm, v in R["극단"].items():
        print(f"  [{nm}] " + " | ".join(f"{b}: n {x['n']} r {x['다음날r']:+.2f} |r| {x['다음날|r|']:.2f} RV {x['다음날RV']:.2f} 2%↑ {x['|r|>2%']:.0%}" for b, x in v.items()))

    # 21 장 마감 기준 vs 달력 기준
    byc, _ = assign(lang, days, IT, "calendar")
    Fc = daily(days, byc)
    R["달력"] = {"마감_같은날": corr(F["S"], M["r"]), "달력_같은날": corr(Fc["S"], M["r"]), "마감_다음날(r뺀)": pcorr(F["S"], lead(M["r"]), M["r"][:, None]),
                 "달력_다음날(r뺀)": pcorr(Fc["S"], lead(M["r"]), M["r"][:, None]), "두지수상관": corr(F["S"], Fc["S"])}
    print(f"21 마감 대 달력: {R['달력']}")

    # 3 안전장치 끄면 지수 진폭
    Fg = daily(days, by, key="s_noguard")
    R["안전장치_지수"] = {"켬_표준편차": round(float(np.nanstd(F["S"])), 4), "끔_표준편차": round(float(np.nanstd(Fg["S"])), 4),
                       "켬_r": corr(F["S"], M["r"]), "끔_r": corr(Fg["S"], M["r"]), "바뀐제목비율": round(float(np.mean([it["guarded"] for it in IT.values()])), 4)}
    print(f"3 안전장치와 지수: {R['안전장치_지수']}")
    if lang == "en":
        SAVE_EN = (days, F, M, A, y1, dd, oos)
    else:
        SAVE_KO = (days, F, M)

# 22 미국 뉴스 → 한국 장 시작 (영어 미국·한국 관점 지수, S&P 그날 등락 뺀 뒤)
kdays, KM = market("코스피")
udays, F, M, *_ = SAVE_EN
ITk = items("en", "최종_한국")
byk, _ = assign("en", udays, ITk)
Fk = daily(udays, byk)
gap = np.full(len(udays), np.nan); kr_next = np.full(len(udays), np.nan)
for i, d in enumerate(udays):
    k = bisect.bisect_right(kdays, d)
    if k < len(kdays) and (dt.date.fromisoformat(kdays[k]) - dt.date.fromisoformat(d)).days <= 4:
        gap[i], kr_next[i] = KM["gap"][k], KM["r"][k]
OUT["미국→한국"] = {"갭_미국관점": corr(F["S"], gap), "갭_한국관점": corr(Fk["S"], gap), "갭_S&P": corr(M["r"], gap),
                 "갭_미국관점(S&P뺀)": pcorr(F["S"], gap, M["r"][:, None]), "갭_한국관점(S&P뺀)": pcorr(Fk["S"], gap, M["r"][:, None]),
                 "한국하루_한국관점(S&P뺀)": pcorr(Fk["S"], kr_next, M["r"][:, None])}
print(f"\n22 미국 뉴스 → 다음 한국 장: {OUT['미국→한국']}")

# 23 공포탐욕
fg = {l.split("\t")[0]: float(l.split("\t")[1]) for l in open(j.HOME / "data" / "fear_greed.tsv", encoding="utf-8").read().splitlines()[1:]}
days, F, M, A, y1, dd, oos = SAVE_EN
FG = np.array([fg.get(d, np.nan) for d in days]); dFG = np.r_[np.nan, np.diff(FG)]
sel = dd >= "2025-01-01"
R2 = lambda X: ols(np.where(sel, M["r"], np.nan), X)["R2"]
OUT["공포탐욕"] = {"같은날R2_뉴스": round(R2(F["S"][:, None]), 3), "같은날R2_공포탐욕변화": round(R2(dFG[:, None]), 3), "같은날R2_둘다": round(R2(np.column_stack([F["S"], dFG])), 3)}
pred = {}
for nm, Xm in (("A", A), ("A+공포탐욕", np.column_stack([A, FG])), ("A+S", np.column_stack([A, F["S"]])), ("A+둘다", np.column_stack([A, FG, F["S"]]))):
    pr = np.full(len(days), np.nan)
    Xt = np.column_stack([np.ones(len(days)), Xm])
    for i in np.where(oos)[0]:
        ok = sel & ~np.isnan(y1) & ~np.isnan(Xt).any(1); ok[i:] = False
        if ok.sum() >= 100 and not np.isnan(Xt[i]).any():
            pr[i] = Xt[i] @ np.linalg.lstsq(Xt[ok], y1[ok], rcond=None)[0]
    pred[nm] = pr
ok = oos & ~np.isnan(y1) & np.all([~np.isnan(v) for v in pred.values()], 0)
OUT["공포탐욕"]["표본외_MSE"] = {k: round(float(((y1 - v)[ok] ** 2).mean()), 4) for k, v in pred.items()} | {"일수": int(ok.sum())}
print(f"23 공포탐욕 (2025~): {OUT['공포탐욕']}")

# 24 영어 장기 되돌림: k일 평균 → 앞으로 k일 등락
OUT["장기"] = {}
for k in (40, 60, 80):
    mS = np.array([np.nanmean(F["S"][max(0, i - k + 1):i + 1]) if (~np.isnan(F["S"][max(0, i - k + 1):i + 1])).sum() >= k * 0.8 else np.nan for i in range(len(days))])
    fwd = np.r_[100 * np.log(M["C"][k:] / M["C"][:-k]), [np.nan] * k]
    past = np.r_[[np.nan] * k, 100 * np.log(M["C"][k:] / M["C"][:-k])]
    row = {"전체(겹침,NW)": pcorr(mS, fwd, past[:, None])}
    idx = np.arange(np.argmax(~np.isnan(mS)), len(days), k)
    v = idx[~np.isnan(mS[idx]) & ~np.isnan(fwd[idx])]
    row["겹치지않음"] = (round(float(np.corrcoef(mS[v], fwd[v])[0, 1]), 3), len(v)) if len(v) > 3 else None
    for y in ("2023", "2024", "2025", "2026"):
        s_ = np.array([d[:4] == y for d in days])
        row[y] = pcorr(np.where(s_, mS, np.nan), fwd, past[:, None]) if (s_ & ~np.isnan(mS) & ~np.isnan(fwd)).sum() > 30 else None
    OUT["장기"][f"{k}일"] = row
print(f"24 영어 장기 (k일 평균 → 앞으로 k일 등락, 지난 k일 등락 뺀 뒤): {json.dumps(OUT['장기'], ensure_ascii=False)}")

json.dump(OUT, open(j.HOME / "실험" / "결과" / "지수분석.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print("\n저장 실험/결과/지수분석.json")
