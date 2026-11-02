# 업체 기억 2차: 같은 날 같은 업체의 실적 제목만, 방향은 쇼크·서프라이즈 낱말로, 반대면 부호만 바꿈 (2026-10-31, docs/실험기록.md 같은 날). 모델은 그대로, 점수 뒤처리.
# python 실험/업체기억.py   → 실험/결과/업체기억.json·.log
import sys, re, json, random, collections, datetime as dt
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
sys.path.insert(0, r"C:\english_folder\somethings\web")
sys.argv = sys.argv[:1]
import judge as j
import index as ix
import build as b
import sectors as S
sys.stdout.reconfigure(encoding="utf-8")
START, END = "2025-11-03", "2026-10-30"
NEG = re.compile(r"어닝\s?쇼크|실적\s?쇼크|실적\s?충격|(기대|예상|추정|컨센서스)\s?치?\s?하회|못\s?미쳐|못\s?미친")
POS = re.compile(r"어닝\s?서프라이즈|깜짝\s?실적|(기대|예상|추정|컨센서스)\s?치?\s?상회|웃돌")
EARN = re.compile(r"영업익|영업이익|순이익|순익|매출|실적|어닝|분기")
DESPITE = re.compile(r"(쇼크|충격|하회|서프라이즈|깜짝\s?실적|상회)[^…·,]{0,6}에도")


def remember(L, D):  # 2차: 같은 날 같은 업체의 실적 제목만, 방향은 낱말로, 반대면 부호만 바꿈 → {제목: 바뀐 점수}
    by_co = collections.defaultdict(list)
    for t, it, _ in L:
        if EARN.search(t):
            for c in S.tag(t, D, "ko")[1]:
                by_co[c].append((t, it))
    out = {}
    for c, xs in by_co.items():
        sg = {(-1 if NEG.search(t) else 1) for t, _ in xs if (NEG.search(t) or POS.search(t)) and not DESPITE.search(t) and not (NEG.search(t) and POS.search(t))}
        if len(sg) != 1:
            continue
        v = sg.pop()
        for t, it in xs:
            if it["s"] * v <= -0.25:
                out[t] = -it["s"]
    return out


if __name__ == "__main__":
    days, _, kchg = b.prices("코스피")
    ev = [i for i, d in enumerate(days) if START <= d <= END]
    E = [days[i] for i in ev]; kos = kchg[ev]
    px = collections.defaultdict(dict)
    for l in open(j.HOME / "data" / "섹터업체주가.tsv", encoding="utf-8").read().splitlines()[1:]:
        p = l.split("\t"); px[p[1]][p[0]] = float(p[3])
    RET = {n: np.array([100 * (d[x] / d[days[i - 1]] - 1) if d.get(x) and d.get(days[i - 1]) else np.nan for i, x in zip(ev, E)]) for n, d in px.items()}
    CAP = {l.split("\t")[0]: float(l.split("\t")[2]) for l in open(j.HOME / "data" / "섹터업체시총.tsv", encoding="utf-8").read().splitlines()[1:]}
    D = S.load("ko"); home = {co: s for _, s, co in D if co}
    IT = b.items("ko", "최종")
    by = b.assign("ko", days, IT, ("daily", "sector"), since="2025-10-20")
    dset = {t for l in open(j.HOME / "headlines" / "ko_daily.tsv", encoding="utf-8") for t in [l.split("\t")[2] if l.count("\t") >= 2 else ""]}
    pool = [x for x in days if "2025-10-24" <= x <= END]
    ADJ, flips = {}, []
    for d in pool:
        a = remember(by.get(d, []), D); ADJ[d] = a
        flips += [(d, t, IT[t]["s"], v) for t, v in a.items()]
    n_all = sum(len(by.get(d, [])) for d in E)
    print(f"제목 {n_all:,} 중 바뀐 제목 {sum(1 for f in flips if f[0] >= START):,}개 ({sum(1 for f in flips if f[0] >= START) / n_all:.2%})")
    sc = lambda d, t, it, mem: ADJ[d].get(t, it["s"]) if mem else it["s"]
    r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
    OUT = {"섹터": {}, "업체": {}, "시장": {}}
    for mem in (False, True):
        k = "기억" if mem else "V0"
        per = collections.defaultdict(lambda: collections.defaultdict(list)); com = collections.defaultdict(lambda: collections.defaultdict(list)); mk = collections.defaultdict(list)
        for d in pool:
            for t, it, _ in by.get(d, []):
                v = sc(d, t, it, mem); ss, cos = S.tag(t, D, "ko")
                for s in ss: per[s][d].append(v)
                for c in cos: com[c][d].append(v)
                if t in dset: mk[d].append(v)
        res = {}
        for s in per:
            mem_ = [n for n in home if home[n] == s and n in CAP and n in RET]
            R = np.vstack([RET[n] for n in mem_]); w = np.array([CAP[n] for n in mem_])
            ex = np.array([np.average(R[np.isfinite(R[:, q]), q], weights=w[np.isfinite(R[:, q])]) if np.isfinite(R[:, q]).any() else np.nan for q in range(len(E))]) - kos
            x1 = np.array([np.mean(per[s][d]) if len(per[s][d]) >= 5 else np.nan for d in E])
            x7 = np.array([np.mean(v) if len(v := [q for xx in pool if (dt.date.fromisoformat(d) - dt.timedelta(6)).isoformat() <= xx <= d for q in per[s][xx]]) >= 10 else np.nan for d in E])
            y7 = np.array([np.nansum(ex[[q for q, xx in enumerate(E) if (dt.date.fromisoformat(d) - dt.timedelta(6)).isoformat() <= xx <= d]]) for d in E])
            res[s] = {"하루": r3(ix.corr(x1, ex)[0]), "7일": r3(ix.corr(x7, y7)[0])}
        OUT["섹터"][k] = res
        cr = {}
        for c in com:
            if c not in RET: continue
            x = np.array([np.mean(com[c][d]) if len(com[c][d]) >= 3 else np.nan for d in E])
            if np.isfinite(x).sum() >= 60:
                cr[c] = r3(ix.corr(x, RET[c] - kos)[0])
        OUT["업체"][k] = cr
        xm = np.array([np.mean(mk[d]) if len(mk[d]) >= b.MIN_DAY else np.nan for d in E])
        OUT["시장"][k] = r3(ix.corr(xm, kos)[0])
    med = lambda k, h: float(np.nanmedian([v[h] for v in OUT["섹터"][k].values() if v[h] is not None]))
    s0, s1 = med("V0", "하루"), med("기억", "하루")
    print(f"\n① 섹터 하루 중앙값 V0 {s0:+.3f} → 기억 {s1:+.3f} | 7일 {med('V0', '7일'):+.3f} → {med('기억', '7일'):+.3f}")
    for s in OUT["섹터"]["V0"]:
        print(f"   {s:14} 하루 {OUT['섹터']['V0'][s]['하루']:+.2f} → {OUT['섹터']['기억'][s]['하루']:+.2f}   7일 {OUT['섹터']['V0'][s]['7일']:+.2f} → {OUT['섹터']['기억'][s]['7일']:+.2f}")
    print("② 업체 초과 상관 V0 → 기억:", ", ".join(f"{c} {OUT['업체']['V0'][c]:+.2f}→{OUT['업체']['기억'][c]:+.2f}" for c in OUT["업체"]["V0"]))
    print(f"③ 시장 지수 ↔ 코스피 같은 날: V0 {OUT['시장']['V0']:+.3f} → 기억 {OUT['시장']['기억']:+.3f}")
    random.seed(0)
    samp = random.sample(F := [f for f in flips if f[0] >= START], min(30, len(F)))
    print("\n④ 바뀐 제목 무작위 30개 (날짜, 원래 → 바뀐 점수):")
    for d, t, a, v in sorted(samp):
        print(f"   {d} {a:+.2f} → {v:+.2f}  {t[:70]}")
    print("\n10/30 기아:", [(round(a, 2), round(v, 2), t[:40]) for d, t, a, v in flips if d == "2026-10-30" and "기아" in t][:6])
    OUT["바뀐제목수"] = len(flips); OUT["표본"] = [list(x) for x in samp]
    OUT["판단"] = {"섹터하루": [s0, s1], "시장": [OUT["시장"]["V0"], OUT["시장"]["기억"]]}
    json.dump(OUT, open(j.HOME / "실험" / "결과" / "업체기억2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
