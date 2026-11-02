# 지금 지수(v2s) ↔ 기존 지수 (2026-10-31, docs/실험기록.md 같은 날 '기존 지수 견주기'). 하루 값은 web/build.py와 같은 방식.
# python 실험/기존지수비교.py
import sys, json
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
sys.path.insert(0, r"C:\english_folder\somethings\web")
import judge as j
import index as ix
import build as b
sys.stdout.reconfigure(encoding="utf-8")
OUT = {}


def ours(lang):
    c = b.CONF[lang]
    days, C, chg = b.prices(c["price"])
    by = b.assign(lang, days, b.items(lang, c["col"]))
    S = np.array([np.mean([it["s"] for _, it, _ in by[d]]) if len(by.get(d, [])) >= b.MIN_DAY else np.nan for d in days])
    return days, S, chg


def ext(path, days, col=1, sep="\t", flip=False):
    d = {}
    for l in open(path, encoding="utf-8").read().splitlines()[1:]:
        p = l.split(sep)
        try:
            d[p[0]] = float(p[col])
        except (ValueError, IndexError):
            pass
    v = np.array([d.get(x, np.nan) for x in days])
    return -v if flip else v


def mean_k(v, k):
    return np.array([np.nanmean(v[i - k + 1:i + 1]) if i >= k - 1 and np.isfinite(v[i - k + 1:i + 1]).sum() >= k // 2 + 1 else np.nan for i in range(len(v))])


diff = lambda v: np.r_[np.nan, np.diff(v)]
R = lambda t: [round(t[0], 3), t[1], round(t[2], 4)]  # 상관, 표본 수, p


def compare(name, S, E, chg):
    dS, dE = diff(S), diff(E)
    nxt = ix.shift(chg, -1)
    o = {"수준_일별": R(ix.corr(S, E)), "수준_5일": R(ix.corr(mean_k(S, 5), mean_k(E, 5))), "수준_20일": R(ix.corr(mean_k(S, 20), mean_k(E, 20))),
         "변화_일별": R(ix.corr(dS, dE)),
         "같은날_우리": R(ix.corr(S, chg)), "같은날_기존변화": R(ix.corr(dE, chg)), "같은날_기존수준": R(ix.corr(E, chg)),
         "같은날_우리_기존변화뺌": R(ix.partial(S, chg, dE)), "같은날_기존변화_우리뺌": R(ix.partial(dE, chg, S)),
         "다음날_우리": R(ix.partial(S, nxt, chg)), "다음날_기존수준": R(ix.partial(E, nxt, chg)), "다음날_기존변화": R(ix.partial(dE, nxt, chg))}
    lv = lambda r: "같은 것" if abs(r) >= 0.5 else "일부 겹침" if abs(r) >= 0.2 else "다른 것"
    o["판단_같이"] = {k: lv(o[k][0]) for k in ("수준_일별", "수준_5일", "수준_20일", "변화_일별")}
    own = o["같은날_우리_기존변화뺌"]
    o["판단_못담는몫"] = "있음" if abs(own[0]) >= 0.1 and own[2] < 0.05 else "없음"
    OUT[name] = o
    print(f"\n== {name}")
    for k, v in o.items():
        print(f"  {k:24} {v}")


H = j.HOME / "data"
days, S, chg = ours("ko")
compare("한국어 ↔ NSI", S, ext(H / "nsi.tsv", days), chg)
days, S, chg = ours("en")
compare("영어 미국 ↔ 공포탐욕", S, ext(H / "fear_greed.tsv", days), chg)
compare("영어 미국 ↔ −VIX", S, ext(H / "외부" / "vix" / "VIXCLS.csv", days, sep=",", flip=True), chg)
json.dump(OUT, open(j.HOME / "실험" / "결과" / "기존지수비교.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
