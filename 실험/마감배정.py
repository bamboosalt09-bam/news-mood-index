# 마감 시황 제목 배정 시험 (docs/실험기록.md 2026-11-05 사전 등록). python 실험/마감배정.py
import sys, datetime as dt
from pathlib import Path
HOME = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HOME), str(HOME / "web")]
import numpy as np
import build as b, index as ix
sys.stdout.reconfigure(encoding="utf-8")
for lang in ("ko", "en"):
    c = b.CONF[lang]; days, C, chg = b.prices(c["price"]); IT = b.items(lang, c["col"]); tset = set(days)
    now, moved_in, wo = {}, {}, {}   # 지금 / 옮긴 뒤 / 옮긴 제목을 뺀 것
    seen, n = set(), 0
    for l in open(HOME / "headlines" / f"{lang}_daily.tsv", encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) < 3 or p[0][:2] != "20" or p[2] not in IT or p[2] in seen:
            continue
        seen.add(p[2])
        st = p[0][:16]; d = ix.market_day(st, days, lang)
        if not d or ix.is_off(st, days, lang) and not ix.close_day(p[2], st, days, lang):
            continue
        it = IT[p[2]]; cd = ix.close_day(p[2], st, days, lang)
        if not ix.is_off(st, days, lang):
            now.setdefault(d, []).append(it)
        moved_in.setdefault(cd or d, []).append(it)
        if cd:
            n += 1
        else:
            wo.setdefault(d, []).append(it)
    def series(by):
        idx = [i for i, x in enumerate(days) if len(by.get(x, [])) >= b.MIN_DAY]
        x = np.full(len(days), np.nan); x[idx] = [b.wmean(by[days[i]]) for i in idx]
        return x
    print(f"[{lang}] 옮긴 제목 {n}개")
    cut = len(days) // 2 + 100
    for name, by in (("지금", now), ("옮긴 제목 뺌", wo), ("옮긴 뒤", moved_in)):
        x = series(by)
        out = []
        for a, z in ((0, cut), (cut, len(days)), (0, len(days))):
            xx, yy = x[a:z], chg[a:z]
            out.append(f"같은 날 {ix.corr(xx, yy)[0]:+.5f} 다음 날 {ix.partial(xx, ix.shift(yy, -1), yy)[0]:+.5f}")
        print(f"  {name:10} 앞 {out[0]} | 뒤 {out[1]} | 전체 {out[2]}")
