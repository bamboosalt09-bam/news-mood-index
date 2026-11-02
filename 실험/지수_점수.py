# 점수 파일(제목 → 점수, en_seqexp/seqexp의 show 결과)로 일별 감성 지수를 만들고 시장과 견준다 (2026-10-29).
# index.py의 거래일 맞추기·비교(tests_for)를 그대로 쓴다. 수준만 말한 제목은 빼고(nan), 하루 제목 20개 미만인 날도 뺀다.
# python 실험/지수_점수.py en|ko 점수.json [칸=미국] [시장=spx|kospi|nasdaq|sox] 이름   → data/지수_<이름>.tsv, 큰 날 목록
import sys, json
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
import index as ix
sys.stdout.reconfigure(encoding="utf-8")
lang, src, name = sys.argv[1], sys.argv[2], sys.argv[-1]
col = next((a[2:] for a in sys.argv[3:-1] if a.startswith("칸=")), None)
which = next((a[3:] for a in sys.argv[3:-1] if a.startswith("시장=")), "spx" if lang == "en" else "kospi")
S = json.load(open(src, encoding="utf-8"))
mname, market = ix.MARKETS[which][0], ix.MARKETS[which][1]()
trading = sorted(market)
rows = [l.rstrip("\n").split("\t") for l in open(j.HOME / "headlines" / f"{lang}_daily.tsv", encoding="utf-8")]
rows = [(ix.market_day(r[0], trading, lang), r[2]) for r in rows if len(r) > 2 and r[0][:2] == "20"]
rows = [(d, t) for d, t in rows if d and t in S]
ts = sorted({t for _, t in rows})
lvl = dict(zip(ts, j.level_only(ts)))
sc = {t: np.nan if lvl[t] else (S[t][col] if col else S[t]) for t in ts}
by = {}
for d, t in rows:
    if not np.isnan(sc[t]):
        by.setdefault(d, []).append(t)
days = [d for d in trading if d >= min(d for d, _ in rows)]
idx = np.array([np.mean([sc[t] for t in by[d]]) if len(by.get(d, [])) >= ix.MIN_PER_DAY else np.nan for d in days])
M = np.array([market[d] for d in days])
R = np.r_[np.nan, np.diff(np.log(M))] * 100
out = j.HOME / "data" / f"지수_{name}.tsv"
out.write_text(f"날짜\t{mname}\t등락%\t지수\t제목수\n" + "".join(f"{d}\t{M[i]:.2f}\t{R[i]:.2f}\t{idx[i]:.3f}\t{len(by.get(d, []))}\n" for i, d in enumerate(days)), encoding="utf-8")
ok = ~np.isnan(idx)
print(f"[{name}] 제목 {len(rows):,}개, 지수 있는 거래일 {ok.sum()}일 ({days[np.argmax(ok)]} ~ {days[len(ok) - 1 - np.argmax(ok[::-1])]}), 시장 {mname}")
for nm, f in ix.tests_for(mname, M):
    r, n, p = f(idx)
    print(f"  {nm:45} {r:+.2f} (p={p:.3f}, {n}일)")
z = (idx - np.nanmean(idx)) / np.nanstd(idx)
order = np.argsort(np.where(ok, z, 0))
for title, sel in (("가장 나쁜 날", order[:5]), ("가장 좋은 날", order[::-1][:5])):
    print(f"  {title}:")
    for i in sel:
        tops = sorted(by[days[i]], key=lambda t: sc[t])
        ex = tops[0] if title == "가장 나쁜 날" else tops[-1]
        print(f"    {days[i]} 지수 {idx[i]:+.2f} (z {z[i]:+.1f}), {mname} {R[i]:+.2f}%, 제목 {len(by[days[i]])}개 | 대표: {ex[:90]}")
