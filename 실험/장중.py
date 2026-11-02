# 장중 맛보기 (2026-10-29): 시각이 찍힌 제목(UTC → 한국 시각)을 B분 구간으로 묶어 평균 감성 ↔ 같은 구간 코스피 등락.
# 누가 먼저인지: 이 구간 뉴스 → 다음 구간 등락 / 이 구간 등락 → 다음 구간 뉴스. 밤사이(전날 15:30 ~ 오늘 09:00) 뉴스 ↔ 시초가 갭도.
# 모델은 판정 창 주력 (한국어 ko_v2-감성, 영어 en_v2-감성 한국 관점). 시세는 data/kospi_minute.tsv (네이버 1분, 최근 며칠만).
# python 실험/장중.py [구간분=30]
import sys, math, datetime as dt
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j, evaluate as e
import index as ix
sys.stdout.reconfigure(encoding="utf-8")
B = int(sys.argv[1]) if len(sys.argv) > 1 else 30
MIN_N = 3  # 구간에 제목이 이보다 적으면 뺌
px = {}
for l in (j.HOME / "data" / "kospi_minute.tsv").read_text(encoding="utf-8").splitlines()[1:]:
    t, v = l.split("\t")
    px[dt.datetime.strptime(t, "%Y-%m-%d %H:%M")] = float(v)
days = sorted({t.date() for t in px})
close = ix.kospi()  # 일별 종가 (첫날 갭용)


def price_at(t):
    """t 이전(같은 날) 마지막 1분 값"""
    ks = [k for k in px if k.date() == t.date() and k <= t]
    return px[max(ks)] if ks else None


edges = []  # (구간 이름, 시작, 끝, 등락%)
for d in days:
    prev = max(x for x in close if x < d.isoformat())
    open_ = dt.datetime.combine(d, dt.time(9, 0))
    edges.append(("밤사이", dt.datetime.combine(d - dt.timedelta(days=1), dt.time(15, 30)), open_, 100 * math.log(price_at(open_) / close[prev])))
    s = open_
    while s < dt.datetime.combine(d, dt.time(15, 30)):
        f = min(s + dt.timedelta(minutes=B), dt.datetime.combine(d, dt.time(15, 30)))
        a, b = price_at(s), price_at(f)
        edges.append((f"{s:%H:%M}", s, f, 100 * math.log(b / a)))
        s = f
p_of = lambda r, n: math.erfc(abs(r * math.sqrt((n - 2) / max(1 - r * r, 1e-12))) / math.sqrt(2)) if n > 2 else float("nan")


def corr(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 4:
        return float("nan"), int(ok.sum()), float("nan")
    r = float(np.corrcoef(x[ok], y[ok])[0, 1])
    return r, int(ok.sum()), p_of(r, ok.sum())


for lang, model in (("ko", "ko_v2-감성"), ("en", "en_v2-감성")):
    seen, items = set(), []
    for l in open(j.HOME / "headlines" / f"{lang}_daily.tsv", encoding="utf-8"):
        p = l.split("\t")
        if len(p) < 3 or p[0][:2] != "20" or p[2] in seen or p[0][11:16] in ("", "07:00", "08:00", "00:00"):
            continue
        seen.add(p[2])
        t = dt.datetime.strptime(p[0][:16], "%Y-%m-%d %H:%M") + dt.timedelta(hours=9)  # UTC → 한국 시각
        if edges[0][1] <= t < edges[-1][2]:
            items.append((t, p[2]))
    sc = dict(zip([x for _, x in items], e.scores(model, [x for _, x in items])))
    S, R, N, kind = [], [], [], []
    for name, a, b, r in edges:
        v = [sc[x] for t, x in items if a <= t < b]
        S.append(np.mean(v) if len(v) >= MIN_N else np.nan); R.append(r); N.append(len(v)); kind.append(name == "밤사이")
    S, R, kind = np.array(S), np.array(R), np.array(kind)
    intra = ~kind
    nxt = lambda v: np.r_[v[1:], np.nan]  # 다음 구간 (같은 날 안에서만)
    same_day_next = np.r_[[edges[i + 1][1].date() == edges[i][1].date() and not kind[i + 1] for i in range(len(edges) - 1)], False]
    print(f"\n[{'한국어' if lang == 'ko' else '영어 (한국 관점)'}] 제목 {len(items):,}개, {len(days)}거래일, 장중 {B}분 구간 {intra.sum()}개 (제목 {MIN_N}개 이상 {int((~np.isnan(S) & intra).sum())}개)")
    for nm, x, y in (("같은 구간: 뉴스 ↔ 코스피 등락", S[intra], R[intra]),
                     ("뉴스 → 다음 구간 등락", S[same_day_next], nxt(R)[same_day_next]),
                     ("등락 → 다음 구간 뉴스", R[same_day_next], nxt(S)[same_day_next]),
                     ("밤사이 뉴스 ↔ 시초가 갭", S[kind], R[kind])):
        r, n, p = corr(x, y)
        print(f"  {nm:28} {r:+.2f} (p={p:.3f}, {n}개)")
    print(f"  구간당 제목 수 중앙값 {int(np.median(np.array(N)[intra]))}, 밤사이 {int(np.median(np.array(N)[kind]))}")
