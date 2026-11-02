# 하루 시장 움직임의 '이유' 요약 시제품 (2026-10-29). 그날 하루(한국어: 한국 시각, 영어: 미국 동부 시각 0~24시)에 발행된 제목만 쓴다.
# 1) 시황(시장 움직임을 전하는 제목)과 재료(원인 쪽)를 나누고 2) 거의 같은 제목을 묶어 3) 묶음마다 그날 지수에 대한 기여(점수 합 / 그날 제목 수)로 줄 세우고
# 4) 주제별 기여를 앞 7일 평균과 견주고 5) 낱말별 기여(그 낱말이 든 제목 점수 합 / 그날 제목 수)를 앞 7일과 견줘 원인 낱말을 뽑는다 (시황 제목 속 원인도 잡힘). 모델은 판정 창 주력 (영어는 관점을 고름).
# python 실험/이유.py 2026-10-22 ko|en [관점=미국]
import sys, re, datetime as dt, collections
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
import index as ix
sys.stdout.reconfigure(encoding="utf-8")
DAY, LANG = sys.argv[1], sys.argv[2]
HEAD = next((a[3:] for a in sys.argv[3:] if a.startswith("관점=")), "미국")
MODEL = {"ko": "ko_v2-감성", "en": "en_v2-감성"}[LANG]
TOPIC = {"ko": "ko-주제-미세", "en": "en-주제"}[LANG]
shift = dt.timedelta(hours=9) if LANG == "ko" else -dt.timedelta(hours=4)  # UTC → 한국 / 미국 동부(서머타임)
day0 = dt.date.fromisoformat(DAY)
WIN = [day0 - dt.timedelta(days=k) for k in range(0, 8)]  # 그날 + 앞 7일(기준선용)

seen, items = set(), []
for l in open(j.HOME / "headlines" / f"{LANG}_daily.tsv", encoding="utf-8"):
    p = l.split("\t")
    if len(p) < 3 or p[0][:2] != "20" or p[0][11:16] in ("", "07:00", "08:00", "00:00"):
        continue
    t = dt.datetime.strptime(p[0][:16], "%Y-%m-%d %H:%M") + shift
    key = j.clean(p[2]).lower()
    if t.date() in WIN and key not in seen:
        seen.add(key); items.append((t.date(), p[2]))
ts = [x for _, x in items]
if LANG == "ko":
    P, m = j.predict(MODEL, ts)
else:
    P, m = j.predict(MODEL, ts, m_head := j.load(MODEL)[2]["heads"].index(HEAD))
sc = np.where(j.level_only(ts), np.nan, j.soften(ts, j.guard(ts, j.score_of(P, m), m.get("guard"), m.get("guard_lexicon"))[0])[0])
Pt, mt = j.predict(TOPIC, ts)
topic = [j.OLD_TOPIC.get(mt["labels"][k], mt["labels"][k]) for k in Pt.argmax(1)]
is_mkt = [bool(ix.MARKET[LANG].search(t)) for t in ts]
rows = [(d, t, s, tp, mk) for (d, t), s, tp, mk in zip(items, sc, topic, is_mkt) if not np.isnan(s)]
today = [r for r in rows if r[0] == day0]
N = len(today)
idx = np.mean([r[2] for r in today])

# 거의 같은 제목 묶기 (3글자 조각 자카드 ≥ 0.35, 앞에서부터 욕심껏)
tri = lambda t: {t[i:i + 3] for i in range(len(t) - 2)} if len(t := "".join(j.clean(t).lower().split())) > 2 else {t}
clusters = []  # [대표 조각, [행…]]
for r in sorted(today, key=lambda r: -abs(r[2])):
    a = tri(r[1])
    for c in clusters:
        if len(a & c[0]) / len(a | c[0]) >= 0.35:
            c[1].append(r); break
    else:
        clusters.append([a, [r]])
cl = [(sum(x[2] for x in c[1]) / N, c[1]) for c in clusters]

mkt = [r for r in today if r[4]]
print(f"\n===== {DAY} {'한국어 (한국 시각)' if LANG == 'ko' else '영어 ' + HEAD + ' 관점 (미국 동부 시각)'}: 제목 {N}개, 그날 뉴스 지수 {idx:+.3f}")
print(f"  시황 제목 {len(mkt)}개 (평균 {np.mean([r[2] for r in mkt]):+.2f}) / 재료 제목 {N - len(mkt)}개 (평균 {np.mean([r[2] for r in today if not r[4]]):+.2f})")
print("\n  [그냥 모으기] 점수 절댓값 상위 6개")
for r in sorted(today, key=lambda r: -abs(r[2]))[:6]:
    print(f"    {r[2]:+.2f} {r[1][:80]}")
for nm, sign in (("끌어내린 재료", -1), ("받쳐 준 재료", 1)):
    print(f"\n  [다듬기] {nm} (같은 소식 묶음, 그날 지수 기여 순)")
    pick = sorted([c for c in cl if not c[1][0][4] and np.sign(c[0]) == sign], key=lambda c: -abs(c[0]))[:5 if sign < 0 else 3]
    for contrib, rs in pick:
        rep = max(rs, key=lambda r: abs(r[2]))
        print(f"    기여 {contrib:+.3f} · {len(rs)}건 · 평균 {np.mean([r[2] for r in rs]):+.2f} · [{rep[3]}] {rep[1][:70]}")
print("\n  [주제별] 재료 제목의 그날 기여 ↔ 앞 7일 평균 기여 (차이가 큰 순)")
base = collections.defaultdict(list)
for d in WIN[1:]:
    rd = [r for r in rows if r[0] == d and not r[4]]
    nd = len([r for r in rows if r[0] == d])
    if nd < 20:
        continue
    tot = collections.Counter()
    for r in rd:
        tot[r[3]] += r[2] / nd
    for tp in set(topic):
        base[tp].append(tot.get(tp, 0.0))
tot = collections.Counter(); cnt = collections.Counter()
for r in today:
    if not r[4]:
        tot[r[3]] += r[2] / N; cnt[r[3]] += 1
diff = sorted(((tot[tp] - np.mean(base[tp] or [0]), tp) for tp in tot), key=lambda x: -abs(x[0]))[:6]
for dlt, tp in diff:
    print(f"    {tp:12} 그날 {tot[tp]:+.3f} ({cnt[tp]}건) · 평소 {np.mean(base[tp] or [0]):+.3f} · 차이 {dlt:+.3f}")

# 낱말별: 오늘 유독 많이·세게 나온 낱말 (시황·재료 모두). 한국어는 명사(kiwi), 영어는 낱말·두 낱말.
STOP = set("the a an of to in on for and as at by with from is are be after amid over its it this that s us u says said new stock stocks market markets shares today what why how will could may more than up down".split())
if LANG == "ko":
    from kiwipiepy import Kiwi
    kw = Kiwi()
    def terms(t):
        ws = [x.form for x in kw.tokenize(j.clean(t)) if x.tag in ("NNG", "NNP", "SL") and len(x.form) > 1]
        return set(ws) | {a + " " + b for a, b in zip(ws, ws[1:])}
else:
    def terms(t):
        ws = [w for w in re.findall(r"[a-z][a-z'&]+", j.clean(t).lower()) if w not in STOP and len(w) > 2]
        return set(ws) | {a + " " + b for a, b in zip(ws, ws[1:])}
TM = {r[1]: terms(r[1]) for r in rows}
def contrib(rs):
    n = len(rs); c = collections.Counter(); k = collections.Counter()
    for r in rs:
        for w in TM[r[1]]:
            c[w] += r[2] / n; k[w] += 1
    return c, k
ct, kt = contrib(today)
bases = [contrib([r for r in rows if r[0] == d])[0] for d in WIN[1:] if len([r for r in rows if r[0] == d]) >= 20]
LEXD = {w.strip().lower() for k in ("오름", "내림", "급오름", "급내림", "약오름", "약내림") for w in j.load(MODEL)[2]["lexicon"].get(k, [])}
DESC = set("마감 장중 지수 개장 출발 마감시황 시황 포인트 거래 종가 장초반 오후 오전 하루 반등 반락 급락 급등 하락 상승 강세 약세 close closes closed index points session day trading".split())
desc = lambda w: any(ix.MARKET[LANG].search(x) or x in LEXD or x in DESC for x in w.split())  # 시장 이름·방향 말 = 움직임 자체 (원인 아님)
cand = [(ct[w] - np.mean([b.get(w, 0.0) for b in bases]), w) for w in ct if kt[w] >= 5 and not desc(w)]
for nm, sign in (("끌어내린 낱말", -1), ("받쳐 준 낱말", 1)):
    print(f"\n  [낱말] {nm} (그날 기여 − 평소, {len(bases)}일 기준)")
    seen_w = []
    for dlt, w in sorted([c for c in cand if np.sign(c[0]) == sign], key=lambda c: -abs(c[0])):
        if any(w in x or x in w for x in seen_w):
            continue
        seen_w.append(w)
        ex = max((r for r in today if w in TM[r[1]]), key=lambda r: r[2] * sign)  # 그 쪽으로 가장 센 제목
        print(f"    {w:22} 차이 {dlt:+.3f} · {kt[w]}건 · 예: {ex[2]:+.2f} {ex[1][:60]}")
        if len(seen_w) == 6:
            break
