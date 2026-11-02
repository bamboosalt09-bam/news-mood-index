# 사전이 놓치는 말 캐기 (2026-10-27): 8월 이전 학습 라벨에서
#  가. 사전이 아무 말도 못 한 제목(규칙 신호 0) 중 정답이 ±인 제목에 몰린 형태소 → 방향·사건 말 후보
#  나. 규칙 신호와 정답 부호가 반대인 제목에 몰린 명사 → 빠진 대상·잘못 붙은 대상 후보
# python 실험/lexmine.py   (시험지 안 봄, lexicon.tsv 안 바꿈)
import sys, os, collections
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
from kiwipiepy import Kiwi
sys.stdout.reconfigure(encoding="utf-8")
CUT = "2026-08-01"
first = {}
for l in open(j.HOME / "headlines" / "ko_daily.tsv", encoding="utf-8"):
    p = l.split("\t")
    if p[0] and p[2] not in first:
        first[p[2]] = p[0][:10]
labf = lambda f: {p[-1]: float(p[0]) for p in (l.split("\t") for l in open(j.HOME / "labeled" / f, encoding="utf-8").read().splitlines()[1:])}
lab = labf("ko_ai_haiku_mid.tsv") | labf("ko_ai_haiku_rest.tsv") | labf("ko_ai_opus.tsv") | labf("ko_more8.tsv")
tests = {l.split("\t")[-1] for f in (j.HOME / "labeled" / "시험").glob("ko_*.tsv") for l in f.read_text(encoding="utf-8").splitlines()[1:]}
titles = [t for t in lab if first.get(t, "9999") < CUT and t not in tests]
lex = j.load_lexicon(j.LEXICON)
words = {w.strip() for k, ws in lex.items() if not k.startswith(("크기:", "민감:")) for w in ws}
st = [j.clean(t) for t in titles]
r = np.concatenate([j.rule_signal(st[i:i + 2000], lex) for i in range(0, len(st), 2000)])
y = np.array([lab[t] for t in titles])
kiwi = Kiwi(num_workers=max(1, os.cpu_count() // 2))
toks = [{x.form for x in ts if x.tag in ("NNG", "NNP", "VV", "VA", "XR", "SL")} for ts in kiwi.tokenize(st)]
known = lambda f: any(f in w or w in f for w in words)


def table(mask, title, n_min, purity):
    c = collections.defaultdict(lambda: [0, 0, 0])
    for i in np.flatnonzero(mask):
        for f in toks[i]:
            c[f][int(np.sign(y[i])) + 1] += 1
    rows = []
    for f, (neg, zero, pos) in c.items():
        n = neg + zero + pos
        if n >= n_min and len(f) >= 2 and not known(f):
            pol = (pos - neg) / n
            if abs(pol) >= purity:
                rows.append((pol, n, f, neg, zero, pos))
    rows.sort(key=lambda x: -abs(x[0]) * np.log(x[1]))
    print(f"\n=== {title}: {len(rows)}개 (형태소, 극성, 개수 −/0/+) ===")
    for pol, n, f, neg, zero, pos in rows[:120]:
        print(f"{f}\t{pol:+.2f}\t{n}\t{neg}/{zero}/{pos}")


silent = np.abs(r) < 1e-9
print(f"8월 이전 라벨 제목 {len(titles):,}, 사전이 침묵한 제목 {silent.sum():,} (그중 정답 ± {(silent & (y != 0)).sum():,})")
table(silent, "가. 사전 침묵 제목에서 한쪽으로 몰린 말", 30, 0.5)
wrong = (np.sign(r) * np.sign(y) < 0)
print(f"\n규칙과 정답이 반대인 제목 {wrong.sum():,}")
c_all = collections.Counter(f for ts in toks for f in ts)
c_w = collections.Counter(f for i in np.flatnonzero(wrong) for f in toks[i])
rate = wrong.mean()
rows = sorted(((c_w[f] / c_all[f], c_w[f], c_all[f], f) for f in c_w if c_w[f] >= 8 and len(f) >= 2), reverse=True)
print("=== 나. 규칙·정답 반대 제목에 몰린 말 (반대 비율, 반대 개수, 전체 개수) ===")
for q, a, n, f in rows[:80]:
    if q >= 4 * rate:
        print(f"{f}\t{q:.0%}\t{a}\t{n}\t{'사전' if known(f) else ''}")
