# 사전 표지 후보 고르기 (2026-10-27): 후보마다 사전 사본에 하나씩 더해, 값이 바뀐 학습 제목에서 규칙 방향이 정답 방향과 더 맞는지 본다.
# 정답 = 8월 이전 학습 라벨 (사람 > Opus > Haiku). 시험지는 안 봄. lexicon.tsv는 안 바꿈.
# python 실험/lexcand.py [후보파일: 줄마다 종류<TAB>말]
import sys, copy
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
sys.stdout.reconfigure(encoding="utf-8")
CANDS = {"지난": "했으나 했지만 였으나 였지만 하더니 했던 하던 하루_만에 이틀_만에 사흘_만에",
         "부정": "꺾이 꺾여 꺾였 실패 좌절 무색 물거품 주춤 불발 무산 반납 멈칫"}
CUT = "2026-08-01"
if len(sys.argv) > 1:
    CANDS = {}
    for l in open(sys.argv[1], encoding="utf-8"):
        if l.strip() and not l.startswith("#"):
            k, w = l.rstrip("\n").split("\t")
            CANDS[k] = CANDS.get(k, "") + " " + w.replace(" ", "_")

first = {}
for l in open(j.HOME / "headlines" / "ko_daily.tsv", encoding="utf-8"):
    p = l.split("\t")
    if p[0] and p[2] not in first:
        first[p[2]] = p[0][:10]
labf = lambda f: {p[-1]: float(p[0]) for p in (l.split("\t") for l in open(j.HOME / "labeled" / f, encoding="utf-8").read().splitlines()[1:])}
lab = labf("ko_ai_haiku_mid.tsv") | labf("ko_ai_haiku_rest.tsv") | labf("ko_ai_opus.tsv") | labf("ko_more8.tsv")
tests = {l.split("\t")[-1] for f in (j.HOME / "labeled" / "시험").glob("ko_*.tsv") for l in f.read_text(encoding="utf-8").splitlines()[1:]}
titles = [t for t in lab if first.get(t, "9999") < CUT and t not in tests]
print(f"8월 이전 라벨 제목 {len(titles):,}개")
lex = j.load_lexicon(j.LEXICON)


def agree(rows, lx):
    st = [j.clean(t) for t in rows]
    r = j.rule_signal(st, lx)
    y = np.array([lab[t] for t in rows])
    return r, np.sign(np.round(r, 6)) == np.sign(y)


keep = {}
for kind, ws in CANDS.items():
    for w in ws.split():
        w = w.replace("_", " ")
        rows = [t for t in titles if w in t.lower()]
        if not rows:
            print(f"{kind} {w:6}: 제목 없음"); continue
        lx = copy.deepcopy(lex); lx[kind] = lx.get(kind, []) + [w]
        r0, a0 = agree(rows, lex); r1, a1 = agree(rows, lx)
        ch = np.abs(r1 - r0) > 1e-9
        n = int(ch.sum())
        ok = n >= 10 and a1[ch].mean() > a0[ch].mean()
        print(f"{kind} {w:6}: 들어간 제목 {len(rows):5}, 값 바뀜 {n:4}, 방향 맞음 {a0[ch].mean() if n else 0:.1%} → {a1[ch].mean() if n else 0:.1%}  {'채택' if ok else ''}")
        if ok:
            keep.setdefault(kind, []).append(w)
print("\n채택:", keep)
lx = copy.deepcopy(lex)
for k, ws in keep.items():
    lx[k] = lx.get(k, []) + ws
r0, a0 = agree(titles, lex); r1, a1 = agree(titles, lx)
ch = np.abs(r1 - r0) > 1e-9
print(f"함께 넣으면: 값 바뀜 {ch.sum()}, 그 제목 방향 맞음 {a0[ch].mean():.1%} → {a1[ch].mean():.1%}, 전체 {a0.mean():.2%} → {a1.mean():.2%}")
