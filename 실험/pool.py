# 라벨 없는 제목 모음 (2026-10-28, 한국어는 2026-10-29): {en,ko}_daily 2026-08-01 이전 제목 중 영어 20자·한국어 8자 이상, 그 언어 시험지들과 거의 같은 것
# (3글자 조각 자카드 ≥ 0.5) 빼고, 이미 라벨 붙은 제목(labeled/{en,ko}*.tsv) 빼고, 완전히 같은 것은 하나만. (서로 거의 같은 것 거르기는 en_trainset과 달리 안 함: 2만 개 넘으면 너무 느림)
# python 실험/pool.py 시험지제목파일 출력파일 [ko]
import sys, collections
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
sys.stdout.reconfigure(encoding="utf-8")
TEST, OUT = sys.argv[1], sys.argv[2]
LANG = sys.argv[3] if len(sys.argv) > 3 else "en"
MINLEN = 20 if LANG == "en" else 8
first = {}
for l in open(j.HOME / "headlines" / f"{LANG}_daily.tsv", encoding="utf-8"):
    p = l.split("\t")
    if len(p) > 2 and p[0][:2] == "20" and p[2] not in first:
        first[p[2]] = p[0][:10]
key = lambda t: j.clean(t).lower()
tri = lambda t: {t[i:i + 3] for i in range(len(t) - 2)} if len(t := "".join(key(t).split())) > 2 else {t}
tests = [l.split("\t")[-1] for l in open(TEST, encoding="utf-8").read().splitlines() if l.strip()]
tests += [l.split("\t")[-1] for f in (j.HOME / "labeled" / "시험").glob(f"{LANG}*.tsv") for l in f.read_text(encoding="utf-8").splitlines()[1:]]
S = [tri(t) for t in tests]
inv = collections.defaultdict(list)
for k, s in enumerate(S):
    for g in s:
        inv[g].append(k)
near = lambda a: any(n / (len(a) + len(S[k]) - n) >= 0.5 for k, n in collections.Counter(k for g in a for k in inv.get(g, ())).items())
seen = {key(l.split("\t")[-1]) for f in (j.HOME / "labeled").glob(f"{LANG}*.tsv") for l in f.read_text(encoding="utf-8").splitlines()[1:]}
pick = []
for t, d in sorted(first.items()):
    if d < "2026-08-01" and len(j.clean(t)) >= MINLEN and j.lang_of(t) == LANG and key(t) not in seen and not near(tri(t)):
        seen.add(key(t)); pick.append(t)
print(f"라벨 없는 제목 {len(pick):,}개")
open(OUT, "w", encoding="utf-8").write("\n".join(pick) + "\n")
