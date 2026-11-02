# 영어 학습 라벨용 제목 뽑기 (2026-10-28): en_daily 2026-08-01 이전 제목 중 영어 시험지들과 거의 같은 것(3글자 조각 자카드 ≥ 0.5)과
# 20자 미만을 빼고 무작위 N개 (시드 2032). 뽑은 것끼리도 거의 같으면 하나만.
# python 실험/en_trainset.py 시험지제목파일 20100 출력폴더 [묶음=300]   → 출력폴더/b000.txt … (번호<TAB>제목)
import sys, os, collections
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
sys.stdout.reconfigure(encoding="utf-8")
TEST, N, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
SIZE = int(sys.argv[4]) if len(sys.argv) > 4 else 300
first = {}
for l in open(j.HOME / "headlines" / "en_daily.tsv", encoding="utf-8"):
    p = l.split("\t")
    if len(p) > 2 and p[0][:2] == "20" and p[2] not in first:
        first[p[2]] = p[0][:10]
key = lambda t: j.clean(t).lower()
tri = lambda t: {t[i:i + 3] for i in range(len(t) - 2)} if len(t := "".join(key(t).split())) > 2 else {t}
tests = [l.split("\t")[-1] for l in open(TEST, encoding="utf-8").read().splitlines() if l.strip()]
tests += [l.split("\t")[-1] for f in (j.HOME / "labeled" / "시험").glob("en*.tsv") for l in f.read_text(encoding="utf-8").splitlines()[1:]]
S = [tri(t) for t in tests]
inv = collections.defaultdict(list)
for k, s in enumerate(S):
    for g in s:
        inv[g].append(k)
near = lambda a: any(n / (len(a) + len(S[k]) - n) >= 0.5 for k, n in collections.Counter(k for g in a for k in inv.get(g, ())).items())
pool = sorted(t for t, d in first.items() if d < "2026-08-01" and len(j.clean(t)) >= 20 and j.lang_of(t) == "en")
pick, inv2, taken = [], collections.defaultdict(list), []
for i in np.random.default_rng(2032).permutation(len(pool)):
    a = tri(pool[i])
    if near(a) or any(n / (len(a) + len(taken[k]) - n) >= 0.5 for k, n in collections.Counter(k for g in a for k in inv2.get(g, ())).items()):
        continue
    taken.append(a); [inv2[g].append(len(taken) - 1) for g in a]; pick.append(pool[i])
    if len(pick) == N:
        break
print(f"후보 {len(pool):,}개 → {len(pick):,}개", dict(sorted(collections.Counter(first[t][:7] for t in pick).items())))
os.makedirs(OUT, exist_ok=True)
for b in range(0, len(pick), SIZE):
    open(os.path.join(OUT, f"b{b // SIZE:03d}.txt"), "w", encoding="utf-8").write("\n".join(f"{b + i + 1}\t{t}" for i, t in enumerate(pick[b:b + SIZE])) + "\n")
open(os.path.join(OUT, "dates.tsv"), "w", encoding="utf-8").write("\n".join(f"{first[t]}\t{t}" for t in pick) + "\n")
