# 영어 새 시험지 제목 뽑기 (2026-10-28): en_daily 2026-08-01 이후 제목 중 기존 영어 라벨 제목과 거의 같은 것
# (3글자 조각 자카드 ≥ 0.5)과 20자 미만을 빼고 무작위 N개 (시드 2031). 뽑은 것끼리도 거의 같으면 하나만. 라벨은 따로 붙인다.
# python 실험/en_newtest.py 1500 출력폴더   → 출력폴더/b00.txt … (150개씩, 번호<TAB>제목), dates.tsv
import sys, os, collections
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
sys.stdout.reconfigure(encoding="utf-8")
N, OUT = int(sys.argv[1]), sys.argv[2]
first = {}
for l in open(j.HOME / "headlines" / "en_daily.tsv", encoding="utf-8"):
    p = l.split("\t")
    if len(p) > 2 and p[0][:2] == "20" and p[2] not in first:
        first[p[2]] = p[0][:10]
key = lambda t: j.clean(t).lower()
seen = {key(l.split("\t")[-1]) for f in (j.HOME / "labeled").glob("en*.tsv") for l in f.read_text(encoding="utf-8").splitlines()[1:]}
tri = lambda t: {t[i:i + 3] for i in range(len(t) - 2)} if len(t := "".join(key(t).split())) > 2 else {t}
S = [tri(t) for t in seen]
inv = collections.defaultdict(list)
for k, s in enumerate(S):
    for g in s:
        inv[g].append(k)
near = lambda t: any(n / (len(a) + len(S[k]) - n) >= 0.5 for a in [tri(t)] for k, n in collections.Counter(k for g in a for k in inv.get(g, ())).items())
pool = sorted(t for t, d in first.items() if d >= "2026-08-01" and len(j.clean(t)) >= 20 and key(t) not in seen and j.lang_of(t) == "en")
pool = [t for t in pool if not near(t)]
pick, taken = [], []
for i in np.random.default_rng(2031).permutation(len(pool)):
    a = tri(pool[i])
    if all(len(a & b) / len(a | b) < 0.5 for b in taken):
        pick.append(pool[i]); taken.append(a)
    if len(pick) == N:
        break
print(f"후보 {len(pool):,}개 → {len(pick)}개", collections.Counter(first[t][:7] for t in pick))
os.makedirs(OUT, exist_ok=True)
for b in range(0, N, 150):
    open(os.path.join(OUT, f"b{b // 150:02d}.txt"), "w", encoding="utf-8").write("\n".join(f"{b + i + 1}\t{t}" for i, t in enumerate(pick[b:b + 150])) + "\n")
open(os.path.join(OUT, "dates.tsv"), "w", encoding="utf-8").write("\n".join(f"{first[t]}\t{t}" for t in pick) + "\n")
