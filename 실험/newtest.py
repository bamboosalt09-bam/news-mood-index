# 새 시험지 제목 뽑기 (2026-10-27): 2026-08-01 이후 ko_daily 제목 중 큰 시험지·모든 시험지·모든 학습 라벨 제목과
# 거의 같은 것(3글자 조각 자카드 ≥ 0.5)과 8자 미만을 빼고 무작위 N개 (시드 2028). 라벨은 따로 붙인다.
# python 실험/newtest.py 1500 출력폴더   → 출력폴더/b00.txt … (150개씩, 번호<TAB>제목)
import sys, collections
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
sys.stdout.reconfigure(encoding="utf-8")
N, OUT = int(sys.argv[1]), sys.argv[2]
first = {}
for l in open(j.HOME / "headlines" / "ko_daily.tsv", encoding="utf-8"):
    p = l.split("\t")
    if p[0] and p[2] not in first:
        first[p[2]] = p[0][:10]
lab = lambda f: [l.split("\t")[-1] for l in (j.HOME / "labeled" / f).read_text(encoding="utf-8").splitlines()[1:]]
seen = set(lab("ko_ai_opus.tsv") + lab("ko_more8.tsv") + lab("ko.tsv"))
seen |= {t for t in lab("ko_ai_haiku_mid.tsv") + lab("ko_ai_haiku_rest.tsv") if first.get(t, "") < "2026-08-01"}
seen |= {l.split("\t")[-1] for f in (j.HOME / "labeled" / "시험").glob("*.tsv") for l in f.read_text(encoding="utf-8").splitlines()[1:]}
tri = lambda t: {t[i:i + 3] for i in range(len(t) - 2)} if len(t := "".join(t.lower().split())) > 2 else {t}
S = [tri(t) for t in seen]
inv = collections.defaultdict(list)
for k, s in enumerate(S):
    for g in s:
        inv[g].append(k)
near = lambda t: any(n / (len(a) + len(S[k]) - n) >= 0.5 for a in [tri(t)] for k, n in collections.Counter(k for g in a for k in inv.get(g, ())).items())
pool = sorted(t for t, d in first.items() if d >= "2026-08-01" and len(t) >= 8 and t not in seen)
pool = [t for t in pool if not near(t)]
pick = []
for i in np.random.default_rng(2028).permutation(len(pool)):  # 뽑은 것끼리도 거의 같은 제목은 하나만
    a = tri(pool[i])
    if all(len(a & b) / len(a | b) < 0.5 for b in map(tri, pick)):
        pick.append(pool[i])
    if len(pick) == N:
        break
print(f"후보 {len(pool):,}개 → {len(pick)}개")
import os
os.makedirs(OUT, exist_ok=True)
for b in range(0, N, 150):
    open(os.path.join(OUT, f"b{b // 150:02d}.txt"), "w", encoding="utf-8").write("\n".join(f"{b + i + 1}\t{t}" for i, t in enumerate(pick[b:b + 150])) + "\n")
open(os.path.join(OUT, "dates.tsv"), "w", encoding="utf-8").write("\n".join(f"{first[t]}\t{t}" for t in pick) + "\n")
