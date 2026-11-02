# 과거 기간 제목 뽑기 (2026-10-27): 기간 강건성 시험지 + 과거 Opus 라벨용 학습 제목.
# ko_daily 2023-10 ~ 2025-12 제목 중 모든 시험지·Opus·사람 라벨 제목과 거의 같은 것(3글자 조각 자카드 ≥ 0.5)과 8자 미만을 빼고,
# 해마다 정한 수만큼 무작위 (시드 2029). 시험지와 학습용끼리도 거의 같은 제목은 겹치지 않게 한다.
# python 실험/pastset.py 출력폴더   → 출력폴더/시험/b00.txt…(150개씩), 출력폴더/학습/b00.txt…(300개씩), 각 dates.tsv
import sys, os, collections
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
sys.stdout.reconfigure(encoding="utf-8")
OUT = sys.argv[1]
PLAN = {"시험": ({"2023": 300, "2024": 600, "2025": 600}, 150), "학습": ({"2023": 600, "2024": 2400, "2025": 3000}, 300)}
first = {}
for l in open(j.HOME / "headlines" / "ko_daily.tsv", encoding="utf-8"):
    p = l.split("\t")
    if p[0] and p[2] not in first:
        first[p[2]] = p[0][:10]
lab = lambda f: [l.split("\t")[-1] for l in (j.HOME / "labeled" / f).read_text(encoding="utf-8").splitlines()[1:]]
seen = set(lab("ko_ai_opus.tsv") + lab("ko_more8.tsv") + lab("ko.tsv"))
seen |= {l.split("\t")[-1] for f in (j.HOME / "labeled" / "시험").glob("*.tsv") for l in f.read_text(encoding="utf-8").splitlines()[1:]}
tri = lambda t: {t[i:i + 3] for i in range(len(t) - 2)} if len(t := "".join(t.lower().split())) > 2 else {t}
S = [tri(t) for t in seen]
inv = collections.defaultdict(list)
for k, s in enumerate(S):
    for g in s:
        inv[g].append(k)
near = lambda t: any(n / (len(a) + len(S[k]) - n) >= 0.5 for a in [tri(t)] for k, n in collections.Counter(k for g in a for k in inv.get(g, ())).items())
pool = sorted(t for t, d in first.items() if "2023-10" <= d < "2026" and len(t) >= 8 and t not in seen and not near(t))
print(f"후보 {len(pool):,}개")
rng = np.random.default_rng(2029)
taken = []  # 뽑은 제목의 조각 (시험·학습 모두): 서로 거의 같은 제목은 하나만
inv2 = collections.defaultdict(list)


def ok(t):
    a = tri(t)
    return all(n / (len(a) + len(taken[k]) - n) < 0.5 for k, n in collections.Counter(k for g in a for k in inv2.get(g, ())).items())


for part, (want, size) in PLAN.items():
    pick = []
    for y, n in want.items():
        got = 0
        for i in rng.permutation(len(pool)):
            t = pool[i]
            if first[t][:4] == y and ok(t):
                taken.append(tri(t)); [inv2[g].append(len(taken) - 1) for g in taken[-1]]
                pick.append(t); got += 1
                if got == n:
                    break
    order = rng.permutation(len(pick))  # 해가 섞이게
    pick = [pick[i] for i in order]
    d = os.path.join(OUT, part); os.makedirs(d, exist_ok=True)
    for b in range(0, len(pick), size):
        open(os.path.join(d, f"b{b // size:02d}.txt"), "w", encoding="utf-8").write("\n".join(f"{b + i + 1}\t{t}" for i, t in enumerate(pick[b:b + size])) + "\n")
    open(os.path.join(d, "dates.tsv"), "w", encoding="utf-8").write("\n".join(f"{first[t]}\t{t}" for t in pick) + "\n")
    print(part, len(pick), collections.Counter(first[t][:4] for t in pick))
