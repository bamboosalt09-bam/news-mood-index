# 한국어 Opus 2차 라벨용 제목 (2026-10-29): Haiku 라벨만 있는 2023-10 ~ 2025-12 ko_daily 제목 중 Opus·사람·Claude 라벨 제목과 모든 시험지와
# 거의 같은 것(3글자 조각 자카드 ≥ 0.5)과 8자 미만을 빼고, 과거 위주로 N개: 2023·2024는 전부, 모자라는 만큼 2025에서 무작위 (시드 2035).
# python 실험/ko_opus2set.py 20000 출력폴더   → 출력폴더/b000.txt … (300개씩, 번호<TAB>제목), dates.tsv
import sys, os, collections
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
sys.stdout.reconfigure(encoding="utf-8")
N, OUT = int(sys.argv[1]), sys.argv[2]
first = {}
for l in open(j.HOME / "headlines" / "ko_daily.tsv", encoding="utf-8"):
    p = l.split("\t")
    if len(p) > 2 and p[0][:2] == "20" and p[2] not in first:
        first[p[2]] = p[0][:10]
lab = lambda f: [l.split("\t")[-1] for l in (j.HOME / "labeled" / f).read_text(encoding="utf-8").splitlines()[1:]]
hk = set(lab("ko_ai_haiku_mid.tsv") + lab("ko_ai_haiku_rest.tsv"))
seen = set(lab("ko_ai_opus.tsv") + lab("ko_ai_opus_과거.tsv") + lab("ko_more8.tsv") + lab("ko.tsv"))
seen |= {l.split("\t")[-1] for f in (j.HOME / "labeled" / "시험").glob("ko*.tsv") for l in f.read_text(encoding="utf-8").splitlines()[1:]}
tri = lambda t: {t[i:i + 3] for i in range(len(t) - 2)} if len(t := "".join(t.lower().split())) > 2 else {t}
S = [tri(t) for t in seen]
inv = collections.defaultdict(list)
for k, s in enumerate(S):
    for g in s:
        inv[g].append(k)
near = lambda t: any(n / (len(a) + len(S[k]) - n) >= 0.5 for a in [tri(t)] for k, n in collections.Counter(k for g in a for k in inv.get(g, ())).items())
pool = sorted(t for t in hk if t in first and "2023-10" <= first[t] < "2026" and len(t) >= 8 and t not in seen and not near(t))
old, y25 = [t for t in pool if first[t] < "2025"], [t for t in pool if first[t] >= "2025"]
rng = np.random.default_rng(2035)
pick = old + [y25[i] for i in sorted(rng.choice(len(y25), max(0, N - len(old)), replace=False))]
print(f"후보 {len(pool):,}개 (2023~24 {len(old):,}, 2025 {len(y25):,}) → {len(pick):,}개", dict(sorted(collections.Counter(first[t][:4] for t in pick).items())))
os.makedirs(OUT, exist_ok=True)
for b in range(0, len(pick), 300):
    open(os.path.join(OUT, f"b{b // 300:03d}.txt"), "w", encoding="utf-8").write("\n".join(f"{b + i + 1}\t{t}" for i, t in enumerate(pick[b:b + 300])) + "\n")
open(os.path.join(OUT, "dates.tsv"), "w", encoding="utf-8").write("\n".join(f"{first[t]}\t{t}" for t in pick) + "\n")
