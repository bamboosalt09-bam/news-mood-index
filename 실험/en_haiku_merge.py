# 영어 Haiku 두 칸 라벨 합치기 (2026-10-28): 묶음마다 줄 수·번호 순서·값을 검사해 통과한 묶음만 labeled/en_ai_haiku.tsv로.
# python 실험/en_haiku_merge.py 묶음폴더 [출력=labeled/en_ai_haiku.tsv]   → 문제 묶음 목록을 찍는다 (다시 돌릴 것)
import sys, os, glob, collections
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
sys.stdout.reconfigure(encoding="utf-8")
D = sys.argv[1]
OUTF = sys.argv[2] if len(sys.argv) > 2 else str(j.HOME / "labeled" / "en_ai_haiku.tsv")
OK = {"-2", "-1", "-0.5", "0", "0.5", "1", "2"}
norm = lambda v: (lambda v: v[:-2] if v.endswith(".0") else v)(v.strip().replace("−", "-").replace("+", ""))
dates = [l.split("\t", 1) for l in open(os.path.join(D, "dates.tsv"), encoding="utf-8").read().splitlines()]
rows, bad, missing = [], [], []
for src in sorted(glob.glob(os.path.join(D, "b*.txt"))):
    b = os.path.basename(src)[1:-4]
    out = os.path.join(D, f"out_b{b}.tsv")
    if not os.path.exists(out):
        missing.append(b); continue
    s = [l.split("\t", 1) for l in open(src, encoding="utf-8").read().splitlines()]
    o = [l.split("\t") for l in open(out, encoding="utf-8").read().splitlines() if l.strip()]
    if len(s) != len(o) or any(len(x) != 3 or x[0].strip() != y[0] or norm(x[1]) not in OK or norm(x[2]) not in OK for x, y in zip(o, s)):
        bad.append(b); continue
    for (n, t), x in zip(s, o):
        assert dates[int(n) - 1][1] == t
        rows.append((norm(x[1]), norm(x[2]), dates[int(n) - 1][0], t))
print(f"합친 줄 {len(rows):,}, 문제 묶음 {bad}, 아직 없는 묶음 {len(missing)}개")
for k, h in [(0, "미국"), (1, "한국")]:
    print(h, sorted(collections.Counter(r[k] for r in rows).items(), key=lambda x: float(x[0])))
open(OUTF, "w", encoding="utf-8", newline="\n").write("미국\t한국\t날짜\t제목\n" + "\n".join("\t".join(r) for r in rows) + "\n")
