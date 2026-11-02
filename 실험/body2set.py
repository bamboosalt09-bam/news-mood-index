# 본문 2차 학습 문장 뽑기 (2026-10-28): 모은 한경 본문에서 시험지·1차 학습 기사를 빼고, 기사마다 최대 K문장을 무작위로 (시드 2030).
# 기사 수를 넓게 보려고 기사당 문장 수를 묶는다. 링크가 같은 기사는 하나만, 같은 문장 글자는 한 번만.
# python 실험/body2set.py 본문.tsv 출력폴더 [K=6] [묶음=300]   → 출력폴더/b00.txt…(번호<TAB>문장), 출력폴더/index.tsv(번호, 기사, 문장번호, 날짜)
import sys, csv, re, random, importlib.util, collections
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
sys.stdout.reconfigure(encoding="utf-8")
SRC, OUT = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 6
SIZE = int(sys.argv[4]) if len(sys.argv) > 4 else 300
spec = importlib.util.spec_from_file_location("ss", j.HOME / ".claude" / "worktrees" / "report-body-495331" / "split_sentences.py")
ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)
done = {l.split("\t")[1] for f in ("시험/ko_본문시험1500.tsv", "ko_ai_opus_본문.tsv")
        for l in (j.HOME / "labeled" / f).read_text(encoding="utf-8").splitlines()[1:]}
csv.field_size_limit(10 ** 8)
arts = {}
for r in csv.DictReader(open(SRC, encoding="utf-8"), **ss.TSV):
    rid = (r["링크"] or "").rstrip("/").rsplit("/", 1)[-1]
    if r["날짜"][:2] == "20" and rid and rid not in done and rid not in arts:
        arts[rid] = (r["날짜"][:10], r["본문"] or "")
rng, seen, pick = random.Random(2030), set(), []
for rid in sorted(arts):
    d, text = arts[rid]
    sents = [(n, s) for n, s in enumerate((s for s in ss.split(text) if len(s) >= 10 and re.search("[가-힣]", s) and not ss.AD.search(s)), 1)]
    sents = [x for x in sents if x[1] not in seen]
    for n, s in sorted(rng.sample(sents, min(K, len(sents)))):
        seen.add(s); pick.append((rid, n, d, s))
order = rng.sample(range(len(pick)), len(pick))  # 기사가 섞이게
pick = [pick[i] for i in order]
import os
os.makedirs(OUT, exist_ok=True)
for b in range(0, len(pick), SIZE):
    open(os.path.join(OUT, f"b{b // SIZE:02d}.txt"), "w", encoding="utf-8").write(
        "\n".join(f"{b + i + 1}\t{p[3]}" for i, p in enumerate(pick[b:b + SIZE])) + "\n")
open(os.path.join(OUT, "index.tsv"), "w", encoding="utf-8").write("번호\t기사\t문장번호\t날짜\n" + "\n".join(
    f"{i + 1}\t{p[0]}\t{p[1]}\t{p[2]}" for i, p in enumerate(pick)) + "\n")
print(f"기사 {len(arts):,}개(시험·1차 뺀 것) → {len(pick):,}문장, 묶음 {-(-len(pick) // SIZE)}개",
      dict(sorted(collections.Counter(p[2][:7] for p in pick).items())))
