# 변형 D Haiku 라벨 합치기: 첫 실행 + 다시 돌린 실행(나중 것이 이김). 개수가 파일 줄 수와 같은 묶음만 → labeled/ko_ai_haiku_rest.tsv
import sys, json, pathlib, glob, collections
sys.stdout.reconfigure(encoding="utf-8")
D = pathlib.Path(sys.argv.pop(1))  # 묶음 폴더 (d000.txt…, all_titles.txt)
W = r"~\.claude\projects\C--english-folder-somethings\c9da82c9-cd21-48a1-a16f-0b5fe40fa8be\subagents\workflows"
RUNS = ["wf_783966e9-197"] + sys.argv[1:]  # 다시 돌린 실행 id를 차례로
titles = (D / "all_titles.txt").read_text(encoding="utf-8").split("\n")
got = {}
for run in RUNS:
    lab = {}
    for l in open(f"{W}\\{run}\\journal.jsonl", encoding="utf-8"):
        d = json.loads(l)
        if d.get("type") == "started": lab[d["agentId"]] = d["label"]
        if d.get("type") == "result" and isinstance(d.get("result"), dict):
            b = int(lab[d["agentId"]][1:]); ls = d["result"]["labels"]
            if len(ls) == len((D / f"d{b:03d}.txt").read_text(encoding="utf-8").splitlines()):
                got[b] = ls
missing = [b for b in range(661) if b not in got]
print("개수 맞는 묶음", len(got), "/ 661, 빠진 묶음", missing)
if not missing:
    out = [(v, titles[b * 113 + i]) for b in range(661) for i, v in enumerate(got[b])]
    p = pathlib.Path(r"C:\english_folder\somethings\labeled\ko_ai_haiku_rest.tsv")
    p.write_text("감성\t제목\n" + "".join(f"{v}\t{t}\n" for v, t in out), encoding="utf-8")
    print(f"저장 {len(out):,}개", dict(collections.Counter(v for v, _ in out)))
