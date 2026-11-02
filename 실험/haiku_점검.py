# 변형 D Haiku 라벨 점검: 줄 수, 묶음별 쏠림, 선생이 확신하는 구간에서 방향 충돌
import sys, json, pathlib, collections, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
D = pathlib.Path(sys.argv.pop(1))  # 묶음 폴더 (d000.txt…, all_titles.txt)
J = r"~\.claude\projects\C--english-folder-somethings\c9da82c9-cd21-48a1-a16f-0b5fe40fa8be\subagents\workflows\wf_783966e9-197\journal.jsonl"
titles = (D / "all_titles.txt").read_text(encoding="utf-8").split("\n")
label_of, res = {}, {}
for l in open(J, encoding="utf-8"):
    d = json.loads(l)
    if d.get("type") == "started": label_of[d["agentId"]] = d["label"]
    if d.get("type") == "result" and isinstance(d.get("result"), dict): res[label_of[d["agentId"]]] = d["result"]["labels"]
bad, got = [], {}
for lab, ls in res.items():
    b = int(lab[1:]); n = len((D / f"d{b:03d}.txt").read_text(encoding="utf-8").splitlines())
    if len(ls) != n: bad.append((lab, len(ls), n)); continue
    for i, v in enumerate(ls): got[titles[b * 113 + i]] = float(v)
print(f"끝난 묶음 {len(res)}, 줄 수 안 맞는 묶음 {len(bad)} {bad[:5]}")
per = {lab: collections.Counter(ls) for lab, ls in res.items()}
odd = [(lab, c.most_common(1)[0]) for lab, c in per.items() if c.most_common(1)[0][1] / sum(c.values()) > 0.6]
print("한 값이 60% 넘는 묶음", len(odd), odd[:5])
v = np.array(list(got.values())); print("분포", {k: f"{np.mean(v == k):.1%}" for k in (-2, -1, -0.5, 0, 0.5, 1, 2)})
te = {t.rstrip("\n"): float(s) for s, t in (l.split("\t", 1) for l in open(r"C:\english_folder\somethings\data\ko_scores_ko-감성-미세.tsv", encoding="utf-8"))}
both = [(got[t], te[t]) for t in got if t in te]
h, s = np.array(both).T
for lo, hi in ((0.9, 9), (0.4, 0.9), (0, 0.4)):
    m = (np.abs(s) >= lo) & (np.abs(s) < hi)
    print(f"선생 |점수| {lo}~{hi}: {m.sum():,}개 | Haiku 같은 방향 {np.mean(np.sign(h[m]) == np.sign(s[m])):.1%}, 0 {np.mean(h[m] == 0):.1%}, 반대 {np.mean(np.sign(h[m]) * np.sign(s[m]) < 0):.1%}")
if len(sys.argv) > 1:  # 표본 보기
    rng = np.random.default_rng(int(sys.argv[1])); ks = list(got)
    for i in rng.permutation(len(ks))[:40]: print(f"{got[ks[i]]:+g}\t{te.get(ks[i], float('nan')):+.2f}\t{ks[i][:70]}")
