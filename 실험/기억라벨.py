# 기억 라벨 준비·합치기 (2026-10-31, docs/실험기록.md 같은 날 '기억 라벨'). Haiku 라벨러가 (업체, 거래일) 묶음을 보고 업체 관점으로 판정.
# python 실험/기억라벨.py prep [시범]     → data/기억라벨/묶음/*.txt + jobs.json (워크플로 args)
# python 실험/기억라벨.py merge 워크플로id[,id…] → labeled/ko_기억라벨.tsv (날짜, 업체, 감성, 제목)
import sys, os, re, json, random, collections
sys.path.insert(0, r"C:\english_folder\somethings")
sys.path.insert(0, r"C:\english_folder\somethings\web")
import judge as j
import build as b
import sectors as S
sys.stdout.reconfigure(encoding="utf-8")
START, END, SIZE = "2025-11-03", "2026-10-30", 60
OUT = j.HOME / "data" / "기억라벨"
W = r"~\.claude\projects\C--english-folder-somethings\c9da82c9-cd21-48a1-a16f-0b5fe40fa8be\subagents\workflows"
from memjudge import GUIDE, CUE, block  # 판정 기준·묶음 형식은 web/memjudge.py 한 곳에 (새 날 판정과 같게)


def groups():
    days, _, _ = b.prices("코스피")
    IT = b.items("ko", "최종")
    by = b.assign("ko", days, IT, ("daily", "sector"), since="2025-10-25")
    D = S.load("ko")
    G = collections.defaultdict(list)
    E = [d for d in days if START <= d <= END]
    prev = {d: days[days.index(d) - 1] for d in E}
    for d in [days[days.index(E[0]) - 1]] + E:
        for t, it, _ in by.get(d, []):
            for c in S.tag(t, D, "ko")[1]:
                G[c, d].append((t, it["mkt"]))
    return G, E, prev


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        pilot = sys.argv[2:3] == ["시범"]
        G, E, prev = groups()
        keys = [(c, d) for (c, d) in G if d >= START]
        if pilot:
            random.seed(1)
            want = [("기아", "2026-10-30"), ("삼성전자", "2026-10-29"), ("현대차", "2026-07-23"), ("카카오", "2026-02-06")]
            want = [k for k in want if k in G] + random.sample([k for k in keys if 5 <= len(G[k]) <= 40], 2)
            keys = want
        keys.sort(key=lambda k: (k[1], k[0]))
        items = [(c, d, t) for c, d in keys for t, _ in G[c, d]]
        out = OUT / ("시범" if pilot else "묶음")
        os.makedirs(out, exist_ok=True)
        jobs, rows = [], []
        for k in range(0, len(items), SIZE):
            seg = items[k:k + SIZE]
            body, n, cur = "", 0, None
            for c, d, t in seg:
                if (c, d) != cur:
                    cur = (c, d); todo = {tt for cc, dd, tt in seg if (cc, dd) == cur}
                    body += "\n" + block(c, d, todo, G[c, d], G.get((c, prev[d]), [])) + "판정할 제목 (번호<TAB>제목):\n"
                n += 1; body += f"{n}\t{t}\n"; rows.append((d, c, t))
            name = f"ko_b{k // SIZE:04d}"
            open(out / f"b{k // SIZE:04d}.txt", "w", encoding="utf-8").write(f"{GUIDE}\n\n아래 묶음들의 '판정할 제목' {n}개 모두에, 번호마다 하나씩 답한다.\n{body}")
            jobs.append({"name": name, "file": str(out / f"b{k // SIZE:04d}.txt"), "n": n})
        json.dump({"jobs": jobs, "rows": rows}, open(out / "jobs.json", "w", encoding="utf-8"), ensure_ascii=False)
        print(f"묶음(업체·날) {len(keys):,}, 판정할 제목 {len(items):,}, 묶음 파일 {len(jobs)} → {out}")
    else:
        runs, which = sys.argv[2].split(","), (sys.argv[3] if len(sys.argv) > 3 else "묶음")
        meta = json.load(open(OUT / which / "jobs.json", encoding="utf-8"))
        lab, ans = {}, {}
        for r in runs:
            for l in open(os.path.join(W, r, "journal.jsonl"), encoding="utf-8"):
                x = json.loads(l)
                if x.get("type") == "started": lab[x["agentId"]] = x.get("label", "")
                elif x.get("type") == "result": ans[x["agentId"]] = x.get("result")
        got = {lab[a]: ans[a] for a in lab if isinstance(ans.get(a), dict)}
        out, miss, k = [], [], 0
        for jb in meta["jobs"]:
            seg = meta["rows"][k:k + jb["n"]]; k += jb["n"]
            o = got.get(jb["name"])
            if not o or len(o) != jb["n"]:
                miss.append(jb["name"]); continue
            out += [(d, c, o[str(i + 1)], t) for i, (d, c, t) in enumerate(seg)]
        path = j.HOME / "labeled" / f"ko_기억라벨{'_시범' if which == '시범' else ''}.tsv"
        open(path, "w", encoding="utf-8").write("날짜\t업체\t감성\t제목\n" + "".join("\t".join(r) + "\n" for r in out))
        print(f"합친 줄 {len(out):,}, 빠진 묶음 {len(miss)}: {miss[:20]} → {path}")
