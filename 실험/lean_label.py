# 가벼운 서브에이전트 라벨러 준비·합치기 (2026-10-29): 묶음마다 '지시 + 기준(필요한 절만) + 번호 붙은 제목' 파일 하나를 만들고,
# 에이전트는 그 파일을 한 번 읽어 구조화 출력(개수 강제)으로만 답한다. 파일 쓰기·확인 단계가 없다. 워크플로 스크립트는 실험/lean_label.js.
# python 실험/lean_label.py prep en|ko 제목파일 묶음크기 출력폴더    → 묶음 파일들 + 워크플로 args 한 줄 (lang, dir, total, size)
# python 실험/lean_label.py merge en|ko 제목파일 워크플로id 출력tsv 묶음크기  → journal.jsonl의 답을 검사해 합침 (문제 묶음 목록을 찍음, 빠진 묶음은 '합친 줄'이 모자람)
import sys, json, re, os, collections
sys.stdout.reconfigure(encoding="utf-8")
HOME = r"C:\english_folder\somethings"
W = r"~\.claude\projects\C--english-folder-somethings\c9da82c9-cd21-48a1-a16f-0b5fe40fa8be\subagents\workflows"
OK = ["-2", "-1", "-0.5", "0", "0.5", "1", "2"]


def guide(lang):
    """라벨기준.md에서 감성 부분만: 0~4번, 6번 숫자 표 (+ 영어는 7번, 한국어는 결정 기록)"""
    t = open(os.path.join(HOME, "labeled", "라벨기준.md"), encoding="utf-8").read()
    parts = re.split(r"(?m)^(?=## )", t)
    want = ("## 0.", "## 1.", "## 2.", "## 3.", "## 4.", "## 6.") + (("## 7.",) if lang == "en" else ("## 결정 기록",))
    return "\n".join(p.strip() for p in parts if p.startswith(want))


def titles(f):
    ls = [l.rstrip("\n").split("\t")[-1] for l in open(f, encoding="utf-8") if l.strip()]
    return ls[1:] if ls and ls[0] == "제목" else ls  # 머리줄 빼기


HEAD = {"en": "영어 경제 뉴스 제목마다 감성을 두 관점으로 정한다. 미국: 미국 주식 투자자 관점(S&P 500 등). 한국: 한국 주식 투자자 관점(코스피). "
              "미국 지수 등락은 한국 관점에서 최대 ±1, 한국과 무관한 지역 소식은 대개 0.",
        "ko": "한국 경제 뉴스 제목마다 감성을 정한다."}
TAIL = ("값은 -2, -1, -0.5, 0, 0.5, 1, 2 중 하나. 제목만 보고 판단한다. 경제·시장과 무관한 제목은 0. \"애매하면 0\". "
        "전망·의견·질문형은 한 단계 약하게, 개별 기업은 최대 ±1. ±2는 6번 표·1번 목록에 맞는 시장 전체 사건만.")

if sys.argv[1] == "prep":  # 묶음마다 파일 하나 = 지시 + 기준 + 번호 붙은 제목. 워크플로 args는 {lang, dir, total, size}
    lang, src, size, out = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    ts = titles(src)
    os.makedirs(out, exist_ok=True)
    g = guide(lang)
    for k in range(0, len(ts), size):
        seg = ts[k:k + size]
        body = "\n".join(f"{i + 1}\t{t}" for i, t in enumerate(seg))
        open(os.path.join(out, f"b{k // size:04d}.txt"), "w", encoding="utf-8").write(
            f"{HEAD[lang]} {TAIL}\n아래 제목 {len(seg)}개 모두에, 번호 순서대로 하나씩 답한다.\n\n# 기준\n{g}\n\n# 제목 ({len(seg)}개, 번호<TAB>제목)\n{body}\n")
    print(json.dumps({"lang": lang, "dir": out.replace("/", "\\"), "total": len(ts), "size": size}, ensure_ascii=False))
else:  # 기록의 started(라벨 → agentId)와 result(agentId → 답)를 이어 묶음 이름으로 찾는다
    lang, src, run, out, size = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6])
    ts = titles(src)
    lab, ans = {}, {}
    for l in (x for r in run.split(",") for x in open(os.path.join(W, r, "journal.jsonl"), encoding="utf-8")):  # 워크플로 id 여러 개(쉼표)
        d = json.loads(l)
        if d.get("type") == "started":
            lab[d["agentId"]] = d.get("label", "")
        elif d.get("type") == "result":
            ans[d["agentId"]] = d.get("result")
    got = {}
    for aid, name in lab.items():
        if name.startswith(f"{lang}_b") and isinstance(ans.get(aid), dict):
            k = int(name.split("_b")[1])
            got[name] = {"start": k * size, "n": min(size, len(ts) - k * size), "out": ans[aid]}
    rows, bad = [], []
    for name in sorted(got):  # 답은 {"번호": "값"} (영어는 "미국/한국")
        r = got[name]
        n, seg, o = r["n"], ts[r["start"]:r["start"] + r["n"]], r["out"]
        vals = [str(o.get(str(i + 1), "")).split("/") for i in range(n)]
        if len(seg) != n or any(len(v) != (2 if lang == "en" else 1) or any(x not in OK for x in v) for v in vals):
            bad.append(name); continue
        rows += [v + [t] for v, t in zip(vals, seg)]
    cols = ("us", "kr") if lang == "en" else ("s",)
    print(f"합친 줄 {len(rows):,} / {len(ts):,}, 문제 묶음 {bad}")
    for i, c in enumerate(cols):
        print(c, sorted(collections.Counter(r[i] for r in rows).items(), key=lambda x: float(x[0])))
    head = "미국\t한국\t제목" if lang == "en" else "감성\t제목"
    open(out, "w", encoding="utf-8", newline="\n").write(head + "\n" + "\n".join("\t".join(r) for r in rows) + "\n")
