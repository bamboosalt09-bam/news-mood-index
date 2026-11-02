# 섹터 제목에서 지금 모델이 약한가 (2026-10-31, docs/실험기록.md 같은 날 '섹터 전용 작은 모델 필요 여부'). 동결 시험지는 읽기만.
# python 실험/섹터모델점검.py   → 실험/결과/섹터모델점검.json
import sys, json, collections
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
sys.argv = sys.argv[:1]  # 실험/섹터.py가 import 때 전체 계산을 하지 않게 함수만 가져온다
import judge as j
import evaluate as e
sys.stdout.reconfigure(encoding="utf-8")
src = open(j.HOME / "실험" / "섹터.py", encoding="utf-8").read()
ns = {}; exec(src[:src.index("OUT = {}")], ns)  # load, tag, STRIP만
T = j.HOME / "labeled" / "시험"
SETS = {"ko": ("ko_v2s-감성", ["ko_무작위300.tsv", "ko_무작위400.tsv", "ko_큰시험2000.tsv", "ko_새시험1500.tsv"]),
        "en": ("en_v2s-감성", ["en_새시험1500.tsv"])}


def rows(f):
    return [l.split("\t") for l in (T / f).read_text(encoding="utf-8").splitlines()[1:] if l.strip()]


def m(s, y):
    r = e.screen(s, y)
    return {"수": len(y), "맞음": round(float(r["맞음"]), 3), "헛경보": round(float(r["헛경보"]), 3), "놓침": round(float(r["놓침"]), 3),
            "방향틀림": r["방향 틀림"], "방향틀림비율": round(r["방향 틀림"] / len(y), 4)}


OUT = {}
for lang, (name, files) in SETS.items():
    D = ns["load"](lang)
    seen, T_, Y = set(), [], []
    for f in files:  # 시험지끼리 겹친 제목은 한 번만
        for r in rows(f):
            if r[-1] not in seen:
                seen.add(r[-1]); T_.append(r[-1]); Y.append(float(r[0]))
    s, y = e.scores(name, T_), np.array(Y)
    secs = [ns["tag"](t, D, lang)[0] for t in T_]
    has = np.array([bool(x) for x in secs])
    r = {"섹터": m(s[has], y[has]), "비섹터": m(s[~has], y[~has]), "섹터별": {}}
    by = collections.defaultdict(list)
    for i, x in enumerate(secs):
        for sec in x:
            by[sec].append(i)
    for sec, ix in sorted(by.items(), key=lambda x: -len(x[1])):
        r["섹터별"][sec] = m(s[ix], y[ix]) if len(ix) >= 30 else {"수": len(ix)}
    lab = set()
    for p in (j.HOME / "labeled").glob(f"{lang}*.tsv"):
        for l in p.read_text(encoding="utf-8").splitlines()[1:]:
            t = l.rstrip("\n").split("\t")[-1]
            if t and t not in seen:
                lab.add(t)
    c = collections.Counter(sec for t in lab for sec in ns["tag"](t, D, lang)[0])
    r["학습라벨"] = {"전체": len(lab), "섹터제목": sum(1 for t in lab if ns["tag"](t, D, lang)[0]), "섹터별": dict(c.most_common())}
    a, b = r["섹터"], r["비섹터"]
    weak = (b["맞음"] - a["맞음"] >= 0.05) or (a["방향틀림"] >= 5 and a["방향틀림비율"] >= 2 * b["방향틀림비율"])
    r["판단"] = "섹터에서 약함" if weak else "약하지 않음"
    OUT[lang] = r
    print(f"\n== {lang} ({name}) 시험 제목 {len(y):,}: {r['판단']}")
    for k in ("섹터", "비섹터"):
        print(f"  {k:4} {r[k]}")
    for sec, v in r["섹터별"].items():
        print(f"    {sec:14} {v}")
    print(f"  학습 라벨 {r['학습라벨']['전체']:,}개 중 섹터 제목 {r['학습라벨']['섹터제목']:,}개: {r['학습라벨']['섹터별']}")
json.dump(OUT, open(j.HOME / "실험" / "결과" / "섹터모델점검.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
