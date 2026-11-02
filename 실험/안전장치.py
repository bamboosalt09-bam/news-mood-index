# 안전장치 약화안 (2026-10-31, docs/실험기록.md): 문턱 × 모델 확신 조건 × 처리(0·절반) 조합을 Opus 검증으로 고르고 시험지로 확인.
# python 실험/안전장치.py [test]   (test: 고른 변형을 시험지에서도 잼)
import sys, json, itertools
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
import evaluate as e
sys.stdout.reconfigure(encoding="utf-8")
V = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
L = j.HOME / "labeled"
tsv = lambda p: [l.rstrip("\n").split("\t") for l in open(p, encoding="utf-8").read().splitlines()[1:] if l.strip()]
VARS = {"끔": None} | {f"cut{c}_conf{m}_{a}": (c, m, a) for c, m, a in itertools.product((1, 1.5, 2), (None, 1.0, 0.5), ("0", "절반"))}
NOW = "cut1_confNone_0"


def prep(name, texts, head):
    P, meta = j.predict(name, texts, head) if head is not None else j.predict(name, texts)
    ts = [j.clean(t) for t in texts]
    return {"ts": ts, "raw": j.score_of(P, meta), "rule": j.rule_signal(ts, meta.get("guard_lexicon")), "lv": j.level_only(ts)}


def apply(d, v):
    s = d["raw"].copy()
    if v is not None:
        c, m, a = v
        hit = (np.abs(d["rule"]) >= c) & (np.sign(d["rule"]) * np.sign(s) < 0) & ((np.abs(s) < m) if m else True)
        s = np.where(hit, 0.0 if a == "0" else s / 2, s)
    return np.where(d["lv"], 0, j.soften(d["ts"], s)[0])


def metrics(s, y):
    m = e.screen(s, y); s7 = V[np.abs(s[:, None] - V).argmin(1)]; nz = y != 0
    return {"7단계": float(np.mean(s7 == y)), "헛경보": float(m["헛경보"]), "놓침": float(m["놓침"]),
            "방향틀림3": int(((np.abs(s) >= 0.25) & nz & (np.sign(s) * np.sign(y) < 0)).sum())}


def tradeoff(d, y, v):  # 지금 안전장치 대비가 아니라 '안전장치 없음' 대비: 바꾼 제목 중 맞는 방향을 버린 것 / 뒤집힘을 막은 것
    s = apply(d, v); r = np.where(d["lv"], 0, j.soften(d["ts"], d["raw"])[0])
    ch = s != r; nz = y != 0
    return int((ch & nz & (np.sign(r) == np.sign(y))).sum()), int((ch & nz & (np.sign(r) == -np.sign(y))).sum())


OUT = {}
CFG = {"ko": ("ko_v2s-감성", [None], [0]), "en": ("en_v2s-감성", [0, 1], [0, 1])}
for lang, (name, heads, cols) in CFG.items():
    rows = tsv(L / f"{lang}_opus검증1000.tsv"); texts = [r[-1] for r in rows]
    D = [(prep(name, texts, h), np.array([float(r[c]) for r in rows])) for h, c in zip(heads, cols)]
    pool = {}  # Opus 모음 (학습 제목 포함, 참고)
    for f in ({"ko": ["ko_ai_opus.tsv", "ko_ai_opus_과거.tsv", "ko_ai_opus_과거2.tsv", "ko_ai_opus_과거3.tsv"], "en": ["en_ai_opus_pool.tsv", "en_ai_opus_pool2.tsv", "en_ai_opus_과거.tsv"]}[lang]):
        for r in tsv(L / f):
            pool[r[-1]] = [float(r[c]) for c in cols]
    pt = list(pool); PD = [(prep(name, pt, h), np.array([pool[t][k] for t in pt])) for k, h in enumerate(heads)]
    res = {}
    for k, v in VARS.items():
        ms = [metrics(apply(d, v), y) for d, y in D]
        to = [tradeoff(d, y, v) for d, y in PD]
        res[k] = {"7단계": round(float(np.mean([m["7단계"] for m in ms])), 4), "관점별": ms, "버림:막음(모음)": [list(x) for x in to]}
    now = res[NOW]
    ok = [k for k, r in res.items() if all(m["방향틀림3"] <= n["방향틀림3"] and m["헛경보"] <= n["헛경보"] + 0.003 for m, n in zip(r["관점별"], now["관점별"]))]
    best = max(r_["7단계"] for k, r_ in res.items() if k in ok)
    strength = lambda k: (-99, 0, False) if k == "끔" else (-VARS[k][0], VARS[k][1] or 9, VARS[k][2] == "0")  # 클수록 강함 (문턱 낮음·확신 조건 없음·0으로)
    near = [k for k in ok if res[k]["7단계"] >= best - 0.001]
    pick = max(near, key=strength)
    cand = pick if res[pick]["7단계"] >= now["7단계"] + 0.002 else None
    OUT[lang] = {"변형": res, "조건맞음": ok, "고른것": pick, "후보": cand}
    print(f"\n== {lang} (Opus 검증): 지금 {NOW} 7단계 {now['7단계']:.2%} 방향틀림3 {[m['방향틀림3'] for m in now['관점별']]}")
    for k, r in sorted(res.items(), key=lambda x: -x[1]["7단계"]):
        print(f"  {'✓' if k in ok else ' '} {k:22} 7단계 {r['7단계']:.2%}  방향틀림3 {[m['방향틀림3'] for m in r['관점별']]}  헛경보 {[round(m['헛경보'] * 100, 1) for m in r['관점별']]}  놓침 {[round(m['놓침'] * 100, 1) for m in r['관점별']]}  버림:막음 {r['버림:막음(모음)']}")
    print(f"  → 조건 맞는 것 중 고른 것 {pick} ({res[pick]['7단계']:.2%}), 후보 {cand}")

if "test" in sys.argv:
    TESTS = {"ko": [("큰 시험지(깨끗한)", L / "시험" / "ko_큰시험2000.tsv"), ("과거 시험지", L / "시험" / "ko_과거시험1500.tsv")], "en": [("영어 새 시험지", L / "시험" / "en_새시험1500.tsv")]}
    bad = set(json.load(open(j.HOME / "실험" / "결과" / "큰시험_학습겹침.json", encoding="utf-8")))
    for lang, (name, heads, cols) in CFG.items():
        c = OUT[lang]["후보"]
        if not c:
            continue
        for tn, path in TESTS[lang]:
            rows = tsv(path)
            if "큰시험" in path.name:
                rows = [r for i, r in enumerate(rows) if i not in bad]
            texts = [r[-1] for r in rows]
            for h, col in zip(heads, cols):
                d, y = prep(name, texts, h), np.array([float(r[col]) for r in rows])
                a, b = metrics(apply(d, VARS[NOW]), y), metrics(apply(d, VARS[c]), y)
                passed = b["방향틀림3"] <= a["방향틀림3"] + 1 and b["7단계"] >= a["7단계"]
                OUT[lang].setdefault("시험", {})[f"{tn} {h}"] = {"지금": a, c: b, "통과": passed}
                print(f"[시험 {lang} {tn} 관점 {h}] 지금 {a} | {c} {b} | {'통과' if passed else '불통과'}")
json.dump(OUT, open(j.HOME / "실험" / "결과" / "안전장치.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
