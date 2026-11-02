# 모델 점검 (2026-10-31, docs/실험기록.md '지수·모델 점검 묶음' 3·4·6·8번). 판정 창 주력(ko_v2s, en_v2s)으로:
# 3 안전장치 켬/끔, 6 사전 특징 0(추론 때), 8 Macro-F1·평균 절대 차이·부호 정확도·판정률·ECE, 4 안전장치가 0으로 바꾼 제목의 정답 분포.
# python 실험/모델점검.py
import sys, json
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
import evaluate as e
sys.stdout.reconfigure(encoding="utf-8")
V = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
L = j.HOME / "labeled"
tsv = lambda p: [l.rstrip("\n").split("\t") for l in open(p, encoding="utf-8").read().splitlines()[1:] if l.strip()]
OUT = {}
_pf = j.phrase_features


def run(name, texts, head, nolex=False):
    j.phrase_features = (lambda st, lex, hint: np.zeros_like(_pf(st[:1], lex, hint)).repeat(len(st), 0)) if nolex else _pf
    try:
        P, meta = j.predict(name, texts, head) if head is not None else j.predict(name, texts)
    finally:
        j.phrase_features = _pf
    ts = [j.clean(t) for t in texts]
    raw = j.score_of(P, meta)
    g, ch = j.guard(ts, raw, e.GUARD, meta.get("guard_lexicon"))
    lv = j.level_only(ts)
    return P, {"켬": np.where(lv, 0, j.soften(ts, g)[0]), "끔": np.where(lv, 0, j.soften(ts, raw)[0])}, ch & ~lv, raw


def f1(yp, yt, labels):
    fs = []
    for c in labels:
        tp, fp, fn = np.sum((yp == c) & (yt == c)), np.sum((yp == c) & (yt != c)), np.sum((yp != c) & (yt == c))
        fs.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan)
    return float(np.nanmean(fs))


def metrics(s, y):
    m = e.screen(s, y); s7 = V[np.abs(s[:, None] - V).argmin(1)]; nz = y != 0
    sg = lambda v: np.sign(np.where(np.abs(v) >= 0.25, v, 0))
    return {"7단계": round(float(np.mean(s7 == y)), 4), "3분류": round(float(np.mean(sg(s) == np.sign(y))), 4),
            "MacroF1_7": round(f1(s7, y, V), 4), "MacroF1_3": round(f1(sg(s), np.sign(y), [-1, 0, 1]), 4),
            "평균절대차이": round(float(np.abs(s - y).mean()), 3), "0아닌_부호정확도": round(float(np.mean(np.sign(s[nz]) == np.sign(y[nz]))), 4),
            "판정률": round(float(np.mean(np.abs(s) >= 0.25)), 4), "맞음": round(m["맞음"], 4), "헛경보": round(m["헛경보"], 4), "놓침": round(m["놓침"], 4),
            "방향틀림3": int(((np.abs(s) >= 0.25) & nz & (np.sign(s) * np.sign(y) < 0)).sum())}


def ece(P, y, bins=10):
    yi = np.abs(y[:, None] - V).argmin(1); conf, pred = P.max(1), P.argmax(1)
    out = 0.0
    for b in range(bins):
        s = (conf > b / bins) & (conf <= (b + 1) / bins)
        if s.any():
            out += s.mean() * abs((pred[s] == yi[s]).mean() - conf[s].mean())
    return round(float(out), 4)


SETS = [("ko", "ko_v2s-감성", None, "한국어 Opus 검증", L / "ko_opus검증1000.tsv", 0),
        ("ko", "ko_v2s-감성", None, "한국어 큰 시험지(깨끗한)", L / "시험" / "ko_큰시험2000.tsv", 0),
        ("en", "en_v2s-감성", 0, "영어 Opus 검증 미국", L / "en_opus검증1000.tsv", 0),
        ("en", "en_v2s-감성", 1, "영어 Opus 검증 한국", L / "en_opus검증1000.tsv", 1)]
bad = set(json.load(open(j.HOME / "실험" / "결과" / "큰시험_학습겹침.json", encoding="utf-8")))
for lang, name, head, title, path, col in SETS:
    rows = tsv(path)
    if "큰시험" in path.name:
        rows = [r for i, r in enumerate(rows) if i not in bad]
    texts, y = [r[-1] for r in rows], np.array([float(r[col]) for r in rows])
    P, S, ch, raw = run(name, texts, head)
    _, Sn, _, _ = run(name, texts, head, nolex=True)
    R = OUT[title] = {"n": len(y), "ECE": ece(P, y)}
    for k, s in (("현재", S["켬"]), ("안전장치 끔", S["끔"]), ("사전특징 0", Sn["켬"]), ("사전특징 0 + 안전장치 끔", Sn["끔"])):
        R[k] = metrics(s, y)
    # 4 안전장치가 0으로 바꾼 제목: 정답이 0인가, 모델 방향이 맞았나, 규칙 방향이 맞았나
    yc, rc = y[ch], np.sign(raw[ch])
    R["안전장치_바뀐제목"] = {"개수": int(ch.sum()), "정답0": int((yc == 0).sum()), "모델방향이_맞음": int(((yc != 0) & (np.sign(yc) == rc)).sum()),
                         "모델방향이_틀림(뒤집힘 막음)": int(((yc != 0) & (np.sign(yc) == -rc)).sum())}
    print(f"\n== {title} ({len(y)}개, ECE {R['ECE']})")
    for k in ("현재", "안전장치 끔", "사전특징 0", "사전특징 0 + 안전장치 끔"):
        print(f"  {k:16} {R[k]}")
    print(f"  안전장치가 0으로 바꾼 제목: {R['안전장치_바뀐제목']}")

# 4 더 많은 표본: Opus 라벨이 붙은 모음 전부 (학습에 쓴 제목이라 모델 점수는 정답 쪽으로 기울어 있음, 참고)
for lang, name, head, files, col in (("ko", "ko_v2s-감성", None, ["ko_ai_opus.tsv", "ko_ai_opus_과거.tsv", "ko_ai_opus_과거2.tsv", "ko_ai_opus_과거3.tsv"], 0),
                                     ("en", "en_v2s-감성", 1, ["en_ai_opus_pool.tsv", "en_ai_opus_pool2.tsv", "en_ai_opus_과거.tsv"], 1)):
    lab = {}
    for f in files:
        for r in tsv(L / f):
            lab[r[-1]] = float(r[col])
    texts = list(lab); y = np.array([lab[t] for t in texts])
    _, _, ch, raw = run(name, texts, head)
    yc, rc = y[ch], np.sign(raw[ch])
    OUT[f"{lang} Opus 모음 안전장치"] = d = {"제목": len(y), "바뀜": int(ch.sum()), "정답0": int((yc == 0).sum()), "모델방향이_맞음": int(((yc != 0) & (np.sign(yc) == rc)).sum()),
                                        "모델방향이_틀림(뒤집힘 막음)": int(((yc != 0) & (np.sign(yc) == -rc)).sum())}
    print(f"\n4 {lang} Opus 모음 (학습 제목 포함, 참고): {d}")
json.dump(OUT, open(j.HOME / "실험" / "결과" / "모델점검.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
