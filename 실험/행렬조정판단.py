# 낱말 행렬 넓은 미세 조정(v2u) 판단 (2026-11-02 사전 등록): 지금 주력(한국어 ko_v2s, 영어 en_v2t, 둘 다 변환기) 대 v2u.
# python 실험/행렬조정판단.py safety          → 일반 시험지 (해 없음)
# python 실험/행렬조정판단.py gain_make       → 이득 표본 (부호 −1 대상 말 + 방향 말이 함께 든 캐시 제목, 시드 2043)
# python 실험/행렬조정판단.py gain_score      → 이득 채점 (Haiku 판정 haiku_<언어>.json 필요)
# python 실험/행렬조정판단.py index           → 지수 같은 날 상관 (지금 대 v2u)
import sys, json, random
from pathlib import Path
HOME = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HOME), str(HOME / "web")]
import numpy as np
import judge as j, evaluate as e, rewrite
import importlib.util as u
sp = u.spec_from_file_location("bp", HOME / "실험" / "변환점검.py"); bp = u.module_from_spec(sp); sp.loader.exec_module(bp)
sp = u.spec_from_file_location("mz", HOME / "실험" / "방향말조정.py"); mz = u.module_from_spec(sp); sp.loader.exec_module(mz)
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
LAB = HOME / "labeled"
NOW = {"ko": "ko_v2s-감성", "en": "en_v2t-감성"}
CAND = {"ko": "ko_v2u-감성", "en": "en_v2u-감성"}
G = HOME / "실험" / "결과" / "방향말조정2"
seven = lambda s, y: float(np.mean(bp.V[np.abs(s[:, None] - bp.V).argmin(1)] == y))


def use(names):
    j.MAIN.update({f"{l}-감성": n for l, n in names.items()})


def safety():
    rows = [r for f in e.TESTS["무작위700 (최종)"] for r in bp.tsv(LAB / "시험" / f)]
    hr = bp.tsv(LAB / "en_ai_haiku.tsv"); hv = [hr[i] for i in np.random.default_rng(0).permutation(len(hr))[:len(hr) // 10]]
    olds = [r for f in ("en_test.tsv", "en_newtest.tsv") for r in bp.tsv(LAB / f)]
    new = bp.tsv(LAB / "시험" / "en_새시험1500.tsv")
    for tag, names in (("지금", NOW), ("v2u", CAND)):
        use(names)
        t, y = [r[-1] for r in rows], np.array([float(r[0]) for r in rows]); m = e.screen(bp.scores("ko", t), y)
        ok = {"<": lambda a, b: a < b, "=": lambda a, b: abs(a - b) < 0.05, ">0": lambda a, b: a > 0, "<0": lambda a, b: a < 0, ">=0": lambda a, b: a >= -0.1, "<=0": lambda a, b: a <= 0.1}
        fails = [d for d, a, b, rel in e.BEHAVIOR if not ok[rel](bp.scores("ko", [a])[0], bp.scores("ko", [b])[0] if b else None)]
        print(f"[{tag}] 한국어 700: 맞음 {m['맞음']:.1%} 헛경보 {m['헛경보']:.1%} 놓침 {m['놓침']:.1%} 방향 틀림 {m['방향 틀림']}, 기본 행동 {16 - len(fails)}/16 {fails}")
        for name, rr in (("영어 Haiku 검증 2,010", hv), ("옛 영어 303", olds), ("영어 새 시험지 1,500 (참고)", new)):
            t, y = [r[-1] for r in rr], np.array([float(r[0]) for r in rr]); s = bp.scores("en", t, head=0); m = e.screen(s, y)
            print(f"[{tag}] {name}: 7단계 {seven(s, y):.1%} 맞음 {m['맞음']:.1%} 헛경보 {m['헛경보']:.1%} 방향틀림3 {int(bp.flips(s, y).sum())}")
    use(NOW)


def gain_make():
    import repcheck as rc
    seen = mz.judged() | {l.split("\t")[-1] for f in LAB.rglob("*.tsv") for l in open(f, encoding="utf-8").read().splitlines()[1:]}
    G.mkdir(parents=True, exist_ok=True)
    for lang in ("ko", "en"):
        p, T, meta = j._load(str(HOME / "models" / f"{lang}_v2s-감성.npz"), 0)
        lex, voc = meta["lexicon"], meta["vocab"]
        key = (lambda k: k.split("/")[0]) if lang == "ko" else (lambda k: k)
        neg = {w.strip().lower() for k, v in lex.items() if k.startswith("대상:") and k.endswith(":-1") for w in v}
        dirw = {w.strip().lower() for k, v in lex.items() if k in j.STEP for w in v} | {key(w) for w in mz.WORDS[lang]}
        tset = {i for k, i in voc.items() if key(k) in neg}; dset = {i for k, i in voc.items() if key(k) in dirw}
        t = sorted({l.rstrip("\n").split("\t")[-1] for l in open(HOME / "data" / "cache" / f"지수자료_{lang}.tsv", encoding="utf-8")} - seen - {"제목"})
        C = mz.ids_of(lang, rewrite.apply([j.clean(x) for x in t], lang), meta)
        hit = [x for x, c in zip(t, C) if tset & set(c.tolist()) and dset & set(c.tolist())]
        pick = random.Random(2043).sample(hit, min(300, len(hit)))
        (G / f"{lang}.tsv").write_text("".join(x + "\n" for x in pick), encoding="utf-8")
        mix = list(enumerate(pick)); random.Random(lang).shuffle(mix)
        for q in range(3):
            (G / f"묶음_{lang}_{q}.txt").write_text(rc.GUIDE.format(view=rc.VIEW[lang]) + "".join(f"{k}\t{x}\n" for k, x in mix[q * 100:(q + 1) * 100]), encoding="utf-8")
        print(lang, "대상(−1)+방향 말 제목", len(hit), "→ 표본", len(pick))


def gain_score():
    for lang in ("ko", "en"):
        t = (G / f"{lang}.tsv").read_text(encoding="utf-8").splitlines()
        hk = json.loads((G / f"haiku_{lang}.json").read_text(encoding="utf-8"))
        y = np.array([{"호재": 1, "악재": -1, "중립": 0}.get(hk.get(str(k)), 0) for k in range(len(t))], float)
        use(NOW); a = bp.scores(lang, t, head=0)
        use(CAND); b = bp.scores(lang, t, head=0)
        use(NOW)
        fa, fb = bp.flips(a, y), bp.flips(b, y)
        ok = lambda s: int(((np.sign(s) == y) & (np.abs(s) >= 0.25) & (y != 0)).sum())
        print(f"[{lang}] 표본 {len(t)}, Haiku 호재 {int((y > 0).sum())} 악재 {int((y < 0).sum())} 중립 {int((y == 0).sum())}")
        print(f"  방향 틀림: 지금 {int(fa.sum())} → v2u {int(fb.sum())}, 고쳐짐 {int((fa & ~fb).sum())}, 새로 틀림 {int((~fa & fb).sum())}; 방향 맞음 {ok(a)} → {ok(b)}")
        for i in np.where(fa | fb)[0]:
            print(f"   {'고침' if fa[i] and not fb[i] else '새로' if fb[i] and not fa[i] else '남음'} {y[i]:+.0f} {a[i]:+.2f}→{b[i]:+.2f} {t[i][:90]}")


def index():  # 지금 캐시(지수자료_<언어>.tsv) 대 v2u로 다시 매긴 점수 → 하루 평균 ↔ 등락 같은 날 상관
    import build as b
    for lang in ("ko", "en"):
        c = b.CONF[lang]; days, C, chg = b.prices(c["price"])
        IT = b.items(lang, c["col"])
        by = b.assign(lang, days, IT)
        idx = [i for i, d in enumerate(days) if len(by.get(d, [])) >= b.MIN_DAY]
        ts = sorted({t for i in idx for t, _, _ in by[days[i]]})
        use(CAND)
        P, meta = j.predict(CAND[lang], ts, **({"head": 0} if lang == "en" else {}))
        g = j.guard(ts, j.score_of(P, meta), meta.get("guard"), meta.get("guard_lexicon"))[0]
        new = dict(zip(ts, j.soften(ts, g)[0]))
        use(NOW)
        y = np.array([chg[i] for i in idx]); okm = np.isfinite(y)
        m0 = np.array([np.mean([it["s"] for _, it, _ in by[days[i]]]) for i in idx])
        m1 = np.array([np.mean([new[t] for t, _, _ in by[days[i]]]) for i in idx])
        print(f"[{lang}] 같은 날 상관 지금 {np.corrcoef(m0[okm], y[okm])[0, 1]:+.3f} → v2u {np.corrcoef(m1[okm], y[okm])[0, 1]:+.3f}, 전후 지수 상관 {np.corrcoef(m0, m1)[0, 1]:.4f}")


if __name__ == "__main__":
    {"safety": safety, "gain_make": gain_make, "gain_score": gain_score, "index": index}[sys.argv[1]]()
