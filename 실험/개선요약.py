# 2026-11-02 개선 요약: 오늘 아침 주력(ko_v2s·en_v2s, 변환 없음) 대 지금(변환기 사전 + 영어 en_v2t)을 가진 판단 자료 전부에서.
# python 실험/개선요약.py
import sys, json
from pathlib import Path
HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))
import numpy as np
import judge as j, evaluate as e, rewrite
import importlib.util as u
sp = u.spec_from_file_location("bp", HOME / "실험" / "변환점검.py"); bp = u.module_from_spec(sp); sp.loader.exec_module(bp)
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
LAB, R = HOME / "labeled", HOME / "실험" / "결과"
NOW, APPLY = dict(j.MAIN), rewrite.apply
seven = lambda s, y: float(np.mean(bp.V[np.abs(s[:, None] - bp.V).argmin(1)] == y))


def use(state):
    if state == "전":
        j.MAIN.update({"ko-감성": "ko_v2s-감성", "en-감성": "en_v2s-감성"}); rewrite.apply = lambda t, lang: list(t)
    else:
        j.MAIN.update(NOW); rewrite.apply = APPLY


def haiku_set(tsv, js):  # (제목, 호재 +1 / 악재 −1 / 중립 0)
    t = tsv.read_text(encoding="utf-8").splitlines(); hk = json.loads(js.read_text(encoding="utf-8"))
    return t, np.array([{"호재": 1, "악재": -1, "중립": 0}.get(hk.get(str(k)), 0) for k in range(len(t))], float)


rows700 = [r for f in e.TESTS["무작위700 (최종)"] for r in bp.tsv(LAB / "시험" / f)]
hr = bp.tsv(LAB / "en_ai_haiku.tsv"); hv = [hr[i] for i in np.random.default_rng(0).permutation(len(hr))[:len(hr) // 10]]
old303 = [r for f in ("en_test.tsv", "en_newtest.tsv") for r in bp.tsv(LAB / f)]
new1500 = bp.tsv(LAB / "시험" / "en_새시험1500.tsv")
rep = [l.split("\t") for l in (R / "대표점검.tsv").read_text(encoding="utf-8").splitlines()[1:]]
rh = json.loads((R / "대표점검_haiku.json").read_text(encoding="utf-8"))
out = {}
for state in ("전", "후"):
    use(state); o = out[state] = {}
    t, y = [r[-1] for r in rows700], np.array([float(r[0]) for r in rows700]); m = e.screen(bp.scores("ko", t), y)
    o["한국어 무작위 700"] = f"맞음 {m['맞음']:.1%} · 헛경보 {m['헛경보']:.1%} · 방향 틀림 {m['방향 틀림']}"
    for name, rr in (("영어 Haiku 검증 2,010", hv), ("옛 영어 303", old303), ("영어 새 시험지 1,500 (참고)", new1500)):
        t, y = [r[-1] for r in rr], np.array([float(r[0]) for r in rr]); s = bp.scores("en", t, head=0)
        o[name] = f"7단계 {seven(s, y):.1%} · 방향 틀림 {int(bp.flips(s, y).sum())}"
    for lang in ("ko", "en"):
        for name, d in (("변환 이득 표본", R / "변환" / "이득"), ("방향 말 이득 표본", R / "방향말조정")):
            t, y = haiku_set(d / f"{lang}.tsv", d / f"haiku_{lang}.json"); s = bp.scores(lang, t, head=0)
            ok = int(((np.sign(s) == y) & (np.abs(s) >= 0.25) & (y != 0)).sum())
            o[f"{'한국어' if lang == 'ko' else '영어'} {name} ({len(t)})"] = f"방향 틀림 {int(bp.flips(s, y).sum())} · 방향 맞음 {ok}"
        rr = [r for r in rep if r[1] == lang and r[0] in rh]
        t, y = [r[-1] for r in rr], np.array([{"호재": 1, "악재": -1, "중립": 0}[rh[r[0]]] for r in rr], float); s = bp.scores(lang, t, head=0)
        o[f"{'한국어' if lang == 'ko' else '영어'} 대표 제목 400 (만들 때 봄)"] = f"방향 틀림 {int(bp.flips(s, y).sum())}"
use("후")
for k in out["전"]:
    print(f"{k:32} | {out['전'][k]:38} → {out['후'][k]}")
