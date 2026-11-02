# 환율 기억 규칙 판단 (2026-11-02 사전 등록). python 실험/환율기억점검.py make | score | safety | index
import sys, json, random
from pathlib import Path
HOME = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HOME), str(HOME / "web")]
import numpy as np
import judge as j, evaluate as e, fxmemory as fx
import importlib.util as u
sp = u.spec_from_file_location("mz", HOME / "실험" / "방향말조정.py"); mz = u.module_from_spec(sp); sp.loader.exec_module(mz)
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
LAB, G = HOME / "labeled", HOME / "실험" / "결과" / "환율기억"
GUIDE = """너는 경제 뉴스 제목 판정가다. 아래 제목마다, 그 제목이 보도한 내용이 한국 주식 투자자(코스피)에게 그날 호재인지 악재인지 고른다.
- 제목 뒤 [ ] 안에 그 제목이 나오기 전날의 원·달러 환율을 적어 두었다. 제목의 환율 숫자를 이것과 견줘 환율이 오른 소식인지 내린 소식인지 판단해도 된다.
  원·달러 환율 상승(원화 약세)은 대체로 코스피에 악재, 하락(원화 강세)은 호재로 본다. 다만 제목이 다른 방향을 분명히 말하면 그것을 따른다.
- 제목에 적힌 사실만 본다. 영향이 없거나 반반이면 중립.
- label: '호재' | '악재' | '중립'. 번호(id)를 그대로 돌려준다.

# 제목
"""


def cache():
    rows = [l.rstrip("\n").split("\t") for l in open(HOME / "data" / "cache" / "지수자료_ko.tsv", encoding="utf-8")]
    k = {c: i for i, c in enumerate(rows[0])}
    return [(r[-1], float(r[k["최종"]]), r[k["수준만"]] == "1", r[k["전망표지"]] == "1") for r in rows[1:]]


def first_day():
    d = {}
    for f in ("daily", "sector"):
        for l in open(HOME / "headlines" / f"ko_{f}.tsv", encoding="utf-8"):
            p = l.rstrip("\n").split("\t")
            if len(p) > 2 and p[0][:2] == "20":
                d[p[2]] = min(d.get(p[2], p[0][:10]), p[0][:10])
    return d


def changed():
    day = first_day()
    out = []
    for t, s, lv, op in cache():
        if t in day:
            a = fx.adjust(t, day[t], s, lv, op)
            if a:
                out.append((t, day[t], s, lv, a[0]))
    return out


def make():
    seen = mz.judged() | {l.split("\t")[-1] for f in LAB.rglob("*.tsv") for l in open(f, encoding="utf-8").read().splitlines()[1:]}
    ch = [x for x in changed() if x[0] not in seen]
    print("규칙이 점수를 주는 제목", len(ch), "(수준만", sum(x[3] for x in ch), ")")
    pick = random.Random(2044).sample(ch, min(300, len(ch)))
    G.mkdir(parents=True, exist_ok=True)
    (G / "표본.json").write_text(json.dumps(pick, ensure_ascii=False), encoding="utf-8")
    mix = list(enumerate(pick)); random.Random("ko").shuffle(mix)
    for q in range(3):
        body = "".join(f"{k}\t{t} [전날 원·달러 {fx.before(d):,.1f}원]\n" for k, (t, d, *_) in mix[q * 100:(q + 1) * 100])
        (G / f"묶음_ko_{q}.txt").write_text(GUIDE + body, encoding="utf-8")


def score():
    pick = json.loads((G / "표본.json").read_text(encoding="utf-8"))
    hk = json.loads((G / "haiku_ko.json").read_text(encoding="utf-8"))
    y = np.array([{"호재": 1, "악재": -1, "중립": 0}.get(hk.get(str(k)), 0) for k in range(len(pick))])
    r = np.array([np.sign(x[4]) for x in pick])
    n = len(pick); same, opp = int((r == y).sum()), int((r * y < 0).sum())
    print(f"표본 {n}: 규칙과 Haiku 같음 {same} ({same / n:.0%}), 반대 {opp} ({opp / n:.0%}), Haiku 중립 {int((y == 0).sum())}")
    print(f"  (지금 점수로는 Haiku와 같은 부호 |점수|≥0.25: {int(sum(1 for x, yy in zip(pick, y) if yy and abs(x[2]) >= 0.25 and np.sign(x[2]) == yy))})")
    for x, yy in zip(pick, y):
        if np.sign(x[4]) * yy < 0:
            print(f"   반대 Haiku {yy:+d} 규칙 {x[4]:+.2f} 전날 {fx.before(x[1]):,.0f} | {x[0][:80]}")


def safety():  # 무작위 700: 시험지 제목엔 발행일이 없어 headlines에서 찾음, 없으면 규칙 안 씀
    import importlib.util as u2
    sp = u2.spec_from_file_location("bp", HOME / "실험" / "변환점검.py"); bp = u2.module_from_spec(sp); sp.loader.exec_module(bp)
    rows = [r for f in e.TESTS["무작위700 (최종)"] for r in bp.tsv(LAB / "시험" / f)]
    t, y = [r[-1] for r in rows], np.array([float(r[0]) for r in rows])
    s = bp.scores("ko", t); day = first_day()
    lv, op = j.level_only(t), j.soften(t, np.zeros(len(t)))[1]
    s2 = s.copy(); n = 0
    for i, x in enumerate(t):
        a = fx.adjust(x, day[x], s[i], lv[i], op[i]) if x in day else None
        if a:
            s2[i] = a[0]; n += 1
    for tag, v in (("지금", s), ("환율 기억", s2)):
        m = e.screen(v, y); print(f"[{tag}] 무작위 700: 맞음 {m['맞음']:.1%} 헛경보 {m['헛경보']:.1%} 방향 틀림 {m['방향 틀림']} (규칙 적용 {n}개)")


def index():
    import build as b
    c = b.CONF["ko"]; days, C, chg = b.prices(c["price"])
    IT = b.items("ko", c["col"])
    ch = {t: s for t, _, _, _, s in changed()}
    by0 = b.assign("ko", days, IT)
    IT2 = dict(IT)
    rows = {t: (s, lv) for t, s, lv, _ in cache()}
    for t, s in ch.items():
        IT2[t] = dict(IT.get(t, {"mkt": False, "cause": False, "opin": False, "topic": "환율·달러"}), s=s)
    by1 = b.assign("ko", days, IT2)
    for tag, by in (("지금", by0), ("환율 기억", by1)):
        idx = [i for i, d in enumerate(days) if len(by.get(d, [])) >= b.MIN_DAY]
        m = np.array([np.mean([it["s"] for _, it, _ in by[days[i]]]) for i in idx]); yv = np.array([chg[i] for i in idx]); ok = np.isfinite(yv)
        print(f"[{tag}] 같은 날 상관 {np.corrcoef(m[ok], yv[ok])[0, 1]:+.4f} (지수일 {len(idx)}), 바뀐 제목 {len(ch)}")


if __name__ == "__main__":
    {"make": make, "score": score, "safety": safety, "index": index}[sys.argv[1]]()
