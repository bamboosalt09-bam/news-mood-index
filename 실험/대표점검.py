# 화면 '끌어내린·끌어올린 묶음' 대표 제목의 방향 점검 (2026-11-02): 최근 40 지수일, 날마다 위·아래 5개씩을 Haiku(labeler)가 호재/악재/중립으로 다시 봄
# python 실험/대표점검.py make  → 실험/결과/대표점검/묶음_<n>.txt + 대표점검.tsv
# python 실험/대표점검.py score → 실험/결과/대표점검_haiku.json으로 방향 틀림 비율 (그날 지수 z 위·아래로 나눠)
import sys, json, collections, random
from pathlib import Path
HOME = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HOME / "web"), str(HOME)]
import numpy as np
import build as b

OUT = HOME / "실험" / "결과" / "대표점검"
TSV = HOME / "실험" / "결과" / "대표점검.tsv"
HAIKU = HOME / "실험" / "결과" / "대표점검_haiku.json"
N, SIZE = 40, 100
from repcheck import GUIDE, VIEW  # build의 대표 확인과 같은 지시 (점검 판정을 캐시 씨앗으로 씀)


def make():
    rows, items = ["번호\t언어\t날짜\t쪽\t점수\t건수\t그날z\t제목"], []
    for lang in ("ko", "en"):
        c = b.CONF[lang]
        days, C, chg = b.prices(c["price"])
        IT = b.items(lang, c["col"])
        by = b.assign(lang, days, IT)
        D = [d for d in days if len(by.get(d, [])) >= b.MIN_DAY]
        mood = [float(np.mean([it["s"] for _, it, _ in by[d]])) for d in D]
        for i in range(len(D) - N, len(D)):
            d, z = D[i], b.rel(mood, i)
            cl = b.clusters(by[d])
            neg = [x for x in sorted(cl, key=lambda x: x["contribution"]) if x["contribution"] < 0][:5]
            pos = [x for x in sorted(cl, key=lambda x: -x["contribution"]) if x["contribution"] > 0][:5]
            for side, lst in (("내림", neg), ("올림", pos)):
                for x in lst:
                    k = len(items); items.append((lang, x["title"]))
                    rows.append(f"{k}\t{lang}\t{d}\t{side}\t{x['score']}\t{x['count']}\t{'' if z is None else round(z, 2)}\t{x['title']}")
    TSV.write_text("\n".join(rows) + "\n", encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for lang in ("ko", "en"):
        mine = [(k, t) for k, (l, t) in enumerate(items) if l == lang]
        random.Random(lang).shuffle(mine)  # 날마다 아래 5·위 5 순서라 순서로 짐작하지 않게 섞음
        for s in range(0, len(mine), SIZE):
            body = "\n".join(f"{k}\t{t}" for k, t in mine[s:s + SIZE])
            (OUT / f"묶음_{n}.txt").write_text(GUIDE.format(view=VIEW[lang]) + body + "\n", encoding="utf-8"); n += 1
    print(len(items), "제목,", n, "묶음")


def score():
    hk = {int(k): v for k, v in json.loads(HAIKU.read_text(encoding="utf-8")).items()}
    rows = [l.split("\t") for l in TSV.read_text(encoding="utf-8").splitlines()[1:]]
    cnt = collections.defaultdict(collections.Counter)
    bad = []
    for k, lang, d, side, s, n, z, t in rows:
        h = hk.get(int(k))
        if h is None:
            continue
        day = "내린 날(z<-0.5)" if z and float(z) < -0.5 else "오른 날(z>0.5)" if z and float(z) > 0.5 else "보통 날"
        wrong = (side == "올림" and h == "악재") or (side == "내림" and h == "호재")
        for key in ((lang, side, "전체"), (lang, side, day)):
            cnt[key]["틀림" if wrong else "중립" if h == "중립" else "맞음"] += 1
        if wrong:
            bad.append((lang, d, side, s, n, t))
    for key in sorted(cnt):
        c = cnt[key]; tot = sum(c.values())
        print(*key, f"n {tot}, 방향 틀림 {c['틀림']} ({c['틀림'] / tot:.0%}), 중립 {c['중립']}")
    print()
    for x in bad:
        print(*x)


if __name__ == "__main__":
    {"make": make, "score": score}[sys.argv[1]]()
