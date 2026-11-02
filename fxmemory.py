# 환율 기억 규칙 (2026-11-02, docs/실험기록.md 같은 날 사전 등록): 모델이 애매하게 본 한국어 환율 제목에서만,
# 제목의 원·달러 숫자를 '제목 발행일 전날까지의' 원·달러 값(data/원달러.tsv)과 견줘 방향을 정한다. 그날 값·주가는 쓰지 않는다.
import re, bisect, functools
from pathlib import Path
HOME = Path(__file__).resolve().parent
FX = re.compile(r"환율|원[·/\-]?달러|달러[·/\-]?원|원화")
NUM = re.compile(r"(?<![\d.,])(1[,.]?\d{3}(?:\.\d+)?)\s*원\s*(대)?\s*(초반|중반|후반)?")
PART = {"초반": 2, "중반": 5, "후반": 8}  # '1320원대' = 1320~1329원 → 가운데로 읽음
GAP, SCORE, VAGUE = 0.003, 0.5, 0.25


@functools.lru_cache(1)
def series():
    rows = [l.split("\t") for l in (HOME / "data" / "원달러.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    return [r[0] for r in rows], [float(r[1]) for r in rows]


def before(day):  # 발행일(YYYY-MM-DD)보다 앞선 마지막 날의 원·달러
    d, v = series()
    i = bisect.bisect_left(d, day) - 1
    return v[i] if i >= 0 else None


def sign(title, day):  # +1 (환율이 기억보다 낮음 = 호재) / −1 / 0 (판단 안 함)
    if not FX.search(title):
        return 0
    ns = [float(x.replace(",", "")) + (PART.get(p, 5) if (d or p) and x.replace(",", "")[-1] == "0" else 0) for x, d, p in NUM.findall(title)]
    ns = [n for n in ns if 1000 <= n <= 1999]
    R = before(day)
    if not ns or R is None:
        return 0
    s = {(-1 if n > R * (1 + GAP) else 1 if n < R * (1 - GAP) else 0) for n in ns}
    return s.pop() if len(s) == 1 else 0


def adjust(title, day, score, level, opinion):  # 바뀌면 (새 점수, False=수준만 뗌), 아니면 None
    if not (level or abs(score) < VAGUE):
        return None
    g = sign(title, day)
    if g == 0:
        return None
    return g * SCORE * (0.5 if opinion else 1.0), False


if __name__ == "__main__":  # 규칙 동작 확인 (기억 값은 실제 파일)
    d, v = series()
    day = d[-1]; R = before(day)
    assert R == v[-2], (R, v[-2])
    assert sign(f"원·달러 환율 {R * 1.02:,.0f}원 마감", day) == -1
    assert sign(f"원·달러 환율 {R * 0.98:,.0f}원대로", day) == 1
    assert sign(f"원·달러 환율 {R:,.0f}원 보합", day) == 0
    assert sign("코스피 2,500선 회복", day) == 0
    assert NUM.findall("환율 1320원대 후반") == [("1320", "대", "후반")] and sign(f"환율 {round(R) // 10 * 10}원대", day) == 0
    assert sign(f"환율 {R * 0.98:,.0f}원→{R * 1.02:,.0f}원 출렁", day) == 0
    assert adjust(f"환율 {R * 1.02:,.0f}원", day, 0.8, False, False) is None
    assert adjust(f"환율 {R * 1.02:,.0f}원", day, 0.1, False, True) == (-0.25, False)
    print("환율 기억 규칙 확인 통과, 기억 마지막", d[-2], R)
