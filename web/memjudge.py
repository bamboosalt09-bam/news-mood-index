# 업체 기억 판정 (2026-11-01, docs/실험기록.md '기억 라벨'·'업체·섹터 기억'): 같은 날 같은 업체 제목 묶음 + 전 거래일 제목을 함께 보여 주고
# claude-haiku-5-5가 '그 업체 주가에 좋은가'를 판정. 지난 1년은 labeled/ko_기억라벨.tsv(서브에이전트 Haiku), 새 날은 이 모듈이 API로 → data/cache/기억판정_ko.tsv.
# 판정 기준·묶음 형식은 라벨 만들 때(실험/기억라벨.py)와 같다. API 자격이 없으면 아무것도 안 하고, build.py는 판정 없는 제목에 지금 점수를 쓴다.
import os, re, json, collections
import judge as j

MODEL, SIZE = "claude-haiku-5-5", 60
CACHE = j.HOME / "data" / "cache" / "기억판정_ko.tsv"
LABELS = j.HOME / "labeled" / "ko_기억라벨.tsv"
CUE = re.compile(r"쇼크|충격|하회|상회|서프라이즈|깜짝|웃돌|못\s?미|급락|급등|폭락|폭등|상한가|하한가|%\s?(↑|↓|상승|하락|올|내)")
GUIDE = """너는 한국 경제 뉴스 제목에 '그 업체 주가에 좋은가'를 판정한다. 묶음마다 머리에 업체 이름이 있고, 그 업체 기준으로만 판정한다.
값은 -2, -1, -0.5, 0, 0.5, 1, 2 중 하나.

# 기준
- 관점: 머리 업체의 주가에 좋은 소식인가 나쁜 소식인가. 제목에 다른 업체가 함께 나와도 머리 업체 기준.
- 기억: 묶음마다 '같은 날 다른 제목(참고)'과 '전 거래일 제목(참고)'이 있다. 참고 제목은 판정하지 않고 기억으로만 쓴다.
  - 제목 하나로 보면 긍정인 실적 숫자(영업익 증가, 역대 최대 매출)라도, 같은 실적 발표에 대해 어닝 쇼크·기대(예상·컨센서스) 하회·실적 발표 뒤 주가 급락 제목이
    같은 날이나 전날에 있으면 그 사건의 실제 평가를 따라 부정으로 판정한다. 반대로 숫자는 나빠 보여도 어닝 서프라이즈·기대 상회·주가 급등이 있으면 긍정.
  - 기억은 같은 사건에만 쓴다. 같은 업체라도 다른 일(홍보, 사회공헌, 행사, 인사, 다른 사업 소식, 증권사 장기 전망)에는 옮기지 않는다.
- 홍보·사회공헌·행사·채용·광고·제품 소개처럼 주가와 거의 무관한 제목 = 0.
- '~에도' 꼴(예: "실적 쇼크에도 주가 상승")은 결과(주가) 쪽을 따른다.
- 증권사 목표가·투자의견·전망은 의견이므로 한 단계 약하게 (목표가 상향 = +0.5, 하향 = -0.5).
- 등급: ±1 = 분명한 호재·악재 (실적 서프라이즈·쇼크, 큰 수주·계약, 주가 급등락, 소송·파업·제재), ±0.5 = 방향은 있지만 약함, 0 = 중립·무관·엇갈림.
  ±2 = 하루 10% 넘는 급등락, 상한가·하한가, 존립이 걸린 일 같은 아주 큰 일만.
- 애매하면 0. 방향을 거꾸로 붙이는 것이 가장 나쁘다."""
V = ["-2", "-1", "-0.5", "0", "0.5", "1", "2"]


def block(c, d, todo, same, before):  # 묶음 머리 + 참고 제목 (같은 날은 쇼크·급등락 꼴 먼저 15개, 전날 5개)
    ref = sorted((t for t, m in same if t not in todo), key=lambda t: not CUE.search(t))[:15]
    s = f"## 업체: {c} · {d}\n"
    s += "전 거래일 제목 (참고, 판정 안 함):\n" + ("".join(f"  - {t}\n" for t, _ in before[:5]) or "  (없음)\n")
    s += "같은 날 다른 제목 (참고, 판정 안 함):\n" + ("".join(f"  - {t}\n" for t in ref) or "  (없음)\n")
    return s


def batches(items, G, prev):  # items: [(업체, 날, 제목)] → [(본문, [(날, 업체, 제목)])], 60개씩
    out = []
    for k in range(0, len(items), SIZE):
        seg = items[k:k + SIZE]; body, cur, rows = "", None, []
        for c, d, t in seg:
            if (c, d) != cur:
                cur = (c, d); todo = {tt for cc, dd, tt in seg if (cc, dd) == cur}
                body += "\n" + block(c, d, todo, G[c, d], G.get((c, prev.get(d)), [])) + "판정할 제목 (번호<TAB>제목):\n"
            rows.append((d, c, t)); body += f"{len(rows)}\t{t}\n"
        out.append((f"{GUIDE}\n\n아래 묶음들의 '판정할 제목' {len(rows)}개 모두에, 번호마다 하나씩 답한다.\n{body}", rows))
    return out


def load():  # {(날, 업체, 제목): 점수} 지난 라벨 + 새 판정
    out = {}
    for path in (LABELS, CACHE):
        if path.exists():
            for l in path.read_text(encoding="utf-8").splitlines()[1:]:
                d, c, y, t = l.split("\t", 3); out[d, c, t] = float(y)
    return out


def update(by, days, D, since):
    """since 이후 거래일의 업체 제목 중 판정이 없는 것만 Haiku로 판정해 CACHE에 덧붙인다. 판정한 제목 수를 돌려준다 (자격이 없으면 0)."""
    import sectors as S
    done = load()
    G = collections.defaultdict(list)
    E = [d for d in days if d >= since]
    prev = {d: days[days.index(d) - 1] for d in E}
    for d in [prev[E[0]]] + E if E else []:
        for t, it, _ in by.get(d, []):
            for c in S.tag(t, D, "ko")[1]:
                G[c, d].append((t, it["mkt"]))
    items = sorted({(c, d, t) for (c, d), xs in G.items() if d in prev for t, _ in xs if (d, c, t) not in done}, key=lambda x: (x[1], x[0]))
    if not items:
        return 0
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print(f"  업체 기억 판정: 판정 안 된 업체 제목 {len(items)}개, API 자격 없음 → 지금 점수로")
        return 0
    import anthropic
    cl, n = anthropic.Anthropic(), 0
    if not CACHE.exists():
        CACHE.write_text("날짜\t업체\t감성\t제목\n", encoding="utf-8")
    for text, rows in batches(items, G, prev):
        keys = [str(i + 1) for i in range(len(rows))]
        schema = {"type": "object", "additionalProperties": False, "required": keys, "properties": {k: {"type": "string", "enum": V} for k in keys}}
        try:
            r = cl.messages.create(model=MODEL, max_tokens=4000, messages=[{"role": "user", "content": text}],
                                   output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}})
            if r.stop_reason != "end_turn":
                print(f"  업체 기억 판정: 멈춤 이유 {r.stop_reason}, 이 묶음 건너뜀"); continue
            o = json.loads(next(b_.text for b_ in r.content if b_.type == "text"))
        except Exception as e:  # 키가 메시지에 섞이지 않게 종류만 출력
            print(f"  업체 기억 판정: 호출 실패 ({type(e).__name__}), 이 묶음 건너뜀"); continue
        with open(CACHE, "a", encoding="utf-8") as f:
            f.writelines(f"{d}\t{c}\t{o[str(i + 1)]}\t{t}\n" for i, (d, c, t) in enumerate(rows))
        n += len(rows)
    print(f"  업체 기억 판정: {n}/{len(items)}개 판정 ({MODEL})")
    return n
