# 대표 제목 방향 확인 (2026-11-02, docs/실험기록.md 같은 날 '대표 제목 방향 점검'): 화면에 올릴 대표 제목 후보를
# claude-haiku-5-5가 그 시장 투자자 관점으로 호재/악재/중립만 다시 본다. 끌어올린 쪽인데 악재, 끌어내린 쪽인데 호재면 build가 빼고 다음 후보를 올린다.
# 제목 단위 캐시 (지난 판정은 다시 안 부름). API 자격이 없으면 캐시에 있는 것만 쓰고 나머지는 거르지 않음.
import os, json
import judge as j

MODEL = "claude-haiku-5-5"
CACHE = j.HOME / "data" / "cache" / "대표확인.json"
WAIT = j.HOME / "data" / "cache" / "대표확인_대기.tsv"  # 자격 없을 때 판정 못 한 후보 (언어, 제목) → 손으로 판정해 CACHE에 넣을 수 있게
LABELS = ("호재", "악재", "중립")
VIEW = {"ko": "한국 주식 투자자(코스피)", "en": "미국 주식 투자자(S&P 500)"}
GUIDE = """너는 경제 뉴스 제목 판정가다. 아래 제목마다, 그 제목이 보도한 내용이 {view}에게 그날 호재인지 악재인지 고른다.
- 제목에 적힌 사실만 본다. 주가가 이미 오르거나 내렸다고 적혀 있으면 그 방향이다 (예: 'wipe out $1 trillion' = 시가총액이 사라짐 = 악재, '손실을 만회' = 호재).
- 업체 하나의 소식이면 그 업체 주가 쪽으로 본다. 영향이 없거나 반반이면 중립.
- label: '호재' | '악재' | '중립'. 번호(id)를 그대로 돌려준다.

# 제목
"""


def load():
    return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


def check(lang, titles):
    """titles → {제목: '호재'|'악재'|'중립'|None}. None = 판정 없음 (거르지 않음)."""
    cache = load()
    key = lambda t: f"{lang}|{t}"
    todo = list(dict.fromkeys(t for t in titles if key(t) not in cache))
    if todo and (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        schema = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["id", "label"],
            "properties": {"id": {"type": "integer"}, "label": {"type": "string", "enum": list(LABELS)}}}}}}
        try:
            import anthropic
            r = anthropic.Anthropic().messages.create(
                model=MODEL, max_tokens=4000,
                messages=[{"role": "user", "content": GUIDE.format(view=VIEW[lang]) + "\n".join(f"{k}\t{t}" for k, t in enumerate(todo)) + "\n"}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}})
            if r.stop_reason == "end_turn":
                for x in json.loads(next(b.text for b in r.content if b.type == "text"))["items"]:
                    if 0 <= x["id"] < len(todo):
                        cache[key(todo[x["id"]])] = x["label"]
                CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
            else:
                print(f"  대표 확인 {lang}: 멈춤 이유 {r.stop_reason} → 거르지 않음")
        except Exception as e:  # 키가 메시지에 섞이지 않게 종류만 출력
            print(f"  대표 확인 {lang}: 호출 실패 ({type(e).__name__}) → 거르지 않음")
    elif todo:
        print(f"  대표 확인 {lang}: 판정 없는 후보 {len(todo)}개, API 자격 없음 → 그 제목은 거르지 않음")
        old = set(WAIT.read_text(encoding="utf-8").splitlines()) if WAIT.exists() else set()
        with open(WAIT, "a", encoding="utf-8") as f:
            f.writelines(f"{lang}\t{t}\n" for t in todo if f"{lang}\t{t}" not in old)
    return {t: cache.get(key(t)) for t in titles}


def keep(lst, ok, sign, k):  # 후보 lst(기여 순)에서 방향이 반대로 판정된 것을 빼고 앞 k개, 각자 판정을 'check'에 붙임
    bad = "악재" if sign > 0 else "호재"
    return [dict(x, check=ok.get(x["title"])) for x in lst if ok.get(x["title"]) != bad][:k]
