# 섹터 동향 판정 (2026-11-01, docs/실험기록.md 같은 날 '섹터 동향은 Haiku로'): 그날 섹터마다 서로 다른 소식(기사 묶음 대표 제목·건수·제목 점수)만 보고
# claude-haiku-5-5가 세 단계 라벨과 근거 한 문장을 낸다. 주가는 주지 않는다 (카드에서 '뉴스는 이렇게, 주가는 이렇게'를 나란히 보이려고).
# API 자격이 없거나 실패하면 규칙 라벨(model = "dry-run"). (날, 시장) 단위로 캐시해 같은 날 다시 부르지 않는다.
import os, json
import judge as j

MODEL = "claude-haiku-5-5"
# 10/30 시범(docs/실험기록.md 2026-11-01)에서 방향 틀림 2/11 (규칙 라벨 0) → 여러 날로 확인하기 전까지 꺼 둠. SECTOR_AI=1이면 켬
ON = os.environ.get("SECTOR_AI") == "1"
LABELS = ("좋은 소식 많음", "엇갈림", "나쁜 소식 많음")
CACHE = j.HOME / "data" / "cache" / "섹터판정.json"
SYSTEM = """너는 경제 뉴스 섹터 동향 판정가다. 사용자가 한 시장의 그날 섹터별 주요 소식을 JSON으로 준다.
소식마다 대표 제목, 같은 내용 기사 건수, 제목 점수(−2~+2, 그 업체·섹터에 좋으면 +)가 있다. 섹터마다 그날 뉴스가 그 섹터 주가에 좋은 소식이 많았는지 판정한다.
규칙:
- 서로 다른 소식을 기준으로 판단한다. 건수는 얼마나 크게 보도됐는지 참고할 뿐이다. 같은 소식이 여러 번 나와도 소식 하나다.
- 섹터 전체나 그 섹터 대형 업체에 영향이 큰 소식에 무게를 둔다. 홍보·행사·사회공헌·인사 같은 소식은 무시한다.
- 좋은 소식과 나쁜 소식이 비슷하거나, 방향 있는 소식이 작고 적으면 '엇갈림'.
- 주어진 제목에 있는 내용만 쓴다. 예측·투자 권유를 하지 않는다. 주가가 어떻게 될지 말하지 않는다.
- label: '좋은 소식 많음' | '엇갈림' | '나쁜 소식 많음' 중 하나. line: 판단 근거 한 문장 (제목의 소식을 짧게, '~가 보도됐다' 꼴, 40자 안팎)."""


def stories(cl, k=15):  # 기사 묶음 → 서로 다른 소식 (건수 많은 것, 그다음 점수 큰 것)
    top = sorted(cl, key=lambda x: (-x["count"], -abs(x["score"])))[:k]
    return [{"제목": x["title"], "건수": x["count"], "점수": x["score"]} for x in top]


def rule(z, mood):  # 자격 없을 때: 평소 대비(z)와 그날 점수 부호가 같을 때만 좋은/나쁜
    if z is not None and z > 0.5 and mood > 0:
        return "좋은 소식 많음"
    if z is not None and z < -0.5 and mood < 0:
        return "나쁜 소식 많음"
    return "엇갈림"


def payload(market, d, items):  # items: {섹터: {"stories": [...]}} → 사용자 메시지
    return json.dumps({"시장": market, "날짜": d, "섹터": [{"이름": n, "소식": v["stories"]} for n, v in items.items()]}, ensure_ascii=False)


def judge(lang, market, d, items):
    """items: {섹터: {"stories": [...], "z": ..., "mood": ...}} → {섹터: {"label", "line", "model"}}"""
    key = f"{lang}|{d}"
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    old = cache.get(key, {})
    out = {n: old[n] for n in items if n in old} if ON else {}
    todo = {n: v for n, v in items.items() if n not in out}
    if ON and todo and (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        schema = {"type": "object", "additionalProperties": False, "required": ["sectors"], "properties": {"sectors": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "label", "line"],
            "properties": {"name": {"type": "string", "enum": list(todo)}, "label": {"type": "string", "enum": list(LABELS)}, "line": {"type": "string"}}}}}}
        try:
            import anthropic
            r = anthropic.Anthropic().messages.create(model=MODEL, max_tokens=4000, system=SYSTEM,
                                                      messages=[{"role": "user", "content": payload(market, d, todo)}],
                                                      output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}})
            if r.stop_reason == "end_turn":
                for x in json.loads(next(b.text for b in r.content if b.type == "text"))["sectors"]:
                    out[x["name"]] = {"label": x["label"], "line": x["line"], "model": MODEL}
                cache[key] = {**old, **{n: out[n] for n in todo if n in out}}
                CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            else:
                print(f"  섹터 판정 {lang}: 멈춤 이유 {r.stop_reason} → 규칙 라벨")
        except Exception as e:  # 키가 메시지에 섞이지 않게 종류만 출력
            print(f"  섹터 판정 {lang}: 호출 실패 ({type(e).__name__}) → 규칙 라벨")
    for n, v in items.items():
        if n not in out:
            out[n] = {"label": rule(v.get("z"), v.get("mood", 0)), "line": None, "model": "dry-run"}
    return out
