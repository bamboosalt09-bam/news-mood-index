# AI 해설 (web/DATA.md의 summary 칸): 그날 숫자와 기사 묶음 → claude-haiku-5-5 구조화 출력. build.py가 부름.
# API 자격(ANTHROPIC_API_KEY·ANTHROPIC_AUTH_TOKEN)이 없거나 호출이 실패하면 규칙으로 만든 시험용 해설(model = "dry-run").
import os, json, datetime as dt

MODEL = "claude-haiku-5-5"
SYSTEM = """너는 경제 뉴스 제목 요약가다. 사용자가 한 시장의 그날 뉴스 분위기 숫자와 기사 묶음(대표 제목)을 JSON으로 준다. 한국어로 해설을 쓴다.
규칙:
- 주어진 제목과 숫자에 있는 내용만 쓴다. 제목에 없는 사실·수치·배경지식을 더하지 않는다.
- 예측·전망을 하지 않는다. 매수·매도·보유 등 투자 권유를 하지 않는다.
- 인과를 단정하지 않는다. 원인은 제목에 '보도된' 것으로만 쓴다 ("~로 보도됐다", "~라는 제목이 많았다"). "A에도 급락"은 A가 원인이 아니다.
- headline: 그날 분위기를 한 줄로. bullets: 2~4문장, 각 문장은 근거 묶음의 내용. watch: 지켜볼 보도 주제 한 문장 (예측 아님).
- sources: bullets의 근거로 쓴 묶음 id (top_negative·top_positive의 id만)."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["headline", "bullets", "watch", "sources"],
          "properties": {"headline": {"type": "string"}, "bullets": {"type": "array", "items": {"type": "string"}},
                         "watch": {"type": "string"}, "sources": {"type": "array", "items": {"type": "string"}}}}


def facts(m):
    t = m["today"]
    pick = lambda lst: [{k: x[k] for k in ("id", "title", "score", "count")} for x in lst]
    return {"시장": m["name"], "뉴스": m["news"], "날짜": m["date"],
            "숫자": {k: t[k] for k in ("mood", "mood_label", "shock_label", "change_pct")},
            "갈래별_평균": {k: v["mood"] for k, v in m["kinds"].items()},
            "top_negative": pick(m["top_negative"]), "top_positive": pick(m["top_positive"]),
            "주제_비중_변화": [{"주제": x["topic"], "오늘": x["share"], "앞7일": x["share_prior7"]} for x in m["topics"]]}


def dry_run(m):
    t, neg, pos = m["today"], m["top_negative"], m["top_positive"]
    chg = "등락률 없음" if t["change_pct"] is None else f"지수 {t['change_pct']:+.2f}%"
    b = [f"{m['name']} 기준 뉴스 분위기 '{t['mood_label']}', 충격도 '{t['shock_label']}' ({chg}, 제목 {t['titles']}개)."]
    b += [f"{w} 쪽으로는 '{x['title']}' 등 {x['count']}건이 보도됐다." for w, lst in (("부정", neg), ("긍정", pos)) for x in lst[:1]]
    up = max(m["topics"], key=lambda x: x["share"] - x["share_prior7"], default=None)
    if up:
        b.append(f"'{up['topic']}' 보도 비중이 앞 7일 {up['share_prior7']:.0%}에서 {up['share']:.0%}로 바뀌었다.")
    return {"headline": f"{m['name']} 뉴스 분위기 {t['mood_label']}, 충격도 {t['shock_label']}", "bullets": b[:4],
            "watch": f"'{up['topic'] if up else '시장'}' 관련 보도가 이어지는지 지켜본다.", "sources": [x["id"] for x in neg[:1] + pos[:1]]}


def summarize(m):
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).replace(microsecond=0).isoformat()
    out, model = None, "dry-run"
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        try:
            import anthropic
            r = anthropic.Anthropic().messages.create(
                model=MODEL, max_tokens=4000, system=SYSTEM,
                messages=[{"role": "user", "content": json.dumps(facts(m), ensure_ascii=False)}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}})
            if r.stop_reason == "end_turn":
                out, model = json.loads(next(b.text for b in r.content if b.type == "text")), MODEL
            else:
                print(f"  해설 {m['id']}: 멈춤 이유 {r.stop_reason} → 시험용 해설")
        except Exception as e:  # 키가 메시지에 섞이지 않게 종류만 출력
            print(f"  해설 {m['id']}: 호출 실패 ({type(e).__name__}) → 시험용 해설")
    if out is None:
        out = dry_run(m)
    ids = [x["id"] for x in m["top_negative"] + m["top_positive"]]
    out["bullets"] = out["bullets"][:4] if len(out["bullets"]) >= 2 else dry_run(m)["bullets"]
    out["sources"] = [s for s in dict.fromkeys(out["sources"]) if s in ids]
    return out | {"model": model, "generated": now}
