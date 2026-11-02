# 대조 구조 실패 원인: 사전 특징(방향 말 → 대상 붙이기) 들여다보기, 사전 특징 끄기, 절 나눔 바꾸기
import sys, numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
sys.stdout.reconfigure(encoding="utf-8")
name = j.MAIN["ko-감성"]; p, T, meta = j.load(name); lex = meta["lexicon"]
names = sorted({k.split(":")[1] for k in lex if k.startswith("대상:")} | {"기타"})
step = {w: v for k, v in j.STEP.items() for w in lex.get(k, [])}
rd = j.word_regex(step)
CASES = [(-1, "코스피, 상승 출발했으나 하락 마감"), (1, "코스피, 하락 출발했으나 상승 마감"), (1, "실적 쇼크에도 주가는 반등"),
         (-1, "금리 인하 기대 꺾이며 뉴욕증시 하락"), (-1, "코스피 7000 돌파 실패…장 막판 밀려 하락"), (-1, "환율 하락에 수출주 약세"),
         (1, "적자 폭 줄이자 주가 급등"), (-1, "반등 하루 만에 코스피 다시 하락"), (-1, "[이란 미국 전쟁] 국제유가 95.63달러…호르무즈 통항 다시 급감")]


def model_score(texts, zero_ph=False):
    st = [j.clean(t) for t in texts]
    X = j.featurize(st, meta)
    if zero_ph:
        X[:, meta["buckets"]:] = 0
    return j.score_of(j.softmax(j.logits(p, X)[0] / T), meta)


def show(y, t):
    c = j.clean(t); ph = j.phrase_features([c], lex, j.HINT)[0]
    hits = [(m.group(), step[m.group()]) for m in rd.finditer(c.lower())]
    f = dict(zip(names, ph[2:2 + len(names)]))
    a, b = model_score([t])[0], model_score([t], True)[0]
    print(f"[정답 {y:+d}] {t}\n   방향 말 {hits} → 대상별 {{{', '.join(f'{k} {v:+.1f}' for k, v in f.items() if v)}}}, "
          f"전체 {ph[0]:+.1f}, 마지막 절 {ph[1]:+.1f}, 사건 {ph[2 + len(names)]:+.1f}\n   A' 점수 {a:+.2f} | 사전 특징 끄면 {b:+.2f}")


for y, t in CASES:
    show(y, t)

if "fix" in sys.argv:  # 3. 고쳐 보기: 지나간 상태·부정 표지를 사전에 넣으면? (메모리 안의 사전 사본만 바꿈, 파일은 그대로)
    import copy
    lex2 = copy.deepcopy(lex)
    lex2["지난"] = lex2.get("지난", []) + ["출발했으나", "했으나", "했지만", "하루 만에", "꺾이"]
    lex2["무산"] = lex2.get("무산", []) + ["실패"]
    tgt = next(k for k in lex2 if k.startswith("대상:증시:"))
    lex2[tgt] = lex2[tgt] + ["통항"]  # 통항이 늘면 좋음 (증시와 같은 부호)
    print("\n=== 사전 사본에 지난: 했으나·하루 만에·꺾이, 무산: 실패, 대상: 통항 을 더했을 때 (모델은 그대로) ===")
    for y, t in CASES:
        c = j.clean(t)
        X = j.featurize([c], meta); X[:, meta["buckets"]:] = j.phrase_features([c], lex2, j.HINT)
        s2 = j.score_of(j.softmax(j.logits(p, X)[0] / T), meta)[0]
        ph = j.phrase_features([c], lex2, j.HINT)[0]
        print(f"[정답 {y:+d}] {t[:40]:42} A' {model_score([t])[0]:+.2f} → {s2:+.2f}   (사전 전체 {ph[0]:+.1f}, 마지막 절 {ph[1]:+.1f})")
