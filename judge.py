"""목적별 작은 판정기 공장: 예시 파일을 주면 확률로 답하는 작은 모델을 만든다.

데이터: 첫 줄은 열 이름, 이후 한 줄에 `라벨들<TAB>...<TAB>상태`. 상태(마지막 열)는 텍스트,
또는 공백으로 구분한 숫자들. 라벨이 전부 숫자면 기대값 점수(score)도 낸다.

python judge.py train ko 감성                    # labeled/ko.tsv의 '감성' 열 → 모델 ko-감성
python judge.py train data/messages.tsv 라벨      # 파일 경로도 됨 → 모델 messages-라벨
python judge.py ask "제목" ko-감성 ko-범위 ko-주제  # 여러 질문을 한 번에
python judge.py eval ko_test 감성 ko_base-감성 ko_aug-감성  # 같은 시험 파일로 모델 비교
python judge.py check                             # 기울기 검산
python judge.py                                   # 대화형: 제목을 칠 때마다 감성·범위·주제 (판정하기.bat)
                                                  #   여러 줄을 한 번에 붙여 넣으면 한꺼번에 판정 + 요약
python judge.py file 제목들.txt                   # 한 줄에 제목 하나 → 표·요약, 제목들_판정.tsv 저장
"""
import functools
import json
import re
import sys
import time
import unicodedata
import zlib
from pathlib import Path

import numpy as np

# ── 내가 정하는 것들 ─────────────────────────────────────
BUCKETS = 2048    # 텍스트 조각을 담을 칸 수 = 텍스트 입력 크기
NGRAM   = 3       # 1~NGRAM 글자 조각을 특징으로 씀
LEXICON = "lexicon.tsv"  # 해석 사전 (방향·뒤집힘·사건). ""이면 안 씀
HINT    = 1.0     # 해석 특징의 세기 (글자 조각 특징 대비)
EMBED   = ""      # 사전 학습된 다국어 문장 모델 이름 (fastembed). ""이면 안 씀
EMBED_W = 1.0     # 문장 모델 특징의 세기
QUANT   = ("ko-감성-미세",)  # 불러올 때 문장 모델을 8비트로 줄일 미세 조정 모델 이름들 (CPU 판정이 빨라짐, docs/실험기록.md)
MAIN    = {"ko-감성": "ko_v2s-감성", "en-감성": "en_v2s-감성"}  # 2026-10-29: 한국어 K2v, 영어 F4 (둘 다 사전 묶음). 2026-10-31: 둘 다 8비트판 (한국어 25 → 7.5MB, 영어 234 → 60MB, 실험/다이어트.py). 이전: ko_v1-감성, en-감성-미세. 2026-11-02: 영어 en_v2t(방향 말 행렬 조정)를 잠깐 썼다가, 판단 자료 겹침을 뺀 재학습이 Haiku 검증 기준에 걸려 되돌림 (docs/실험기록.md).
            # 판정 창이 질문별로 쓸 모델 (없으면 <언어>-<질문>-미세, 그다음 <언어>-<질문>)
                  # 한국어 감성은 작은 증류 모델 (판정 모델과 실력이 같고 약 20배 빠름, docs/실험기록.md 2026-10-26)
EMBED_PREFIX = "" # 제목 앞에 붙일 지시문 (Qwen3 같은 모델은 '무슨 일인지' 알려 주면 나아질 수 있음)
HIDDEN  = 16      # 은닉층 뉴런 수, 0이면 은닉층 없음
BRIER   = 0.1     # 손실 = 교차엔트로피 + BRIER × Brier
NONZERO = 2.0     # 숫자 라벨에서 0이 아닌 사례의 손실 가중치. 1보다 크면 '애매하면 0'으로 피하기 어려워짐
                  # (1→3에서 '놓침'과 '헛경보'가 거의 맞바뀜. 2가 균형이 가장 좋았음)
LR      = 0.01    # Adam 학습률
WD      = 0.0     # 가중치 감쇠(규제): 클수록 우연히 맞아떨어진 조각에 덜 기댐
STEPS   = 3000
EVAL    = 10      # 이 간격마다 검증 손실을 재고 가장 좋았던 순간을 남김
PATIENCE = 300    # 이만큼 나아지지 않으면 학습을 멈춤
SPLIT   = (0.7, 0.15, 0.15)  # 학습 / 검증(조기 종료·온도) / 시험(최종 보고)
SEED    = 0
# ─────────────────────────────────────────────────────────

HOME = Path(__file__).parent


_CRC = {}  # 글자 조각 → crc32 (같은 조각을 다시 해시하지 않음)


def text_features(texts, buckets, ngram):
    # 1~ngram 글자 조각마다 crc32로 칸 번호를 정해 개수를 센다 (어휘 사전 없음). 칸 0이면 안 씀
    X = np.zeros((len(texts), buckets), np.float32)
    if buckets == 0:
        return X
    if len(_CRC) > 2_000_000:  # ponytail: 조각 기억은 200만 개에서 비움. 메모리가 문제면 lru로
        _CRC.clear()
    grams, rows = [], []  # 모든 제목의 조각을 한 줄로 모아 numpy로 한꺼번에 셈
    for r, t in enumerate(texts):
        t = f" {unicodedata.normalize('NFC', t.lower())} "
        g = [t[i:i + n] for n in range(1, ngram + 1) for i in range(len(t) - n + 1)]
        grams += g
        rows += [r] * len(g)
    for g in set(grams).difference(_CRC):
        _CRC[g] = zlib.crc32(g.encode())
    key = np.array(rows, np.int64) * buckets + np.fromiter(map(_CRC.__getitem__, grams), np.int64, len(grams)) % buckets
    key, cnt = np.unique(key, return_counts=True)
    r, c, v = key // buckets, key % buckets, cnt.astype(np.float32)
    norm = np.sqrt(np.bincount(r, v * v, len(texts)).astype(np.float32))  # 행마다 길이 1로 (전체 행렬 norm 없이)
    X[r, c] = v / np.maximum(norm[r], np.float32(1e-9))
    return X


_embedders, _embedded = {}, {}


def embed_features(texts, name, w, prefix=""):
    """사전 학습된 다국어 문장 모델로 제목을 뜻을 담은 숫자(벡터)로 바꾼다.
    한 번 바꾼 제목은 models/embed_cache/vectors_*.npz에 저장해 두고 다음부터는 다시 계산하지 않음."""
    texts = [prefix + t for t in texts]
    store = HOME / "models" / "embed_cache" / f"vectors_{re.sub(r'\W', '_', name)}.npz"
    if name not in _embedded:
        _embedded[name] = {}
        if store.exists():
            d = np.load(store)
            _embedded[name] = dict(zip(d["texts"].tolist(), d["vecs"]))
    seen = _embedded[name]
    new = [t for t in dict.fromkeys(texts) if t not in seen]
    if new:
        if name not in _embedders:
            import os
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            from fastembed import TextEmbedding  # 쓸 때만 불러옴: 설치 안 해도 나머지는 동작
            _embedders[name] = TextEmbedding(name, cache_dir=str(HOME / "models" / "embed_cache"))
        for t, e in zip(new, _embedders[name].embed(new)):
            seen[t] = (e / np.linalg.norm(e)).astype(np.float32)
        store.parent.mkdir(parents=True, exist_ok=True)
        np.savez(store, texts=np.array(list(seen)), vecs=np.array(list(seen.values())))
    return w * np.array([seen[t] for t in texts], np.float32).reshape(len(texts), -1)


def load_lexicon(path):
    """사전 파일을 읽는다. 파일이 바뀌지 않았으면 전에 읽은 것을 그대로 돌려줌 (판정마다 다시 읽지 않게)."""
    return _load_lexicon(path, (HOME / path).stat().st_mtime_ns)


@functools.lru_cache(maxsize=8)
def _load_lexicon(path, mtime):
    lex = {}
    for line in open(HOME / path, encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            kind, word = line.rstrip("\n").split("\t")
            lex.setdefault(kind, []).append(word.lower())
    return lex


def word_regex(words):
    # 영어 단어는 단어 경계로(high ≠ highway), 한국어·기호는 글자 그대로. 긴 말부터 찾음. 같은 낱말 목록이면 만든 것을 재사용
    return _word_regex(tuple(words))


@functools.lru_cache(maxsize=256)
def _word_regex(words):
    words = sorted(words, key=len, reverse=True)
    return alt_regex(tuple(rf"\b{re.escape(w)}\b" if w.isascii() and w[0].isalpha() else re.escape(w) for w in words), tuple(words))


@functools.lru_cache(maxsize=256)
def alt_regex(parts, words):
    """'A|B|C…' 정규식. 앞에 첫 글자 거름을 붙여, 첫 글자가 안 맞는 자리에서는 후보를 하나씩 대 보지 않음 (찾는 결과는 같음)."""
    if not parts:
        return re.compile("(?!)")
    if not all(words):
        return re.compile("|".join(parts))
    return re.compile(f"(?=[{''.join(sorted({re.escape(w[0]) for w in words}))}])(?:{'|'.join(parts)})")


CLAUSE = re.compile(r"…|\.\.\.|[,;|]| - ")  # 절 나눔 (마지막 절 = 보통 결과)
EN_BG = re.compile(r"\s(?:even as|as|after|on|amid|despite|following)\s")  # 영어 배경절 표지 ('규칙 영어배경')
CAUSE = re.compile(r"(?<!만)(?:에|에도)\s")  # "중동 긴장에 급락": 대상 뒤에 '~에' 어절이 있으면 그 대상은 원인 ("하루 만에"는 아님)
ARROW = re.compile(r"(\d[\d,]*\.?\d*)\s*(%p?|원|달러|bp)?\s*(?:→|->)\s*(\d[\d,]*\.?\d*)")  # "관세 10%→12.5%"
num = lambda s: float(s.replace(",", ""))
# 숫자로 말한 하루 움직임: "0.3% 하락", "15원 오른", "0.25%p 인상", "2.6%↓" (라벨기준.md 6번 표와 같은 단계로 읽음)
NUMCHG = re.compile(r"(\d+(?:\.\d+)?)\s?(%p|%|bp|원)\s?(오른|올라|상승|급등|인상|뛰|↑|▲|내린|내려|하락|급락|인하|떨어|↓|▼)")
UPWORD = ("오른", "올라", "상승", "급등", "인상", "뛰", "↑", "▲")
STEP = {"오름": 1, "내림": -1, "급오름": 2, "급내림": -2, "약오름": 0.5, "약내림": -0.5}


def phrase_features(texts, lex, hint):
    """단어구 특징: 방향 말마다 가장 가까운 대상에 붙여(뒤쪽 대상은 거리 2배, 앞쪽 우선)
    '대상 부호 × 방향 세기'를 증시 영향으로 본다. 대상이 없으면 '기타'(+1, 보통 그 회사·지표 자체).
    제목에 섹터가 하나만 나오면 그 섹터의 민감도가 대상 부호를 덮어쓴다 (예: 정유주에겐 유가 +1).
    [전체 영향, 마지막 절 영향, 대상 묶음별 영향…, 사건 좋고 나쁨, 섹터별 있음…], 각각 ±2로 자름."""
    step = {w: v for k, v in STEP.items() for w in lex.get(k, [])}
    group = {w: k.split(":")[1] for k in lex if k.startswith("대상:") for w in lex[k]}
    sign = {k.split(":")[1]: float(k.split(":")[2]) for k in lex if k.startswith("대상:")} | {"기타": 1.0}
    names = sorted(sign)
    sector = {w: k.split(":")[1] for k in lex if k.startswith("섹터:") for w in lex[k]}
    sens = {k.split(":")[1]: {g: float(s) for g, s in map(str.split, lex[k])} for k in lex if k.startswith("민감:")}
    sectors = sorted(set(sector.values()))
    rd, rs, rsec = word_regex(step), word_regex(group), word_regex(sector)
    good, bad = word_regex(lex.get("좋음", [])), word_regex(lex.get("나쁨", []))
    war, peace = word_regex(lex.get("분쟁", [])), word_regex(lex.get("평화", []))
    neg, solved, failed, past = (word_regex(lex.get(k, [])) for k in ("부정", "해소", "무산", "지난"))
    # 배경·양보 절 (사전에 '배경' 줄이 있을 때만): "급락 때 오른", "반등에도 팔자"의 앞 방향 말은 결과가 아님. "홀로 상승"은 예외 종목
    has_bg = "배경" in lex
    bgd, exc = word_regex(lex.get("배경", [])), word_regex(lex.get("예외", []))
    ev_ctx = "사건배경" in lex.get("규칙", [])  # 사전에 '규칙 사건배경' 줄이 있을 때만: 사건 말(좋음·나쁨)에도 배경·지난 처리
    if has_bg:  # 영어 대상 말 뒤에 한글이 바로 붙어도 찾음 ("WTI선물"): 단어 경계 대신 앞뒤가 영문자가 아닌지로
        gw = sorted(group, key=len, reverse=True)
        rs = alt_regex(tuple(rf"(?<![a-z]){re.escape(w)}(?![a-z])" if w.isascii() and w[0].isalpha() else re.escape(w) for w in gw), tuple(gw))
    # 크기 기준표: "크기:묶음:단위" 줄의 두 수 a b → a 미만 0.5, a~b 1, b 이상 2단계. 표가 있는 사전부터 '크기' 칸이 생김
    size = {(k.split(":")[1], k.split(":")[2]): tuple(map(float, lex[k][0].split())) for k in lex if k.startswith("크기:")}
    after = lambda rx, t, m, n: rx.search(t, m.end(), m.end() + n) is not None  # m 바로 뒤 n글자 안에 rx가 있나
    out = []
    for t in texts:
        t = unicodedata.normalize("NFC", t.lower())
        if "지난" in lex:  # 새 사전부터: "10%→12.5%"는 숫자가 커졌으니 '상향', 작아졌으면 '하향'으로 읽음
            t = ARROW.sub(lambda m: f"{m[1]}{m[2] or ''} {'상향' if num(m[3]) > num(m[1]) else '하향'} {m[3]}", t)
        if has_bg:
            t = t.replace("급등락", "등락")  # 오르내림 반복은 방향 없음
            t = re.sub(r"(서킷브레이커|사이드카)['’\"”]?\s?해제", r"\1", t)  # 거래 재개일 뿐, 급락은 이미 일어남
            t = re.sub(r"^\[([^\]]{2,10})\]\s*", r"\1, ", t)  # "[국제유가] …"의 머리말은 주어 (뒤의 '에'를 그 주어의 원인으로 보지 않게)
        found = {sector[m.group()] for m in rsec.finditer(t)}
        # 섹터가 둘 이상이면 보통 '희비 교차' 제목이라 덮어쓰지 않음
        over = sens.get(next(iter(found)), {}) if len(found) == 1 else {}
        sg = sign | {g: s for g, s in over.items() if g in sign}
        subj = [(m.start(), m.end(), group[m.group()]) for m in rs.finditer(t)
                if not (has_bg and group[m.group()] == "환율" and re.search(r"[\d조억만천]$", t[:m.start()]))]  # "1조달러"의 달러는 금액 단위
        cuts = [m.end() for m in CLAUSE.finditer(t)]
        f, seg = dict.fromkeys(names, 0.0), {}
        def clause(pos):
            return sum(c <= pos for c in cuts)

        def en_solved_before(rx, t, m):  # 나쁨 말 앞 20글자 안에 해소 말, 그 사이에 배경절 표지가 없을 때 ('ease policy amid recession'은 아님)
            h = [x for x in rx.finditer(t, max(0, m.start() - 20), m.start())]
            return bool(h) and not EN_BG.search(t, h[-1].start(), m.start() + 1)
        en_res = "영어해소" in lex.get("규칙", []) and lang_of(t) == "en"  # 사전 '규칙 영어해소': 해소 말이 나쁨 말 앞에 와도 ('calm jitters', 'shrugs off fears')
        en_bg = None  # 영어는 결과가 앞, 배경이 " as / after / on …" 뒤: 앞에 방향 말이 있으면 그 표지 뒤의 방향 말은 배경 (사전 '규칙 영어배경')
        if "영어배경" in lex.get("규칙", []) and lang_of(t) == "en":
            first = next(rd.finditer(t), None)
            mb = first and EN_BG.search(t, first.end())
            en_bg = mb.start() if mb else None
        hits = []  # (대상 묶음, 값, 절, 배경인가)
        for m in rd.finditer(t):
            if after(past, t, m, 6):  # "4% 급락서 강보합", "급락 딛고 반등": 지나간 상태라 결과에서 뺌
                continue
            start = max([c for c in cuts if c <= m.start()], default=0)
            is_bg = has_bg and (bgd.match(t, m.end()) is not None or exc.search(t, max(start, m.start() - 8), m.start()) is not None)
            is_bg = is_bg or (en_bg is not None and m.start() > en_bg)

            def dist(s):
                return m.start() - s[1] if s[1] <= m.start() else 2 * (s[0] - m.end()) if s[0] >= m.end() else np.inf
            k = clause(m.start())
            # 원인 대상("유가에 급락"의 유가)은 빼고, 같은 절의 대상 우선, 없으면 앞 절의 대상("국제유가, 이틀째 급락").
            # 뒤 절 대상에는 안 붙임
            def cause(s):  # 대상 뒤 ~ 방향 말 앞, 단 대상과 같은 절 안에서만 '~에'를 찾음
                stop = min([c for c in cuts if c > s[1]] + [m.start()])
                return s[1] <= m.start() and CAUSE.search(t, s[1], stop) is not None
            ok = [s for s in subj if not cause(s)]
            if en_bg is not None and m.start() < en_bg:  # 영어 결과 쪽 방향 말은 배경절 안의 대상에 붙이지 않음 ("Fall on Oil Gains")
                ok = [s for s in ok if s[0] < en_bg]
            cands = [s for s in ok if clause(s[0]) == k] or [s for s in ok if s[1] <= m.start()]
            near = min(cands, key=dist, default=None)
            if en_res:  # 영어 '동사 + 목적어': 방향 말 바로 뒤(20글자 안)의 위험·낙폭 대상이 바뀌는 것 ('cuts its losses', 'calm funding jitters')
                obj = [s_ for s_ in ok if 0 <= s_[0] - m.end() <= 20 and s_[2] in ("리스크", "낙폭")]
                near = obj[0] if obj else near
            g = near[2] if near and dist(near) < np.inf else "기타"
            v = sg[g] * step[m.group()] * (-1 if after(neg, t, m, 6) else 1)  # "실업률 오르지 않아" → 뒤집힘
            hits.append((g, v, k, is_bg))
        main = [h for h in hits if not h[3]] or hits  # 결과 방향 말이 있으면 그것만, 없으면 원인 방향 (라벨기준 4번)
        for g, v, k, _ in main:
            f[g] += v
            seg[k] = seg.get(k, 0.0) + v
        cs = over.get("분쟁", -1.0)  # 분쟁 말 하나의 영향: 시장엔 -1, 방산엔 +1
        n_war = sum(1 for m in war.finditer(t) if not (has_bg and re.match(r"\s?(?:전|前|이전)(?:\s|$)", t[m.end():])))  # "전쟁 전 수준"
        # 나쁨 말 뒤에 해소("위기설 부인", "파업 철회")면 좋음으로, 좋음 말 뒤에 무산("타결 무산")이면 나쁨으로
        evs = [(sg_ * (-1 if after(flip, t, m, 8) or (en_res and en_solved_before(flip, t, m)) else 1), ev_ctx and bgd.match(t, m.end()) is not None)
               for rx, flip, sg_ in ((good, failed, 1), (bad, solved, -1)) for m in rx.finditer(t)
               if not (ev_ctx and (after(past, t, m, 6) or t.startswith("에도", m.end())))]  # "충격 딛고", "긴축에도 웃었다": 결과 아님
        if any(not b for _, b in evs) or any(not h[3] for h in hits):  # 결과 쪽 말이 있으면 배경 절("충격에 급락"의 충격) 사건은 뺌
            evs = [e for e in evs if not e[1]]
        event = sum(v for v, _ in evs) + cs * (n_war - len(peace.findall(t)))
        row = [sum(f.values()), seg[max(seg)] if seg else 0.0, *f.values(), event,
               *(float(s in found) for s in sectors)]
        if size:  # 숫자 크기: 숫자 앞의 가장 가까운 대상(같은 절 우선)에 붙여 단계 × 방향 × 대상 부호
            mag = 0.0
            for m in NUMCHG.finditer(t):
                if after(past, t, m, 6):  # "4% 급락서 강보합": 지나간 상태의 숫자도 뺌 (방향 말과 같은 규칙)
                    continue
                before = [s for s in subj if s[1] <= m.start()]
                same = [s for s in before if clause(s[0]) == clause(m.start())]
                s = (same or before)[-1] if (same or before) else None
                g = "기타" if s is None or t[s[0]:s[1]] in ("주가", "주식", "shares", "stock") else s[2]  # "○○ 주가"는 개별 종목
                a, b = size.get((g, m.group(2)), size.get(("기타", m.group(2)), (None, None)))
                if a is None:
                    continue
                n = float(m.group(1))
                level = 0.5 if n < a else 1.0 if n < b else 2.0
                mag += level * (1 if m.group(3) in UPWORD else -1) * sg[g]
            row.append(mag)
        out.append(np.clip(row, -2, 2))
    return hint * np.array(out, np.float32)


def hint_features(texts, lex, hint):
    """(예전 모델용) 사전으로 만든 해석 특징 4개: 보통 대상의 방향, 뒤집힘 대상의 방향, 사건의 좋고 나쁨, 뒤집힘 대상 유무."""
    rx = {kind: word_regex(words) for kind, words in lex.items()}

    def count(kind, t):
        return len(rx[kind].findall(t)) if kind in rx else 0

    out = []
    for t in texts:
        t = unicodedata.normalize("NFC", t.lower())
        d = np.sign(count("오름", t) - count("내림", t))
        e = np.sign(count("좋음", t) - count("나쁨", t))
        inv = float(count("뒤집힘", t) > 0)
        out.append([d * (1 - inv), d * inv, e, inv])
    return hint * np.array(out, np.float32)


def parse_numbers(states):
    try:
        return np.array([[float(v) for v in s.split()] for s in states], np.float32)
    except ValueError:
        return None


def featurize(states, meta):
    if meta["kind"] == "num":
        return (parse_numbers(states) - meta["mean"]) / meta["std"]
    X = text_features(states, meta["buckets"], meta["ngram"])
    if "lexicon" in meta:
        feats = phrase_features if meta.get("phrase") else hint_features
        X = np.hstack([X, feats(states, meta["lexicon"], meta["hint"])])
    if meta.get("embed"):
        X = np.hstack([X, embed_features(states, meta["embed"], meta["embed_w"], meta.get("embed_prefix", ""))])
    return X


def init(F, K, rng):
    if HIDDEN == 0:
        return {"W2": np.zeros((F, K)), "b2": np.zeros(K)}
    return {
        "W1": rng.normal(0, 1, (F, HIDDEN)) / np.sqrt(F),
        "b1": np.zeros(HIDDEN),
        "W2": rng.normal(0, 1, (HIDDEN, K)) / np.sqrt(HIDDEN),
        "b2": np.zeros(K),
    }


def logits(p, X):
    h = np.tanh(X @ p["W1"] + p["b1"]) if "W1" in p else X
    return h @ p["W2"] + p["b2"], h


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def loss_grad(p, X, Y, w=None):
    """w: 사례별 손실 가중치 (없으면 모두 같음). 가중 평균으로 합친다."""
    n = len(X)
    w = np.full(n, 1 / n) if w is None else w / w.sum()
    z, h = logits(p, X)
    P = softmax(z)
    Y1 = np.eye(z.shape[1])[Y]
    ce = -(w * np.log(P[np.arange(n), Y] + 1e-12)).sum()
    brier = (w * ((P - Y1) ** 2).sum(1)).sum()

    gp = 2 * (P - Y1)  # dBrier/dP, softmax를 거쳐 dz로
    dz = (P - Y1 + BRIER * P * (gp - (gp * P).sum(1, keepdims=True))) * w[:, None]
    g = {"W2": h.T @ dz, "b2": dz.sum(0)}
    if "W1" in p:
        dh = dz @ p["W2"].T * (1 - h**2)
        g["W1"] = X.T @ dh
        g["b1"] = dh.sum(0)
    return ce + BRIER * brier, g


def calibration(P, Y):
    """확신도 구간별로 '말한 확률'과 '실제 정답률'을 비교."""
    conf, hit = P.max(1), P.argmax(1) == Y
    rows, ece = [], 0.0
    edges = [0, .5, .6, .7, .8, .9, 1.0001]
    for lo, hi in zip(edges, edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            rows.append((lo, min(hi, 1), m.sum(), conf[m].mean(), hit[m].mean()))
            ece += m.sum() * abs(conf[m].mean() - hit[m].mean())
    return rows, ece / len(Y)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def train(data, col):
    path = Path(data) if Path(data).exists() else HOME / "labeled" / f"{data}.tsv"
    header, *rows = [l.rstrip("\n").split("\t") for l in open(path, encoding="utf-8") if "\t" in l]
    c = header.index(col)
    rows = [r for r in rows if r[c]]  # 이 질문의 라벨이 빈 줄은 제외
    numeric = all(is_number(r[c]) for r in rows)  # 숫자 라벨이면 점수(Score) 질문
    labels = sorted({r[c] for r in rows}, key=float if numeric else None)
    states = [r[-1] for r in rows]
    Y = np.array([labels.index(r[c]) for r in rows])

    nums = parse_numbers(states)
    if nums is None:
        meta = {"kind": "text", "buckets": BUCKETS, "ngram": NGRAM}
        if LEXICON:  # 사전을 모델에 같이 저장해 두면 나중에 사전을 고쳐도 기존 모델은 그대로
            meta.update(lexicon=load_lexicon(LEXICON), hint=HINT, phrase=True)
        if EMBED:
            meta.update(embed=EMBED, embed_w=EMBED_W, embed_prefix=EMBED_PREFIX)
    else:
        meta = {"kind": "num", "mean": nums.mean(0).tolist(), "std": (nums.std(0) + 1e-9).tolist()}
    X = featurize(states, meta)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(Y))
    a, b = int(len(Y) * SPLIT[0]), int(len(Y) * (SPLIT[0] + SPLIT[1]))
    tr, va, te = idx[:a], idx[a:b], idx[b:]

    W = np.where(np.abs(np.array(labels, float)[Y]) < 1, 1.0, NONZERO) if numeric else np.ones(len(Y))  # 0·±0.5는 1배
    p = init(X.shape[1], len(labels), rng)
    print(f"데이터 {len(Y)}개 (학습 {len(tr)} / 검증 {len(va)} / 시험 {len(te)}), "
          f"입력 {meta['kind']} {X.shape[1]}칸, 라벨 {labels}")
    print(f"매개변수 {sum(v.size for v in p.values())}개")

    # ponytail: 전체 배치 Adam, 데이터가 수만 줄을 넘으면 미니배치로
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(x) for k, x in p.items()}
    best, best_step, best_p = np.inf, 0, p
    for t in range(1, STEPS + 1):
        _, g = loss_grad(p, X[tr], Y[tr], W[tr])
        for k in p:
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
            p[k] -= LR * (m[k] / (1 - 0.9**t)) / (np.sqrt(v[k] / (1 - 0.999**t)) + 1e-8)
            if k[0] == "W":  # AdamW식 감쇠, 편향은 제외
                p[k] -= LR * WD * p[k]
        if t % EVAL == 0:
            vl = loss_grad(p, X[va], Y[va], W[va])[0]
            if vl < best:
                best, best_step, best_p = vl, t, {k: x.copy() for k, x in p.items()}
            elif t - best_step >= PATIENCE:
                break
    p = best_p
    print(f"검증 손실이 가장 낮았던 step {best_step}의 매개변수를 씀")

    def nll(z, y):
        return -np.log(softmax(z)[np.arange(len(y)), y] + 1e-12).mean()

    zva = logits(p, X[va])[0]
    T = float(min(np.exp(np.linspace(-2, 2, 81)), key=lambda T: nll(zva / T, Y[va])))
    if numeric:
        # 기대값 점수는 망설일수록 0으로 쪼그라든다. 검증 데이터에서 '점수 × 배율'이 라벨에 가장 가깝게 펴 준다
        v = np.array(labels, float)
        ev, yt = softmax(zva / T) @ v, v[Y[va]]
        meta["stretch"] = float(np.clip((ev @ yt) / max(ev @ ev, 1e-9), 1.0, 4.0))

    zte = logits(p, X[te])[0]
    P = softmax(zte / T)
    print(f"\n[시험] 정확도 {(P.argmax(1) == Y[te]).mean():.1%}  (무작위로 찍으면 {1 / len(labels):.1%})")
    if numeric:
        print(f"점수 평균 오차 {np.abs(score_of(P, {**meta, 'labels': labels}) - v[Y[te]]).mean():.2f}  "
              f"(전부 평균값으로 찍으면 {np.abs(v[Y[tr]].mean() - v[Y[te]]).mean():.2f}), 점수 배율 {meta['stretch']:.2f}")
    rows_, ece = calibration(P, Y[te])
    print(f"온도 T={T:.2f}  보정 오차 ECE {calibration(softmax(zte), Y[te])[1]:.3f} → {ece:.3f}")
    print("  확신도 구간   개수  말한 확률  실제 정답률")
    for lo, hi, n, c, h in rows_:
        print(f"  {lo:.1f}~{hi:.1f}      {n:4}    {c:.2f}       {h:.2f}")

    out = HOME / "models" / f"{path.stem}-{col}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **p, T=T, meta=json.dumps({**meta, "labels": labels, "numeric": numeric}, ensure_ascii=False))
    print(f"\n저장: 모델 이름 {out.stem}")


def score_of(P, meta):
    """Jev의 Score처럼 등급 확률의 기대값, 학습 때 정한 배율로 편 뒤 라벨 범위 안으로 자름."""
    v = np.array(meta["labels"], float)
    return np.clip(meta.get("stretch", 1.0) * (P @ v), v.min(), v.max())


def load(name):
    """모델 파일(.npz)을 읽는다. 파일이 바뀌지 않았으면 전에 읽은 것을 그대로 돌려줌 (판정마다 다시 읽지 않게)."""
    path = HOME / "models" / f"{name}.npz"
    return _load(path, path.stat().st_mtime_ns)


@functools.lru_cache(maxsize=16)
def _load(path, mtime):
    d = np.load(path)
    p = {k: d[k] for k in d.files if k not in ("T", "meta")}
    for k in [k for k in p if k.endswith("_q8")]:  # 줄여 저장한 표 되돌리기 (실험/다이어트.py): 8비트 + 줄 배율, 낮은 계수 U·V, 16비트
        p[k[:-3]] = p.pop(k).astype(np.float32) * p.pop(k[:-3] + "_s")[:, None]
    for k in [k for k in p if k.endswith("_U")]:
        p[k[:-2]] = p.pop(k).astype(np.float32) @ p.pop(k[:-2] + "_V").astype(np.float32)
    p = {k: v.astype(np.float32) if v.dtype == np.float16 else v for k, v in p.items()}
    return p, float(d["T"]), json.loads(str(d["meta"]))


# ── 미세 조정 모델 (finetune.py가 만드는 models/<이름>.pt) ─────────────
# 문장 모델 전체 + 사전 특징 → 판정층. torch·transformers는 이 모델을 쓸 때만 불러온다.
_finetuned = {}


def ft_build(base, features, hidden, n_labels):
    from torch import nn
    from transformers import AutoModel, AutoTokenizer
    cache = str(HOME / "models" / "embed_cache")
    tok = AutoTokenizer.from_pretrained(base, cache_dir=cache)
    enc = AutoModel.from_pretrained(base, cache_dir=cache)
    head = nn.Sequential(nn.Linear(enc.config.hidden_size + features, hidden), nn.Tanh(), nn.Linear(hidden, n_labels))
    return tok, enc, head


def ft_logits(tok, enc, head, texts, lex, hint, max_len=64):
    import torch
    b = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    h = enc(**b)[0]  # 마지막 층 출력 (KoMiniLM처럼 튜플로 돌려주는 모델도 있어서 첫 칸으로)
    m = b["attention_mask"].unsqueeze(-1).float()
    pooled = (h * m).sum(1) / m.sum(1)  # 조각별 숫자의 평균 = 제목 하나의 요약
    return head(torch.cat([pooled, torch.tensor(phrase_features(texts, lex, hint))], 1))


def load_finetuned(name):
    q = name in QUANT
    if (name, q) not in _finetuned:
        import torch
        d = torch.load(HOME / "models" / f"{name}.pt", weights_only=False)
        meta = d["meta"]
        tok, enc, head = ft_build(meta["base"], meta["features"], meta["hidden"], len(meta["labels"]))
        enc.load_state_dict(d["enc"]); head.load_state_dict(d["head"])
        enc.eval(); head.eval()
        if q:
            enc = torch.quantization.quantize_dynamic(enc, {torch.nn.Linear}, dtype=torch.qint8)
        _finetuned[name, q] = (tok, enc, head, meta)
    return _finetuned[name, q]


TAG = re.compile(r"^\s*(?:\d{1,3}\s*[.)\]:]?\s+|[-•·*▶▷■□◆◇○●]+\s*)?(?:[\[【(](?:속보|단독|종합|1보|2보|3보|상보|긴급|포토)[\]】)]\s*)*")
TAIL = re.compile(r"\s*(?:\((?:종합|\d보|상보)\)|\s-\s[^-]{1,15})$")


def clean(s):
    """제목 앞 번호·기호·[속보] 같은 말머리, 끝의 (종합)·' - 언론사'를 뗀다. 판정 내용은 그대로."""
    return TAIL.sub("", TAG.sub("", s)).strip() or s.strip()


def lang_of(s):
    """한글이 글자(한글+영문)의 30% 이상이면 ko. '엔비디아 shares jump 8%'처럼 영어가 대부분이면 en."""
    ko, en = len(re.findall("[가-힣]", s)), len(re.findall("[A-Za-z]", s))
    return "ko" if ko and ko >= 0.3 * (ko + en) else "en"


def predict(name, states, head=None):
    """모델 이름으로 등급 확률을 낸다. models/<이름>.pt(미세 조정)가 있으면 그것, 없으면 .npz.
    head: 판정층이 여럿인 모델(영어 cmsm_en: 미국·한국)에서 쓸 관점 번호. 없으면 모델의 기본 관점."""
    states = [clean(s) for s in states]
    if name in MAIN.values():  # 주력 모델만: 못 알아듣는 관용 표현을 쉬운 말로 (rewrite.py, docs/실험기록.md 2026-11-02 변환기 사전)
        import rewrite
        states = rewrite.apply(states, name[:2])
    if (HOME / "models" / f"{name}.pt").exists():
        import torch
        tok, enc, head, meta = load_finetuned(name)
        with torch.no_grad():
            z = torch.cat([ft_logits(tok, enc, head, states[i:i + 64], meta["lexicon"], meta["hint"])
                           for i in range(0, len(states), 64)]).numpy()
        return softmax(z / meta["T"]), meta
    p, T, meta = load(name)
    if meta["kind"] == "cmsm_en":  # 판정층마다 온도·배율이 따로 있음
        h = meta["head"] if head is None else head
        Th = meta["T_heads"][h]
        P = np.vstack([softmax(cmsm_en_logits(p, states[i:i + 2000], meta)[:, h] / Th)
                       for i in range(0, len(states), 2000)] or [np.zeros((0, len(meta["labels"])))])
        return P, meta | {"stretch": meta["stretch_heads"][h], "head": h}
    run = (lambda st, mt: cmsm_logits(p, st, mt)) if meta["kind"] == "cmsm" else (lambda st, mt: logits(p, featurize(st, mt))[0])
    # 2,000개씩: 글자 조각 칸이 1.6만이면 한꺼번에 13만 개는 8GB가 넘음 (조각마다 약 130MB로 제한)
    return np.vstack([softmax(run(states[i:i + 2000], meta) / T)
                      for i in range(0, len(states), 2000)] or [np.zeros((0, len(meta["labels"])))]), meta


_kiwi = []


def cmsm_logits(p, states, meta):
    """형태소 CMSM (실험/seqexp.py cmsm morph): 글자 조각 + 사전 특징 갈래에, 형태소마다 16×16 행렬을
    읽는 순서대로 곱한 상태를 더한다."""
    if not _kiwi:
        import os
        from kiwipiepy import Kiwi  # 이 모델을 쓸 때만 불러옴
        _kiwi.append(Kiwi(num_workers=max(1, os.cpu_count() // 2)))
    h = text_features(states, meta["buckets"], meta["ngram"]) @ p["emb"] + phrase_features(states, meta["lexicon"], meta["hint"]) @ p["lin_w"].T + p["lin_b"]
    L, D, voc = meta["L"], meta["D"], meta["vocab"]
    norm = lambda s: unicodedata.normalize("NFC", s.lower())
    C = np.zeros((len(states), L), np.int64)
    for r, toks in enumerate(_kiwi[0].tokenize(list(states))):
        ids = [voc.get(f"{norm(x.form)}/{x.tag}", 1) for x in toks][:L]
        C[r, :len(ids)] = ids
    st = np.repeat(p["v0"][None], len(states), 0)
    for t in range(int((C > 0).sum(1).max(initial=0))):  # 읽는 순서대로: 상태 ← M(형태소) · 상태, 크기는 cap을 넘지 않게
        M = p["A"][C[:, t]].reshape(-1, D, D) + np.eye(D, dtype=np.float32)
        new = np.einsum("bij,bj->bi", M, st)
        new = new / np.maximum(np.linalg.norm(new, axis=1, keepdims=True) / meta["cap"], 1.0)
        st = np.where(C[:, t:t + 1] > 0, new, st)
    h = h + st @ p["seq_w"].T + p["seq_b"]
    return np.tanh(h) @ p["out_w"].T + p["out_b"]


EN_TOK = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)?%?|[$%]")  # 실험/en_seqexp.py toks와 같음


def cmsm_en_logits(p, states, meta):
    """영어 낱말 CMSM (실험/en_seqexp.py, 2026-10-29): 글자 조각 + 사전 특징 갈래에, 낱말마다 D×D 행렬을 앞으로 (bi면 뒤로도) 곱한 상태를 더하고
    판정층 둘(미국·한국)을 낸다 → (제목 수, 2, 7)."""
    h = text_features(states, meta["buckets"], meta["ngram"]) @ p["emb"] + phrase_features(states, meta["lexicon"], meta["hint"]) @ p["lin_w"].T + p["lin_b"]
    L, D, voc = meta["L"], meta["D"], meta["vocab"]
    C = np.zeros((len(states), L), np.int64)
    for r, t in enumerate(states):
        ids = [voc.get(w, 1) for w in EN_TOK.findall(unicodedata.normalize("NFC", t.lower()))][:L]
        C[r, :len(ids)] = ids
    n = int((C > 0).sum(1).max(initial=0))
    C = C[:, :max(1, n)]
    for A, w, b, Cx in [(p["A"], p["seq_w"], p["seq_b"], C)] + ([(p["A2"], p["seq2_w"], p["seq2_b"], C[:, ::-1])] if meta["bi"] else []):
        st = np.repeat(p["v0"][None], len(states), 0)
        for t in range(Cx.shape[1]):  # 읽는 순서대로 (거꾸로 읽을 때 앞쪽 빈칸은 건너뜀): 상태 ← M(낱말) · 상태, 크기는 cap을 넘지 않게
            M = A[Cx[:, t]].reshape(-1, D, D) + np.eye(D, dtype=np.float32)
            new = np.einsum("bij,bj->bi", M, st)
            new = new / np.maximum(np.linalg.norm(new, axis=1, keepdims=True) / meta["cap"], 1.0)
            st = np.where(Cx[:, t:t + 1] > 0, new, st)
        h = h + st @ w.T + b
    return np.einsum("nh,kch->nkc", np.tanh(h), p["out_w"]) + p["out_b"][None]


def model_exists(name):
    return any((HOME / "models" / f"{name}{ext}").exists() for ext in (".pt", ".npz"))


def evaluate(data, col, names):
    """같은 시험 파일로 여러 모델 비교. 방향 = 실제 점수가 0이 아닌 제목에서 예측 부호가 맞은 비율."""
    path = Path(data) if Path(data).exists() else HOME / "labeled" / f"{data}.tsv"
    header, *rows = [l.rstrip("\n").split("\t") for l in open(path, encoding="utf-8") if "\t" in l]
    c = header.index(col)
    rows = [r for r in rows if r[c]]
    print(f"시험 {len(rows)}개 ({path.name})")
    for name in names:
        P, meta = predict(name, [r[-1] for r in rows])
        Y = np.array([meta["labels"].index(r[c]) if r[c] in meta["labels"] else -1 for r in rows])
        line = f"  {name:16} 정확도 {(P.argmax(1) == Y).mean():.1%}  ECE {calibration(P, Y)[1]:.3f}"
        if meta["numeric"]:
            score, true = score_of(P, meta), np.array([float(r[c]) for r in rows])
            nz = true != 0
            line += (f"  점수 오차 {np.abs(score - true).mean():.2f}"
                     f"  방향 {(np.sign(score[nz]) == np.sign(true[nz])).mean():.1%} ({nz.sum()}개)"
                     f"  크기 |예측| {np.abs(score[nz]).mean():.2f} / |실제| {np.abs(true[nz]).mean():.2f}")
        print(line)


def answer(name, state):
    P, meta = predict(name, [state])
    P = P[0]
    i = int(P.argmax())
    a = {"choice": meta["labels"][i], "top_probability": round(float(P[i]), 3)}
    if meta["numeric"]:
        a["score"] = round(float(score_of(P[None], meta)[0]), 2)
        a["neutral"] = meta.get("neutral", 0.5)  # 이 모델의 중립 경계 (±)
    a["probabilities"] = {l: round(float(x), 3) for l, x in zip(meta["labels"], P)}
    return a


def ask(state, names):
    print(json.dumps({n: answer(n, state) for n in names}, ensure_ascii=False, indent=2))


def cuts(neutral):
    """중립 경계 → (호재 경계, 악재 경계). 숫자 하나면 양쪽 같고, [호재, 악재] 두 개면 따로."""
    return (neutral, neutral) if np.isscalar(neutral) else tuple(neutral)


def mood(score, neutral=0.5):
    """neutral: 이 안쪽이면 중립. 넓힐수록 헛경보가 줄고 놓침이 는다 (모델마다 meta에 저장, [호재, 악재]로 따로 줄 수도)."""
    up, dn = cuts(neutral)
    if -dn < score < up:  # 중립이어도 기운 쪽은 알려 줌
        return "중립 (호재 쪽)" if score >= up / 2 else "중립 (악재 쪽)" if score <= -dn / 2 else "중립"
    return "큰 호재" if score >= 1.5 else "호재" if score > 0 else "악재" if score > -1.5 else "큰 악재"


def rule_signal(states, lex=None):
    """사전 규칙만으로 본 신호: 방향 영향(첫 칸) + 사건 좋고 나쁨, ±2로 자름."""
    lex = lex or load_lexicon(LEXICON)
    h = phrase_features(states, lex, 1.0)
    tail = len({k.split(":")[1] for k in lex if k.startswith("섹터:")}) + any(k.startswith("크기:") for k in lex)  # 사건 칸 뒤의 칸 수
    return np.clip(h[:, 0] + h[:, -tail - 1], -2, 2)


def guard(states, scores, cut=None, lex=None):
    """안전장치: 지금 사전 규칙의 신호(방향 영향 + 사건 좋고 나쁨)가 ±cut 이상인데 모델 점수와 방향이 반대면 중립(0)으로.
    방향을 거꾸로 말하느니 중립이 낫다는 기준. 사전 파일을 고치면 재학습 없이 바로 반영된다. cut=None이면 끔.
    lex: 고정판 모델은 meta['guard_lexicon']에 묶어 둔 사전을 넘긴다 (lexicon.tsv를 고쳐도 판정이 안 바뀌게)."""
    scores = np.asarray(scores, float)
    if not cut:
        return scores, np.zeros(len(scores), bool)
    rule = rule_signal(states, lex)
    crossed = (np.abs(rule) >= cut) & (np.sign(rule) * np.sign(scores) < 0)
    return np.where(crossed, 0.0, scores), crossed


# 전망·의견 표지 (라벨기준.md 2번). 공식 수치 발표("전망 하향", "예상치 상회")는 사실이라 뺌, "우려"는 사실 제목에도 흔해서 뺌
# 질문 표지(?·까)는 뺌: "…급등…갈까", "주가 강세... 왜?"처럼 사실 + 질문 꼬리 제목이 많고 라벨은 이를 사실로 봄 (docs/실험기록.md)
OPINION = re.compile(r"가능성|관측|전망(?!치|을|보다|\s*(?:상향|하향|올려|내려|높여|낮춰|웃|밑|상회|하회))"
                     r"|예상(?!\s*(?:밖|외|보다|치|을|를|과|와|대로|상회|하회))|예측(?!치|을|를|보다)|[\[【(](?:칼럼|사설|인터뷰|분석|기고|시론)[\]】)]")


def soften(states, scores):
    """라벨기준.md 2번을 판정 뒤에: 전망·의견·질문형 제목은 한 단계 약하게 (점수 절반: ±2→±1, ±1→±0.5)."""
    hit = np.array([bool(OPINION.search(s)) for s in states])
    return np.where(hit, np.asarray(scores, float) / 2, scores), hit


# 수준만 말하는 제목: "원·달러 환율, 17년 3개월 만에 '1530원'". 그때 수준을 알아야 오름·내림을 알 수 있어 판단하지 않음
# (판정 창·채점은 중립, 지수에서는 뺌. docs/실험기록.md)
LEVEL_SUBJ = ("환율", "원화", "유가", "금리", "증시")
YEARS = re.compile(r"\d+\s*(?:년|개월)(?:\s*\d+\s*개월)?\s*(?:반|여)?\s*만에")
LEVEL_NUM = re.compile(r"\d[\d,.]*\s*(?:원|달러|선|포인트|%)")
LEVEL_DIR = ("최고", "최저", "최대", "최소", "돌파", "붕괴", "깨", "넘", "밑돌", "웃돌", "회복", "탈환", "아래", "위로", "고점", "저점",
             "도약", "악몽", "공포", "충격", "쇼크", "훈풍",
             "뚫", "깬", "깼", "상회", "하회", "육박", "밑으로", "올랐", "올렸", "올리", "올려", "내렸", "내리", "낮춰", "뛴", "훌쩍",
             "가장", "추월", "랠리", "귀환", "반납", "강해", "약해", "약달러", "강달러", "높아", "낮아", "치솟", "곤두박질", "올릴", "초읽기")  # 좋고 나쁨을 말하는 말이 있으면 수준만 말한 게 아님


def level_only(states, lex=None):
    lex = lex or load_lexicon(LEXICON)
    subj = word_regex([w for k in lex if k.startswith("대상:") and k.split(":")[1] in LEVEL_SUBJ for w in lex[k]])
    dirs = word_regex([w for k in (*STEP, "좋음", "나쁨") for w in lex.get(k, [])] + list(LEVEL_DIR))
    ts = [unicodedata.normalize("NFC", s.lower()) for s in states]
    return np.array([bool(subj.search(t) and YEARS.search(t) and LEVEL_NUM.search(t) and not dirs.search(t)) for t in ts])


# 2026-10-26 주제 21칸 → 17칸 (라벨기준.md 5번). 옛 칸으로 학습된 주제 모델의 답을 새 이름으로
# ponytail: 주제 모델을 17칸 라벨로 다시 학습하면 지운다
OLD_TOPIC = {"국내 물가·고용·경기": "국내 경기·물가", "기타 국내 경제": "국내 경기·물가", "부동산": "국내 경기·물가",
             "중국·아시아 증시": "기타 해외 증시", "유럽·기타 해외 증시": "기타 해외 증시",
             "연준·해외 금리": "해외 금리·통화", "달러·해외 통화": "해외 금리·통화",
             "해외 물가·경기지표": "해외 경기·물가", "기타 국제 경제": "해외 경기·물가", "한국은행·금리": "국내 금리·통화정책"}


# 배경 태그 (라벨기준.md 5번): 제목에 원인으로 적힌 말 → 태그 최대 2개. 주 주제와 같은 내용은 뺀다
BG_TAGS = {"전쟁·지정학": r"이란|이스라엘|전쟁|전면전|확전|휴전|종전|공습|폭격|교전|호르무즈|홍해|후티|우크라|러시아|중동|지정학|북한|하마스|미사일",
           "유가·원자재": r"유가|원유|wti|브렌트|금값|원자재|구리|천연가스|석유|기름값",
           "금리·통화정책": r"금리|연준|fomc|한은|금통위|국채|매파|비둘기|긴축|통화정책|파월|ecb|boj|일본은행",
           "환율": r"환율|(?<![\d.,만억조천백])달러|원화|엔화|엔저|위안",  # "100달러"처럼 값의 단위인 달러는 환율이 아님
           "AI·반도체": r"\bai\b|인공지능|반도체|hbm|메모리|엔비디아|빅테크|기술주|딥시크",
           "정치·정책": r"선거|계엄|탄핵|셧다운|밸류업|상법|세제|금투세|규제|정부|국회|대통령|정책|공조|행정명령|표심|배당금",
           "관세·통상": r"관세|무역|통상|수출규제|수출통제|fta",
           "기업 실적": r"실적|어닝|영업익|영업이익|순이익|매출",
           "경기·물가 지표": r"cpi|물가|인플레|고용|실업|gdp|성장(?:률| 우려| 둔화| 쇼크)|침체|k자|경기|소비|pmi|pce|내수|착공|소매판매|디플레",
           "수급·심리": r"외국인|외인|기관|개인|개미|순매수|순매도|매도세|매수세|투심|투자심리|차익실현|레버리지|환헤지|공매도|투자 소식"}
BG_RE = {k: re.compile(v) for k, v in BG_TAGS.items()}
BG_SAME = {"원화·환율": "환율", "에너지·원자재": "유가·원자재", "국내 금리·통화정책": "금리·통화정책", "해외 금리·통화": "금리·통화정책",
           "무역·관세": "관세·통상", "국제정치·지정학": "전쟁·지정학", "국내 경기·물가": "경기·물가 지표", "해외 경기·물가": "경기·물가 지표",
           "국내 정치·정책": "정치·정책"}
# 원인 꼴: "A에 B", "A에도", "A 속", "A 여파", "A 덕분/탓", "A에 힘입어", "A하자", "A가 끌어올린", "중동발·A發".
# "A에 기회·보고·민감·달렸다"처럼 대상·조건을 가리키는 '에'는 원인이 아님
NOT_CAUSE_E = r"(?!\s?(?:기회|보고|민감|달렸|달려|따라|대해|대한|관해|관한|있|없|진출|입점|공급|납품|편입|상장|매각|합류|참여|대비|비해|불과|그쳐|육박|달해|이르|맞서))"
BG_CAUSE = re.compile(r"(?<=\S)에도?(?=[\s…,·'\"’”])" + NOT_CAUSE_E + r"|\s속(?=[\s…,·])|\s?여파|덕분|탓|힘입어"
                   r"|(?<=[가-힣])(?:하|되|르|지|히|오|커|와)자(?=[\s…,·])|(?:이|가)\s?(?:끌어올린|끌어내린|이끈|견인한|밀어올린|짓누른|흔든)"
                   r"|(?<=중동|미국|중국|일본|유럽|이란|해외|관세|금리|유가|환율|전쟁)발(?=[\s'‘\"“])|(?<=[美中日])발|發")
QUOTED_CAUSE = re.compile(r"['‘\"“]([^'’\"”]{2,20}(?:우려|공포|쇼크|리스크|악재|호재|훈풍|기대|불안|충격)[^'’\"”]{0,6})['’\"”]")


def background(states, topics=None):
    """제목마다 배경 태그 목록 (원인 자리에 나온 순서, 최대 2개). 원인 자리 = 원인 꼴 바로 앞의 절, 또는 우려·공포 같은 말이 든 따옴표."""
    out = []
    for k, s in enumerate(states):
        t = clean(s).lower()
        spans, start = [], 0
        for m in BG_CAUSE.finditer(t):
            spans.append((start, re.split(r"[…,]|\.\.\.", t[start:m.start()])[-1]))  # 앞 절은 빼고 원인이 붙은 절만
            start = m.end()
        spans += [(m.start(), m.group(1)) for m in QUOTED_CAUSE.finditer(t)]
        found = sorted((pos + r.search(span).start(), tag) for pos, span in spans for tag, r in BG_RE.items() if r.search(span))
        same = BG_SAME.get(topics[k]) if topics else None
        out.append(list(dict.fromkeys(tag for _, tag in found if tag != same))[:2])
    return out


def judge_many(states):
    """제목 여러 개를 언어별로 묶어 한 번에 판정 → 제목마다 {감성, 중립경계, 주제, 주제확률, 범위, 범위확률}.
    언어는 lang_of로 (영어가 대부분이면 en). <언어>-<질문>-미세가 있으면 그것을 쓰고, 범위는 주제에서 따라간다."""
    scope = {n: s for s, n in (l.split("\t") for l in (HOME / "topics.txt").read_text(encoding="utf-8").splitlines() if l.strip())}
    out = [{} for _ in states]
    for lang in ("ko", "en"):
        idx = [i for i, s in enumerate(states) if lang_of(s) == lang]
        for q in ("감성", "주제", "범위") if idx else ():
            name = next((n for n in (MAIN.get(f"{lang}-{q}"), f"{lang}-{q}-미세") if n and model_exists(n)), f"{lang}-{q}")
            todo = [i for i in idx if not (q == "범위" and out[i].get("주제") in scope)]
            for i in set(idx) - set(todo):
                out[i] |= {"범위": scope[out[i]["주제"]], "범위확률": None}  # 주제가 정하면 범위는 따라감
            if not todo or not model_exists(name):
                continue
            P, meta = predict(name, [states[i] for i in todo])
            if meta["numeric"]:
                s, crossed = guard([states[i] for i in todo], score_of(P, meta), meta.get("guard"), meta.get("guard_lexicon"))
                s = soften([states[i] for i in todo], s)[0]
                s = np.where(level_only([states[i] for i in todo]), 0.0, s)  # 수준만 말한 제목은 중립
            if meta["numeric"] and "heads" in meta:  # 판정층이 여럿이면 다른 관점 점수도 (영어: 기본 한국 관점 + 미국 관점)
                ts_ = [states[i] for i in todo]
                for h, hn in enumerate(meta["heads"]):
                    if h != meta["head"]:
                        P2, m2 = predict(name, ts_, h)
                        s2 = soften(ts_, guard(ts_, score_of(P2, m2), m2.get("guard"), m2.get("guard_lexicon"))[0])[0]
                        s2 = np.where(level_only(ts_), 0.0, s2)
                        for k, i in enumerate(todo):
                            out[i][f"감성_{hn}"] = float(s2[k])
                for i in todo:
                    out[i]["관점"] = meta["heads"][meta["head"]]
            for k, i in enumerate(todo):
                if meta["numeric"]:
                    out[i] |= {"감성": float(s[k]), "중립경계": meta.get("neutral", 0.5), "엇갈림": bool(crossed[k])}
                else:
                    y = meta["labels"][int(P[k].argmax())]
                    out[i] |= {q: OLD_TOPIC.get(y, y) if q == "주제" else y, f"{q}확률": float(P[k].max())}
    # 배경은 참고 표시만 (규칙 정밀도 약 85%, 사용자 결정 2026-10-26). 한국어 규칙이라 한국어 제목만
    ko = [i for i, s in enumerate(states) if lang_of(s) == "ko"]
    for i, tags in zip(ko, background([states[i] for i in ko], [out[i].get("주제") for i in ko])):
        out[i]["배경"] = tags
    return out


def show_one(a):
    if "감성" in a:
        print(f"  감성 {a['감성']:+.2f}  {mood(a['감성'], a['중립경계'])}" + (f"  ({a['관점']} 투자자 관점)" if a.get("관점") else "")
              + ("  (규칙과 방향이 엇갈려 중립 처리)" if a["엇갈림"] else ""))
        for k, v in a.items():
            if k.startswith("감성_"):
                print(f"  {k[3:]} 관점 {v:+.2f}  {mood(v, a['중립경계'])}")
    for q in ("범위", "주제"):
        if q in a:
            p = a[f"{q}확률"]
            print(f"  {q} {a[q]}  " + ("(주제에서)" if p is None else f"(확률 {p:.0%})"))
    if a.get("배경"):
        print(f"  배경(참고) {' · '.join(a['배경'])}")


def show_many(states, res):
    """한 줄에 하나씩, 끝에 요약 (감성 평균, 호재·중립·악재 수, 범위·주제별 개수와 평균 감성)."""
    for k, (s, a) in enumerate(zip(states, res), 1):
        senti = f"{a['감성']:+.2f} {mood(a['감성'], a['중립경계']) + ('*' if a['엇갈림'] else ''):9}" if "감성" in a else ""
        bg = f" ({'·'.join(a['배경'])})" if a.get("배경") else ""
        print(f"{k:>3}. {senti} | {a.get('범위', '')} | {a.get('주제', '')}{bg} | {s}")
    scored = [a for a in res if "감성" in a]
    if not scored:
        return
    v = np.array([a["감성"] for a in scored])
    up, dn = np.array([cuts(a["중립경계"]) for a in scored]).T
    print(f"\n── 요약 ({len(states)}개)\n  감성 평균 {v.mean():+.2f}   호재 {(v >= up).sum()} · 중립 {((v < up) & (v > -dn)).sum()} · 악재 {(v <= -dn).sum()}"
          + (f"   (* 규칙과 방향이 엇갈려 중립 처리 {sum(a['엇갈림'] for a in scored)}개)" if any(a["엇갈림"] for a in scored) else ""))
    for q in ("범위", "주제"):
        groups = {}
        for a in scored:
            groups.setdefault(a.get(q, "?"), []).append(a["감성"])
        print(f"  {q}: " + " / ".join(f"{g} {len(x)}개 {np.mean(x):+.2f}"
                                      for g, x in sorted(groups.items(), key=lambda kv: -len(kv[1]))))


def judge_file(path):
    """제목이 한 줄에 하나씩 든 파일 → 화면에 표·요약, 같은 폴더에 <이름>_판정.tsv 저장."""
    path = Path(path)
    states = [l.strip() for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    res = judge_many(states)
    show_many(states, res)
    out = path.with_name(f"{path.stem}_판정.tsv")
    out.write_text("감성\t판정\t범위\t주제\t제목\n" + "".join(
        f"{a.get('감성', float('nan')):+.2f}\t{mood(a['감성'], a['중립경계']) if '감성' in a else ''}\t"
        f"{a.get('범위', '')}\t{a.get('주제', '')}\t{s}\n" for s, a in zip(states, res)), encoding="utf-8")
    print(f"\n저장: {out}")


def interactive():
    """제목을 받아 감성·범위·주제를 보여 준다. 여러 줄을 한 번에 붙여 넣으면 한꺼번에 판정하고 요약까지."""
    try:
        import msvcrt  # 윈도 콘솔: 붙여 넣은 나머지 줄이 입력 대기열에 남아 있는지 볼 수 있음
    except ImportError:
        msvcrt = None
    if not sys.stdin.isatty():  # 파일을 흘려 넣은 경우 (python judge.py < 제목들.txt)
        states = [l.strip() for l in sys.stdin if l.strip()]
        show_many(states, judge_many(states))
        return
    print("경제 뉴스 제목을 입력하고 Enter. 여러 줄을 한 번에 붙여 넣으면 한꺼번에 판정. 끝내려면 빈 줄에서 Enter.")
    if model_exists("ko-주제-미세"):
        print("(한국어 주제는 미세 조정 모델을 씀. 첫 한국어 제목은 불러오느라 10~15초 걸림)")
    while True:
        try:
            states = [input("\n제목> ").strip()]
            time.sleep(0.05)
            while msvcrt and msvcrt.kbhit():  # 붙여 넣은 나머지 줄 (중간의 빈 줄은 건너뜀)
                states.append(input().strip())
        except EOFError:
            break
        states = [s for s in states if s]
        if not states:
            break
        res = judge_many(states)
        show_one(res[0]) if len(states) == 1 else show_many(states, res)


def check():
    global HIDDEN
    assert CAUSE.pattern == r"(?<!만)(?:에|에도)\s", "사전 특징의 CAUSE가 다른 정의로 덮어써짐 (배경 태그용은 BG_CAUSE)"
    op = ["코스피 상승 관측", "다음주 코스피 상승 전망", "금리 인하 가능성", "[사설] 환율 대책 시급", "증시 반등 예상돼",
          "삼성전자 실적 예상치 상회", "IMF, 성장률 전망 하향", "실적 예상 밖 호조", "주가 강세... 왜?", "코스피 3000 갈까"]
    assert soften(op, [1.0] * len(op))[1].tolist() == [True] * 5 + [False] * 5, soften(op, [1.0] * len(op))[1]
    lv = ["원·달러 환율, 17년 3개월 만에 '1530원'", "약 2년 만에 원·달러 환율 1,330원대 진입", "원·달러 환율 2년 만에 최저",
          "금융위기 후 17년 만에 달러 환율 1500원 ‘악몽’", "삼성전자 3년 만에 영업익 10조", "원·달러 환율 1530원"]
    assert level_only(lv).tolist() == [True, True, False, False, False, False], level_only(lv)
    rng = np.random.default_rng(1)
    X, Y = rng.normal(0, 1, (6, 5)), rng.integers(0, 3, 6)
    for HIDDEN, w in ((0, None), (4, None), (4, rng.uniform(0.5, 3, 6))):
        p = {k: v + rng.normal(0, 0.5, v.shape) for k, v in init(5, 3, rng).items()}
        _, g = loss_grad(p, X, Y, w)
        for k in p:
            idx = tuple(rng.integers(0, s) for s in p[k].shape)
            old = p[k][idx]
            p[k][idx] = old + 1e-5
            lp = loss_grad(p, X, Y, w)[0]
            p[k][idx] = old - 1e-5
            lm = loss_grad(p, X, Y, w)[0]
            p[k][idx] = old
            assert abs((lp - lm) / 2e-5 - g[k][idx]) < 1e-6, (HIDDEN, k)
    assert parse_numbers(["1 2", "3 4"]).shape == (2, 2) and parse_numbers(["1 2", "a"]) is None
    m = {"labels": ["-1", "0", "1"], "stretch": 3.0}
    assert score_of(np.array([[0, 0.5, 0.5], [0, 0, 1]]), m).tolist() == [1.0, 1.0]  # 1.5는 1로 잘림
    lex = {"오름": ["상승", "high"], "내림": ["하락"], "뒤집힘": ["금리"], "나쁨": ["쇼크"]}
    h = hint_features(["금리 하락", "코스피 상승", "실적 쇼크", "Highway"], lex, 1.0)
    assert h.tolist() == [[0, -1, 0, 1], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 0]], h
    lex = {"오름": ["상승", "인상"], "내림": ["하락"], "급내림": ["급락"], "나쁨": ["쇼크"],
           "대상:증시:1": ["증시", "stocks"], "대상:유가:-1": ["유가", "oil"], "대상:금리:-1": ["rates"]}
    h = phrase_features(["증시 상승·유가 하락", "유가 급락…증시 하락", "실적 상승", "fed 인상 rates"], lex, 1.0)
    # 열: 전체, 마지막 절, 금리, 기타, 유가, 증시, 사건
    assert h.tolist() == [[2, 2, 0, 0, 1, 1, 0], [1, -1, 0, 0, 2, -1, 0], [1, 1, 0, 1, 0, 0, 0],
                          [-1, -1, -1, 0, 0, 0, 0]], h
    # 절을 넘어 뒤쪽 대상에 붙지 않음: 급락은 '…' 뒤 유가가 아니라 앞의 증시에, 쉼표 뒤 급락은 앞 절 유가에
    h = phrase_features(["증시 장중 크게 급락…유가 상승", "유가, 이틀째 급락"], lex, 1.0)
    assert h.tolist() == [[-2, -1, 0, 0, -1, -2, 0], [2, 2, 0, 0, 2, 0, 0]], h
    # 원인 '~에'의 대상은 건너뜀: 급락은 유가가 아니라 증시에. '에서'는 원인 아님
    h = phrase_features(["증시, 유가에 급락", "증시, 유가 상승에 하락", "증시에서 급락"], lex, 1.0)
    assert h.tolist() == [[-2, -2, 0, 0, 0, -2, 0], [-2, -2, 0, 0, -1, -1, 0], [-2, -2, 0, 0, 0, -2, 0]], h
    lex |= {"섹터:정유": ["정유주"], "섹터:방산": ["방산주"], "민감:정유": ["유가 +1"], "민감:방산": ["분쟁 +1"],
            "분쟁": ["전쟁"], "평화": ["휴전"], "대상:증시:1": ["증시", "정유주", "방산주"]}
    h = phrase_features(["유가 상승에 정유주 상승", "유가 상승에 증시 하락", "휴전에 방산주 하락", "전쟁에 증시 하락"], lex, 1.0)
    # 열: 전체, 마지막 절, 금리, 기타, 유가, 증시, 사건, 방산 있음, 정유 있음
    assert h.tolist() == [[2, 2, 0, 0, 1, 1, 0, 0, 1], [-2, -2, 0, 0, -1, -1, 0, 0, 0],
                          [-1, -1, 0, 0, 0, -1, -1, 1, 0], [-1, -1, 0, 0, 0, -1, -1, 0, 0]], h
    # 부정·약한 움직임·해소·무산 (진단 세트에서 나온 유형)
    lex = {"오름": ["오르", "상승"], "약오름": ["소폭 상승"], "약내림": ["약보합"], "부정": ["지 않"], "나쁨": ["위기", "파업"],
           "해소": ["부인", "철회"], "좋음": ["타결"], "무산": ["무산"], "대상:실업:-1": ["실업률"], "대상:증시:1": ["코스피"]}
    h = phrase_features(["실업률 오르지 않아", "코스피 소폭 상승", "코스피 약보합", "위기설 공식 부인", "파업 철회",
                         "타결 무산", "노조 파업"], lex, 1.0)
    assert h[:, 0].tolist() == [1, 0.5, -0.5, 0, 0, 0, 0] and h[:, -1].tolist() == [0, 0, 0, 1, 1, -1, -1], h
    # 배경·양보 절 (사전에 '배경'이 있을 때만): 결과 방향 말이 있으면 배경 방향 말은 빼고, 없으면 원인 방향. 홀로 = 예외 종목. 전쟁 전 = 사건 아님
    lex = {"오름": ["오른", "상승", "반등"], "내림": ["하락", "약세"], "급내림": ["급락"], "약오름": ["소폭 상승"], "분쟁": ["전쟁"],
           "배경": [" 때", "장에", "에도", "에 "], "예외": ["홀로"], "대상:증시:1": ["코스피", "코스닥", "나스닥"], "대상:유가:-1": ["유가"]}
    h = phrase_features(["시장 급락 때 오른 정유주", "하락장에 3% 상승한 은행주", "나스닥 반등에도 코스피 약세",
                         "코스닥 하락…hlb 홀로 소폭 상승", "유가 상승에 촉각", "국제유가, 전쟁 전 수준으로", "코스피 급등락 반복"], lex, 1.0)
    assert h[:, 0].tolist() == [1, 1, -1, -1, -1, 0, 0] and h[:, -1].tolist() == [0] * 7, h
    assert phrase_features(["시장 급락 때 오른 정유주"], {k: v for k, v in lex.items() if k not in ("배경", "예외")}, 1.0)[0, 0] == -1  # 옛 사전은 그대로
    assert clean("3. [속보] 코스피 급락(종합)") == "코스피 급락" and clean("1분기 GDP 성장") == "1분기 GDP 성장"
    assert lang_of("엔비디아 shares jump 8% after earnings") == "en" and lang_of("엔비디아 주가 8% 급등") == "ko"
    assert [mood(x, [0.9, 0.85]) for x in (0.88, 0.95, -0.86, -0.8)] == ["중립 (호재 쪽)", "호재", "악재", "중립 (악재 쪽)"]
    # 지난 상태("급락서", "급락 딛고")는 빼고, 화살표 숫자는 상향·하향으로, 빠진 방향 말("흘러내리", "위로")
    lex = {"오름": ["상승", "상향", "위로"], "내림": ["하락", "하향", "흘러내리"], "급내림": ["급락"], "약오름": ["강보합"],
           "지난": ["서 ", "딛고"], "대상:증시:1": ["코스피", "주가"], "대상:관세:-1": ["관세"], "대상:환율:-1": ["환율"]}
    h = phrase_features(["코스피 4% 급락서 강보합 마감", "코스피 급락 딛고 상승", "관세 10%→12.5%", "관세 25%→15%",
                         "주가 흘러내리는 이유", "환율 다시 1500원 위로"], lex, 1.0)
    assert h[:, 0].tolist() == [0.5, 1, -1, 1, -1, -1], h[:, 0]
    # 숫자 크기 (기본 행동 시험): 0.1% < 1% < 8% 하락, 환율 5원 < 15원, 개별 종목 1% 미만은 0.5단계
    lex = {"오름": ["상승"], "내림": ["하락"], "대상:증시:1": ["코스피", "주가"], "대상:환율:-1": ["환율"],
           "크기:증시:%": ["0.5 2"], "크기:환율:원": ["10 30"], "크기:기타:%": ["1 1000000"]}
    h = phrase_features(["코스피 0.1% 하락", "코스피 1% 하락", "코스피 8% 하락", "환율 5원 오른", "환율 15원 오른",
                         "대한항공 주가 0.64% 상승", "에코프로 20%↑"], lex, 1.0)
    assert h[:, -1].tolist() == [-0.5, -1, -2, -0.5, -1, 0.5, 1], h[:, -1]
    lex |= {"지난": ["서 "], "약오름": ["강보합"], "급내림": ["급락"]}
    h = phrase_features(["코스피 4% 급락서 강보합 마감"], lex, 1.0)
    assert h[0, 0] == 0.5 and h[0, -1] == 0, h  # 지나간 "4% 급락"은 방향·크기 둘 다 빠짐
    print("기울기 검산 통과")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1:2]
    if cmd == ["train"]:
        train(sys.argv[2], sys.argv[3])
    elif cmd == ["ask"]:
        ask(sys.argv[2], sys.argv[3:])
    elif cmd == ["eval"]:
        evaluate(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif cmd == ["check"]:
        check()
    elif cmd == ["file"]:
        judge_file(sys.argv[2])
    elif cmd == []:
        interactive()
    else:
        print(__doc__)
