# 방향 말 행렬만 미세 조정 (2026-11-02, docs/실험기록.md 같은 날 사전 등록): 주력 모델(ko_v2s·en_v2s)에서 고른 방향 말의
# 낱말 행렬 줄(A, 영어는 A2도)만 학습하고 나머지는 고정 → models/<언어>_v2t-감성.npz (같은 8비트 꼴).
# python 실험/방향말조정.py train ko|en
import sys, re, json, random
from pathlib import Path
HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))
import numpy as np
import torch
import judge as j, rewrite
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
torch.set_num_threads(4)
LAB = HOME / "labeled"
WORDS = {"ko": ["깨지/VV", "꺾이/VV", "내주/VV", "들썩/MAG", "들썩이/VV", "반색/NNG", "비틀/VV", "주춤/MAG"],
         "en": ["buckle", "buckles", "dive", "dives", "diving", "nosedive", "nosedives", "falter", "falters", "faltering",
                "stumble", "stumbles", "stumbled", "stumbling", "leap", "leaps"]}
# 학습 자료 (앞이 우선). 시험지·검증·기억 라벨·본문 문장은 넣지 않음
SRC = {"ko": ["ko.tsv", "ko_more7.tsv", "ko_more8.tsv", "ko_ai_opus.tsv", "ko_ai_opus_과거.tsv", "ko_ai_opus_과거2.tsv", "ko_ai_opus_과거3.tsv",
              "ko_ai_haiku_mid.tsv", "ko_ai_haiku_rest.tsv", "ko_ai_haiku_과거2.tsv"],
       "en": ["en_ai_opus_pool.tsv", "en_ai_opus_pool2.tsv", "en_ai_opus_과거.tsv", "en_ai_haiku.tsv", "en_ai_haiku_pool.tsv", "en_ai_haiku_과거.tsv"]}
V = np.array([-2, -1, -0.5, 0, 0.5, 1, 2], float)


def ids_of(lang, texts, meta):
    L, voc = meta["L"], meta["vocab"]
    C = np.zeros((len(texts), L), np.int64)
    if lang == "ko":
        if not j._kiwi:  # judge.cmsm_logits와 같은 형태소 분석기
            from kiwipiepy import Kiwi
            j._kiwi.append(Kiwi(num_workers=4))
        norm = lambda s: j.unicodedata.normalize("NFC", s.lower())
        for r, toks in enumerate(j._kiwi[0].tokenize(list(texts))):
            x = [voc.get(f"{norm(t.form)}/{t.tag}", 1) for t in toks][:L]; C[r, :len(x)] = x
    else:
        for r, t in enumerate(texts):
            x = [voc.get(w, 1) for w in j.EN_TOK.findall(j.unicodedata.normalize("NFC", t.lower()))][:L]; C[r, :len(x)] = x
    return C


def judged():  # 판단에 쓰는 제목 (학습에서 뺌): 시험지·검증·영어 Haiku 검증 2,010·이득 표본 전부
    out = set()
    for f in list((LAB / "시험").glob("*.tsv")) + [LAB / x for x in ("en_test.tsv", "en_newtest.tsv", "en_opus검증1000.tsv", "ko_opus검증1000.tsv")]:
        out |= {l.split("\t")[-1] for l in open(f, encoding="utf-8").read().splitlines()[1:]}
    hr = [l.split("\t") for l in open(LAB / "en_ai_haiku.tsv", encoding="utf-8").read().splitlines()[1:]]
    out |= {hr[i][-1] for i in np.random.default_rng(0).permutation(len(hr))[:len(hr) // 10]}
    for f in (HOME / "실험" / "결과").glob("*/**/*.tsv"):
        if f.parent.name in ("이득", "방향말조정", "방향말조정2"):
            out |= set(f.read_text(encoding="utf-8").splitlines())
    return out


def broad(lang, meta):  # 넓은 판: 사전의 방향 말(오름·내림 6범주) + 오르면 악재인 대상 말(부호 −1) + 1차 고른 말
    lex, voc = meta["lexicon"], meta["vocab"]
    words = {w.strip().lower() for k, v in lex.items() if k in j.STEP or (k.startswith("대상:") and k.endswith(":-1")) for w in v if " " not in w.strip()}
    if lang == "ko":
        got = [k for k in voc if k.split("/")[0] in words]
    else:
        got = [w for w in words if w in voc]
    return sorted(set(got) | set(WORDS[lang]))


def data(lang, meta, words):
    tid = {meta["vocab"][w] for w in words}
    seen, rows = judged(), []
    nj = len(seen)
    for f in SRC[lang]:
        lines = open(LAB / f, encoding="utf-8").read().splitlines()
        head = lines[0].split("\t")
        for l in lines[1:]:
            r = l.split("\t")
            t = r[-1]
            if not t or t in seen:
                continue
            if lang == "ko":
                y = (float(r[0]), None)
            else:
                y = (float(r[0]), float(r[1])) if head[0] == "미국" else (float(r[0]), None)
            rows.append((t, y)); seen.add(t)
    texts = rewrite.apply([j.clean(t) for t, _ in rows], lang)
    C = ids_of(lang, texts, meta)
    keep = np.isin(C, list(tid)).any(1)
    print(f"{lang}: 라벨 제목 {len(rows)} (판단용 {nj}개 뺌), 고른 말이 든 것 {int(keep.sum())}")
    idx = np.where(keep)[0]
    return [texts[i] for i in idx], C[idx], np.array([rows[i][1] for i in idx], dtype=object)


def static(lang, p, meta, texts):
    return j.text_features(texts, meta["buckets"], meta["ngram"]) @ p["emb"] + j.phrase_features(texts, meta["lexicon"], meta["hint"]) @ p["lin_w"].T + p["lin_b"]


def forward(lang, p, meta, h0, C, tabs):
    D, cap = meta["D"], meta["cap"]
    h = torch.tensor(h0, dtype=torch.float32)
    Ct = torch.tensor(C)
    pairs = [(tabs["A"], "seq_w", "seq_b", Ct)] + ([(tabs["A2"], "seq2_w", "seq2_b", Ct.flip(1))] if lang == "en" and meta["bi"] else [])
    for A, w, b, Cx in pairs:
        st = torch.tensor(p["v0"], dtype=torch.float32).repeat(len(C), 1)
        for t in range(Cx.shape[1]):
            M = A[Cx[:, t]].view(-1, D, D) + torch.eye(D)
            new = torch.einsum("bij,bj->bi", M, st)
            new = new / torch.clamp(new.norm(dim=1, keepdim=True) / cap, min=1.0)
            st = torch.where(Cx[:, t:t + 1] > 0, new, st)
        h = h + st @ torch.tensor(p[w], dtype=torch.float32).T + torch.tensor(p[b], dtype=torch.float32)
    th = torch.tanh(h)
    if lang == "en":
        return torch.einsum("nh,kch->nkc", th, torch.tensor(p["out_w"], dtype=torch.float32)) + torch.tensor(p["out_b"], dtype=torch.float32)[None]
    return (th @ torch.tensor(p["out_w"], dtype=torch.float32).T + torch.tensor(p["out_b"], dtype=torch.float32))[:, None]


def train(lang, kind="v1", lam=1.0, lr=3e-3, epochs=300, seed=0, batch=None, dst=None):
    path = HOME / "models" / f"{lang}_v2s-감성.npz"  # 출발점은 늘 아침 주력 (조정판 위에 또 조정하지 않음)
    p, T, meta = j._load(str(path), path.stat().st_mtime)
    words = WORDS[lang] if kind == "v1" else broad(lang, meta)
    print(f"{lang} {kind}: 학습할 낱말 {len(words)}개")
    texts, C, Y = data(lang, meta, words)
    n = (C > 0).sum(1).max(); C = C[:, :n]
    h0 = static(lang, p, meta, texts)
    Ts = meta["T_heads"] if lang == "en" else [T]
    lab = np.array([[np.abs(V - y[k]).argmin() if y[k] is not None else -1 for k in range(len(Ts))] for y in Y])
    rng = np.random.default_rng(seed); perm = rng.permutation(len(texts)); nv = len(texts) * 15 // 100
    va, tr = perm[:nv], perm[nv:]
    tid = torch.tensor(sorted({meta["vocab"][w] for w in words}))
    keys = ["A", "A2"] if lang == "en" and meta["bi"] else ["A"]
    base = {k: torch.tensor(p[k], dtype=torch.float32) for k in keys}
    W0 = {k: base[k][tid].clone() for k in keys}
    W = {k: torch.nn.Parameter(W0[k].clone()) for k in keys}
    opt = torch.optim.Adam(list(W.values()), lr=lr)
    tabs = lambda: {k: base[k].index_put((tid,), W[k]) for k in keys}
    def loss_sum(sel):
        z = forward(lang, p, meta, h0[sel], C[sel], tabs())
        tot, cnt = 0.0, 0
        for k, Tk in enumerate(Ts):
            m = lab[sel, k] >= 0
            if m.any():
                tot = tot + torch.nn.functional.cross_entropy(z[torch.tensor(m), k] / Tk, torch.tensor(lab[sel, k][m]), reduction="sum"); cnt += int(m.sum())
        return tot, cnt
    B = batch or len(tr)
    def evaluate_(sel):
        with torch.no_grad():
            t = [loss_sum(sel[i:i + 2048]) for i in range(0, len(sel), 2048)]
        return float(sum(a for a, _ in t)) / max(sum(c for _, c in t), 1)
    best, best_W, bad = evaluate_(va), {k: W0[k].clone() for k in keys}, 0
    print(f"  처음 검증 손실 {best:.4f}, 학습 자료 {len(tr)}, 검증 {len(va)}")
    for ep in range(epochs):
        order = rng.permutation(tr)
        for i in range(0, len(order), B):
            sel = order[i:i + B]
            opt.zero_grad()
            l, c = loss_sum(sel)
            (l / max(c, 1) + lam * sum(((W[k] - W0[k]) ** 2).sum() for k in keys) / len(tr)).backward(); opt.step()
        v = evaluate_(va)
        if v < best - 1e-4:
            best, best_W, bad = v, {k: W[k].detach().clone() for k in keys}, 0
        else:
            bad += 1
        if ep % (1 if batch else 20) == 0:
            print(f"  에폭 {ep}: 검증 {v:.4f}", flush=True)
        if bad >= (3 if batch else 30):
            break
    print(f"  멈춤 에폭 {ep}, 가장 좋은 검증 손실 {best:.4f}")
    d = np.load(path); out = {k: d[k] for k in d.files}  # 저장: 고친 줄만 다시 8비트로
    for k in keys:
        q, s = out[f"{k}_q8"].copy(), out[f"{k}_s"].copy()
        for r, i in zip(best_W[k].numpy(), tid.tolist()):
            s[i] = max(np.abs(r).max() / 127, 1e-12); q[i] = np.clip(np.round(r / s[i]), -127, 127).astype(np.int8)
        out[f"{k}_q8"], out[f"{k}_s"] = q, s
    m2 = json.loads(str(out["meta"])); m2["source"] = m2.get("source", "") + f" → 낱말 행렬 미세 조정 {kind} ({len(words)}개, 실험/방향말조정.py)"
    out["meta"] = np.array(json.dumps(m2, ensure_ascii=False))
    dst = HOME / "models" / (dst or f"{lang}_v2t-감성.npz")
    np.savez(dst, **out)
    print(f"  저장 {dst}, 고친 줄 {len(tid)}개 × {len(keys)}표")


def compare():  # 해 없음 판단: 지금(v2s) 대 조정판(v2t), 사전 등록 기준 자료
    sys.path.insert(0, str(HOME / "실험"))
    import evaluate as e
    import importlib.util as u
    sp = u.spec_from_file_location("bp", HOME / "실험" / "변환점검.py"); bp = u.module_from_spec(sp); sp.loader.exec_module(bp)
    old = dict(j.MAIN)
    seven = lambda s, y: float(np.mean(V[np.abs(s[:, None] - V).argmin(1)] == y))
    rows = [r for f in e.TESTS["무작위700 (최종)"] for r in bp.tsv(LAB / "시험" / f)]
    hr = bp.tsv(LAB / "en_ai_haiku.tsv"); hv = [hr[i] for i in np.random.default_rng(0).permutation(len(hr))[:len(hr) // 10]]
    olds = [r for f in ("en_test.tsv", "en_newtest.tsv") for r in bp.tsv(LAB / f)]
    for tag in ("v2s", "v2t"):
        j.MAIN.update({k: v.replace("v2s", tag) for k, v in old.items()})
        t, y = [r[-1] for r in rows], np.array([float(r[0]) for r in rows]); m = e.screen(bp.scores("ko", t), y)
        fails = [d for d, a, b, rel in e.BEHAVIOR if not {"<": lambda sa, sb: sa < sb, "=": lambda sa, sb: abs(sa - sb) < 0.05, ">0": lambda sa, sb: sa > 0,
                 "<0": lambda sa, sb: sa < 0, ">=0": lambda sa, sb: sa >= -0.1, "<=0": lambda sa, sb: sa <= 0.1}[rel](bp.scores("ko", [a])[0], bp.scores("ko", [b])[0] if b else None)]
        print(f"[{tag}] 한국어 700: 맞음 {m['맞음']:.1%} 헛경보 {m['헛경보']:.1%} 놓침 {m['놓침']:.1%} 방향 틀림 {m['방향 틀림']}, 기본 행동 {16 - len(fails)}/16 {fails}")
        for name, rr in (("영어 Haiku 검증 2,010", hv), ("옛 영어 303", olds)):
            t, y = [r[-1] for r in rr], np.array([float(r[0]) for r in rr]); s = bp.scores("en", t, head=0); m = e.screen(s, y)
            print(f"[{tag}] {name}: 7단계 {seven(s, y):.1%} 맞음 {m['맞음']:.1%} 헛경보 {m['헛경보']:.1%} 방향틀림3 {int(bp.flips(s, y).sum())}")
    j.MAIN.update(old)


GAIN = HOME / "실험" / "결과" / "방향말조정"


def gain_make():  # 이득 표본: 캐시 제목 중 고른 말이 든 것, 라벨 파일·변환 이득 표본 제목 뺌, 무작위 최대 300 (시드 2042)
    sys.path.insert(0, str(HOME / "web")); import repcheck as rc
    seen = {l.split("\t")[-1] for f in LAB.rglob("*.tsv") for l in open(f, encoding="utf-8").read().splitlines()[1:]}
    for lang in ("ko", "en"):
        seen |= set((HOME / "실험" / "결과" / "변환" / "이득" / f"{lang}.tsv").read_text(encoding="utf-8").splitlines())
    GAIN.mkdir(parents=True, exist_ok=True)
    for lang in ("ko", "en"):
        _, _, meta = j.load(j.MAIN[f"{lang}-감성"])
        t = sorted({l.rstrip("\n").split("\t")[-1] for l in open(HOME / "data" / "cache" / f"지수자료_{lang}.tsv", encoding="utf-8")} - seen - {"제목"})
        C = ids_of(lang, rewrite.apply([j.clean(x) for x in t], lang), meta)
        hit = [x for x, k in zip(t, np.isin(C, [meta["vocab"][w] for w in WORDS[lang]]).any(1)) if k]
        pick = random.Random(2042).sample(hit, min(300, len(hit)))
        (GAIN / f"{lang}.tsv").write_text("".join(x + "\n" for x in pick), encoding="utf-8")
        mix = list(enumerate(pick)); random.Random(lang).shuffle(mix)
        for p_ in range(3):
            part = mix[p_ * 100:(p_ + 1) * 100]
            (GAIN / f"묶음_{lang}_{p_}.txt").write_text(rc.GUIDE.format(view=rc.VIEW[lang]) + "".join(f"{k}\t{x}\n" for k, x in part), encoding="utf-8")
        print(lang, "고른 말이 든 제목", len(hit), "→ 표본", len(pick))



if __name__ == "__main__":
    A_ = sys.argv
    {"train": lambda: train(A_[2]), "train2": lambda: train(A_[2], "v2", lr=1e-3, epochs=15, batch=1024, dst=f"{A_[2]}_v2u-감성.npz"),
     "compare": compare, "gain_make": gain_make}[A_[1]]()
