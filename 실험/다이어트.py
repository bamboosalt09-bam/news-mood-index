# 영어 주력 다이어트 (2026-10-31, docs/실험기록.md): 다시 학습하지 않고 en_v2-감성을 줄인다.
# Q16(16비트) / Q8(줄마다 8비트 + 배율) × P(판정층 앞 채널 가지치기 512→344·256) × R(A·A2 낮은 계수 SVD 256·128)
# 판마다 파일로 저장 → judge._load로 다시 읽어(되돌리기까지 확인) Opus 검증 1,000을 채점. 파일 크기·속도도 잰다.
# python 실험/다이어트.py [save=판이름]   (save: 그 판을 models/en_v2s-감성.npz로 저장)
import sys, json, time, itertools, os
from pathlib import Path
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j, evaluate as e
sys.stdout.reconfigure(encoding="utf-8")
V = np.array([-2, -1, -0.5, 0, 0.5, 1, 2], float)
SRC = j.HOME / "models" / "en_v2-감성.npz"
OUT = Path(os.environ.get("TEMP", ".")) / "en_diet"; OUT.mkdir(exist_ok=True)
d = np.load(SRC); base = {k: d[k] for k in d.files if k not in ("T", "meta")}; T0, META = d["T"], d["meta"]
meta = json.loads(str(META))
inc = np.array(json.load(open(j.HOME / "실험" / "결과" / "채널_en.json", encoding="utf-8"))["손실증가"])  # Opus 3차 절반에서 잰 값
rows = [l.rstrip("\n").split("\t") for l in open(j.HOME / "labeled" / "en_opus검증1000.tsv", encoding="utf-8").read().splitlines()[1:]]
texts = [j.clean(r[-1]) for r in rows]; Y = np.array([[float(r[0]), float(r[1])] for r in rows])


SVD = {}  # 낮은 계수 근사는 표마다 한 번만 분해 (가지치기와 무관)


def build(q, keep, rank):
    p = dict(base)
    if keep < 512:  # 덜 중요한 채널부터 뺀다 (판정층 앞 칸 = emb 열, lin·seq 줄, out 열)
        k = np.sort(np.argsort(-inc)[:keep])
        p |= {"emb": p["emb"][:, k], "lin_w": p["lin_w"][k], "lin_b": p["lin_b"][k], "seq_w": p["seq_w"][k], "seq_b": p["seq_b"][k],
              "seq2_w": p["seq2_w"][k], "seq2_b": p["seq2_b"][k], "out_w": p["out_w"][:, :, k]}
    store = {}
    for name in ("A", "A2"):
        if rank:
            if name not in SVD:
                SVD[name] = np.linalg.svd(p[name], full_matrices=False)
            U, s, Vt = SVD[name]
            store[name + "_U"], store[name + "_V"] = (U[:, :rank] * s[:rank]).astype(np.float32), Vt[:rank].astype(np.float32)
            p.pop(name)
    for name, v in p.items():
        if q == 8 and name in ("A", "A2", "emb", "A_U", "A2_U"):
            sc = np.maximum(np.abs(v).max(1), 1e-12) / 127
            store[name + "_q8"], store[name + "_s"] = np.round(v / sc[:, None]).astype(np.int8), sc.astype(np.float32)
        elif q in (8, 16) and v.size > 10000:
            store[name] = v.astype(np.float16)
        else:
            store[name] = v
    for name in [k for k in store if k.endswith(("_U", "_V"))]:
        if q == 16:
            store[name] = store[name].astype(np.float16)
        elif q == 8 and name.endswith("_U"):
            v = store.pop(name); sc = np.maximum(np.abs(v).max(1), 1e-12) / 127
            store[name + "_q8"], store[name + "_s"] = np.round(v / sc[:, None]).astype(np.int8), sc.astype(np.float32)
    return store


def score(p, h):
    z = np.vstack([j.cmsm_en_logits(p, texts[i:i + 2000], meta) for i in range(0, len(texts), 2000)])[:, h]
    s = np.clip(meta["stretch_heads"][h] * (j.softmax(z / meta["T_heads"][h]) @ V), -2, 2)
    s = j.soften(texts, j.guard(texts, s, e.GUARD, meta.get("guard_lexicon"))[0])[0]
    return np.where(j.level_only(texts), 0.0, s)


def metrics(s, y):
    m = e.screen(s, y); s7 = V[np.abs(s[:, None] - V).argmin(1)]; nz = y != 0
    return {"7단계": float(np.mean(s7 == y)), "헛경보": m["헛경보"], "놓침": m["놓침"],
            "방향틀림3": int(((np.abs(s) >= 0.25) & nz & (np.sign(s) * np.sign(y) < 0)).sum())}


res, ref = {}, None
for q, keep, rank in itertools.product((32, 16, 8), (512, 344, 256), (0, 256, 128)):
    name = f"Q{q}_P{keep}_R{rank}"
    f = OUT / f"{name}.npz"
    np.savez(f, T=T0, meta=META, **build(q, keep, rank))
    p, _, _ = j._load.__wrapped__(f, f.stat().st_mtime_ns)  # 캐시 안 씀 (판마다 230MB가 쌓임)
    t0 = time.time(); S = np.stack([score(p, h) for h in (0, 1)], 1); sec = time.time() - t0
    if ref is None:
        ref = S
    r = {"MB": round(f.stat().st_size / 1e6, 1), "제목당ms": round(sec * 1000 / len(texts), 2), "최대차이": float(np.abs(S - ref).max()),
         **{h: metrics(S[:, k], Y[:, k]) for k, h in enumerate(("미국", "한국"))}}
    b = res.get("Q32_P512_R0", r)
    r["통과"] = all(r[h]["7단계"] >= b[h]["7단계"] - 0.003 and r[h]["방향틀림3"] <= b[h]["방향틀림3"] + 1 and r[h]["헛경보"] <= b[h]["헛경보"] + 0.003 for h in ("미국", "한국"))
    res[name] = r
    print(name, json.dumps(r, ensure_ascii=False))
    del p; f.unlink()
json.dump(res, open(j.HOME / "실험" / "결과" / "다이어트_en.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
ok = [k for k, v in res.items() if v["통과"]]
print("통과", len(ok), "개, 가장 작은 것:", min(ok, key=lambda k: res[k]["MB"]) if ok else None)
SAVE = next((a[5:] for a in sys.argv[1:] if a.startswith("save=")), None)
if SAVE:
    q, keep, rank = (int(x[1:]) for x in SAVE.split("_"))
    np.savez(j.HOME / "models" / "en_v2s-감성.npz", T=T0, meta=json.dumps(meta | {"source": f"{meta['source']} → 다이어트 {SAVE}"}, ensure_ascii=False), **build(q, keep, rank))
    print("저장: models/en_v2s-감성.npz")
