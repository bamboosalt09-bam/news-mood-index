# en_seqexp.py로 학습한 영어 낱말 CMSM(.pt, 판정층 둘 = 미국·한국)을 judge가 읽는 numpy 모델(.npz)로 옮긴다 (2026-10-29).
# 특징 사전과 안전장치 사전(지금 lexicon.tsv)을 모델에 묶는다 → lexicon.tsv를 고쳐도 판정이 안 바뀐다.
# python 실험/en_to_numpy.py models/en_seq_f4_0.pt en_v2-감성 [head=한국]   (head: 판정 창 '감성'에 쓸 관점, 다른 관점은 함께 표시)
import json, sys
import numpy as np
import torch
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j

src, name = sys.argv[1], sys.argv[2]
head = next((a[5:] for a in sys.argv[3:] if a.startswith("head=")), "한국")
d = torch.load(src, weights_only=False)
H, D, DEEP, BI, ATT = (list(d["cfg"]) + [False, False])[:5]
assert DEEP == 0 and not ATT, "층을 더 쌓은 모델·어텐션 모델은 아직 못 옮김"
s = {k: v.numpy().astype(np.float32) for k, v in d["state"].items()}
heads = ["미국", "한국"]
meta = {"kind": "cmsm_en", "buckets": 16384, "ngram": 3, "lexicon": j.load_lexicon(j.LEXICON), "guard_lexicon": j.load_lexicon(j.LEXICON),
        "hint": j.HINT, "phrase": True, "labels": ["-2", "-1", "-0.5", "0", "0.5", "1", "2"], "numeric": True, "guard": 1.0,
        "neutral": [0.9, 0.85], "vocab": d["vocab"], "L": 40, "D": D, "cap": 4.0, "bi": bool(BI), "heads": heads, "head": heads.index(head),
        "T_heads": d["T"], "stretch_heads": d["stretch"], "stretch": d["stretch"][heads.index(head)], "tokens": "영어 낱말 (en_seqexp)", "source": src}
arr = dict(emb=s["emb.weight"], lin_w=s["lin.weight"], lin_b=s["lin.bias"], A=s["A.weight"], v0=s["v0"], seq_w=s["seq.weight"], seq_b=s["seq.bias"],
           out_w=np.stack([s["out.0.weight"], s["out.1.weight"]]), out_b=np.stack([s["out.0.bias"], s["out.1.bias"]]))
if BI:
    arr |= dict(A2=s["A2.weight"], seq2_w=s["seq2.weight"], seq2_b=s["seq2.bias"])
np.savez(j.HOME / "models" / f"{name}.npz", **arr, T=d["T"][heads.index(head)], meta=json.dumps(meta, ensure_ascii=False))
print(f"저장: models/{name}.npz (낱말 {len(d['vocab']):,}개, 기본 관점 {head})")
