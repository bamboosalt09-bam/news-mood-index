# seqexp.py로 학습한 형태소 CMSM(.pt)을 judge가 읽는 numpy 모델(.npz)로 옮긴다
# python 실험/cmsm_to_numpy.py models/seq_cmsm_d_morph_lexinit_seqlr=0.001.pt ko_cmsm_m2-감성 [featlex=특징사전]
#   특징 사전과 안전장치 사전(지금 lexicon.tsv)을 모델에 묶는다 → lexicon.tsv를 고쳐도 판정이 안 바뀐다. featlex: 학습 때 특징에 쓴 사전 (2026-10-29)
import json, sys
import numpy as np
import torch
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j

src, name = sys.argv[1], sys.argv[2]
feat = next((a[8:] for a in sys.argv[3:] if a.startswith("featlex=")), j.LEXICON)
d = torch.load(src, weights_only=False)
s = {k: v.numpy().astype(np.float32) for k, v in d["state"].items()}
meta = {"kind": "cmsm", "buckets": 16384, "ngram": 3, "lexicon": j.load_lexicon(feat), "guard_lexicon": j.load_lexicon(j.LEXICON), "hint": j.HINT, "phrase": True,
        "labels": ["-2", "-1", "-0.5", "0", "0.5", "1", "2"], "numeric": True, "stretch": d["stretch"], "guard": 1.0,
        "neutral": [0.9, 0.85], "vocab": d["vocab"], "L": 40, "D": int(s["A.weight"].shape[1] ** 0.5), "cap": 4.0, "tokens": "kiwi 형태/품사", "source": src}
np.savez(j.HOME / "models" / f"{name}.npz", emb=s["emb.weight"], lin_w=s["lin.weight"], lin_b=s["lin.bias"], A=s["A.weight"],
         v0=s["v0"], seq_w=s["seq.weight"], seq_b=s["seq.bias"], out_w=s["out.weight"], out_b=s["out.bias"],
         T=d["T"], meta=json.dumps(meta, ensure_ascii=False))
print(f"저장: models/{name}.npz (형태소 {len(d['vocab']):,}개)")
