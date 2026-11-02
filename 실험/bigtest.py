# 시험지 채점. 기본은 큰 시험지(깨끗한 부분 = 학습 라벨과 안 겹친 1,498개가 판단 기준, 전체는 참고).
# python 실험/bigtest.py [시험=ko_새시험1500.tsv] [lex=사전파일] 점수.json|판정모델이름 ...
#   json = 제목→점수 (seqexp.py eval show=…가 만든 것). 이름이면 evaluate.scores로 잼. lex=는 그 뒤 항목의 안전장치 사전
import sys, json
import numpy as np
sys.path.insert(0, r"C:\english_folder\somethings")
import judge as j
import evaluate as e
sys.stdout.reconfigure(encoding="utf-8")
args = sys.argv[1:]
test = next((a[3:] for a in args if a.startswith("시험=")), "ko_큰시험2000.tsv")
rows = [l.split("\t") for l in (j.HOME / "labeled" / "시험" / test).read_text(encoding="utf-8").splitlines()[1:]]
texts, y = [r[-1] for r in rows], np.array([float(r[0]) for r in rows])
parts = {"전체": np.ones(len(rows), bool)}
if test == "ko_큰시험2000.tsv":
    clean = np.ones(len(rows), bool); clean[json.load(open(j.HOME / "실험" / "결과" / "큰시험_학습겹침.json", encoding="utf-8"))] = False
    parts = {"깨끗한": clean} | parts
V = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
for a in args:
    if a.startswith("시험="):
        continue
    if a.startswith("lex="):
        j.LEXICON = a[4:]; continue
    s = np.array([json.load(open(a, encoding="utf-8"))[t] for t in texts]) if a.endswith(".json") else e.scores(a, texts)
    out = []
    for name, k in parts.items():
        m = e.screen(s[k], y[k]); s7 = V[np.abs(s[k][:, None] - V).argmin(1)]
        out.append(f"{name} {k.sum()}: 7단계 {np.mean(s7 == y[k]):.1%} 맞음 {m['맞음']:.1%} 헛경보 {m['헛경보']:.1%} 놓침 {m['놓침']:.1%} "
                   f"방향 틀림 {m['방향 틀림']} 평균 차이 {np.abs(s[k] - y[k]).mean():.3f}")
    print(f"[{a.replace(chr(92), '/').split('/')[-1]} | 사전 {j.LEXICON}]\n  " + "\n  ".join(out))
