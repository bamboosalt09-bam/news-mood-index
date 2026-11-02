# 공개용 고정본 만들기 (2026-11-02, 사용자 결정): Firebase에 올릴 화면을 '특정 시점 기준'으로 멈춘 사본을 web/dist/에 만든다.
# API 키는 쓰지 않는다 (AI 해설·대표 확인·업체 판정은 build 때 캐시에 있던 것만). 배포는 web/DEPLOY.md 순서대로 사람이 직접.
# python web/freeze.py                          # 지금 web/public/data/latest.json을 고정
# python web/freeze.py --from data/archive/2026-11-04.json   # 지난 날 자료(web/public 기준 경로)로 고정
# python web/freeze.py --note "해커톤 시연용"     # 화면 위 알림에 덧붙일 한 줄
import sys, json, shutil
from pathlib import Path
WEB = Path(__file__).resolve().parent
PUB, DIST = WEB / "public", WEB / "dist"
sys.path.insert(0, str(WEB))
import build

arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
src = PUB / (arg("--from") or "data/latest.json")
o = json.loads(src.read_text(encoding="utf-8"))
build.check(src)  # 형식 검사를 통과한 자료만 고정
o["frozen"] = {"at": o["updated"], "source": src.relative_to(PUB).as_posix()} | ({"note": arg("--note")} if arg("--note") else {})
if DIST.exists():
    shutil.rmtree(DIST)
DIST.mkdir()
for f in ("index.html", "about.html", "app.js", "style.css"):
    shutil.copy2(PUB / f, DIST / f)
(DIST / "data").mkdir()
(DIST / "data" / "latest.json").write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"고정본 {DIST} ← {src.name} (기준 {o['updated']}). 배포는 web/DEPLOY.md")
