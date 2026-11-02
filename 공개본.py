# 공개 GitHub 저장소용 사본 (2026-11-02, 사용자 결정 '코드·문서만 공개'): git이 추적하는 파일 중 공개해도 되는 것만 대상 폴더로 복사.
# 빼는 것: 수집한 기사 제목(headlines/), 라벨·본문 자료(labeled/, 기준 문서·검사 스크립트만 남김), 시세·캐시 자료(data/),
#          제목이 많이 든 실험 결과(시험지 점수 목록·제목 표본), 에이전트 설정. 기록(history)은 옮기지 않음.
# python 공개본.py <대상 폴더>     (대상 폴더의 .git은 그대로 두고 나머지를 새로 채움)
import sys, subprocess, shutil, re
from pathlib import Path
HOME = Path(__file__).resolve().parent
KEEP_LABELED = {"labeled/라벨기준.md", "labeled/check_labels.py"}
DROP = re.compile(r"^(headlines/|data/|labeled/|\.claude/|\.codex/|CLAUDE\.md$|AGENTS\.md$)"
                  r"|^실험/결과/.*(\.tsv$|_eval_show\.json$|/묶음[^/]*\.txt$|/입력_[^/]*\.txt$|발견_틀림\.txt$)")
MAX = 2_000_000  # 이보다 큰 파일은 빼고 알림
TEXT = {".md", ".py", ".js", ".html", ".json", ".txt", ".tsv", ".log", ".bat", ".css"}
HOMEDIR = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+([\\/])")


def files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=HOME, capture_output=True, check=True).stdout.decode("utf-8").split("\0")
    return [f for f in out if f and (f in KEEP_LABELED or not DROP.search(f))]


if __name__ == "__main__":
    dst = Path(sys.argv[1]).resolve()
    dst.mkdir(parents=True, exist_ok=True)
    for p in dst.iterdir():  # .git만 남기고 비움
        if p.name != ".git":
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    n, big = 0, []
    for f in files():
        src = HOME / f
        if not src.exists():
            continue
        if src.stat().st_size > MAX:
            big.append(f); continue
        (dst / f).parent.mkdir(parents=True, exist_ok=True)
        try:  # 내 컴퓨터 사용자 폴더 경로(실명이 든)는 ~로
            assert src.suffix in TEXT
            (dst / f).write_bytes(HOMEDIR.sub(lambda m: "~" + m.group(1), src.read_text(encoding="utf-8")).encode("utf-8"))
        except (AssertionError, UnicodeDecodeError):
            shutil.copy2(src, dst / f)
        n += 1
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in dst.rglob("*") if p.is_file() and p.suffix in (".md", ".py", ".js", ".html", ".json", ".txt", ".tsv", ".log"))
    assert not re.search(r"sk-ant-[A-Za-z0-9_-]{10}|gh[op]_[A-Za-z0-9]{20}|@ts\.hs\.kr|Users[\\/]+최", text), "키·개인 메일 모양이 남아 있음"
    print(f"{n}개 파일 → {dst}" + (f", 큰 파일 뺌: {big}" if big else ""))
