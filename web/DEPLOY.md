# 공개 배포 (Firebase Hosting)

- 공개 주소: https://news-mood-index-2026-9c174.web.app (프로젝트 `news-mood-index-2026-9c174`, `web/.firebaserc`)
- 첫 배포: 2026-11-02 22:00, 고정 시각 11/02 21:50 (AI 해설은 키 없이 시험용 문구)

공개 사이트는 **특정 시점 기준으로 멈춘 고정본**(`web/dist/`)만 올린다. 실시간 갱신(주기 수집기 + build)은 이 PC의 `web/public/`에서만 돈다.
API 키는 쓰지 않는다: AI 해설·대표 제목 확인·업체 판정은 build 때 캐시에 있던 것만 들어가고, 없으면 시험용 해설·확인 없음으로 나온다.

## 처음 한 번 (사람이 직접)

1. Node.js가 있으면 `npm install -g firebase-tools`
2. `firebase login` (브라우저 로그인 — Claude가 대신 하지 않음)
3. `cd web` 후 `firebase use --add` 로 올릴 프로젝트 고르기 (`web/.firebaserc`가 생김)

## 올릴 때마다

1. `python web/freeze.py` — 지금 `web/public/data/latest.json`을 고정 (지난 날은 `--from data/archive/<날짜>.json`, 알림 한 줄은 `--note "..."`)
2. `cd web` 후 `firebase deploy --only hosting` — `firebase.json`의 public이 `dist`라 고정본만 올라감

화면 위에 "고정본: <시각> 기준으로 멈춘 화면입니다"가 뜨고, 실시간 칸은 '잠정 · 고정본'으로 표시된다.
