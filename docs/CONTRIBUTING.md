# 협업 규칙 (CONTRIBUTING)

3일짜리 조별 프로젝트라 규칙은 최소한으로 둡니다. 핵심은 **충돌 없이, 항상 돌아가는 main** 입니다.

## 1. 브랜치 전략

```
main                         ← 항상 실행 가능한 상태. 직접 push 금지, PR로만 병합
 ├─ feat/api-mlflow-loader   ← 기능 단위 브랜치 (한 사람, 한 파일 묶음, 하루 이내)
 ├─ feat/monitor-versions
 ├─ feat/dashboard-forecast-card
 ├─ data/real-arrivals-csv
 └─ docs/plan-section-1-3
```

- 브랜치 이름: `feat/…`, `fix/…`, `data/…`, `docs/…` + 짧은 kebab-case 설명.
- 작업 시작 전 `git pull origin main` → 브랜치 생성. 끝나면 PR. 브랜치는 병합 후 삭제.
- 담당 파일 밖을 고쳐야 하면 먼저 소유자에게 말하기 (`docs/PROJECT_PLAN.md` 6번 표).

## 2. 커밋 메시지

`<type>: <무엇을, 한국어 또는 영어 한 줄>` — type은 `feat` `fix` `refactor` `data` `docs` `chore` `test`.

```
feat: implement _load_from_mlflow for Production model
fix: keep recent_predictions window at 21 entries
data: replace sample CSV with 2023-2025 Jeju arrivals
docs: fill plan section ① pain point
```

- TODO 이식 커밋은 어떤 TODO(실습 번호)인지 본문에 적기 (예: `실습 2-1`).
- 모델 파일(`*.keras`, `scaler.pkl`), `mlruns/`, `mlflow.db`, `logs/`, `data/uploads/*`는 커밋하지 않음 (`.gitignore` 처리됨).

## 3. PR 규칙

- PR 템플릿(`.github/PULL_REQUEST_TEMPLATE.md`)의 체크리스트를 채운다.
- **리뷰어 1명 승인** 후 병합 (Squash merge 권장). 급하면 슬랙에서 구두 승인 후 본인 병합, PR에 한 줄 남기기.
- PR 하나 = 기능 하나. 300줄을 넘기면 나눈다.
- 병합 전 최소 확인:
  ```bash
  python -m compileall -q data scripts serving_app     # 문법
  uvicorn serving_app.main:app --port 8000             # 서버가 뜨는가
  curl localhost:8000/health                           # {"status":"ok",...}
  ```
  TODO를 구현한 PR은 관련 완료 기준(README "완료 기준")의 실제 실행 로그/캡처를 PR에 첨부.

## 4. 이슈 / 작업 관리

- 할 일은 GitHub Issue로 만들고(템플릿: `작업`, `버그`), PR 본문에 `Closes #번호`.
- 라벨: `sprint-1`, `sprint-2`, `api`, `monitoring`, `deploy`, `dashboard`, `data`, `docs`, `blocked`.
- 막히면 30분 이상 혼자 붙들지 말고 `blocked` 라벨 + 슬랙 공유. 트러블슈팅은 "오류 메시지 → 원인 → 해결 명령 → 전후 결과" 형식으로 이슈에 남기면 기획서·서브노트에 그대로 재사용 가능.

## 5. 로컬 실행 공통 규칙

- Python **3.11** (Dockerfile과 동일). 가상환경은 프로젝트 루트의 `.venv/`로 통일한다 (.gitignore 처리됨).

  ```bash
  # macOS / Linux
  git clone https://github.com/wonn2k/skala-model-serving.git && cd skala-model-serving
  python3.11 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
  ```powershell
  # Windows (PowerShell)
  git clone https://github.com/wonn2k/skala-model-serving.git; cd skala-model-serving
  py -3.11 -m venv .venv
  .venv\Scripts\Activate.ps1      # 실행 정책 오류 시: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```
  tensorflow 설치에 수 분 걸릴 수 있다. VS Code에서는 `Python: Select Interpreter`로 `.venv`를 선택한다.
- 항상 **프로젝트 루트**에서 실행 (`serving_app/`, `scripts/` 안에서 실행하면 상대경로가 깨짐).
- 포트는 **8000** 으로 통일 (대시보드 `http://localhost:8000/`).
- 환경변수: `LOADING_MODE=lazy|eager`, `MODEL_SOURCE=local|mlflow`.
- 스케일러는 Day1에서 한 번 만들고 다시 fit하지 않음. 실데이터 CSV를 교체했으면 `scripts/train_baseline_v1.py`부터 다시 실행.

## 6. 도메인 용어 통일

| 코드 | 한국어 표기 | 비고 |
|---|---|---|
| `arrivals` / `Arrivals` | 도착 여객 수 (명) | 예측 대상 |
| `departures` / `Departures` | 출발 여객 수 (명) | 보조 피처 |
| `RMSE_GATE`, `RMSE_THRESHOLD` | 배포 게이트 / 드리프트 임계값 = 2,700명 | 같은 값 유지 |
| `Airport_Arrivals_Predictor` | MLflow 모델 이름 | 바꾸면 `model_loader.py`, `train_and_register.py` 함께 |
| 혼잡 등급 | 여유 / 보통 / 혼잡 | 대시보드 카드 (Sprint 2) |
