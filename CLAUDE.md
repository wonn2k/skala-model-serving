# CLAUDE.md — 프로젝트 규칙 (Claude Code가 자동으로 읽는 파일)

## 프로젝트 요약
- 공항 운영 담당자(B2B)를 위한 **제주공항 다음날 도착 여객 수 예측** LSTM 모델을 FastAPI로 서빙하고,
  MLflow 게이트 → 드리프트 감지 → 자동 fine-tuning → 재배포(AIOps)까지 이어지는 조별 미니 프로젝트.
- SKALA "모델 서빙 및 AIOps" 실습 스켈레톤(HAIC 종가 예측)을 도메인 치환한 것. 원 구조를 최대한 유지한다.
- 기획·역할·스프린트: `docs/PROJECT_PLAN.md` · 협업 규칙: `docs/CONTRIBUTING.md` · API: `docs/API_SPEC.md` · 데이터: `data/README.md`

## 반드시 지킬 것
- **폴더 구조와 백엔드 동작 방식을 바꾸지 않는다.** 새 기능은 기존 패턴을 따른다
  (예: 새 API는 `serving_app/routers/<name>.py` 신설 + `main.py`에 `include_router` 한 줄).
- 모델 아키텍처(`lstm_model.py`), 피처 정의(`data/features.py`, SEQ_LEN=20), 학습/재학습 흐름은 변경 금지.
- 실습 TODO 5개는 개인 실습(HAIC) 구현을 **필드명만 바꿔 이식**한다. 위치:
  `model_loader._load_from_mlflow`, `routers/predict.batch_test`, `monitoring/drift_detector.compute_rmse`,
  `monitoring/retrain_trigger.check_and_trigger`, `scripts/simulate_drift.send_batch`.
- 항상 **프로젝트 루트**에서 실행한다 (상대경로 `data/`, `serving_app/models/`, `logs/` 기준).
- 포트는 **8000**. 대시보드 `http://localhost:8000/`, Swagger `/docs`.
- 커밋 금지: `serving_app/models/*.keras`, `scaler.pkl`, `mlruns/`, `mlflow.db`, `logs/`, `data/uploads/*` (.gitignore 처리됨).

## 도메인 용어 / 상수 (한 곳에서만 바꾼다)
| 개념 | 코드 | 값 / 위치 |
|---|---|---|
| 예측 대상 | `arrivals` / CSV `Arrivals` | 도착 여객 수(명) |
| 보조 피처 | `departures` / CSV `Departures` | 출발 여객 수(명) |
| 배포 게이트 | `RMSE_GATE` | 2700.0 — `serving_app/train_and_register.py` (`scripts/train_baseline_v1.py`에도 동일 값) |
| 드리프트 임계값 | `RMSE_THRESHOLD` | 2700.0 — `serving_app/monitoring/drift_detector.py` (게이트와 같은 값 유지) |
| 시뮬레이션 고정 출발 여객 | `SIMULATED_DEPARTURES` | 37_000 — `serving_app/routers/predict.py` |
| 대시보드 상수 복제 | `RMSE_THRESHOLD`, `DEFAULT_BASE_ARRIVALS` | `serving_app/static/index.html` 상단 (서버 값 바꾸면 함께) |
| MLflow 모델 이름 | `Airport_Arrivals_Predictor` | `model_loader.py`, `train_and_register.py` |

## 데이터
- 학습: `data/jeju_airport_arrivals.csv` (Date,Arrivals,Departures · 2023-01-01~2025-10-31 · 1,035일) → 대시보드에서 업로드.
- 드리프트 시연: `data/jeju_drift_batch_41rows.csv` (폭설 결항일 6,088명 포함) → `/predict/batch-test`에 `arrivals` 41개.
- 원자료 `data/raw/`는 출처 보관용. 재생성: `python scripts/prepare_jeju_data.py`.

## 로컬 실행 (venv)
```bash
python3.12 -m venv .venv && source .venv/bin/activate   # Windows: py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
# Python 3.11 또는 3.12 (3.13은 tensorflow 휠 호환 불안, 3.10은 numpy 2.4 미지원)
pip install -r requirements.txt
uvicorn serving_app.main:app --host 0.0.0.0 --port 8000
# 대시보드에서 data/jeju_airport_arrivals.csv 업로드 후 별도 터미널에서
python scripts/train_baseline_v1.py
python serving_app/train_and_register.py
MODEL_SOURCE=mlflow uvicorn serving_app.main:app --port 8000
```

## 작업 흐름
- `main`에 직접 push 금지. `feat/…` 브랜치 → PR → 리뷰 1명 → Squash merge (`docs/CONTRIBUTING.md`).
- PR 전: `python -m compileall -q data scripts serving_app`, 서버 기동, `/health` 확인.
- 커밋 메시지: `feat|fix|refactor|data|docs|chore: 한 줄 요약`.
- 현재 확정 전 이슈: 게이트 2,700이 실측(전일값 복사 RMSE 3,600)보다 빡빡할 수 있음, 시뮬레이션 σ(1.2%/3.6%)가
  실변동성(8.4%)보다 낮음 — `docs/PROJECT_PLAN.md` 7번 참고. 상수 변경은 팀 합의 후 한 PR로.
