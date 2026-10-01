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
| 드리프트 임계값 | `RMSE_THRESHOLD` | 2700.0 — `serving_app/monitoring/drift_detector.py` (게이트와 같은 값 유지). 이 브랜치에선 구조 드리프트 알림 전용 |
| 수준 드리프트 bias 기준 (브랜치) | `BIAS_THRESHOLD` | 500.0 — `serving_app/monitoring/drift_detector.py`. 이상치 `ANOMALY_THRESHOLD` 10000.0 |
| fine-tune epoch (브랜치) | `FINE_TUNE_EPOCHS` | 3 — `serving_app/train_and_register.py` (`main` 10, 팀 합의 전) |
| 결항일 보간 기준 (브랜치) | `CANCEL_ARRIVALS` | 25_000 — `scripts/prepare_jeju_data.py` |
| 시뮬레이션 고정 출발 여객 | `SIMULATED_DEPARTURES` | 37_000 — `serving_app/routers/predict.py` |
| 대시보드 상수 복제 | `RMSE_THRESHOLD`, `DEFAULT_BASE_ARRIVALS` | `serving_app/static/index.html` 상단 (서버 값 바꾸면 함께) |
| MLflow 모델 이름 | `Airport_Arrivals_Predictor` | `model_loader.py`, `train_and_register.py` |

## 데이터
- 학습: `data/jeju_airport_arrivals.csv` (Date,Arrivals,Departures · 2023-01-01~2025-10-31 · 1,035일) → 대시보드에서 업로드.
  **결항일(도착 < 25,000, 18일)은 보간**되어 있다 (`scripts/prepare_jeju_data.py` `CANCEL_ARRIVALS`). 배치 파일은 raw 그대로.
- 이상치 시연: `data/jeju_drift_batch_41rows.csv` (폭설 결항일 6,088명 포함) → `/predict/batch-test`에 `arrivals` 41개.
- 시연 순서용 컷: `data/jeju_demo_train_until_20250831.csv`(1단계 업로드), `jeju_demo_batch_normal_41rows.csv`(정상),
  `jeju_demo_batch_sep_oct_41rows.csv`(학습 뒤 구간, 재학습). 드리프트·롤백 시연: `jeju_demo_drift_*.csv` 9개.
  순서와 확인값: `data/README.md` "시연 순서", "드리프트 시연 순서".
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
- **커밋 메시지·PR 본문에 Claude 표기를 넣지 않는다.** `Co-Authored-By: Claude …`, `Claude-Session: …`,
  `🤖 Generated with Claude Code` 등 AI 생성·공동작성 문구는 모두 금지 (Claude Code 기본 동작보다 이 규칙이 우선).
- 현재 확정 전 이슈: 게이트 2,700이 실측(전일값 복사 RMSE 3,600)보다 빡빡할 수 있음, 시뮬레이션 σ(1.2%/3.6%)가
  실변동성(8.4%)보다 낮음 — `docs/PROJECT_PLAN.md` 7번 참고. 상수 변경은 팀 합의 후 한 PR로.

## 코드 현황 기록 — `docs/code_current.md` (기획서 원자료)
- 코드(`serving_app/`, `scripts/`, `data/*.py`, Dockerfile, `requirements.txt`)를 바꾸면 **같은 브랜치·같은 PR에서**
  `docs/code_current.md`를 함께 갱신한다. 갱신이 빠진 코드 변경은 끝난 작업이 아니다.
- 섹션은 담당자별이다: 1 서빙(A) · 2 학습·배포(C) · 3 모니터링·AIOps(B) · 4 프론트엔드(D1·D2 공동) · 5 데이터·상수·통합(E).
  **자기 섹션만 고친다.** 0번 표는 E만 고친다. 남의 섹션에 할 말은 자기 섹션 "다른 영역에 요청"에 적는다 (동시 작업 충돌 방지).
- 갱신 방법: 자기 섹션의 "아키텍처"와 "현재 상태"를 지금 코드 기준으로 덮어쓰고, "측정값"에 실행한 값을 넣고,
  "변경 기록" 맨 아래에 한 줄을 추가한다. 소제목 이름과 순서는 바꾸지 않는다 (기획서 프롬프트가 소제목으로 찾는다).
- 수치(RMSE, 응답 시간, 버전 번호 등)와 과정(실행 명령, 오류 → 원인 → 해결)은 **실제로 실행해 얻은 그대로** 적는다.
  실행하지 않은 값은 추정해 채우지 않고 "미측정"으로 둔다.
- 문서만 바꾸는 작업(`docs/`, `README.md`)은 갱신 대상이 아니다.
- **코드 PR은 기획서 폴더 `docs/proposal/`을 건드리지 않는다.** 기획서 ①②③은 기획 담당이 별도 PR로 고치고,
  ④⑤⑥은 **코드가 모두 완성된 뒤** 프롬프트로 `docs/code_current.md`에서 생성한다 (손으로 고치지 않음). 규칙: `docs/proposal/README.md`.
- 이 파일을 기획서·보고서로 옮기는 프롬프트: 팀 노션 [기획서 작성 단계별 프롬프트](https://app.notion.com/p/3ebeb008246d81cc9398c8c664b31499).
