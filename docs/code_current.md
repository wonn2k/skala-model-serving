# 코드 현황 (code_current)

기획서 ①~⑥의 **원자료**다. 코드를 바꾸는 PR마다 이 파일을 함께 갱신한다 (규칙: `CLAUDE.md` "코드 현황 기록").
이 파일을 보고서로 옮기는 프롬프트는 팀 노션 [기획서 작성 단계별 프롬프트](https://app.notion.com/p/3ebeb008246d81cc9398c8c664b31499)에 있다.

## 작성 규칙

- **자기 영역 섹션만 고친다.** 영역은 아래 1~5번이다. 여러 명이 동시에 고치므로 남의 섹션을 건드리면 병합 충돌이 난다.
- 영역 섹션은 두 부분이다.
  - **현재 상태**: 지금 코드 기준으로 덮어쓴다. 더 이상 사실이 아닌 문장은 지운다.
  - **변경 기록**: 맨 아래에 한 줄을 추가한다. 기존 줄은 지우거나 고치지 않는다.
- 변경 기록 형식: `- YYYY-MM-DD HH:MM | 작성자 | 무엇을 바꿨나 | 확인한 수치·결과와 실행 명령 | 브랜치 또는 PR`
- 수치(RMSE, 응답 시간, 버전 번호, 예측 개수 등)와 과정은 **실제로 실행해 얻은 그대로** 적는다. 반올림하거나 요약하지 않는다. 실행하지 않은 값은 추정해 채우지 않고 `미측정`으로 둔다.
- 오류를 만나면 6번에 `증상 → 원인 → 해결 명령 → 전후 결과`로 남긴다.
- 캡처를 찍으면 7번 표의 해당 행을 채운다.
- 상태가 달라진 단계는 0번 표도 고친다 (0번 표는 누구나 자기 행만 고친다).

## 0. 한눈에 보기

| 파이프라인 단계 | 상태 | 근거 |
|---|---|---|
| CSV 업로드 (`POST /data/upload`, `GET /data/status`) | 동작 | 스켈레톤 제공. `data/jeju_airport_arrivals.csv` 1,035행 업로드 (PC 1대) |
| Day1 baseline 학습 (`scripts/train_baseline_v1.py`) | 동작 | RMSE 2,571명, 게이트 2,700명 통과 (PC 1대) |
| Day2 MLflow 학습·게이트·Production 승격 (`serving_app/train_and_register.py`) | 동작 | RMSE 2,267명, 게이트 통과 후 Production 승격 (PC 1대) |
| 예측 서빙 `POST /predict` (로컬 모델) | 코드 완성 | 응답 확인 기록 없음 (미측정) |
| MLflow Production 모델 서빙 (`MODEL_SOURCE=mlflow`) | 미구현 | TODO 1 `_load_from_mlflow` |
| 드리프트 판정 (`POST /predict/batch-test`) | 미구현 | TODO 2 `compute_rmse`, TODO 3 `batch_test` |
| 드리프트 감지 시 자동 재학습 | 미구현 | TODO 4 `check_and_trigger` |
| 드리프트 시뮬레이션 스크립트 | 미구현 | TODO 5 `send_batch` |
| Production 버전 조회 `GET /monitor/versions` | 미구현 | 설계안만 있음 (`docs/API_SPEC.md`) |
| 대시보드 예측값·혼잡 등급 카드, Production 버전 표기 | 미구현 | 업로드·드리프트 시뮬레이션·재학습 로그 카드는 스켈레톤에 있음 |
| Docker 컨테이너 재현 | 미확인 | TODO 1 구현 전에는 기동 실패 예상 (2번 참고) |

## 1. 서빙 API

대상: `serving_app/main.py`, `serving_app/model_loader.py`, `serving_app/routers/predict.py`, `serving_app/routers/health.py`, `serving_app/schemas.py`

### 현재 상태

- `GET /health`, `POST /predict`는 스켈레톤 그대로 완성되어 있다.
- `/predict` 입력은 최근 20일 시퀀스(`arrivals`, `departures`)이며 길이가 20이 아니거나 값이 음수이면 422로 거부한다 (`schemas.py`).
- 모델 로딩은 환경변수로 바꾼다: `LOADING_MODE=lazy`(기본) 또는 `eager`, `MODEL_SOURCE=local`(기본) 또는 `mlflow`.
- `model_loader._load_from_mlflow()`는 TODO 상태다. `MODEL_SOURCE=mlflow`로 모델을 불러오면 `NotImplementedError`가 난다.
- `routers/predict.batch_test()`는 TODO 상태다. 예측 없이 빈 `predictions`를 돌려준다.
- 재배포 후 서빙 모델 반영: `_model_cache`가 자동으로 갱신되지 않는다 (미해결, 수업 가이드 부록 1의 6번).
- Lazy / Eager 서버 시작 시간, 첫 요청 응답 시간: 미측정.
- `/predict` 응답 시간: 미측정.

### 변경 기록

- 2026-09-30 21:40 | 초기 작성 | 스켈레톤을 공항 도메인으로 치환한 상태를 기록 | 해당 없음 | main

## 2. 학습·배포

대상: `scripts/train_baseline_v1.py`, `serving_app/train_and_register.py`, `serving_app/Dockerfile`, `serving_app/docker-compose.yml`, `serving_app/routers/monitor.py`(신설 예정)

### 현재 상태

- Day1 baseline: RMSE 2,571명으로 게이트 2,700명 통과. 전일값 복사 기준선 3,600명 대비 -29%. PC 1대에서 실행한 결과다.
- Day2 base 학습(100 epoch, seed 42): RMSE 2,267명(2266.94)으로 게이트 통과, `Airport_Arrivals_Predictor`가 Production으로 승격됐다. PC 1대에서 실행한 결과다.
- 모델 파일(`*.keras`, `scaler.pkl`)과 MLflow 기록(`mlflow.db`, `mlruns/`)은 커밋하지 않는다. 각자 PC에서 업로드 → baseline → MLflow 학습을 직접 실행해야 한다.
- fine-tuning(`fine_tune`)은 스켈레톤에 구현되어 있다: Production 가중치에서 이어서 10 epoch, 학습률 1e-4. 호출하는 쪽(TODO 4)이 미구현이라 실행 기록은 없다.
- `GET /monitor/versions`: 미구현.
- Docker: 미확인. Dockerfile은 이미지 빌드 중에 baseline과 MLflow 학습을 실행하고 `MODEL_SOURCE=mlflow`, `LOADING_MODE=eager`로 시작하므로, TODO 1이 구현되기 전에는 컨테이너 기동이 실패할 것으로 예상된다.
- 다른 PC에서의 게이트 통과 여부: 미측정.

### 변경 기록

- 2026-09-30 21:40 | 초기 작성 | Day1·Day2 실행 결과 기록 | Day1 RMSE 2,571명 (`python scripts/train_baseline_v1.py`), Day2 RMSE 2,267명 (`python serving_app/train_and_register.py`) | main

## 3. 모니터링·AIOps

대상: `serving_app/monitoring/drift_detector.py`, `serving_app/monitoring/retrain_trigger.py`, `scripts/simulate_drift.py`

### 현재 상태

- `drift_detector.compute_rmse()`: TODO 상태.
- `retrain_trigger.check_and_trigger()`: 드리프트 판정 시 `[WARN]` 로그까지만 남긴다. 재학습 호출 부분이 TODO 상태다.
- `simulate_drift.send_batch()`: TODO 상태.
- 판정 기준: 최근 21건(`WINDOW_SIZE`)의 예측·실제 쌍으로 RMSE를 구해 2,700명(`RMSE_THRESHOLD`)과 비교한다.
- 시뮬레이션 배치: 41개(시퀀스 20 + 윈도우 21) 랜덤워크, 일별 변동성 정상 1.2% / 드리프트 3.6%.
- 정상 배치 RMSE, 드리프트 배치 RMSE, 실데이터 드리프트 배치(`data/jeju_drift_batch_41rows.csv`) RMSE: 미측정.
- `logs/aiops.log`의 `[WARN]` → `[INFO]` → `[OK]` 기록: 미확인.
- 재학습 후 RMSE와 승격 버전: 미측정.

### 변경 기록

- 2026-09-30 21:40 | 초기 작성 | TODO 상태 기록 | 해당 없음 | main

## 4. 대시보드

대상: `serving_app/static/index.html`

### 현재 상태

- 스켈레톤이 제공하는 카드 4개가 있다: 공항 도착 여객 데이터 업로드, 드리프트 시뮬레이션, 드리프트 감지 기반 재학습 파이프라인, 재학습 로그.
- 상수 `RMSE_THRESHOLD = 2700`, `DEFAULT_BASE_ARRIVALS = 37000`은 서버 값을 복제한 것이다. 서버 값을 바꾸면 함께 바꾼다.
- 내일 예측값·혼잡 등급 카드: 미구현. 등급 경계 초안은 40,177명 초과 혼잡, 34,962명 미만 여유, 그 사이 보통이다.
- 현재 Production 버전 표기: 미구현.
- 카드가 `/predict`에 넣을 최근 20일 시퀀스를 어디서 가져올지: 미정. `GET /data/status`는 요약(행 수, 기간, 최소·최대)만 돌려준다.

### 변경 기록

- 2026-09-30 21:40 | 초기 작성 | 스켈레톤 카드 구성 기록 | 해당 없음 | main

## 5. 데이터·상수

대상: `data/`, 코드 안의 운영 상수

### 현재 상태

| 항목 | 값 | 위치 |
|---|---|---|
| 학습 데이터 | 제주공항 일별 여객 2023-01-01 ~ 2025-10-31, 1,035일 | `data/jeju_airport_arrivals.csv` |
| 드리프트 시연 데이터 | 2025-01-09 ~ 2025-02-18, 41일 (폭설 결항일 17,093명·6,088명 포함) | `data/jeju_drift_batch_41rows.csv` |
| 도착 여객 평균 / 중앙값 | 37,218명 / 37,820명 | `data/README.md` |
| 도착 여객 표준편차 | 4,413명 | `data/README.md` |
| 도착 여객 최소 / 최대 | 1,042명 / 46,954명 | `data/README.md` |
| 사분위 Q1 / Q3 | 34,962명 / 40,177명 | `data/README.md` |
| 일별 변화율 표준편차 | 19.6% (전체) / 8.4% (2만 명 미만 결항일 제외) | `data/README.md` |
| 전일값 복사 RMSE | 3,600명 | `data/README.md` |
| 배포 게이트 `RMSE_GATE` | 2,700명 | `serving_app/train_and_register.py`, `scripts/train_baseline_v1.py` |
| 드리프트 임계값 `RMSE_THRESHOLD` | 2,700명 | `serving_app/monitoring/drift_detector.py` |
| 판정 윈도우 `WINDOW_SIZE` | 21건 | `serving_app/monitoring/drift_detector.py` |
| 입력 시퀀스 길이 `SEQ_LEN` | 20일 | `data/features.py` |
| 시뮬레이션 고정 출발 여객 `SIMULATED_DEPARTURES` | 37,000명 | `serving_app/routers/predict.py` |
| MLflow 모델 이름 | `Airport_Arrivals_Predictor` | `serving_app/model_loader.py`, `serving_app/train_and_register.py` |

- 결측 처리: 2023-01-24 (도착·출발 모두 0) 1건을 전날·다음날 평균으로 보간했다. 결항으로 급감한 날은 실제 값 그대로 둔다.
- 확정 전 이슈: 게이트 여유가 129명으로 얇다. 시뮬레이션 변동성(1.2% / 3.6%)이 실데이터(8.4%)보다 낮다.

### 변경 기록

- 2026-09-30 21:40 | 초기 작성 | 데이터와 상수 현재 값 기록 | 해당 없음 | main

## 6. 트러블슈팅

형식: `증상 → 원인 → 해결 명령 → 전후 결과`. 작성자와 시각을 함께 적는다. 맨 아래에 추가한다.

(아직 없음)

## 7. 증빙 목록 (기획서 ⑥)

캡처 파일은 `docs/snapshots/`에 둔다. 찍은 사람이 자기 행을 채운다.

| 스냅샷 | 무엇을 보여주나 | 상태 | 파일 | 찍은 사람·시각 |
|---|---|---|---|---|
| `/predict` 응답 (로컬 모델) | `model_version`이 `v1-local` | 미촬영 | | |
| `/predict` 응답 (MLflow 모델) | `model_version`이 `production`으로 전환 | 미촬영 | | |
| `/predict` 422 응답 | 시퀀스 19개, 음수 값 입력이 거부됨 | 미촬영 | | |
| `train_and_register.py` 실행 로그 | `[GATE PASSED]`와 RMSE | 미촬영 | | |
| 컨테이너 빌드·실행 로그 | Docker로 재현됨 | 미촬영 | | |
| 컨테이너의 Swagger 화면 | 컨테이너에서도 같은 API가 동작 | 미촬영 | | |
| `simulate_drift.py` 실행 결과 | 정상 배치와 드리프트 배치의 `drift_check` | 미촬영 | | |
| 재학습 로그 패널 | `[WARN]` → `[INFO]` → `[OK]` 순서 | 미촬영 | | |
| 재학습 후 `/predict` 응답 | 새 Production 버전 반영 | 미촬영 | | |
| 대시보드 예측·혼잡 등급 카드 | 내일 예측값과 등급 | 미촬영 | | |
| 대시보드 Production 버전 표기 | 재학습 전후 버전 변화 | 미촬영 | | |
