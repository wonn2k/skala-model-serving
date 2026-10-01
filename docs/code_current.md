# 코드 현황 (code_current)

기획서 ①~⑥의 **원자료**다. 코드를 바꾸는 PR마다 이 파일을 함께 갱신한다 (규칙: `CLAUDE.md` "코드 현황 기록").
이 파일을 보고서로 옮기는 프롬프트는 팀 노션 [기획서 작성 단계별 프롬프트](https://app.notion.com/p/3ebeb008246d81cc9398c8c664b31499)에 있다.

## 작성 규칙

- **섹션 하나에 주인 한 명.** 자기 섹션(1~5번 중 하나)만 고친다. 남의 섹션은 한 글자도 건드리지 않는다. 0번 표는 E만 고친다.
  같은 파일이라도 섹션 사이가 멀리 떨어져 있어 서로 다른 섹션만 고치면 병합 충돌이 나지 않는다.
- 남의 영역에 할 말이 있으면 (예: "A가 캐시를 비워야 내 재배포가 보인다") 자기 섹션 "다른 영역에 요청"에 적고 PR 리뷰에서 말한다. 남의 섹션에 대신 적지 않는다.
- 섹션 구성은 모두 같다. 소제목을 지우거나 순서를 바꾸지 않는다 (프롬프트가 소제목 이름으로 찾는다).

| 소제목 | 어떻게 쓰나 | 어디에 쓰이나 |
|---|---|---|
| 담당 / 대상 파일 | 바뀌면 고친다 | 협업 계획 |
| 아키텍처 | 코드 기준으로 **덮어쓴다.** 항목 이름(역할·구성 요소·흐름·연결·설정·실행)은 고정 | ④ 아키텍처 구성도 (프롬프트 4가 6개 섹션의 "아키텍처"를 합쳐 전체 구성도를 만든다) |
| 현재 상태 | 지금 코드 기준으로 **덮어쓴다.** 사실이 아닌 문장은 지운다 | ③ 운영 설계, ⑤ API 명세 |
| 측정값 | 실행해서 얻은 값만. 반올림·요약 금지. 안 했으면 `미측정` | ② 운영 목표, ③ 운영 설계 |
| 트러블슈팅 | `증상 → 원인 → 해결 명령 → 전후 결과`. 맨 아래에 추가 | 구현 보고서 |
| 증빙 | 찍은 캡처를 `docs/snapshots/`에 두고 표를 채운다 | ⑥ 동작 화면 스냅샷 |
| 다른 영역에 요청 | 남의 코드가 바뀌어야 내 것이 되는 일 | 체크포인트 |
| 변경 기록 | 맨 아래에 한 줄 **추가**. 기존 줄은 안 고친다. `- YYYY-MM-DD HH:MM \| 작성자 \| 무엇을 바꿨나 \| 확인한 수치·결과와 실행 명령 \| 브랜치 또는 PR` | 구현 보고서 |

## 0. 한눈에 보기 (E만 고친다)

체크포인트마다 E가 각 섹션 "현재 상태"를 보고 이 표를 갱신한다. 다른 사람은 자기 섹션만 고친다.

| 파이프라인 단계 | 담당 | 상태 | 근거 |
|---|---|---|---|
| CSV 업로드 (`POST /data/upload`, `GET /data/status`) | C | 동작 | 스켈레톤 제공. `data/jeju_airport_arrivals.csv` 1,035행 업로드 (PC 1대) |
| Day1 baseline 학습 (`scripts/train_baseline_v1.py`) | C | 동작 | RMSE 2,571명, 게이트 2,700명 통과 (PC 1대) |
| Day2 MLflow 학습·게이트·Production 승격 (`serving_app/train_and_register.py`) | C | 동작 | RMSE 2,267명, 게이트 통과 후 Production 승격 (PC 1대) |
| 예측 서빙 `POST /predict` (로컬 모델) | A | 코드 완성 | 응답 확인 기록 없음 (미측정) |
| MLflow Production 모델 서빙 (`MODEL_SOURCE=mlflow`) | A | 미구현 | TODO 1 `_load_from_mlflow` |
| 드리프트 판정 (`POST /predict/batch-test`) | B | 미구현 | TODO 2 `compute_rmse`, TODO 3 `batch_test` |
| 드리프트 감지 시 자동 재학습 | B | 미구현 | TODO 4 `check_and_trigger` |
| 드리프트 시뮬레이션 스크립트 | B | 미구현 | TODO 5 `send_batch` |
| `GET /data/status`의 `recent` (최근 20일) | C | 미구현 | 설계만 있음 (2번 참고) |
| Production 버전 조회 `GET /monitor/versions` | C | 미구현 | 설계안만 있음 (`docs/API_SPEC.md`) |
| 대시보드 예측값·혼잡 등급 카드 | D1 (프론트엔드) | 미구현 | |
| 대시보드 CSV 배치 전송, Production 버전 표기 | D2 (프론트엔드) | 미구현 | 업로드·드리프트 시뮬레이션·재학습 로그 카드는 스켈레톤에 있음 |
| Docker 컨테이너 재현 | C | 미확인 | TODO 1 구현 전에는 기동 실패 예상 (2번 참고) |
| 통합 데모 (업로드 → 학습 → 예측 → 드리프트 → 재학습 → 재배포) | E | 미실행 | 본 레포 코드로는 미실행. 사전 실험은 5번 참고 |

---

## 1. 서빙 API — 담당 A

### 대상 파일

`serving_app/model_loader.py`, `serving_app/routers/predict.py`의 `predict()`, `serving_app/routers/health.py`, `serving_app/schemas.py`
(`predict.py`의 `batch_test()`와 `schemas.py`의 `BatchTest*`는 B 영역. 이번 변경은 이 영역을 수정하지 않는다. `main.py` 라우터 등록은 C와 별도 조율한다.)

### 아키텍처

- **역할**: 최근 20일 시퀀스를 받아 다음 날 도착 여객 수 하나를 돌려주는 HTTP 서버. 모델을 어디서(로컬 파일 / MLflow Production) 언제(기동 시 / 첫 요청 시) 불러올지 정한다.
- **구성 요소**
  - `main.py`: `FastAPI` 앱 생성, 라우터 4개 등록(`predict`, `health`, `data`, `logs`), `aiops` 로거를 `logs/aiops.log`에 연결, `static/`을 `/`에 마운트, `startup`에서 `LOADING_MODE=eager`면 `model_loader.load_eager()`.
  - `model_loader.py`: `get_model()` → `_model_cache`가 비어 있으면 `_load_model()` → `MODEL_SOURCE`에 따라 `_load_from_local()`(`serving_app/models/airport_v1.keras` + `scaler.pkl`, 버전 `v1-local`) 또는 `_load_from_mlflow()`(Production을 조회한 뒤 `models:/Airport_Arrivals_Predictor/<실제 버전>`으로 고정해서 로드, **TODO 1 구현**). 반환은 `LoadedModel(predict_one, version, registry_version)`. 스케일러는 기존 로컬 `scaler.pkl`을 그대로 사용하며 재학습하지 않는다.
  - `routers/predict.py` `predict()`: `PredictRequest.sequence` 20개 → `model.predict_one()` → `PredictResponse(predicted_arrivals, model_version, model_registry_version)`. 기존 `model_version="production"`을 유지하고 실제 등록 번호를 별도 필드로 반환한다. 로컬 모델의 등록 번호는 `null`.
  - `routers/health.py`: `GET /health` → 상태, 로딩 모드, 모델 로드 여부.
  - `schemas.py`: `PredictRequest`(길이 20, `arrivals`·`departures` 0 이상 정수), 위반 시 422.
- **흐름**: 대시보드 또는 클라이언트 → `POST /predict` → Pydantic 검증 → `get_model()`(캐시) → 스케일 → LSTM → 역스케일 → JSON 응답.
- **다른 영역과의 연결**
  - C가 만드는 `GET /data/status`의 `recent`가 이 API의 입력이 된다 (D1이 호출).
  - A가 제공한 `model_loader.reset_cache()`를 B가 승격·롤백 성공 후 호출하면 다음 예측에서 모델을 다시 로드한다. 호출부는 B 영역이며 이번 PR에서는 수정하지 않는다. 캐시 잠금으로 첫 로딩과 초기화의 경합을 막는다.
  - D2는 소스 구분에 `model_version`, 실제 서빙 버전 번호 표시에 `model_registry_version`을 사용할 수 있다. 현재 레지스트리 Production과 메모리에 로드된 버전은 다를 수 있다.
- **설정**: `LOADING_MODE=lazy|eager`(기본 lazy), `MODEL_SOURCE=local|mlflow`(기본 local), `MLFLOW_MODEL_URI`.
- **실행**: `uvicorn serving_app.main:app --host 0.0.0.0 --port 8000` / MLflow 모델: `MODEL_SOURCE=mlflow uvicorn ...`

### 현재 상태

- `GET /health`, `POST /predict`는 스켈레톤 그대로 완성되어 있다.
- `/predict` 입력은 최근 20일 시퀀스(`arrivals`, `departures`)이며 길이가 20이 아니거나 값이 음수이면 422로 거부한다.
- TODO 1 구현 완료. Production 조회 결과와 같은 버전 URI로 가중치를 읽고 응답 번호를 기록한다. Production이 없거나 로딩이 실패하면 로컬 모델로 조용히 대체하지 않는다.
- Lazy/Eager 및 캐시 초기화 함수 구현·검증 완료. B 호출 연결 전에는 승격만으로 캐시가 바뀌지 않는다.
- `reset_cache()`는 호출한 프로세스에만 적용된다. 이미 모델을 받은 요청은 기존 모델로 마치며, 다중 worker 간 동기화는 구현 범위 밖이다.
- 교수 실습가이드 v3의 Day2 `model_version: production`, Day1 고정 스케일러, Lazy/Eager 방식 유지. 모델 구조·피처·학습·게이트·드리프트 정책은 변경하지 않았다.

### 측정값

| 항목 | 값 | 조건 (PC, 명령) |
|---|---|---|
| local/lazy 시작 / 첫 요청 / 두 번째 요청 (초) | 0.2241142499842681 / 2.0358974580012728 / 0.01348475000122562 | macOS arm64, Python 3.12.13, TensorFlow 2.21.0, 포트 8000, 1회 측정 |
| local/lazy `/predict` | `{"predicted_arrivals": 42087.61, "model_version": "v1-local", "model_registry_version": null}` | 원자료 마지막 20행, HTTP 200 |
| mlflow/lazy 시작 / 첫 요청 / 두 번째 요청 (초) | 0.19868695898912847 / 3.511652999994112 / 0.016238417010754347 | macOS arm64, Python 3.12.13, TensorFlow 2.21.0, 포트 8000, 1회 측정 |
| mlflow/lazy `/predict` | `{"predicted_arrivals": 39772.69, "model_version": "production", "model_registry_version": "1"}` | 원자료 마지막 20행, HTTP 200 |
| mlflow/eager 시작 / 첫 요청 / 두 번째 요청 (초) | 2.153921208024258 / 0.09605337501852773 / 0.014345583011163399 | macOS arm64, Python 3.12.13, TensorFlow 2.21.0, 포트 8000, 1회 측정 |
| mlflow/eager `/predict` | `{"predicted_arrivals": 39772.69, "model_version": "production", "model_registry_version": "1"}` | 원자료 마지막 20행, HTTP 200 |
| 입력 오류 | 19행·21행·음수·departures 누락 모두 422 (4모드 × 4종) | local/lazy, mlflow/lazy, mlflow/eager, 별도 registry 모드 |
| 캐시 전환 | 승격 직후 번호 1 → reset 후 2 → 롤백+reset 후 1 | 복제 SQLite DB, 같은 v1 artifact로 v2 등록. 실제 재학습/성능 개선 검증 아님 |
| 캐시·로더 테스트 | 8개 통과 | 버전 2/11 숫자 비교, 고정 scaler, Production 없음, 실패 후 재시도, 동시 첫 요청 1회 로드, 로딩 중 reset, eager 캐시, 반복 reset |
| `/health` | lazy 초기 false → 예측 후 true, eager 초기 true | 모두 `status: ok` |

### 트러블슈팅

- 이전: MLflow 로드는 `NotImplementedError`. 원인: TODO 1 미구현. 해결: Production 조회·정확한 버전 로드·고정 scaler 결합. 이후 실제 `/predict` 200, `model_version=production`, `model_registry_version=1`.
- Production 상태 이름만 출력하면 버전 전환을 구분할 수 없으므로 기존 필드는 보존하고 등록 번호 필드를 추가했다.
- 테스트 중 MLflow stage API의 폐기 예정 경고가 출력됨. 수업의 Production stage 방식을 유지했다. 별칭 전환은 이번 범위 밖이며 기능 오류는 아니었다.
- 이번 테스트의 시작 시간은 프로세스 내부 import 시작부터 Uvicorn startup 완료까지다. OS 프로세스 생성 시간은 포함하지 않으며 환경·캐시 영향을 받는 1회 값이다.

### 증빙

| 증빙 | 무엇을 보여주나 | 상태 | 파일 | 기록자 |
|---|---|---|---|---|
| HTTP local/lazy/eager | 예측 결과·시간·health·422 | 실제 실행 로그 확보, UI 캡처 미촬영 | `logs/a-http-local.json`, `logs/a-http-lazy.json`, `logs/a-http-eager.json` | 윤동현/A |
| 버전 전환·롤백 후 reset | 같은 프로세스에서 등록 번호 1→2→1 | 실제 HTTP 확인, 재학습 호출 없음 | `logs/a-http-cache.json` | 윤동현/A |
| 캐시·로더 테스트 | 경합·실패·재시도 등 8개 | 통과 | `logs/a-unit-results.log` | 윤동현/A |

로그와 검증 스크립트(`logs/verify_a_unit.py`, `logs/verify_a_http.py`)는 이 PC의 Git 제외 경로에 보관한다. 위 측정표와 PR 본문에 결과를 함께 기록하며, UI 스크린샷으로 간주하지 않는다.

### 다른 영역에 요청

- B: 승격·롤백 **성공 후** `model_loader.reset_cache()` 호출 연결 필요. 판정 창 초기화·재학습 정책은 B가 결정/구현한다.
- C/D/E: 단일 예측 응답에 `model_registry_version: str | null`을 추가했다. 기존 필드는 유지한다. C의 버전 조회와 D의 표시, E의 API 명세에 반영 요청.
- C: `GET /data/status`에 `recent`(최근 20일, 오래된 순, `{date, arrivals, departures}`)를 넣어 주면 `/predict` 입력을 그대로 만들 수 있다.

### 변경 기록

- 2026-09-30 21:40 | 초기 작성 | 스켈레톤을 공항 도메인으로 치환한 상태를 기록 | 해당 없음 | main

- 2026-10-01T14:27:20+09:00 | 윤동현/A | 실습 2-1 MLflow 로더, 실제 등록 번호, 캐시 초기화 구현 | `uv run python logs/verify_a_unit.py` 8개 통과; `uv run python logs/verify_a_http.py local/lazy/eager/cache` (모드별 별도 실행)로 위 HTTP 결과 확인 | feat/a-mlflow-serving

검증 실행 명령 (프로젝트 루트, 8000 포트에서 순차 실행):
```bash
uv run python -m compileall -q data scripts serving_app
uv run python logs/verify_a_unit.py
uv run python logs/verify_a_http.py local
uv run python logs/verify_a_http.py lazy
uv run python logs/verify_a_http.py eager
uv run python logs/verify_a_http.py cache
# 일반 실행 (레지스트리는 이 프로젝트 DB):
MLFLOW_TRACKING_URI=sqlite:///mlflow.db MODEL_SOURCE=mlflow uv run uvicorn serving_app.main:app --host 127.0.0.1 --port 8000
```

---

## 2. 학습·배포·버전 조회 — 담당 C

### 대상 파일

`scripts/train_baseline_v1.py`, `serving_app/train_and_register.py`(`fine_tune()` 호출은 B), `serving_app/routers/data.py`, `serving_app/routers/monitor.py`(신설), `serving_app/Dockerfile`, `serving_app/docker-compose.yml`, `requirements.txt`, `data/storage.py`

### 아키텍처

- **역할**: 업로드된 CSV로 모델을 학습하고, 게이트(RMSE ≤ 2,700명)를 통과한 버전만 MLflow Production에 올린다. 어느 PC에서나 같은 절차로 재현되게 컨테이너로 묶는다.
- **구성 요소**
  - `routers/data.py`: `POST /data/upload`(컬럼 `Date,Arrivals,Departures` 검사, 최소 행 수 `SEQ_LEN + WINDOW_SIZE` = 41, `data/uploads/`에 저장) / `GET /data/status`(행 수, 기간, 최소·최대). `data/storage.py`의 `latest_upload()`가 최신 파일 경로를 준다.
  - `scripts/train_baseline_v1.py`: Day1. 최신 업로드로 학습 → 마지막 20% 검증 RMSE → 게이트 → `serving_app/models/airport_v1.keras`, `scaler.pkl` 저장.
  - `train_and_register.py`: `train_and_register()` — seed 42, 100 epoch → `rmse()` → MLflow run 기록 → `_register_if_gate_passed()`가 통과 시 `Airport_Arrivals_Predictor` 등록 + Production 승격. `fine_tune(rows)` — Production 가중치에서 10 epoch, LR 1e-4, 같은 게이트 (호출은 B의 TODO 4).
  - `routers/monitor.py` (신설 예정): `GET /monitor/versions` → MLflow Registry에서 Production 버전·run_id·RMSE·생성 시각 (`docs/API_SPEC.md`).
  - `Dockerfile`: 이미지 빌드 중 baseline + MLflow 학습 실행, `MODEL_SOURCE=mlflow`, `LOADING_MODE=eager`로 기동.
- **흐름**: 대시보드 CSV 업로드 → `data/uploads/` → `train_baseline_v1.py`(로컬 모델) → `train_and_register.py`(MLflow run → 게이트 → Registry → Production) → A의 `_load_from_mlflow()`가 읽는다.
- **다른 영역과의 연결**
  - A: `MLFLOW_MODEL_URI = models:/Airport_Arrivals_Predictor/Production`을 읽는다. Registry 이름·스테이지를 바꾸면 A도 바뀐다.
  - B: `fine_tune(rows)`를 호출하고 반환 `{"promoted", "rmse", "version"}`을 쓴다. 반환 형식을 바꾸면 B도 바뀐다.
  - D1: `GET /data/status`의 `recent`를 읽는다. D2: `GET /monitor/versions`를 읽는다.
- **설정**: `RMSE_GATE = 2700.0`, `SEED = 42`, `BASE_EPOCHS = 100`, `FINE_TUNE_EPOCHS = 10`, `FINE_TUNE_LR = 1e-4`, `MODEL_NAME`. MLflow 저장소는 cwd의 `mlflow.db`, `mlruns/`.
- **실행**: `python scripts/train_baseline_v1.py` → `python serving_app/train_and_register.py` / `docker compose -f serving_app/docker-compose.yml up --build`

### 현재 상태

- Day1 baseline: RMSE 2,571명으로 게이트 2,700명 통과. 전일값 복사 기준선 3,600명 대비 -29%. PC 1대에서 실행한 결과다.
- Day2 base 학습(100 epoch, seed 42): RMSE 2,267명(2266.94)으로 게이트 통과, `Airport_Arrivals_Predictor`가 Production으로 승격됐다. PC 1대에서 실행한 결과다.
- 모델 파일(`*.keras`, `scaler.pkl`)과 MLflow 기록(`mlflow.db`, `mlruns/`)은 커밋하지 않는다. 각자 PC에서 업로드 → baseline → MLflow 학습을 직접 실행해야 한다.
- `fine_tune()`은 스켈레톤에 구현되어 있다. 호출하는 쪽(B의 TODO 4)이 미구현이라 실행 기록은 없다.
- 승격 시 기존 Production 버전을 Archived로 내리지 않는다. Production에 여러 버전이 남을 수 있다 (사전 실험에서 확인, 5번).
- `GET /data/status`의 `recent`: 미구현. 설계: 최신 업로드의 마지막 `SEQ_LEN`(20)행, 오래된 순, `[{date, arrivals, departures}]` 정수.
- `GET /monitor/versions`: 미구현. 설계: `docs/API_SPEC.md`.
- Docker: 미확인. TODO 1이 구현되기 전에는 컨테이너 기동이 실패할 것으로 예상된다.

### 측정값

| 항목 | 값 | 조건 (PC, 명령) |
|---|---|---|
| Day1 baseline RMSE | 2,571명 | PC 1대, `python scripts/train_baseline_v1.py` |
| Day2 MLflow 학습 RMSE | 2,267명 (2266.94) | PC 1대, `python serving_app/train_and_register.py` |
| 다른 PC에서의 게이트 통과 여부 | 미측정 | |
| 학습 소요 시간 (100 epoch) | 미측정 | |
| 컨테이너 빌드 시간 / 기동 성공 여부 | 미측정 | |
| `GET /monitor/versions` 응답 예시 | 미측정 | |

### 트러블슈팅

(아직 없음)

### 증빙

| 스냅샷 | 무엇을 보여주나 | 상태 | 파일 | 찍은 사람·시각 |
|---|---|---|---|---|
| `train_and_register.py` 실행 로그 | `[GATE PASSED]`와 RMSE | 미촬영 | | |
| MLflow UI Registry 화면 | Production 버전 | 미촬영 | | |
| 컨테이너 빌드·실행 로그 | Docker로 재현됨 | 미촬영 | | |
| 컨테이너의 Swagger 화면 | 컨테이너에서도 같은 API가 동작 | 미촬영 | | |
| `GET /monitor/versions` 응답 | Production 버전·RMSE | 미촬영 | | |

### 다른 영역에 요청

(아직 없음)

### 변경 기록

- 2026-09-30 21:40 | 초기 작성 | Day1·Day2 실행 결과 기록 | Day1 RMSE 2,571명 (`python scripts/train_baseline_v1.py`), Day2 RMSE 2,267명 (`python serving_app/train_and_register.py`) | main

---

## 3. 모니터링·AIOps — 담당 B

### 대상 파일

`serving_app/monitoring/drift_detector.py`, `serving_app/monitoring/retrain_trigger.py`, `scripts/simulate_drift.py`, `serving_app/routers/predict.py`의 `batch_test()`, `serving_app/schemas.py`의 `BatchTestRequest`·`BatchTestResponse`

### 아키텍처

- **역할**: 예측·실제 쌍을 쌓아 오차를 감시하고, 임계값을 넘으면 알림 → 최근 데이터로 fine-tuning → 게이트 재검증 → 재배포까지 사람 없이 잇는다.
- **구성 요소**
  - `routers/predict.py` `batch_test()` (**TODO 3**): `BatchTestRequest.arrivals` 41개 → 길이 20 슬라이딩 윈도우 21개 → 각 윈도우로 예측(출발 여객은 `SIMULATED_DEPARTURES` 고정) → `recent_predictions`에 `{"predicted", "actual"}` 누적(최근 21건 유지) → `check_and_trigger()` → `BatchTestResponse(predictions, drift_check)`.
  - `drift_detector.py`: `compute_rmse(recent_predictions)` (**TODO 2**), `is_drift()` = RMSE > `RMSE_THRESHOLD`(2,700명), `WINDOW_SIZE = 21`.
  - `retrain_trigger.py` `check_and_trigger()` (**TODO 4**): 드리프트면 `[WARN]` → `latest_upload()`의 최근 41행 → `train_and_register.fine_tune(rows)` → `[INFO]` → `promoted`면 `[OK]`. 반환 `{"status": "ok" | "retrain_triggered", "promoted", "rmse"}`.
  - `scripts/simulate_drift.py` `send_batch()` (**TODO 5**): 랜덤워크 41일(정상 σ 1.2% / 드리프트 σ 3.6%)을 `/predict/batch-test`로 전송.
  - 로그: `aiops` 로거 → `logs/aiops.log` (`main.py`가 연결) → `GET /logs/aiops.log`로 대시보드가 읽는다.
- **흐름**: 배치 전송 → `batch_test` → 예측 21건 → `is_drift` → (드리프트) `[WARN]` → `fine_tune` → 게이트 → 승격 `[OK]` / 유지.
- **다른 영역과의 연결**
  - A: `model_loader.get_model()`로 예측. 승격 후 A의 캐시 비우기 함수를 호출해야 새 모델이 서빙된다.
  - C: `fine_tune(rows)`의 반환 형식에 의존. `latest_upload()`로 재학습 데이터를 얻는다.
  - D2: 배치를 보내는 쪽. `BatchTestRequest`에 `departures`를 추가하면 D2 화면과 `docs/API_SPEC.md`도 바뀐다.
- **설정**: `RMSE_THRESHOLD = 2700.0`, `WINDOW_SIZE = 21`, `SIMULATED_DEPARTURES = 37_000`, 시뮬레이션 σ 1.2% / 3.6%.
- **실행**: `python scripts/simulate_drift.py` (서버 기동 후) / 실데이터 배치: `data/jeju_drift_batch_41rows.csv`의 `arrivals` 41개를 `POST /predict/batch-test`

### 현재 상태

- `compute_rmse()`, `batch_test()`, `send_batch()`: TODO 상태. `batch_test()`는 예측 없이 빈 `predictions`를 돌려준다.
- `check_and_trigger()`: 드리프트 판정 시 `[WARN]` 로그까지만 남긴다. 재학습 호출 부분이 TODO다.
- 판정 기준: 최근 21건의 예측·실제 쌍 RMSE > 2,700명.
- 확정 전 이슈 (팀 결정 필요, 근거는 5번 사전 실험): 임계값 2,700명이 평상시 오차(2,275~2,570명)와 거의 같아 오탐이 잦다. 출발 여객 고정값이 250~950명의 오차를 더한다. 재학습 후 판정 윈도우에 이전 예측이 남아 다시 드리프트로 판정된다.

### 측정값

| 항목 | 값 | 조건 (PC, 명령) |
|---|---|---|
| 정상 배치 RMSE (`simulate_drift.py`) | 미측정 | |
| 드리프트 배치 RMSE (`simulate_drift.py`) | 미측정 | |
| 실데이터 폭설 배치 RMSE (`data/jeju_drift_batch_41rows.csv`) | 미측정 | |
| 재학습 후 RMSE / 승격 버전 | 미측정 | |
| `[WARN]` → `[INFO]` → `[OK]` 로그 | 미확인 | |
| 재학습 소요 시간 (10 epoch) | 미측정 | |

### 트러블슈팅

(아직 없음)

### 증빙

| 스냅샷 | 무엇을 보여주나 | 상태 | 파일 | 찍은 사람·시각 |
|---|---|---|---|---|
| `simulate_drift.py` 실행 결과 | 정상 배치와 드리프트 배치의 `drift_check` | 미촬영 | | |
| `logs/aiops.log` | `[WARN]` → `[INFO]` → `[OK]` 순서 | 미촬영 | | |
| 재학습 후 `/predict` 응답 | 새 Production 버전 반영 | 미촬영 | | |

### 다른 영역에 요청

- A: 승격 후 `_model_cache`를 비우는 함수.
- C: `fine_tune()` 반환에 `version`이 들어 있는지 확인.

### 변경 기록

- 2026-09-30 21:40 | 초기 작성 | TODO 상태 기록 | 해당 없음 | main

---

## 4. 프론트엔드 (대시보드) — 담당 D1, D2

### 대상 파일

`serving_app/static/index.html` (두 사람이 같은 파일과 이 섹션을 함께 고친다. 충돌은 허용하고 나중에 머지하는 쪽이 rebase로 푼다. 공통 CSS·상수는 E에게 요청)

### 아키텍처

- **역할**: 운영 담당자가 보는 화면 하나. 위쪽은 "내일 도착 여객 예측 N명 · 등급"(D1), 아래쪽은 운영 체계가 돌아가는 것을 보여주는 배치 전송·재학습 로그·모델 버전(D2).
- **구성 요소**
  - [D1] 예측 카드: 예측값, 날짜, 혼잡 등급, 최근 20일 추이(표 또는 간단한 차트), 새로고침 버튼.
  - [D2] 드리프트 시뮬레이션 카드(스켈레톤): 랜덤워크 생성 → `POST /predict/batch-test`. 여기에 **CSV 배치 전송**(`data/jeju_drift_batch_41rows.csv` 등 41행 파일을 읽어 `arrivals` 배열로 전송) 추가.
  - [D2] 재학습 로그 카드(스켈레톤): `GET /logs/aiops.log` 주기 조회, `[WARN]`/`[INFO]`/`[OK]` 색 구분.
  - [D2] Production 버전 표기: `GET /monitor/versions` → 버전·RMSE·생성 시각. 재학습 전후 변화 표시.
  - 데이터 업로드 카드(스켈레톤): `POST /data/upload`.
- **흐름**
  - [D1] 페이지 로드 → `GET /data/status` → `recent` 20행 → `POST /predict` → `predicted_arrivals` → 등급 판정(Q3 40,177명 초과 혼잡 / Q1 34,962명 미만 여유 / 그 사이 보통) → 카드 표시. 업로드 데이터가 없으면 "데이터를 먼저 업로드하세요".
  - [D2] CSV 선택 → 파싱 → `batch-test` → `drift_check` 표시 → 로그 카드가 `[WARN]`→`[OK]` 갱신 → 버전 표기 갱신.
- **다른 영역과의 연결**: C의 `recent` 필드와 `/monitor/versions`. A의 `/predict` 응답 형식(`predicted_arrivals`, `model_version`). B의 `BatchTestRequest` 형식(`arrivals`, 추가되면 `departures`). 스켈레톤 상수 `RMSE_THRESHOLD = 2700`, `DEFAULT_BASE_ARRIVALS = 37000`은 서버 값 복제 — 서버가 바뀌면 함께 (E가 알린다).
- **설정**: `RMSE_THRESHOLD`, `DEFAULT_BASE_ARRIVALS` (index.html 상단), 등급 경계 상수(`CONGESTION_HIGH = 40177`, `CONGESTION_LOW = 34962`, 이름은 구현 시 확정) — `data/README.md` 사분위와 같게.
- **실행**: `http://localhost:8000/`

### 현재 상태

- 스켈레톤 카드 4개가 있다: 데이터 업로드, 드리프트 시뮬레이션, 드리프트 감지 기반 재학습 파이프라인, 재학습 로그.
- [D1] 예측·혼잡 등급 카드: 미구현. 입력 시퀀스 출처는 C의 `GET /data/status` `recent`(미구현). 그 전에는 `data/jeju_airport_arrivals.csv` 마지막 20행을 직접 넣어 화면만 먼저 만든다.
- [D2] CSV 배치 전송: 미구현. Production 버전 표기: 미구현 (`/monitor/versions` 미구현).
- [D2] 드리프트 시뮬레이션 카드는 `batch_test()`가 TODO라 지금은 빈 `predictions`를 받는다.

### 측정값

| 항목 | 값 | 조건 |
|---|---|---|
| [D1] 페이지 로드 → 예측 표시까지 시간 | 미측정 | |
| [D1] 표시된 예측값 / 등급 (실데이터 마지막 20일 기준) | 미측정 | |
| [D2] CSV 배치 전송 후 `drift_check` 표시 | 미측정 | |
| [D2] 로그 카드에 `[WARN]`→`[OK]` 반영까지 시간 | 미측정 | |
| [D2] 재학습 전후 버전 표기 | 미측정 | |

### 트러블슈팅

(아직 없음)

### 증빙

| 스냅샷 | 무엇을 보여주나 | 상태 | 파일 | 찍은 사람·시각 |
|---|---|---|---|---|
| 예측·혼잡 등급 카드 | 내일 예측값과 등급 | 미촬영 | | |
| 최근 20일 추이 | `recent` 20행 | 미촬영 | | |
| CSV 배치 전송 결과 | `drift_check` 응답 | 미촬영 | | |
| 재학습 로그 패널 | `[WARN]` → `[INFO]` → `[OK]` 순서 | 미촬영 | | |
| Production 버전 표기 | 재학습 전후 버전 변화 | 미촬영 | | |

### 다른 영역에 요청

- C: `GET /data/status`의 `recent`, `GET /monitor/versions`.
- B: `BatchTestRequest`에 `departures`를 넣을지 결정.

### 변경 기록

- 2026-09-30 21:40 | 초기 작성 | 스켈레톤 카드 구성 기록 | 해당 없음 | main

## 5. 데이터·상수·통합 — 담당 E

### 대상 파일

`data/` (CSV, `README.md`, `prepare_jeju_data.py`), `CLAUDE.md` 상수 표, `docs/PROJECT_PLAN.md`, 0번 표, 이 섹션

### 아키텍처 (전체 흐름)

프롬프트 4는 1~4번 "아키텍처"를 부품으로, 이 항목을 뼈대로 전체 구성도를 만든다.

- **역할**: 부품이 하나의 파이프라인으로 이어지는지, 상수가 한 곳에서만 바뀌는지 본다.
- **전체 흐름**
  1. 데이터: 한국공항공사 일별 통계 → `scripts/prepare_jeju_data.py` → `data/jeju_airport_arrivals.csv` → 대시보드 업로드 → `data/uploads/` (C)
  2. 학습·배포: baseline → MLflow 학습 → 게이트 2,700명 → Production (C)
  3. 서빙: `MODEL_SOURCE=mlflow` → `/predict` (A) → 예측·혼잡 카드 (D1)
  4. 모니터링: 배치 전송 (D2) → `batch-test` → RMSE > 임계값 → `[WARN]` → `fine_tune` → 게이트 → 승격 `[OK]` (B, C)
  5. 반영: 캐시 비움 (A) → 새 버전으로 `/predict` → 버전 표기 (D2)
- **경계**: 학습 코드(`train_and_register.py`)와 서빙 코드(`model_loader.py`)는 MLflow Registry(`models:/Airport_Arrivals_Predictor/Production`)로만 만난다. 모니터링은 `recent_predictions`(프로세스 메모리)와 `logs/aiops.log`(파일)로 상태를 남긴다. 서버 재시작이면 둘 다 초기화된다 (로그 파일은 남음).
- **한 곳에서만 바꾸는 값**: 아래 상수 표. 바꾸면 `CLAUDE.md` 표, `index.html` 복제 상수(D2), ③ 운영 설계를 같은 PR에서.

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
- 팀 결정 대기: 드리프트 임계값 분리 여부, `batch-test`에 `departures` 전달 여부, 재학습 후 윈도우 초기화 (`docs/proposal/03_operations_design.md` 4번). 결정되면 여기에 날짜와 결론을 적는다.

### 측정값 (사전 실험, 2026-10-01, 임시 복사본, PC 1대, 1회 — 본 레포 코드 아님)

| 항목 | 값 |
|---|---|
| 2025-08-31까지(974행) 업로드 후 MLflow 학습 RMSE | 2,241명 |
| 09-01 예측 / 실제 | 35,715명 / 33,819명 |
| 정상 배치 2025-07-22~08-31 RMSE | 2,226명 (ok) |
| 배치 2025-09-01~10-11 RMSE (출발 고정 / 실제 출발) | 3,715명 / 2,754명 |
| 폭설 배치 RMSE | 6,224명 (v1), 6,385명 (v2) |
| 재학습(09-01~10-11 드리프트 후) RMSE → 승격 | 1,858명 → v2 |
| 학습 컷오프별 게이트 통과 | 2024-06-30 2,353 / 2024-08-31 2,626 / 2025-08-31 2,241 / 09-20 2,160 / 09-30 2,207 / 10-31 2,267 통과. 2025-01-08 3,310, 02-18 4,090, 05-31 3,385 실패 |
| 평상시 하루 오차 (결항 제외) | 모델 2,275명(실제 출발) / 2,570명(고정 출발), 전일값 복사 2,502명 |
| 학습에 안 쓰인 평상시 41일 배치 중 임계값 초과 비율 | 32% (고정 출발) / 13% (실제 출발) / 31% (전일값 복사) |

통합 데모(본 레포 코드, 팀 PC): 미실행. 실행하면 순서·명령·결과를 여기에 적는다.

### 트러블슈팅

(아직 없음)

### 증빙

| 스냅샷 | 무엇을 보여주나 | 상태 | 파일 | 찍은 사람·시각 |
|---|---|---|---|---|
| 통합 데모 한 바퀴 | 업로드 → 예측 → 드리프트 → 재학습 → 버전 변화 | 미촬영 | | |

### 다른 영역에 요청

(아직 없음)

### 변경 기록

- 2026-09-30 21:40 | 초기 작성 | 데이터와 상수 현재 값 기록 | 해당 없음 | main
- 2026-10-01 | E | 섹션을 담당자별로 재편, 아키텍처·측정값·증빙을 섹션 안으로 이동, 사전 실험 수치 기록 | 해당 없음 | main
