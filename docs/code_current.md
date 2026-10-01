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

`serving_app/main.py`, `serving_app/model_loader.py`, `serving_app/routers/predict.py`의 `predict()`, `serving_app/routers/health.py`, `serving_app/schemas.py`
(`predict.py`의 `batch_test()`와 `schemas.py`의 `BatchTest*`는 B 영역. 같은 파일이므로 PR을 작게 나누고 먼저 머지된 쪽에 rebase한다.)

### 아키텍처

- **역할**: 최근 20일 시퀀스를 받아 다음 날 도착 여객 수 하나를 돌려주는 HTTP 서버. 모델을 어디서(로컬 파일 / MLflow Production) 언제(기동 시 / 첫 요청 시) 불러올지 정한다.
- **구성 요소**
  - `main.py`: `FastAPI` 앱 생성, 라우터 4개 등록(`predict`, `health`, `data`, `logs`), `aiops` 로거를 `logs/aiops.log`에 연결, `static/`을 `/`에 마운트, `startup`에서 `LOADING_MODE=eager`면 `model_loader.load_eager()`.
  - `model_loader.py`: `get_model()` → `_model_cache`가 비어 있으면 `_load_model()` → `MODEL_SOURCE`에 따라 `_load_from_local()`(`serving_app/models/airport_v1.keras` + `scaler.pkl`, 버전 `v1-local`) 또는 `_load_from_mlflow()`(`models:/Airport_Arrivals_Predictor/Production`, **TODO 1**). 반환은 `LoadedModel(predict_one, version)`.
  - `routers/predict.py` `predict()`: `PredictRequest.sequence` 20개 → `model.predict_one()` → `PredictResponse(predicted_arrivals, model_version)`.
  - `routers/health.py`: `GET /health` → 상태, 로딩 모드, 모델 로드 여부.
  - `schemas.py`: `PredictRequest`(길이 20, `arrivals`·`departures` 0 이상 정수), 위반 시 422.
- **흐름**: 대시보드 또는 클라이언트 → `POST /predict` → Pydantic 검증 → `get_model()`(캐시) → 스케일 → LSTM → 역스케일 → JSON 응답.
- **다른 영역과의 연결**
  - C가 만드는 `GET /data/status`의 `recent`가 이 API의 입력이 된다 (D1이 호출).
  - B의 재학습이 새 Production을 승격해도 `_model_cache`는 그대로다. 승격 시 캐시를 비우는 함수(예: `model_loader.reset_cache()`)를 A가 제공하고 B가 호출한다.
  - `model_version` 값은 D2가 대시보드에 표시한다.
- **설정**: `LOADING_MODE=lazy|eager`(기본 lazy), `MODEL_SOURCE=local|mlflow`(기본 local), `MLFLOW_MODEL_URI`.
- **실행**: `uvicorn serving_app.main:app --host 0.0.0.0 --port 8000` / MLflow 모델: `MODEL_SOURCE=mlflow uvicorn ...`

### 현재 상태

- `GET /health`, `POST /predict`는 스켈레톤 그대로 완성되어 있다.
- `/predict` 입력은 최근 20일 시퀀스(`arrivals`, `departures`)이며 길이가 20이 아니거나 값이 음수이면 422로 거부한다.
- `model_loader._load_from_mlflow()`는 TODO 1 상태다. `MODEL_SOURCE=mlflow`로 모델을 불러오면 `NotImplementedError`가 난다.
- 재배포 후 서빙 모델 반영: `_model_cache`가 자동으로 갱신되지 않는다 (미해결, 수업 가이드 부록 1의 6번).

### 측정값

| 항목 | 값 | 조건 (PC, 명령) |
|---|---|---|
| `/predict` 첫 요청 응답 시간 (lazy) | 미측정 | |
| `/predict` 두 번째 요청 응답 시간 | 미측정 | |
| 서버 시작 시간 lazy / eager | 미측정 | |
| `/predict` 응답 예시 (로컬 모델) | 미측정 | |
| `/predict` 응답 예시 (MLflow 모델, `model_version`) | 미측정 | |

### 트러블슈팅

(아직 없음)

### 증빙

| 스냅샷 | 무엇을 보여주나 | 상태 | 파일 | 찍은 사람·시각 |
|---|---|---|---|---|
| `/predict` 응답 (로컬 모델) | `model_version`이 `v1-local` | 미촬영 | | |
| `/predict` 응답 (MLflow 모델) | `model_version`이 `production`으로 전환 | 미촬영 | | |
| `/predict` 422 응답 | 시퀀스 19개, 음수 값 입력이 거부됨 | 미촬영 | | |
| `/health` 응답 | 로딩 모드와 모델 로드 여부 | 미촬영 | | |

### 다른 영역에 요청

- C: `GET /data/status`에 `recent`(최근 20일, 오래된 순, `{date, arrivals, departures}`)를 넣어 주면 `/predict` 입력을 그대로 만들 수 있다.

### 변경 기록

- 2026-09-30 21:40 | 초기 작성 | 스켈레톤을 공항 도메인으로 치환한 상태를 기록 | 해당 없음 | main

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
  - `routers/monitor.py` (신설 완료): `GET /monitor/versions` → MLflow Registry의 Production 버전·run_id·rmse·created_at과, 서버가 실제로 들고 있는 모델(`serving_version`, `serving_run_id`)을 함께 반환. 둘이 어긋났는지는 `stale`로 표시한다. `MODEL_SOURCE=local`이면 `production`은 null.
  - `Dockerfile`: 이미지 빌드 중 baseline + MLflow 학습 실행, `MODEL_SOURCE=mlflow`, `LOADING_MODE=eager`로 기동.
- **흐름**: 대시보드 CSV 업로드 → `data/uploads/` → `train_baseline_v1.py`(로컬 모델) → `train_and_register.py`(MLflow run → 게이트 → Registry → Production) → A의 `_load_from_mlflow()`가 읽는다.
- **다른 영역과의 연결**
  - A: `MLFLOW_MODEL_URI = models:/Airport_Arrivals_Predictor/Production`을 읽는다. Registry 이름·스테이지를 바꾸면 A도 바뀐다.
  - B: `fine_tune(rows)`를 호출하고 반환 `{"promoted", "rmse", "version"}`을 쓴다. 반환 형식을 바꾸면 B도 바뀐다.
  - D1: `GET /data/status`의 `recent`를 읽는다. D2: `GET /monitor/versions`를 읽는다.
- **설정**: `RMSE_GATE = 2700.0`, `SEED = 42`, `BASE_EPOCHS = 100`, `FINE_TUNE_EPOCHS = 3`, `FINE_TUNE_LR = 1e-4`, `MODEL_NAME`. MLflow 저장소는 cwd의 `mlflow.db`, `mlruns/`.
- **실행**: `python scripts/train_baseline_v1.py` → `python serving_app/train_and_register.py` / `docker compose -f serving_app/docker-compose.yml up --build`

### 현재 상태

- Day1 baseline: RMSE 2,571명으로 게이트 2,700명 통과. 전일값 복사 기준선 3,600명 대비 -29%. PC 1대에서 실행한 결과다.
- Day2 base 학습(100 epoch, seed 42): RMSE 2,267명(2266.94)으로 게이트 통과, `Airport_Arrivals_Predictor`가 Production으로 승격됐다. PC 1대에서 실행한 결과다.
- 모델 파일(`*.keras`, `scaler.pkl`)과 MLflow 기록(`mlflow.db`, `mlruns/`)은 커밋하지 않는다. 각자 PC에서 업로드 → baseline → MLflow 학습을 직접 실행해야 한다.
- `fine_tune()`은 스켈레톤에 구현되어 있다. 호출하는 쪽(B의 TODO 4)이 미구현이라 실행 기록은 없다.
- 승격 시 기존 Production 버전을 Archived로 내리지 않는다. Production에 여러 버전이 남을 수 있다 (사전 실험에서 확인, 5번).
- `GET /data/status`의 `recent`: **구현 완료.** 최신 업로드의 마지막 `SEQ_LEN`(20)행, 오래된 날 → 최근 날 순서, `[{date, arrivals, departures}]` 정수. D1이 이 값을 그대로 `/predict`의 `sequence`로 보내면 200이 나오는 것까지 확인했다.
- `GET /monitor/versions`: **구현 완료.** 다만 `stale` 판정은 아직 항상 `null`이다. `LoadedModel`이 `version` 문자열("production")만 들고 있고 어느 run에서 왔는지 모르기 때문이다. A가 `run_id`를 보관해 주면 `production.run_id`와 비교해 판정할 수 있다 (아래 "다른 영역에 요청").
- Docker: 미확인. TODO 1이 구현되기 전에는 컨테이너 기동이 실패할 것으로 예상된다.

### 측정값

| 항목 | 값 | 조건 (PC, 명령) |
|---|---|---|
| Day1 baseline RMSE | 2,571명 | PC 1대, `python scripts/train_baseline_v1.py` |
| Day2 MLflow 학습 RMSE | 2,267명 (2266.94) | PC 1대, `python serving_app/train_and_register.py` |
| 다른 PC에서의 게이트 통과 여부 | 미측정 | |
| 학습 소요 시간 (100 epoch) | 미측정 | |
| 컨테이너 빌드 시간 / 기동 성공 여부 | 미측정 | |
| `GET /monitor/versions` 응답 예시 | `{"model_name":"Airport_Arrivals_Predictor","model_source":"mlflow","production":{"version":1,"run_id":"bbec6d0d...","rmse":2266.94,"created_at":1790831552595},"serving_version":"production","serving_run_id":null,"stale":null}` | PC 1대, `MODEL_SOURCE=mlflow uvicorn ... --port 8000` 후 `curl localhost:8000/monitor/versions` |
| `GET /data/status`의 `recent` | 20건, `2025-10-12` ~ `2025-10-31`, 오래된 순, arrivals/departures 모두 int | PC 1대, `curl localhost:8000/data/status` |
| `recent`를 그대로 `/predict`에 전달 | HTTP 200, `{"predicted_arrivals":39772.69,"model_version":"production"}` | PC 1대, `/data/status`의 recent를 `sequence`로 변환해 POST |
| B 머지 뒤 C API 동작 | `/monitor/versions`, `/data/status`의 `recent`, `recent → /predict` 모두 정상 | PC 1대, `origin/main`(B 포함) 위에 리베이스 후 `MODEL_SOURCE=mlflow ... --port 8000` |
| fine-tune 3 epoch 전체 루프 | v1(rmse 2267)에서 드리프트 배치 주입 → `bias=+2015` 감지 → 재학습 → `[OK] new_rmse=1840 v2` 승격. **예측값 39772.69 → 41283.94로 바뀌고 `/monitor/versions`도 v2로 따라감** | PC 1대, mlruns 초기화 후 1회. `reset_cache()`를 임시로 넣고 측정했고 임시 코드는 커밋하지 않음 |
| 배치별 판정 (윈도우 격리, 배치마다 서버 재기동) | normal `ok` rmse 2280 bias 315 / falsealarm `retrain_triggered` / drift `retrain_triggered` | PC 1대, 각 배치 전에 서버를 다시 띄워 `recent_predictions`를 비운 상태에서 측정 |
| 윈도우를 안 비웠을 때 | 판정이 배치 순서에 따라 뒤집힘. 같은 세 배치를 한 서버에 연속 주입하면 normal이 `retrain_triggered`, drift가 `ok`로 나옴 | PC 1대. 앞 배치의 예측이 윈도우에 남아 다음 판정에 섞인다 (가이드 부록1의 7번) |
| Day1 baseline RMSE (이 PC) | 2,945명 | PC 2대째, `python scripts/train_baseline_v1.py`. 다른 PC에서는 2,571명과 2,191명이 나왔다 |
| Day2 MLflow 학습 RMSE (이 PC) | 2,267명 (2266.94) | PC 2대째, `python serving_app/train_and_register.py`. seed 42 고정이라 다른 PC와 같은 값 |

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

- **B에게**: 배치를 연속으로 주입하면 `recent_predictions` 윈도우가 이어져 판정이 뒤집힌다. 같은 세 배치를 한 서버에 연속으로 보내면 normal이 `retrain_triggered`, drift가 `ok`로 나왔다. 배치마다 서버를 다시 띄우면 의도대로 갈린다. 시연 때 배치 사이에 윈도우를 비우는 절차가 필요해 보인다 (가이드 부록1의 7번).
- **A에게 (급함)**: `model_loader.reset_cache()`가 아직 없어 **승격이 일어나는 순간 `AttributeError`로 `/predict/batch-test`가 500**이 난다. B의 `retrain_trigger.py`가 45행과 97행에서 부른다. 임시로 넣어 보니 전체 루프가 정상 동작했고 재학습 후 예측값도 41283.94에서 41547.42로 바뀌었다.
- **A에게**: `LoadedModel`이 `run_id`를 함께 보관해 주면 좋겠다. `_load_from_mlflow()`에서 로드한 모델이 어느 run에서 왔는지 알 수 있으면, `GET /monitor/versions`가 "레지스트리는 v2인데 서버는 v1을 들고 있다"를 자동으로 판정할 수 있다. 지금은 `stale`이 항상 `null`이다.
  재배포 후 캐시가 안 비워지는 문제(수업 가이드 부록1의 6번)를 **대시보드에서 눈으로 볼 수 있게** 만드는 일이라, D2의 버전 표기와도 이어진다. `LoadedModel.__init__`에 `run_id=None` 인자를 하나 늘리는 정도면 충분하다.
- **D1에게**: `GET /data/status`의 `recent`가 올라갔다. 20건, 오래된 날 → 최근 날 순서이고 그대로 `/predict`의 `sequence`로 보내면 된다 (`arrivals`를 float로만 바꾸면 됨). 확인 완료.
- **E에게**: `docs/API_SPEC.md`의 `/monitor/versions` 설계안보다 응답 필드가 늘었다. 설계안은 `model_name`, `production`, `serving_version` 셋인데 구현은 `model_source`, `serving_run_id`, `stale`을 더 돌려준다. 레지스트리가 말하는 버전과 서버가 실제로 들고 있는 버전을 나란히 보여주려고 넣었다. API_SPEC은 E 소유라 직접 고치지 않았으니 갱신 부탁한다. 실제 응답 예시는 아래 측정값 표에 있다.
- **D2에게**: `GET /monitor/versions`가 올라갔다. `production.version`과 `production.rmse`를 쓰면 되고, `stale`은 A의 작업 전까지 `null`이라 표시하지 않는 편이 낫다.

### 변경 기록

- 2026-09-30 21:40 | 초기 작성 | Day1·Day2 실행 결과 기록 | Day1 RMSE 2,571명 (`python scripts/train_baseline_v1.py`), Day2 RMSE 2,267명 (`python serving_app/train_and_register.py`) | main
- 2026-10-01 13:10 | 유경모 | `GET /monitor/versions` 신설(`routers/monitor.py`, `main.py` 한 줄), `GET /data/status`에 `recent` 20건 추가 | `/monitor/versions`가 Production v1, rmse 2266.94 반환. `recent`를 그대로 `/predict`에 보내 HTTP 200 확인. Day1 2,945명 / Day2 2,267명 | feat/c-monitor-versions
- 2026-10-01 14:50 | 유경모 | B 머지본 위로 리베이스. `FINE_TUNE_EPOCHS` 10 → 3 (B의 `BIAS_THRESHOLD` 500이 3 epoch 전제라 같이 움직여야 함) | 충돌 없음(B와 파일이 겹치지 않음). 전체 루프 재학습 전 41283.94 → 후 41547.42. ⚠️ `model_loader.reset_cache()`가 아직 없어 승격 시점에 `AttributeError`로 batch-test가 500이 난다 (A 대기) | feat/c-monitor-versions

---

## 3. 모니터링·AIOps — 담당 B

### 대상 파일

`serving_app/monitoring/drift_detector.py`, `serving_app/monitoring/retrain_trigger.py`, `scripts/simulate_drift.py`, `serving_app/routers/predict.py`의 `batch_test()`, `serving_app/schemas.py`의 `BatchTestRequest`·`BatchTestResponse`

### 아키텍처

- **역할**: 예측·실제 쌍을 쌓아 오차를 감시하고, 임계값을 넘으면 알림 → 최근 데이터로 fine-tuning → 게이트 재검증 → 재배포까지 사람 없이 잇는다.
- **구성 요소**
  - `routers/predict.py` `batch_test()` (**TODO 3**): `BatchTestRequest.arrivals` 41개 → 길이 20 슬라이딩 윈도우 21개 → 각 윈도우로 예측(출발 여객은 `SIMULATED_DEPARTURES` 고정) → `recent_predictions`에 `{"predicted", "actual"}` 누적(최근 21건 유지) → `check_and_trigger()` → `BatchTestResponse(predictions, drift_check)`.
  - `drift_detector.py`: `compute_rmse()` (**TODO 2**), `compute_bias()` (평균 오차 = 실제 − 예측), `assess(window)` → `{rmse, bias, anomalies, rmse_excl_anomalies, bias_excl_anomalies, drift}`. **세 종류 판정**: 하루 오차 > `ANOMALY_THRESHOLD`(10,000)인 날은 이상치로 빼고, 나머지의 |bias| > `BIAS_THRESHOLD`(500)이면 **수준 드리프트**(재학습). bias 작은데 RMSE > `RMSE_THRESHOLD`(2,700)이면 **구조 드리프트**(알림만). RMSE ≥ |bias|가 항상 성립해 드리프트 조건에 RMSE 항은 없다. `is_drift()`는 `assess()["drift"]`.
  - `retrain_trigger.py` `check_and_trigger()` (**TODO 4**): `assess()` → 이상치 있으면 `[WARN] anomaly` → 드리프트 아니면 `ok` / `anomaly` / `structure_drift`(오차 크지만 치우침 없음, 재학습 안 함) 반환 → 드리프트면 `[WARN] drift` → 이전 Production 버전 기억 → `latest_upload()` 최근 41행 → `fine_tune()` → 승격 시 `reset_cache()` + 판정 윈도우 `clear()` + `[OK]`. 게이트 실패는 그냥 실패 (재시도 간격 없음, 다음 배치에서 다시 판정). **롤백**: 승격 뒤 첫 판정이 드리프트이고 그 재학습이 게이트에 실패하면 `[ROLLBACK]` — 새 버전 Archived, 이전 버전 Production, `reset_cache()`. 승격 뒤 드리프트 아닌 윈도우가 한 번 나오면 승격 확정(롤백 대상에서 제외). 승격 기록은 프로세스 메모리(`_last_promotion`)에만 있다.
  - `scripts/simulate_drift.py` `send_batch()` (**TODO 5**): 랜덤워크 41일(정상 σ 1.2% / 드리프트 σ 3.6%)을 `/predict/batch-test`로 전송.
  - 로그: `aiops` 로거 → `logs/aiops.log` (`main.py`가 연결) → `GET /logs/aiops.log`로 대시보드가 읽는다.
- **흐름**: 배치 전송 → `batch_test` → 예측 21건 → `assess` → 이상치 알림 / 치우침 없는 큰 오차 알림 / 드리프트 → `fine_tune` → 게이트 → 승격 `[OK]` / 실패 시 (직전 승격이 미확정이면 롤백, 아니면) 유지.
- **다른 영역과의 연결**
  - A: `model_loader.get_model()`로 예측, 승격·롤백 뒤 `model_loader.reset_cache()` 호출.
  - C: `fine_tune(rows)`의 반환 `{"promoted", "rmse", "version"}`에 의존. `latest_upload()`로 재학습 데이터를 얻는다. 롤백은 `MlflowClient.transition_model_version_stage`로 Registry 스테이지를 직접 바꾼다.
  - D2: 배치를 보내는 쪽. `drift_check.status`가 `ok | anomaly | structure_drift | retrain_triggered | rolled_back` 다섯 가지로 늘었다.
- **설정**: `RMSE_THRESHOLD = 2700.0`, `BIAS_THRESHOLD = 500.0`, `ANOMALY_THRESHOLD = 10000.0`, `WINDOW_SIZE = 21`, `SIMULATED_DEPARTURES = 37_000`, 시뮬레이션 σ 1.2% / 3.6%.
- **실행**: `python scripts/simulate_drift.py` (서버 기동 후) / 실데이터 배치: `data/jeju_drift_batch_41rows.csv`의 `arrivals` 41개를 `POST /predict/batch-test`

### 현재 상태

- TODO 2~5 이식 완료 (힌트 코드 그대로). 이 PR(`feat/b-monitoring`)은 B 담당 파일 4개(`drift_detector.py`, `retrain_trigger.py`, `predict.py`의 `batch_test()`, `simulate_drift.py`)와 이 섹션만 담는다. 실험은 `exp/drift-anomaly` → `exp/clean-cancellation`에서 했다.
- **다른 담당 PR이 있어야 동작하는 것**: A의 TODO 1 `_load_from_mlflow()`와 `model_loader.reset_cache()` (승격·롤백 뒤 호출 — 없으면 승격 시점에 `AttributeError`), C의 `FINE_TUNE_EPOCHS` 3 (bias 500은 3 epoch 기준으로 고른 값), E의 결항일 보간 학습 데이터·시연 컷·상수 표 (bias 500은 보간 데이터 기준 — 결항일 포함 데이터에서는 1,500이 맞았음, 실험 5).
- 판정: 세 종류 (이상치 / 수준 드리프트 / 구조 드리프트) + 롤백. 위 아키텍처 참고. 스켈레톤의 "RMSE > 2,700이면 재학습"에서 바뀐 것이라 **팀 결정 필요**.
- **확정값 (2026-10-01, `exp/clean-cancellation`)**: 수준 드리프트 |bias| > **500**, 게이트 **2,700**, 이상치 10,000, fine-tune **3 epoch**. 재시도 간격은 두지 않는다(실패는 그냥 실패). 근거는 아래 실험 6. 이전 확정값 1,500(실험 5)은 결항일 포함 학습 데이터 기준이었다 — 학습 데이터를 보간본으로 바꾸자(5번) 21일 bias가 ±1,000 안에서 움직여 1,500으론 2025년에 3월 한 번만 걸리고, 2,000 이상은 한 번도 안 걸린다. 500은 윈도우가 찰 때마다 작은 수준 변화도 따라가 사실상 "승격 뒤 3주마다 재학습 + 게이트 실패일엔 매일 재시도"가 된다.
- RMSE만으로 판정하면 안 되는 이유(실험 6): 보간 모델도 21일 RMSE > 2,700인 날이 107일(2~5월, 9~10월)이고, RMSE 트리거 10회 중 7회는 bias ±800 안쪽(수준은 그대로, 흔들림만 큼). 재학습 3배에 RMSE 개선 0 (3,039 vs 재학습 없음 3,038).
- 구조 드리프트의 실체: 기간별로 주간 리듬 ac(7)이 0.17(2024 H1) → 0.67(2025 Q3), 요일 진폭이 1,756 → 5,320으로 커졌다. 수준은 같은데 변동 폭이 바뀐 것이라 bias로 안 잡히고, 41행 fine-tuning으로도 안 줄어든다 (실험 2: 3,142 → 3,238). 9~10월 배치가 이 경우다.
- 남은 이슈: 출발 여객 고정값이 250~950명의 오차를 더한다 (`departures` 전달 여부 미결). `_last_promotion`이 메모리에만 있어 서버 재시작 후 롤백 불가. `simulate_drift.py` 랜덤워크 σ(1.2%/3.6%)는 실변동성(7.3~8.4%)보다 낮아 미측정.
- 폭설 배치를 넣으면 이상치 3일을 뺀 나머지도 드리프트(2025-01~02 수요 하락 실제)라 재학습이 돈다. 이때 재학습 데이터는 "최신 업로드의 마지막 41행"이라 배치와 무관한 기간일 수 있다 — 시연 순서에서 업로드 순서를 지켜야 한다.

### 측정값

브랜치 `exp/drift-anomaly`, PC 1대, 1회, in-process 실행 (서버 아님, 출발 여객 고정 37,000). bias = 실제 − 예측 평균.

| 항목 | 값 | 조건 |
|---|---|---|
| v1 학습 (~2024-06-30, 547행) 게이트 RMSE | 2,353 (baseline 2,365~2,922, 실행마다 다름) | `train_baseline_v1.py`, `train_and_register.py` |
| 정상 배치 2024-04-15~05-25 (v1) | RMSE 2,374 / bias +252 → `ok` | |
| 드리프트 배치 2025-02-20~04-01 (v1) | RMSE 4,321 / bias −3,056 → 드리프트 → fine-tuning 게이트 2,491 → v2 승격 (14초) | 업로드 ~2025-04-01 |
| 드리프트 배치 (v2) | RMSE 3,073 / bias −309 | 재학습이 치우침 제거 |
| 다음 윈도우 2025-04-02~05-12 (v2) | RMSE 3,918 / bias +2,394 → 드리프트 → v2에서 재학습 게이트 실패 3,769 → 승격 미확정이므로 **롤백 → v1** | v1로는 3,126 / −396 (`high_error`). 재학습된 모델 자체는 이후 3개월 2,023 / 2,773 / 2,366으로 좋았으나 5일 검증(05-08~12)에서 3,769. 같은 5일에서 v1 3,732, v2 3,149 — 5일 표본은 모델 우열을 못 가림 |
| 2025-05-13~06-22, 07-22~08-31 (v1) | 2,009 / −476, 2,383 / +715 → `ok` | |
| 오탐 배치 2025-09-01~10-11 (v1) | 이상치 1일(10-11, −11,306) + 나머지 4,381 / +1,318 → `structure_drift` (재학습 안 함) | |
| 폭설 배치 (v1) | 이상치 3일(02-04 −12,587, 02-05 −10,940, 02-07 −29,616) + 나머지 4,323 / −3,636 → 드리프트 → fine-tuning 게이트 실패 3,865 → v1 유지 | 재학습 데이터 = 최신 업로드(~05-12) 마지막 41행 |
| 롤백 없이 v2에서 재재학습 | 게이트 실패 3,769 → v2에 갇힘 (v2는 이후 전 구간 bias +2,300~3,500) | |
| 재학습 윈도우 41/62/90/120/180행 (v1에서) | after 배치 bias +2,394 / +3,432 / +4,633 / +4,506 / +2,365 — 길어도 과적응 | |
| **실험 5** 재학습 정책 시뮬레이션 (2025-01-21~08-31 하루씩 전진, 실제 fine-tune, 승격 시 윈도우 초기화, 결항 포함 RMSE) | v1 고정 4,201 / bias>2,000 4,100 (재학습 13, 승격 4) / **bias>1,500 3,942 (9, 6)** / bias>1,000 3,943 (21, 8) / bias>500 3,993 (28, 9) / bias>500·게이트 3,500 4,032 (20, 9) / bias>500·게이트 없음 4,166 (10, 10) / 21일마다 무조건 3,891 (11, 4) / 20일 이동평균 3,651 / 전일값 3,462 | 스크래치 `sim_policy.py`, `sim2.py`. 1,500이 재학습 최소·RMSE 최저. 게이트 없으면 v1 고정 수준으로 악화 |
| 모델 특성 (2025-05~08, 결항 없음) | LSTM 2,398 vs 전일값 2,083 / 7일 MA 2,380 / 상수(학습평균) 2,446 / 20일 MA 2,543. 예측 std 460 vs 실제 2,444. 20일 MA와 상관 0.927. 전일 대비 방향 적중 67% (MA 63%, 동전 50%) | 변화량 자기상관 lag1 −0.18(평균 회귀), lag7 +0.36(주간). 수준+약한 주기만 배움, 진폭은 MSE가 줄임 |
| `simulate_drift.py` 랜덤워크 배치 | 미측정 | |
| 서버(`/predict/batch-test`) 경유 재현 | 아래 "보간 데이터 실측" | |

**보간 데이터 실측** — 브랜치 `exp/clean-cancellation` (학습 CSV 결항일 보간, bias 500, fine-tune 3 epoch), 임시 워크트리, `MODEL_SOURCE=mlflow uvicorn ... --port 8011` 서버에 실제 요청, PC 1대, 1회. 배치 입력은 raw. 레지스트리 번호는 재실행으로 이어져 로그엔 v4·v5·v6로 찍혔다 (아래는 상대 번호).

| 항목 | 값 | 조건 |
|---|---|---|
| v1 학습 (~2024-06-30, 547행) | baseline 1,999 / MLflow 게이트 **1,961** → v1 | 보간 전 2,353 |
| `/predict` (06-11~06-30 입력) | 33,339.53 · 첫 호출 8.8초(lazy 로드), 재호출 2.1초 | |
| 정상 배치 2024-10-01~11-10 (v1, 학습 밖) | RMSE 899 / bias −241 → `ok` | 보간 전 모델의 정상 배치 2024-04-15~05-25는 보간 모델로 이상치 1일(05-06 30,879 vs 41,795) `anomaly` 라 교체 |
| 드리프트 배치 2024-12-12~2025-01-21 (v1) | 이상치 1일(01-09 결항 17,093 vs 29,523) 제외 RMSE 2,656 / bias **−1,041** → 재학습 게이트 1,338 → **v2** (16.7초) | 업로드 ~2025-01-21 |
| `/predict` 재학습 후 (01-02~01-21 입력) | 30,622.54 (v2) | |
| 확정 배치 2025-01-02~02-11 (v2) | 이상치 1일(02-07) 제외 RMSE 3,169 / bias +199 → `structure_drift` (재학습 없음) → v2 승격 확정 | |
| 반등 배치 2025-01-16~02-25 (v2) | 이상치 1일(02-07) 제외 RMSE 2,456 / bias **+686** → 재학습 게이트 902 → **v3** (14.7초) | 업로드 ~2025-02-25 |
| 롤백 배치 2025-02-06~03-18 (v3) | RMSE 3,474 / bias **−1,008** → 재학습 게이트 **실패 2,742** → `[ROLLBACK]` v3 Archived, **v2** Production (14.7초) | 업로드 ~2025-03-18. 2,742 vs 2,700 간발 — 다른 PC 재확인 필요 |
| `/predict` 롤백 후 (02-27~03-18 입력) | 31,309.10 (v2) | |
| 폭설 배치 2025-01-09~02-18 (v2) | 이상치 1일(02-07 6,088 vs 29,242) 제외 RMSE 3,039 / bias −103 → `structure_drift` | |
| 시연 순서(≤2025-08-31 학습) | v1 게이트 2,094 · 정상 07-22~08-31 2,183 / −117 `ok` · 9~10월 배치 09-01~10-11 3,490 / **−587** → 재학습 2,098 → v2 · 폭설 이상치 1일 제외 2,981 / +214 `structure_drift` | `data/README.md` "시연 순서". 보간 전엔 9~10월 배치가 오탐 사례였음 |
| **실험 6** 보간 모델 트리거 시뮬레이션 (2025-01-01~10-31 하루씩 전진, raw 입력, 출발 37,000, 실제 fine-tune·게이트·롤백, 결항 포함 RMSE) | 재학습 없음 3,038 · bias>1,500 / ep10 3,058 (재학습 3, 승격 2) · bias>2,000·2,500·3,000 = 재학습 0 · RMSE>2,700 3,039 (10, 6) · RMSE>3,000 3,069 (6, 6) · bias>290 ep3 3,010 (16, 12) / ep10 3,031 (21, 13) / ep30 3,092 (30, 12) · bias>400 ep3 3,011 (13, 11) · **bias>500 ep3 3,017 (10, 9, 롤백 1)** | 스크래치 `sim_clean.py`, `sim_rmse.py`, `sim_290.py`, `sim_500.py`. v1 고정 21일 bias 최대 −1,989(3월). 어느 설정도 9~10월 구조 드리프트는 못 줄여 전체 RMSE 3,000대 유지. 그래프: 아티팩트 "보간 모델 임계값 시뮬레이션", "bias 290 epoch 비교" |

### 트러블슈팅

- 2026-10-01 | 실험 | 드리프트 시연 정상 배치로 잡은 2024-05-21~06-30이 v1로 RMSE 2,820 → 임계값 초과 | 원인: `train_test_split`이 마지막 20%를 검증으로 떼어 이 구간이 학습에 안 들어감 + 06-29 하루 −7,598 | 해결: 학습 구간 안쪽 2024-04-15~05-25(2,374 / +252)로 교체 | 전후: 2,820 → 2,374
- 2026-10-01 | 실험 | 재학습 후 다음 윈도우에서 v2가 v1보다 나쁨 (3,918 vs 3,126), v2에서 재재학습은 게이트 실패 | 원인: 41행 fine-tuning이 1분기 저점에 과적응, 2분기 수요 회복. 게이트 5일 검증이 노이즈라 좋은 재재학습 모델도 탈락 | 해결: 승격 뒤 첫 드리프트의 재학습이 게이트 실패하면 이전 버전으로 롤백 (처음엔 "반대 부호 bias" 조건이었으나 "신규 드리프트로 재학습 → 실패 시 롤백"이 설명이 단순해 교체) | 전후: v2 갇힘 → v1 복귀, 이후 윈도우 `ok`
- 2026-10-01 | 실험 | 예측선이 평평하고 v1은 1분기 내내 실제 위, v2는 4월 이후 내내 실제 아래 | 원인: 모델이 입력 20일 수준이 아니라 학습 기간 평균 쪽으로 예측 (학습 범위 35,000~40,000 밖 입력에 외삽 안 됨). 2025-05~08 예측 표준편차 460 vs 실제 2,444, 전일값 복사 2,083 < LSTM 2,398 | 해결 없음 (피처·학습 흐름 변경 범위 밖). 기획서 ②의 "전일값 복사 3,600보다 25% 좋아야" 근거는 결항일 포함 값이라 수정 필요 | —

### 증빙

| 스냅샷 | 무엇을 보여주나 | 상태 | 파일 | 찍은 사람·시각 |
|---|---|---|---|---|
| `simulate_drift.py` 실행 결과 | 정상 배치와 드리프트 배치의 `drift_check` | 미촬영 | | |
| `logs/aiops.log` | `[WARN]` → `[INFO]` → `[OK]` 순서 | 미촬영 | | |
| 재학습 후 `/predict` 응답 | 새 Production 버전 반영 | 미촬영 | | |

### 다른 영역에 요청

- A (**선행 필요**): TODO 1 `_load_from_mlflow()` 이식, `model_loader.reset_cache()` 추가 (`_model_cache = None`). `retrain_trigger`가 승격·롤백 뒤 호출한다. 구현은 `exp/clean-cancellation`의 `model_loader.py` 참고.
- C (**선행 필요**): `FINE_TUNE_EPOCHS` 10 → 3 (실험 6: 3/10/30 epoch → 게이트 실패 3/8/16회). 승격 시 이전 Production을 Archived로 내리지 않아 여러 버전이 Production에 남는다 — `_register_if_gate_passed`에서 `archive_existing_versions=True` 권장.
- E (**선행 필요**): `scripts/prepare_jeju_data.py` 결항일(도착 < 25,000) 보간 + `jeju_airport_arrivals.csv`·시연 컷 재생성, CLAUDE.md 상수 표에 `BIAS_THRESHOLD`·`ANOMALY_THRESHOLD`·`FINE_TUNE_EPOCHS`·`CANCEL_ARRIVALS`, `data/README.md` 시연 순서. 구현·값은 `exp/clean-cancellation` 참고.
- D2: `drift_check.status` 다섯 가지를 카드에 색으로 구분 (`rolled_back`, `anomaly`, `structure_drift` 추가). 대시보드 상단 `RMSE_THRESHOLD` 복제 상수는 그대로(2,700, 구조 드리프트 알림 기준).

### 변경 기록

- 2026-09-30 21:40 | 초기 작성 | TODO 상태 기록 | 해당 없음 | main
- 2026-10-01 | 실험 | TODO 2~5 이식, 두 층 판정(이상치/드리프트, bias 조건), 롤백 추가 | 위 측정값 표 전체. 실행: 스크래치 worktree에서 in-process (`exp4*.py`, `scan_ft.py`) | exp/drift-anomaly
- 2026-10-01 | 실험 | 판정을 bias만으로, 롤백 조건을 "승격 뒤 첫 재학습 게이트 실패"로, `BIAS_THRESHOLD` 1,500 확정, 승격 시 윈도우 초기화, `high_error` → `structure_drift` | 실험 5 (정책 시뮬레이션) 및 전체 루프 재실행 — 정상 ok → 드리프트 v2 → 재학습 실패 롤백 v1 → ok → 구조 드리프트 알림 → 폭설 재학습 실패 유지 | exp/drift-anomaly (main 미반영)
- 2026-10-01 | 실험 | 학습 데이터 결항일 보간(5번)에 맞춰 `BIAS_THRESHOLD` 1,500 → 500, fine-tune 3 epoch(2번). 시연 배치 재선정 | 실험 6 + 서버 경유 전체 루프 실측 (위 "보간 데이터 실측"): ok → −1,041 v2 → 확정 → +686 v3 → −1,008 게이트 2,742 실패 롤백 v2 → 폭설 structure_drift. 하루 전진 시뮬레이션과 수치 일치 | exp/clean-cancellation (main 미반영)
- 2026-10-01 | B | B 담당 파일 4개 + 이 섹션만 `main`용 PR로 분리 (`feat/b-monitoring`). A·C·E 선행 PR 요청은 "다른 영역에 요청" | 코드는 exp/clean-cancellation과 동일, `python -m compileall -q data scripts serving_app` 통과. main에서는 A의 TODO 1 전이라 `MODEL_SOURCE=mlflow` 루프 미실행 | feat/b-monitoring

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
