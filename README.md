# 공항 도착 여객 예측 · 모델 서빙 & AIOps (조별 미니 프로젝트)

> 원본: SKALA "모델 서빙 및 AIOps" 3일 실습 스켈레톤(HAIC 종가 예측, 임성열 교수).
> 이 레포는 그 스켈레톤을 **공항 도착 여객 예측** 도메인으로 치환한 팀 프로젝트입니다.
> 팀 기획·역할·협업 규칙은 [`docs/`](docs/)를 참고하세요.

**공항 운영 담당자(B2B)** 가 내일의 도착 여객 규모를 미리 알고 인력·셔틀·카운터·주차 운영을
조정할 수 있도록, 최근 20일간의 (도착 여객, 출발 여객) 시퀀스로 **다음날 도착 여객 수**를
예측하는 LSTM 모델을 Day1(서빙) → Day2(MLOps) → Day3(AIOps) 순서로 하나의 서빙 서버 위에
쌓아 올립니다. 데이터는 대시보드에서 CSV 파일을 업로드하는 방식으로 공급합니다.

## 도메인 매핑 (HAIC 템플릿 → 공항 도착 여객)

| 템플릿 개념 (HAIC) | 우리 팀 (공항 도착 여객 예측) | 코드 상 위치 |
|---|---|---|
| 예측 대상: 다음날 종가 `Close` | 다음날 **도착 여객 수** `Arrivals` (명) | `data/features.py`, `schemas.py` |
| 보조 피처: 거래량 `Volume` | 같은 날 **출발 여객 수** `Departures` (명) | `data/features.py` |
| 입력 시퀀스: 최근 20거래일 | 최근 **20일** (SEQ_LEN, 변경 없음) | `data/features.py` |
| 데이터 공급: CSV 업로드 | 한국공항공사 일별 여객 통계 CSV 업로드 (`Date,Arrivals,Departures`) | `routers/data.py` |
| 배포 게이트: RMSE ≤ $4.00 | RMSE ≤ **2,700명** | `train_and_register.py` `RMSE_GATE` |
| 드리프트 임계값: RMSE > $4.00 | RMSE > **2,700명** (최근 21일 윈도우) | `monitoring/drift_detector.py` |
| 시뮬레이션 고정 거래량 1,200,000 | 고정 출발 여객 **37,000명** | `routers/predict.py` `SIMULATED_DEPARTURES` |
| 재학습: 최근 21거래일 fine-tuning | 최근 **21일** fine-tuning (변경 없음) | `monitoring/retrain_trigger.py` |
| 알림: `aiops.log [WARN]` | 동일 (대시보드 재학습 로그 패널에서 확인) | `serving_app/main.py`, `routers/logs.py` |
| 모델 이름 `HAIC_Predictor` | `Airport_Arrivals_Predictor` | `model_loader.py`, `train_and_register.py` |
| 이해관계자 | **공항 운영 담당자**(인력·셔틀·카운터), 입점사·렌터카·면세점(B2G2B) | `docs/PROJECT_PLAN.md` |

모델 아키텍처·피처·시퀀스 길이·학습/재학습 방식은 스켈레톤 그대로입니다. 바뀐 것은
**이름(식별자·필드·파일명), 단위, 상수 3개(게이트·임계값·시뮬레이션 고정값), CSV** 입니다.

## 데이터 - 대시보드에서 업로드

CSV 형식: `Date,Arrivals,Departures` (일별, 단위: 명, UTF-8, 최소 41행).

- `data/sample_airport_arrivals.csv`는 **형식 확인용 합성 데이터**입니다(3년치, 요일·계절
  패턴 포함, 일평균 약 37,000명). 실제 분석·발표에는 팀이 선정한 한국공항공사 데이터를
  같은 컬럼 형식으로 가공해 업로드하세요 (출처·가공 방법: `docs/PROJECT_PLAN.md` "데이터").
- 서버를 띄운 뒤 대시보드(`http://localhost:8000/`)의 업로드 카드에서 파일을 올리면
  `data/uploads/`에 타임스탬프 파일명으로 쌓이고, 학습·시뮬레이션 코드는 항상 **가장
  최근에 업로드된 파일**을 사용합니다(`data/storage.py`의 `latest_upload()`).

## 모델 아키텍처

최근 20일(SEQ_LEN)의 (도착 여객, 출발 여객) 시퀀스를 입력받아 다음날 도착 여객 수를
예측하는 3층 LSTM입니다 (`serving_app/lstm_model.py`, Day1·Day2 공유).

```
Input (20, 2)  ->  LSTM(32, return_sequences=True)  ->  LSTM(32, return_sequences=True)
               ->  LSTM(16)  ->  Dense(16, relu)  ->  Dense(1)
```

Day3에서 드리프트가 감지되면 처음부터 다시 학습하지 않고, 전체 데이터로 학습된
**Production 가중치에서 이어서(warm start) 최근 21일로 10 epoch fine-tuning**합니다.
`serving_app/train_and_register.py`의 `train_and_register()`(Day2, scratch)와
`fine_tune()`(Day3)이 이 구분입니다. 스케일러(`scaler.pkl`)는 Day1에서 한 번 fit한 뒤
Day1~3 내내 재사용합니다.

## 디렉토리 구조

```
.
├── requirements.txt
├── docs/                            # 팀 기획서 초안·협업 규칙·API 명세 (조별 프로젝트 추가분)
├── data/
│   ├── sample_airport_arrivals.csv  # 형식 확인용 합성 샘플 (Date,Arrivals,Departures)
│   ├── storage.py                   # 업로드된 CSV 중 최신 파일을 찾는 latest_upload()
│   ├── uploads/                     # 업로드된 CSV가 쌓이는 곳 (시작 시 비어 있음, git 제외)
│   └── features.py                  # 시퀀스 빌더(SEQ_LEN=20) + AirportScaler (전 Day 공용)
├── scripts/
│   ├── train_baseline_v1.py         # Day1 사전 준비: MLflow 없이 로컬 baseline LSTM 생성
│   └── simulate_drift.py            # Day3: 정상/드리프트 배치 생성 + 서버로 주입
└── serving_app/
    ├── main.py                      # app 생성, 라우터 등록, 로딩 모드 분기, aiops 로거
    ├── schemas.py                   # /predict 요청·응답 스키마 (arrivals/departures)
    ├── lstm_model.py                # LSTM 아키텍처 정의
    ├── model_loader.py              # Day1 로컬 .keras → Day2 MLflow Production 로드
    ├── train_and_register.py        # Day2 base 학습/등록 + Day3 fine-tuning
    ├── Dockerfile, docker-compose.yml
    ├── models/                      # airport_v1.keras, scaler.pkl (스크립트가 생성, git 제외)
    ├── routers/
    │   ├── predict.py               # POST /predict, POST /predict/batch-test
    │   ├── health.py                # GET /health
    │   ├── data.py                  # POST /data/upload, GET /data/status
    │   └── logs.py                  # GET /logs, GET /logs/{filename}
    ├── monitoring/
    │   ├── drift_detector.py        # RMSE 기반 드리프트 판정 (TODO)
    │   └── retrain_trigger.py       # 감지 → fine-tuning → 재배포 (TODO)
    ├── static/index.html            # 운영자 대시보드 (업로드·시뮬레이션·파이프라인·로그)
    └── logs/                        # aiops.log (실행 시 자동 생성, git 제외)
```

## 실행 순서

```bash
pip install -r requirements.txt

# --- Day1 ---
uvicorn serving_app.main:app --host 0.0.0.0 --port 8000   # http://localhost:8000/ 대시보드, /docs 에서 API 확인
# 대시보드 업로드 카드에서 data/sample_airport_arrivals.csv(또는 실제 데이터)를 업로드한 뒤, 별도 터미널에서:
python scripts/train_baseline_v1.py                        # 로컬 baseline LSTM + scaler.pkl 생성
# LOADING_MODE=eager uvicorn serving_app.main:app --reload # Eager 방식과 시작 시간 비교

# --- Day2 ---
python serving_app/train_and_register.py                   # 로컬 MLflow(sqlite)에 학습 기록 + 게이트 통과 시 Production 승격
MODEL_SOURCE=mlflow uvicorn serving_app.main:app --host 0.0.0.0 --port 8000

# --- 컨테이너로 재현 (단일 컨테이너) ---
docker compose -f serving_app/docker-compose.yml up --build

# --- Day3 ---
uvicorn serving_app.main:app --host 0.0.0.0 --port 8000
python scripts/simulate_drift.py                           # 정상 배치 → 드리프트 배치 순서로 주입
```

`/predict`는 단일 값이 아니라 **최근 20일치 시퀀스**를 받습니다. 요청 예시:

```json
{
  "sequence": [
    {"arrivals": 36800, "departures": 36100},
    {"arrivals": 37450, "departures": 36900},
    { "...": "18개 더" }
  ]
}
```

응답 예시: `{"predicted_arrivals": 37912.43, "model_version": "production"}`

## TODO 체크리스트 (개인 실습에서 채운 내용을 그대로 이식)

배관(라우팅·업로드·MLflow 학습 로직 등)은 완성되어 있고, 각 Day의 핵심 학습 목표만
TODO로 비어 있습니다. 개인 실습(HAIC)에서 구현한 코드를 필드명만 바꿔 이식하면 됩니다.

- `serving_app/model_loader.py` → `_load_from_mlflow()`: MLflow Production 모델 로드 (Day2)
- `serving_app/routers/predict.py` → `batch_test()`: 슬라이딩 윈도우 예측 + `recent_predictions` 누적 (Day3)
- `serving_app/monitoring/drift_detector.py` → `compute_rmse()`: RMSE 직접 구현 (Day3)
- `serving_app/monitoring/retrain_trigger.py` → `check_and_trigger()`: fine-tuning 트리거 연결 (Day3)
- `scripts/simulate_drift.py` → `send_batch()`: `/predict/batch-test` 호출 (Day3)

## 완료 기준

- [ ] `/data/upload`로 CSV를 올리면 업로드 완료로 표시되는가 (`/data/status`로도 확인 가능)
- [ ] 정상 데이터로는 RMSE 2,700명 이내, 드리프트 데이터로는 2,700명 초과가 재현되는가
- [ ] `logs/aiops.log`에 `[WARN] drift detected` → `[INFO] retrain triggered` →
      `[OK] new_rmse=...` 순서로 기록되는가
- [ ] 재배포 후 `/predict` 호출 시 새 Production 버전이 응답하는가
