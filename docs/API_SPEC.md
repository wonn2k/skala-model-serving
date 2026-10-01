# API 명세 (기획서 ⑤)

서버: `http://localhost:8000` · Swagger UI: `/docs` · 대시보드: `/`
아래 내용은 코드(`serving_app/routers/*`, `schemas.py`) 기준이며, 구현 후 Swagger에서 실제 응답을 캡처해 갱신합니다.

| Method | URL | 용도 | 상태 |
|---|---|---|---|
| GET | `/health` | 헬스체크 · 모델 로드 여부 · 로딩 모드 | 완성 |
| POST | `/predict` | 최근 20일 시퀀스 → 다음날 도착 여객 수 예측 | 완성 (PR #5, `model_registry_version` 추가) |
| POST | `/predict/batch-test` | 드리프트 시뮬레이션: 연속 도착 여객 배치 → 슬라이딩 예측 + 3종 판정 + fine-tuning/롤백 | 완성 (PR #3) |
| POST | `/data/upload` | CSV 업로드 (`Date,Arrivals,Departures`) | 완성 |
| GET | `/data/status` | 최신 업로드 파일 요약 + 최근 20일 `recent` | 완성 (PR #6, `recent` 추가) |
| GET | `/logs` | 로그 파일 목록 | 완성 |
| GET | `/logs/{filename}` | 로그 파일 내용 (`aiops.log`) | 완성 |
| GET | `/monitor/versions` | Production 버전과 서버가 실제 들고 있는 버전 | 완성 (PR #6) |

## GET /health
```json
{"status": "ok", "model_loaded": false, "loading_mode": "lazy"}
```

## POST /predict
요청 — 가장 오래된 날 → 가장 최근 날 순서, 정확히 20개:
```json
{
  "sequence": [
    {"arrivals": 36800, "departures": 36100},
    {"arrivals": 37450, "departures": 36900},
    { "...": "18개 더" }
  ]
}
```
응답 (실측, `MODEL_SOURCE=mlflow`):
```json
{"predicted_arrivals": 39772.69, "model_version": "production", "model_registry_version": "1"}
```
- `model_version`: 소스 구분 — `"production"`(MLflow) 또는 `"v1-local"`(로컬 파일).
- `model_registry_version`: 실제 로드된 Registry 버전 번호(문자열). 로컬 모델이면 `null`. 승격·롤백 뒤 `reset_cache()`가 호출되면 다음 요청부터 바뀐다.
- 에러: 길이가 20이 아니거나(19·21행) 음수 값, `departures` 누락 → `422 Unprocessable Entity` (Pydantic 검증 메시지 포함).

## POST /predict/batch-test
요청 — 최소 21개(SEQ_LEN+1), 시뮬레이션은 41개(SEQ_LEN 20 + WINDOW_SIZE 21) 전송:
```json
{"arrivals": [37012.5, 36880.1, "...", 38104.9]}
```
응답 (실측, 정상 배치):
```json
{
  "predictions": [37102.3, "..."],
  "drift_check": {"status": "ok", "rmse": 898.53, "bias": -241.15, "anomaly_days": 0}
}
```
`drift_check.status` 5종과 함께 오는 필드:

| status | 뜻 | 추가 필드 |
|---|---|---|
| `ok` | 이상치 없음, 21일 \|bias\| ≤ 500, RMSE ≤ 2,700 | `rmse`, `bias`, `anomaly_days` |
| `anomaly` | 하루 오차 > 10,000인 날이 있음(판정에서 제외), 나머지는 정상 | 같음 (`anomaly_days` ≥ 1) |
| `structure_drift` | 이상치 제외 bias는 작은데 RMSE > 2,700 — 알림만, 재학습 없음 | 같음 |
| `retrain_triggered` | 이상치 제외 \|bias\| > 500 → fine-tuning 실행 | `promoted` (bool), `rmse` (fine-tuning 검증 RMSE), 승격 시 `version` |
| `rolled_back` | 승격 직후(미확정) 다시 드리프트였고 fine-tuning이 게이트 실패 → 이전 버전 복귀 | `production_version`, `rmse` |

- `bias` = 실제 − 예측의 평균(이상치 포함 전체 21건). 음수 = 모델이 높게 예측.
- 배치는 41개를 보내면 판정 윈도우가 정확히 그 배치의 21건이 된다. 승격 시 윈도우가 비워진다.
- 재학습 데이터는 배치가 아니라 **마지막으로 업로드된 CSV의 최근 41행**이다 — 시연 순서에서 업로드 순서를 지켜야 한다 (`data/README.md`).
- 실측 예 (`data/README.md` "드리프트 시연 순서"): `{"status": "retrain_triggered", "promoted": true, "rmse": 1338.33, "version": 2}` · `{"status": "rolled_back", "production_version": 2, "rmse": 2741.53}`

## POST /data/upload
`multipart/form-data`, 필드명 `file`. 응답:
```json
{"filename": "airport_1727700000.csv", "rows": 1096}
```
에러 400: UTF-8 아님 / 필수 컬럼(`Date, Arrivals, Departures`) 누락 / 41행 미만.

## GET /data/status
```json
{
  "exists": true, "filename": "airport_1790820812.csv", "rows": 547,
  "start_date": "2023-01-01", "end_date": "2024-06-30",
  "min_arrivals": 25168.0, "max_arrivals": 46954.0,
  "recent": [{"date": "2024-06-11", "arrivals": 36412, "departures": 36870}, {"...": "총 20개, 오래된 날 → 최근 날"}]
}
```
`recent` (PR #6): 최신 업로드의 마지막 20행, 정수. D1이 이 배열을 그대로 `/predict`의 `sequence`로 보내면 된다.

## GET /logs · GET /logs/{filename}
```json
[{"name": "aiops.log", "size": 412}]
```
```json
{"name": "aiops.log", "content": "2026-... [WARNING] [WARN] drift detected - triggering retrain\n..."}
```

## GET /monitor/versions (PR #6, `serving_app/routers/monitor.py`)
MLflow Registry가 말하는 Production 버전과, 서버가 실제로 메모리에 들고 있는 버전을 나란히 돌려준다. 대시보드 구역 3에 한 줄로 표시. 실측:
```json
{
  "model_name": "Airport_Arrivals_Predictor",
  "model_source": "mlflow",
  "production": {"version": 1, "run_id": "bbec6d0d...", "rmse": 2266.94, "created_at": 1790831552595},
  "serving_version": "production",
  "serving_run_id": null,
  "stale": null
}
```
| 필드 | 뜻 |
|---|---|
| `model_source` | `MODEL_SOURCE` 환경변수 값 (`mlflow` / `local`) |
| `production` | Registry의 Production 최신 버전 — `version`(정수), `run_id`, `rmse`(해당 run 지표), `created_at`(epoch ms). `local`이면 `null` |
| `serving_version` | 서버가 들고 있는 모델의 `model_version` (`"production"` / `"v1-local"`) |
| `serving_run_id` | 서빙 모델의 run_id. 현재 `LoadedModel`이 run_id를 보관하지 않아 `null` (A에 요청 중) |
| `stale` | Registry 버전 ≠ 서빙 버전이면 `true`. `serving_run_id`가 `null`인 동안은 `null`(판정 불가), `local`이면 `false` |

- `MODEL_SOURCE=local`: `{"model_name": ..., "model_source": "local", "production": null, "serving_version": "v1-local", "serving_run_id": null, "stale": false}`.
- 승격·롤백 직후 `/predict`를 한 번 호출해야 서빙 버전이 갱신된다(lazy 재로드). 그 전까지 `production.version`과 `/predict`의 `model_registry_version`이 다를 수 있다.
