# API 명세 (기획서 ⑤)

서버: `http://localhost:8000` · Swagger UI: `/docs` · 대시보드: `/`
아래 예시는 **2026-10-01 실제 호출 결과**다 (현재 코드, `MODEL_SOURCE=mlflow`·`LOADING_MODE=lazy`, 측정용 포트 8001, 복사본 레지스트리에 seed 42 학습 v1 rmse 2363 등록 후 호출 — `docs/code_current.md` 1번 변경 기록). 코드에 등록된 엔드포인트 8개와 이 목록은 일치한다 (코드에만 있거나 명세에만 있는 것 없음).

| Method | URL | 용도 | 상태 |
|---|---|---|---|
| GET | `/health` | 헬스체크 · 모델 로드 여부 · 로딩 모드 | 완성 |
| POST | `/predict` | 최근 20일 시퀀스 → 다음날 도착 여객 수 예측 | 완성 (TODO 1 MLflow 로드 구현 완료, `model_registry_version` 추가) |
| POST | `/predict/batch-test` | 연속 도착 여객 배치 → 슬라이딩 예측 + 드리프트 판정 + 자동 재학습 | 완성 (TODO 2·3·4 이식, `status` 5종) |
| POST | `/data/upload` | CSV 업로드 (`Date,Arrivals,Departures`) | 완성 |
| GET | `/data/status` | 최신 업로드 파일 요약 + `recent` 최근 20일 | 완성 (`recent` 추가) |
| GET | `/logs` | 로그 파일 목록 | 완성 |
| GET | `/logs/{filename}` | 로그 파일 내용 (`aiops.log`) | 완성 |
| GET | `/monitor/versions` | Registry Production vs 서빙 중 버전 · `stale` 판정 | 완성 (신규, 설계안보다 필드 추가) |

## GET /health

모델 로드 전(lazy) / 후 — 실제 응답:
```json
{"status": "ok", "model_loaded": false, "loading_mode": "lazy"}
```
```json
{"status": "ok", "model_loaded": true, "loading_mode": "lazy"}
```

## POST /predict

요청 — 가장 오래된 날 → 가장 최근 날 순서, 정확히 20개 (`arrivals`·`departures` 0 이상 정수). `GET /data/status`의 `recent`를 그대로 변환해 보내면 된다:
```json
{"sequence": [
  {"arrivals": 33285, "departures": 44836},
  {"arrivals": 34700, "departures": 45241},
  {"...": "16개 더"},
  {"arrivals": 41716, "departures": 40314},
  {"arrivals": 42323, "departures": 42950}
]}
```
실제 응답 (HTTP 200, 첫 호출 lazy 로드 포함 0.2644초 · 두 번째 0.0160초 — 이 측정은 서버 프로세스가 TensorFlow import를 이미 마친 상태의 값. 콜드 스타트 포함 첫 요청은 2.0~3.5초, `docs/code_current.md` 1번):
```json
{"predicted_arrivals": 39980.96, "model_version": "production", "model_registry_version": "1"}
```
- `model_version`: local이면 `"v1-local"`, mlflow면 항상 `"production"`. 실제 버전 번호는 `model_registry_version`(로컬 모델은 `null`).

에러 — 실제 재현:
- 시퀀스 19개 → `422`:
```json
{"detail": [{"type": "too_short", "loc": ["body", "sequence"], "msg": "List should have at least 20 items after validation, not 19", "ctx": {"field_type": "List", "min_length": 20, "actual_length": 19}}]}
```
(21개 등 초과도 동일하게 422 — `docs/code_current.md` 1번 측정값)
- 음수 값 → `422`:
```json
{"detail": [{"type": "greater_than_equal", "loc": ["body", "sequence", 0, "arrivals"], "msg": "Input should be greater than or equal to 0", "input": -1, "ctx": {"ge": 0}}]}
```
- MLflow Production이 없거나 로딩 실패 → `500 Internal Server Error` (텍스트 본문). 로컬 모델로 조용히 대체하지 않는 설계다. 실제 재현: 빈 레지스트리에서 첫 호출 500 (서버 로그 `MlflowException: Registered Model with name=Airport_Arrivals_Predictor not found`). 로컬 모드에서 모델 파일이 없을 때도 같은 형태의 텍스트 500 (D1이 A에 JSON 오류 응답 검토 요청 중).

## POST /predict/batch-test

요청 — `arrivals` 최소 21개(SEQ_LEN+1), 시연은 41개(SEQ_LEN 20 + WINDOW_SIZE 21). `departures`는 **선택**: 같은 길이의 일별 출발 여객이며, 생략하면 서버가 모든 날을 `SIMULATED_DEPARTURES`(37,000명)로 채운다. 대시보드 CSV 배치는 함께 보내고, 랜덤워크 배치와 `scripts/simulate_drift.py`는 생략한다:
```json
{"arrivals": [37934, 38881, "...", 35928], "departures": [35255, 37145, "...", 39674]}
```
실제 응답 (`data/jeju_demo_shift_batch_normal_41rows.csv` 41행 + departures, 모델 v1, HTTP 200, 0.2850초 — 예측 21개):
```json
{
  "predictions": [37587.42097616196, 36708.04119360447, "...19개 더...", 36536.62003171444],
  "drift_check": {"status": "ok", "rmse": 1191.3401794710953, "bias": 85.66765903858911, "anomaly_days": 0}
}
```
- `drift_check.status` 5종: `"ok"` | `"anomaly"`(이상치만) | `"structure_drift"`(치우침 없는 큰 오차, 알림만) | `"retrain_triggered"`(수준 드리프트 → fine-tuning → 게이트 통과 시 승격, `promoted`·`rmse`·`version` 포함) | `"rolled_back"`(미확정 승격 뒤 재학습 게이트 실패 → 이전 버전 복귀).
- 주의: `drift_check.rmse`·`bias`는 이상치 **포함** 값이고 판정·`aiops.log`는 이상치 **제외** 값을 쓴다 — 화면과 로그 숫자가 다를 수 있다 (`docs/code_current.md` 2번 "다른 영역에 요청").

에러 — 실제 재현: `departures` 길이 불일치(41 vs 40) → `422`:
```json
{"detail": [{"type": "value_error", "loc": ["body"], "msg": "Value error, departures는 arrivals와 길이가 같아야 합니다"}]}
```

## POST /data/upload

`multipart/form-data`, 필드명 `file`. 실제 응답 (`data/jeju_airport_arrivals.csv` 업로드, HTTP 200):
```json
{"filename": "airport_1790845113.csv", "rows": 1035}
```
에러 — 실제 재현: 필수 컬럼 누락(`Departures` 없음) → `400`:
```json
{"detail": "CSV에 ['Arrivals', 'Date', 'Departures'] 컬럼이 모두 있어야 합니다."}
```
그 밖에 UTF-8 아님 / 최소 행 수(`SEQ_LEN + WINDOW_SIZE` = 41행) 미만도 400.

## GET /data/status

실제 응답 (HTTP 200 — `recent`는 최신 업로드의 마지막 20행, 오래된 날 → 최근 날 순서, 정수. 그대로 `/predict`의 `sequence`로 쓸 수 있다. 숫자로 못 읽는 칸은 해당 항목만 `null`):
```json
{
  "exists": true, "filename": "airport_1790835307.csv", "rows": 1035,
  "start_date": "2023-01-01", "end_date": "2025-10-31",
  "min_arrivals": 25168.0, "max_arrivals": 46954.0,
  "recent": [
    {"date": "2025-10-12", "arrivals": 33285, "departures": 44836},
    {"...": "18개 더"},
    {"date": "2025-10-31", "arrivals": 42323, "departures": 42950}
  ]
}
```

## GET /logs · GET /logs/{filename}

실제 응답:
```json
[{"name": "aiops.log", "size": 1782}, {"name": "server.log", "size": 385}]
```
```json
{"name": "aiops.log", "content": "2026-10-01 17:14:52,538 [WARNING] [WARN] drift detected - rmse=2189 bias=+1753 (excluding 0 anomaly day(s)) - triggering retrain\n2026-10-01 17:14:52,875 [INFO] [INFO] retrain triggered (window=last_21_days)\n..."}
```

## GET /monitor/versions (신규 — 구현 완료)

MLflow Registry의 Production과 서버가 실제로 들고 있는 모델을 나란히 반환한다. 실제 응답 (HTTP 200, 0.0070초):
```json
{
  "model_name": "Airport_Arrivals_Predictor",
  "model_source": "mlflow",
  "tracking_uri": "sqlite:///...",
  "production": {"version": 1, "run_id": "9b8ca24e0adb4692a4b37885f146745a", "rmse": 2363.4963557372193, "created_at": 1790845094679},
  "registry_error": null,
  "serving_version": "production",
  "serving_registry_version": "1",
  "serving_run_id": "9b8ca24e0adb4692a4b37885f146745a",
  "stale": false,
  "stale_basis": "run_id"
}
```
- 설계안(`model_name`·`production`·`serving_version`)보다 늘어난 필드: `model_source`, `tracking_uri`, `registry_error`, `serving_registry_version`, `serving_run_id`, `stale`, `stale_basis`.
- `stale`: 레지스트리 Production과 서빙 중 모델의 어긋남. `true` = 재기동 필요(서버 밖 승격 시), `false` = 일치, `null` = 판정 불가(lazy 모드에서 첫 예측 전 — 실제 호출로 확인, 위 `production: null`·`serving_version: null` 상태). 판정 근거는 `stale_basis`(`"run_id"` 우선, 없으면 등록 번호).
- `created_at`은 Unix 밀리초 (대시보드가 한국 시간으로 변환 표시).
- `registry_error`: 레지스트리 조회 실패 메시지(성공 시 null). 조회 타임아웃은 5초 × 재시도 없음. 주의: 오타난 sqlite URI는 빈 DB가 새로 생겨 "성공"으로 보인다 — `tracking_uri`로 구분.
- `MODEL_SOURCE=local`이면 `production: null`, `serving_version: "v1-local"`, `stale: false`.
