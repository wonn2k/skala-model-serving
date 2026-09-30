# API 명세 (기획서 ⑤)

서버: `http://localhost:8000` · Swagger UI: `/docs` · 대시보드: `/`
아래 내용은 코드(`serving_app/routers/*`, `schemas.py`) 기준이며, 구현 후 Swagger에서 실제 응답을 캡처해 갱신합니다.

| Method | URL | 용도 | 상태 |
|---|---|---|---|
| GET | `/health` | 헬스체크 · 모델 로드 여부 · 로딩 모드 | 완성 |
| POST | `/predict` | 최근 20일 시퀀스 → 다음날 도착 여객 수 예측 | 완성 (모델 로드 TODO 의존) |
| POST | `/predict/batch-test` | 드리프트 시뮬레이션: 연속 도착 여객 배치 → 슬라이딩 예측 + 드리프트 판정 | TODO (Day3) |
| POST | `/data/upload` | CSV 업로드 (`Date,Arrivals,Departures`) | 완성 |
| GET | `/data/status` | 최신 업로드 파일 요약 | 완성 |
| GET | `/logs` | 로그 파일 목록 | 완성 |
| GET | `/logs/{filename}` | 로그 파일 내용 (`aiops.log`) | 완성 |
| GET | `/monitor/versions` | **현재 Production 모델 버전** (신규) | 🆕 Sprint 2 |

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
응답:
```json
{"predicted_arrivals": 37912.43, "model_version": "production"}
```
에러: 길이가 20이 아니거나 음수 값 → `422 Unprocessable Entity` (Pydantic 검증 메시지 포함).

## POST /predict/batch-test
요청 — 최소 21개(SEQ_LEN+1), 시뮬레이션은 41개(SEQ_LEN 20 + WINDOW_SIZE 21) 전송:
```json
{"arrivals": [37012.5, 36880.1, "...", 38104.9]}
```
응답:
```json
{
  "predictions": [37102.3, "..."],
  "drift_check": {"status": "ok"}
}
```
`drift_check.status`: `"ok"` | `"retrain_triggered"` (+ `promoted`, `rmse` 필드, TODO 구현 후).

## POST /data/upload
`multipart/form-data`, 필드명 `file`. 응답:
```json
{"filename": "airport_1727700000.csv", "rows": 1096}
```
에러 400: UTF-8 아님 / 필수 컬럼(`Date, Arrivals, Departures`) 누락 / 41행 미만.

## GET /data/status
```json
{
  "exists": true, "filename": "airport_1727700000.csv", "rows": 1096,
  "start_date": "2023-01-01", "end_date": "2025-12-31",
  "min_arrivals": 28900.0, "max_arrivals": 46541.0
}
```

## GET /logs · GET /logs/{filename}
```json
[{"name": "aiops.log", "size": 412}]
```
```json
{"name": "aiops.log", "content": "2026-... [WARNING] [WARN] drift detected - triggering retrain\n..."}
```

## GET /monitor/versions (신규 · 설계안)
MLflow Registry에서 `Airport_Arrivals_Predictor`의 Production 버전을 조회한다. 대시보드 구역 3에 한 줄로 표시.
```json
{
  "model_name": "Airport_Arrivals_Predictor",
  "production": {"version": "3", "run_id": "…", "rmse": 2210.5, "created_at": "2026-…"},
  "serving_version": "production"
}
```
- 구현 위치(안): `serving_app/routers/monitor.py` 신설 후 `main.py`에 `include_router` 한 줄 추가 (기존 라우터 구조 유지).
- MODEL_SOURCE=local 인 경우 `{"model_name": ..., "production": null, "serving_version": "v1-local"}` 반환.
