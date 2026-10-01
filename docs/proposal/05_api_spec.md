# ⑤ API 명세 (요약)

> **생성물.** 프롬프트 5가 2026-10-01 실제 서버 호출 결과로 생성했다. 상세 명세는 `docs/API_SPEC.md`. 손으로 고치지 않는다.
> 호출 조건: 현재 코드(`origin/main` `7364435` + `departures` PR), `MODEL_SOURCE=mlflow`·lazy, seed 42 학습 v1(게이트 rmse 2363) — 명령은 `docs/code_current.md` 1번 변경 기록.
> 코드의 엔드포인트 8개와 명세가 일치한다. 예시의 필드명·값은 실제 호출 응답 그대로다.

| Method · URL | 용도 | 요청 예시 | 실제 응답 예시 | 에러 응답 (실제 재현) |
|---|---|---|---|---|
| GET `/health` | 헬스체크: 상태·모델 로드 여부·로딩 모드 | — | `{"status":"ok","model_loaded":true,"loading_mode":"lazy"}` (lazy는 첫 예측 전 `model_loaded:false`) | — |
| POST `/predict` | 최근 20일 시퀀스 → 다음날 도착 여객 수 1개 | `{"sequence":[{"arrivals":33285,"departures":44836}, …20개]}` (오래된 날→최근 날, 0 이상 정수) | `{"predicted_arrivals":39980.96,"model_version":"production","model_registry_version":"1"}` — 첫 호출 lazy 로드 포함 0.26초(콜드 스타트 2.0~3.5초), 이후 0.016초 | 19개/21개 → 422 `too_short`(`"List should have at least 20 items after validation, not 19"`) · 음수 → 422 `greater_than_equal` · Production 없음/로딩 실패 → 텍스트 500 (대체 로드 안 함) |
| POST `/predict/batch-test` | 배치 41개 → 슬라이딩 예측 21개 + 드리프트 판정 + 자동 재학습 | `{"arrivals":[37934,…41개], "departures":[35255,…41개]}` (`departures` 선택 — 생략 시 37,000 고정) | `{"predictions":[37587.42,…21개],"drift_check":{"status":"ok","rmse":1191.34,"bias":85.67,"anomaly_days":0}}` — `status` 5종: ok·anomaly·structure_drift·retrain_triggered·rolled_back | `departures` 길이 불일치(41 vs 40) → 422 `"departures는 arrivals와 길이가 같아야 합니다"` |
| POST `/data/upload` | 학습용 CSV 업로드 (`Date,Arrivals,Departures`) | multipart `file=jeju_airport_arrivals.csv` | `{"filename":"airport_1790845113.csv","rows":1035}` | 컬럼 누락 → 400 `"CSV에 ['Arrivals', 'Date', 'Departures'] 컬럼이 모두 있어야 합니다."` · UTF-8 아님/41행 미만 → 400 |
| GET `/data/status` | 최신 업로드 요약 + `recent` 최근 20일 | — | `{"exists":true,"filename":"airport_1790835307.csv","rows":1035,"start_date":"2023-01-01","end_date":"2025-10-31","min_arrivals":25168.0,"max_arrivals":46954.0,"recent":[{"date":"2025-10-12","arrivals":33285,"departures":44836},…20건]}` — `recent`를 그대로 `/predict`에 전달 가능 | — |
| GET `/logs` | 로그 파일 목록 | — | `[{"name":"aiops.log","size":1782},{"name":"server.log","size":385}]` | — |
| GET `/logs/{filename}` | 로그 내용 (대시보드 재학습 로그 카드) | `/logs/aiops.log` | `{"name":"aiops.log","content":"2026-10-01 17:14:52,538 [WARNING] [WARN] drift detected - rmse=2189 bias=+1753 … [INFO] retrain triggered …"}` | — |
| GET `/monitor/versions` | Registry Production vs 서빙 중 버전, `stale` 판정 (신규) | — | `{"model_name":"Airport_Arrivals_Predictor","model_source":"mlflow","production":{"version":1,"run_id":"9b8ca24e…","rmse":2363.496,"created_at":1790845094679},"registry_error":null,"serving_version":"production","serving_registry_version":"1","serving_run_id":"9b8ca24e…","stale":false,"stale_basis":"run_id"}` — 첫 예측 전엔 `serving_* null`·`stale null`(판정 불가) | 레지스트리 조회 실패 → 200 + `registry_error` 메시지 (5초 타임아웃·재시도 없음) |

운영 포인트 세 가지:

- **입력 게이트**: 길이·음수·컬럼·행 수 위반은 모두 422/400으로 모델에 닿기 전에 거른다 (4모드 × 4종 실측, `docs/code_current.md` 1번).
- **버전 추적**: `model_version`은 mlflow 모드에서 항상 `"production"`이므로 실제 전환은 `model_registry_version`과 `/monitor/versions`의 `stale`로 본다 — 같은 입력의 예측값이 버전이 바뀌어도 동일(39980.96)할 수 있어 값만으로는 구분 불가 (`docs/code_current.md` 2번).
- **숫자 출처 주의**: `drift_check.rmse`·`bias`는 이상치 포함 값, 판정·`aiops.log`는 이상치 제외 값 — 발표 대본의 표 값은 `logs/aiops.log` 기준으로 명시 (`docs/code_current.md` 2번 "다른 영역에 요청").
