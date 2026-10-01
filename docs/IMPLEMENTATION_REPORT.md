# 구현 보고서

> **생성물.** `docs/code_current.md`(코드 현황 기록)를 프롬프트 7로 변환한 보고서다. 손으로 고치지 않는다 — 고칠 것이 있으면 원자료를 고치고 프롬프트를 다시 돌린다.
> 원자료 기준: 2026-10-01, 브랜치 `feat/e-batch-departures` (= `origin/main` `7364435` + 커밋 `22b004a`).
> 원자료의 "작성 규칙" 절(파일 기록 방법 안내)은 구현 내용이 아니라 옮기지 않았다. 수치·과정·측정 조건은 원자료 그대로이며, 측정 시각이 기록되지 않은 값은 비워 두었다.

이 프로젝트는 제주공항 **다음날 도착 여객 수**를 LSTM으로 예측해 FastAPI로 서빙하고, MLflow 게이트 → 드리프트 감지 → 자동 fine-tuning → 재배포(AIOps)까지 잇는 조별 미니 프로젝트다. 담당: A 서빙 API(윤동현), B 모니터링·AIOps, C 학습·배포·버전 조회(유경모), D1·D2 대시보드(D1 youjin09222), E 데이터·상수·통합.

## 1. 구현 현황 요약

파이프라인 전 단계가 동작 상태다. 유일하게 "코드 완성"에 머문 것은 드리프트 시뮬레이션 스크립트(랜덤워크 σ가 실변동성보다 낮아 실측 미실행)다.

| 파이프라인 단계 | 담당 | 상태 | 근거 |
|---|---|---|---|
| CSV 업로드 (`POST /data/upload`, `GET /data/status`) | C | 동작 | 스켈레톤 제공. `data/jeju_airport_arrivals.csv` 1,035행 업로드 (PC 1대) |
| Day1 baseline 학습 (`scripts/train_baseline_v1.py`) | C | 동작 | RMSE 2,571명, 게이트 2,700명 통과 (PC 1대) |
| Day2 MLflow 학습·게이트·Production 승격 (`serving_app/train_and_register.py`) | C | 동작 | RMSE 2,267명, 게이트 통과 후 Production 승격 (PC 1대) |
| 예측 서빙 `POST /predict` (로컬 모델) | A | 동작 | PR #5. local/lazy 첫 요청 2.04초, 두 번째 0.013초, 입력 오류 422 |
| MLflow Production 모델 서빙 (`MODEL_SOURCE=mlflow`) | A | 동작 | PR #5. TODO 1 이식, `reset_cache()` 추가, `model_registry_version` 응답. mlflow/lazy 첫 요청 3.5초, 두 번째 0.016초 |
| 드리프트 판정 (`POST /predict/batch-test`) | B | 동작 | PR #3. TODO 2·3 이식, 3종 판정(이상치/수준/구조), `status` 5종 |
| 드리프트 감지 시 자동 재학습 | B | 동작 | PR #3. TODO 4 이식, fine-tuning → 게이트 → 승격 시 캐시·윈도우 초기화, 미확정 승격 뒤 실패 시 롤백. A+B 결합 검증 통과 |
| 드리프트 시뮬레이션 스크립트 | B | 코드 완성 | PR #3. TODO 5 이식. 랜덤워크 σ가 실변동성보다 낮아 실측 미실행 |
| `GET /data/status`의 `recent` (최근 20일) | C | 동작 | PR #6. 20건(2025-10-12~10-31), 오래된 순, 정수. `recent` → `/predict` 전달 시 HTTP 200 확인 |
| Production 버전 조회 `GET /monitor/versions` | C | 동작 | A의 `run_id`(PR #9) 연결로 `stale` 실판정 — 컨테이너 3단계 재현 false → true → false |
| 대시보드 예측값·혼잡 등급 카드 | D1 | 동작 | main 머지(#20). 브라우저 카드 실측: v1 39,981명 보통 44 ms → v3 40,501명 혼잡 187 ms |
| 대시보드 CSV 배치 전송, Production 버전 표기 | D2 | 동작 | PR #14·#16·#23 머지. CSV 검증·전송, 버전·로그 5초 조회, 비교 차트 실측 |
| Docker 컨테이너 재현 | C | 동작 | 빌드 8단계(`--no-cache`, 이미지 3.26GB), `up --build` 정상 기동, 컨테이너 전체 시연 실측 완료 |
| 통합 데모 (업로드 → 학습 → 예측 → 드리프트 → 재학습 → 재배포) | E | 동작 | `main` 위 컨테이너에서 `data/README.md` 수준 이동 시연 1~6번이 소수점까지 재현, 전체 시연 실측도 README 첫 표와 정수까지 일치. 코드 상수 확인: `FINE_TUNE_EPOCHS=3`, `BIAS_THRESHOLD=500` (2026-10-01) |

## 2. 단계별 구현 내용

### 2.1 서빙 API — A (윤동현)

**무엇을 구현했나.** 최근 20일 시퀀스를 받아 다음 날 도착 여객 수 하나를 돌려주는 HTTP 서버다. 모델을 어디서(로컬 파일 / MLflow Production), 언제(기동 시 eager / 첫 요청 시 lazy) 불러올지 정한다.

- `main.py`: FastAPI 앱 생성, 라우터 5개 등록(`predict`, `health`, `data`, `logs`, `monitor`), `aiops` 로거를 `logs/aiops.log`에 연결, `static/`을 `/`에 마운트, `LOADING_MODE=eager`면 startup에서 `model_loader.load_eager()` 실행.
- `model_loader.py`: `get_model()`은 `_model_cache`가 비어 있으면 `_load_model()`을 호출하고, `MODEL_SOURCE`에 따라 `_load_from_local()`(`serving_app/models/airport_v1.keras` + `scaler.pkl`, 버전 `v1-local`) 또는 `_load_from_mlflow()`(Production을 조회한 뒤 `models:/Airport_Arrivals_Predictor/<실제 버전>`으로 고정해서 로드 — **실습 TODO 1 구현**)를 탄다. 반환은 `LoadedModel(predict_one, version, registry_version, run_id)`. 스케일러는 Day1의 로컬 `scaler.pkl`을 그대로 쓰고 재학습하지 않는다. Production이 없거나 로딩이 실패하면 로컬 모델로 조용히 대체하지 않는다.
- `routers/predict.py`의 `predict()`: `PredictRequest.sequence` 20개 → `model.predict_one()` → `PredictResponse(predicted_arrivals, model_version, model_registry_version)`. 기존 `model_version="production"`은 유지하고 실제 등록 번호를 별도 필드로 돌려준다(로컬 모델은 `null`).
- `routers/health.py`: `GET /health` → 상태, 로딩 모드, 모델 로드 여부.
- `schemas.py`: `PredictRequest`(길이 20, `arrivals`·`departures` 0 이상 정수, 위반 시 422), `BatchTestRequest`(`arrivals` 21개 이상, 선택 `departures` — 있으면 같은 길이, 다르면 422).
- 캐시 운영: B가 승격·롤백 **성공 후** `model_loader.reset_cache()`를 부르면 다음 예측에서 모델을 다시 로드한다. 캐시 잠금으로 첫 로딩과 초기화의 경합을 막았다. `reset_cache()`는 호출한 프로세스에만 적용되며, 이미 모델을 받은 요청은 기존 모델로 마치고, 다중 worker 간 동기화는 구현 범위 밖이다.
- C 요청 반영: 실제 선택한 등록 버전의 `run_id`를 모델 객체에 함께 보관해, C의 조회 API 수정 없이 `serving_run_id`·`stale` 판정이 연결됐다. 로컬 모델이나 학습 run 정보가 없는 등록은 `run_id=None`, lazy 로딩 전 비교는 `stale=null`이다.
- 교수 실습가이드 v3의 Day2 `model_version: production`, Day1 고정 스케일러, Lazy/Eager 방식은 유지했고, 모델 구조·피처·학습·게이트·드리프트 정책은 바꾸지 않았다.

**어떻게 확인했나.** 프로젝트 루트, 포트 8000에서 순차 실행:

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

단위 테스트는 `python -m unittest discover -s tests -v`(A→C 모델 식별·B 판정 창 회귀 신규 6개 + 기존 8개 통과), 캐시·로더 테스트 8개 통과(버전 숫자 비교, 고정 scaler, Production 없음, 실패 후 재시도, 동시 첫 요청 1회 로드, 로딩 중 reset, eager 캐시, 반복 reset).

**결과(핵심 수치).** local/lazy 첫 요청 2.0359초 → 두 번째 0.0135초, mlflow/lazy 3.5117초 → 0.0162초, mlflow/eager 기동 2.1539초·첫 요청 0.0961초. 잘못된 입력 4종(19행·21행·음수·departures 누락) × 4모드 전부 422. 실제 재학습 통합에서 v1→v2→v3→v2 롤백을 거치며 단계별 `stale=false`, 의도적 불일치 `true` → 복구 `false` 확인. 전체 수치는 3장 측정 결과 모음 §A.

**협업 조율.**
- B: PR #3 병합으로 승격·롤백 성공 후 `reset_cache()` 호출 연결 완료, A+B 결합 검증 통과. 판정 창 초기화·재학습 정책은 B 담당.
- C/D/E: 단일 예측 응답에 `model_registry_version: str | null` 추가 — C의 버전 조회, D의 표시, E의 API 명세에 반영 요청.
- C: `recent`는 PR #6으로 연결 완료, `run_id` 보관 요청도 대응 완료. `monitor.py`의 "항상 null" 주석·문서 갱신은 C에 요청. 같은 run을 여러 버전으로 등록하면 run 비교만으로는 버전 차이를 구분하지 못하므로 정확한 비교에는 `registry_version` 사용 권장.
- C: 실제 승격 후 이전 버전들이 Production에 남는 현상 확인 — 기존 버전 Archived 처리는 학습·배포 담당 후속 PR에서 조율.
- E: 3월 18일 학습 컷 마지막 값 30,608은 3월 19일 원자료를 쓴 보간이다. 시점별 데이터 생성으로 미래 정보 누수를 제거한 후 게이트/롤백 결과 재확인 필요 — 이번 통합 결과는 현재 배포 CSV의 동작 확인이며 데이터 타당성 보증이 아니다.
- E/D2: `serving_run_id`는 로드한 모델의 실제 run, `stale`은 C의 비교 결과 — API_SPEC·0번 표·프론트 갱신은 담당자 영역에 요청.

### 2.2 학습·배포·버전 조회 — C (유경모)

**무엇을 구현했나.** 업로드된 CSV로 모델을 학습하고, 게이트(RMSE ≤ 2,700명)를 통과한 버전만 MLflow Production에 올린다. 어느 PC에서나 같은 절차로 재현되게 컨테이너로 묶는다.

- `routers/data.py`: `POST /data/upload`(컬럼 `Date,Arrivals,Departures` 검사, 최소 행 수 `SEQ_LEN + WINDOW_SIZE` = 41, `data/uploads/` 저장), `GET /data/status`(행 수, 기간, 최소·최대, **`recent`** — 최신 업로드의 마지막 20행을 오래된 날 → 최근 날 순서의 `[{date, arrivals, departures}]` 정수로). `data/storage.py`의 `latest_upload()`가 최신 파일 경로를 준다.
- `scripts/train_baseline_v1.py`: Day1. 최신 업로드로 학습 → 마지막 20% 검증 RMSE → 게이트 → `serving_app/models/airport_v1.keras`·`scaler.pkl` 저장.
- `train_and_register.py`: `train_and_register()`(seed 42, 100 epoch → `rmse()` → MLflow run 기록 → `_register_if_gate_passed()`가 통과 시 `Airport_Arrivals_Predictor` 등록 + Production 승격), `fine_tune(rows)`(Production 가중치에서 `FINE_TUNE_EPOCHS`(3) epoch, LR 1e-4, 같은 게이트 — 호출은 B의 TODO 4).
- `routers/monitor.py`(신설): `GET /monitor/versions` → Registry의 Production 버전·run_id·rmse·created_at과 서버가 실제로 든 모델(`serving_version`, `serving_registry_version`)을 함께 반환, 어긋남은 `stale`로, 판정 근거는 `stale_basis`(run_id 우선, 없으면 등록 번호)로, 조회 실패는 `registry_error`로, 어느 저장소를 봤는지는 `tracking_uri`로 구분. `MODEL_SOURCE=local`이면 `production`은 null. MLflow 요청 타임아웃 5초 × 재시도 없음(`setdefault`) 적용.
- `Dockerfile` / `docker-compose.yml`: 이미지 빌드 중 baseline + MLflow 학습 실행, `MODEL_SOURCE=mlflow`·`LOADING_MODE=eager`로 기동. compose에 `${VAR:-기본값}` 환경변수 통로.

흐름: 대시보드 CSV 업로드 → `data/uploads/` → baseline(로컬 모델) → `train_and_register.py`(MLflow run → 게이트 → Registry → Production) → A의 `_load_from_mlflow()`가 읽는다. 승격 시 기존 Production 버전을 Archived로 내리지 않아 여러 버전이 Production에 남을 수 있다. 모델 파일과 MLflow 기록은 커밋하지 않으므로 각자 PC에서 업로드 → baseline → MLflow 학습을 직접 실행해야 한다.

**어떻게 확인했나.** `python scripts/train_baseline_v1.py` → `python serving_app/train_and_register.py` → `docker compose -f serving_app/docker-compose.yml up --build`. 컨테이너에서 `curl localhost:8000/monitor/versions`, `curl localhost:8000/data/status`, `recent`를 `sequence`로 변환한 `/predict` POST, `docker compose exec serving-app python serving_app/train_and_register.py`(컨테이너 내 재학습으로 stale 재현), `docker compose build --no-cache`(빌드 계측).

**결과(핵심 수치).** Day1 baseline RMSE 2,571명(게이트 2,700 통과, 전일값 복사 3,600명 대비 -29%), Day2 MLflow 학습 RMSE 2,267명(2266.94) → Production 승격. `recent` → `/predict` 왕복 HTTP 200(39772.69). 컨테이너: 빌드 8단계·561줄, pip 35.5초 + 빌드 중 학습·등록 32.6초(`[GATE PASSED] rmse=2363 -> v1`) + export 38.2초, 이미지 3.26GB. `stale` 3단계 재현(v1/v1 false → v2 승격·서버 v1 **true** → 재기동 false), run_id 기준 판정(`stale_basis="run_id"`) 동일 3단계 확인. `main` 위 컨테이너에서 수준 이동 시연 1~6번 소수점까지 재현(2363.50 / 1409.99 / +76.14 / 1892.76 / 1423.24 / 1619.16 / +239.10), 전체 시연 실측도 README 첫 표와 정수까지 일치(2183.06 / −116.79 / 3490 / −587 / 2098.47), 감지→승격 3.7초. 무응답 트래킹 서버 호출은 MLflow 기본값으로 848.03초(14분 8초)를 막다가 재시도 없음 설정으로 5.70초. 전체 수치는 3장 §C.

**협업 조율.**
- B에게: 배치를 연속 주입하면 `recent_predictions` 윈도우가 이어져 판정이 뒤집힌다(같은 세 배치 연속 시 normal이 `retrain_triggered`, drift가 `ok`). 배치마다 서버를 다시 띄우면 의도대로 갈린다 — 시연 때 윈도우 비우기 절차 필요.
- A에게 (해결됨): PR #5의 `_load_from_mlflow()`·`reset_cache()`·`registry_version`, PR #9의 `run_id`로 `stale`이 실제 판정을 한다. `monitor.py`는 run_id 우선, 없으면 등록 번호로 비교하고 `stale_basis`로 근거를 돌려준다.
- A에게 (당시 급함, 이후 해소): `reset_cache()` 부재 시 승격 순간 `AttributeError`로 `/predict/batch-test`가 500 — 임시 패치로 전체 루프 정상 동작 확인(재학습 후 예측 41283.94 → 41547.42).
- A에게 (해소): `LoadedModel`에 `run_id` 보관 요청 — "레지스트리는 v2인데 서버는 v1" 자동 판정과 D2 버전 표기로 이어지는 요청이었고 반영됐다.
- D1에게: `recent`는 20건·오래된 순·정수라 변환 없이 `/predict`의 `sequence`로 쓰면 된다. 단 CSV에 숫자로 못 읽는 칸이 있으면 해당 항목만 `null`로 내려가므로(500 회피) 전송 전 `null` 확인 필요.
- E에게: `/monitor/versions` 구현이 설계안(`model_name`, `production`, `serving_version`)보다 `model_source`, `tracking_uri`, `registry_error`, `serving_registry_version`, `stale`을 더 돌려준다 — API_SPEC 갱신 요청.
- E에게 (급함): `.gitignore`에 `*.db`·`*.sqlite` 추가 요청 — 트래킹 URI 오타 시 MLflow가 빈 SQLite를 새로 만들어 `git add -A`에 쓸려 들어간다(실제 876KB 커밋 후 제거).
- D2에게 (우선순위 낮아짐): 랜덤워크 "드리프트 배치 전송" 버튼은 37,000 중심 좌우 대칭이라 bias가 안 생겨 드리프트를 못 만든다(σ 3배도 무효, 실측 `ok`). 개선안으로 **계단식 수준 이동 생성기**(앞 20일 37,000, 뒤 21일 27,000, 하루 변동 5%)를 10 seed 실측과 함께 이관: 현재 랜덤워크 0/10 판정 vs 계단식 27,000 **10/10**(bias 평균 −1,167, 최약 seed −740). 근거 — 실제 일별 로그변화율 표준편차 7.12% vs 대시보드 1.2%/3.6%, 실제 수요는 평균회귀(자기상관 +0.663)인데 랜덤워크는 누적 표류 가정이라 정반대. 추세형(하루 −0.3%)도 LSTM이 쫓아가 bias −89에 그침 — 계단식이어야 창 안에 옛 수준이 남아 치우침이 생긴다. 고정 시드를 쓰면 1/1 결정론. 실제 CSV 배치가 드리프트가 되는 이유는 모델이 2025-08-31까지만 학습된 상태에서 9~10월 배치를 받기 때문(전체 학습 모델에 같은 CSV는 `structure_drift` bias +23).
- B에게 (시연 숫자 불일치): `check_and_trigger` 반환 `rmse`·`bias`는 이상치 **포함** 값인데 판정·로그는 **제외** 값(`rmse_excl_anomalies`)이라 README 표(3,169)와 API(5,924)가 다르다. 반환에 `rmse_excl_anomalies` 추가 또는 발표 대본에 "표 값은 `logs/aiops.log` 기준" 명시 필요.
- D2에게: `/monitor/versions`의 `stale` 표시 요청 — `true`는 재기동 신호, `null`은 판정 불가(경고 아님), `registry_error` 비면 조회 실패 표시. 필드명은 `serving_registry_version`.

### 2.3 모니터링·AIOps — B

**무엇을 구현했나.** 예측·실제 쌍을 쌓아 오차를 감시하고, 임계값을 넘으면 알림 → 최근 데이터 fine-tuning → 게이트 재검증 → 재배포까지 사람 없이 잇는다.

- `routers/predict.py`의 `batch_test()`(**TODO 3**): `arrivals` 41개 → 길이 20 슬라이딩 윈도우 21개 → 각 윈도우 예측(출발 여객은 요청에 `departures`가 있으면 같은 날짜 실제값, 없으면 `SIMULATED_DEPARTURES` 37,000 고정) → `recent_predictions`에 `{"predicted","actual"}` 누적(최근 21건 유지) → `check_and_trigger()` → `BatchTestResponse(predictions, drift_check)`.
- `drift_detector.py`: `compute_rmse()`(**TODO 2**), `compute_bias()`(평균 오차 = 실제 − 예측), `assess(window)` → `{rmse, bias, anomalies, rmse_excl_anomalies, bias_excl_anomalies, drift}`. **세 종류 판정**: 하루 오차 > `ANOMALY_THRESHOLD`(10,000)는 이상치로 빼고, 나머지 |bias| > `BIAS_THRESHOLD`(500)이면 수준 드리프트(재학습), bias 작은데 RMSE > `RMSE_THRESHOLD`(2,700)이면 구조 드리프트(알림만). RMSE ≥ |bias|가 항상 성립해 드리프트 조건에 RMSE 항은 없다.
- `retrain_trigger.py`의 `check_and_trigger()`(**TODO 4**): `assess()` → 이상치 `[WARN] anomaly` → 드리프트 아니면 `ok`/`anomaly`/`structure_drift` 반환 → 드리프트면 `[WARN] drift` → 이전 Production 버전 기억 → `latest_upload()` 최근 41행 → `fine_tune()` → 승격 시 `reset_cache()` + 판정 윈도우 `clear()` + `[OK]`. 게이트 실패는 그냥 실패(재시도 간격 없음). **롤백**: 승격 뒤 첫 판정이 드리프트이고 그 재학습이 게이트 실패면 `[ROLLBACK]` — 새 버전 Archived, 이전 버전 Production, `reset_cache()`. 승격 뒤 드리프트 아닌 윈도우가 한 번 나오면 승격 확정. 승격 기록은 프로세스 메모리(`_last_promotion`)에만 있다.
- `scripts/simulate_drift.py`의 `send_batch()`(**TODO 5**): 랜덤워크 41일(정상 σ 1.2% / 드리프트 σ 3.6%)을 `/predict/batch-test`로 전송.
- `batch-test`의 `departures`: 선택 항목(`feat/e-batch-departures`). 대시보드 CSV 배치는 실제 출발 여객을 함께 보내고, 랜덤워크·`simulate_drift.py`는 생략해 37,000 고정 유지.

확정값(2026-10-01, `exp/clean-cancellation` 실험 근거): 수준 드리프트 |bias| > **500**, 게이트 **2,700**, 이상치 10,000, fine-tune **3 epoch**, 재시도 간격 없음. 이전 확정값 1,500(실험 5)은 결항일 포함 학습 데이터 기준 — 보간본에서는 21일 bias가 ±1,000 안에서 움직여 1,500으로는 2025년에 3월 한 번만 걸리고 2,000 이상은 한 번도 안 걸린다. RMSE만으로 판정하지 않는 이유(실험 6): 보간 모델도 21일 RMSE > 2,700인 날이 107일이고, RMSE 트리거 10회 중 7회는 bias ±800 안쪽이며 재학습 3배에 RMSE 개선 0(3,039 vs 3,038). 구조 드리프트의 실체는 수준은 같은데 변동 폭이 커진 것(주간 리듬 ac(7) 0.17 → 0.67, 요일 진폭 1,756 → 5,320)이라 bias로 안 잡히고 41행 fine-tuning으로도 안 줄어든다(실험 2: 3,142 → 3,238).

남은 이슈: `_last_promotion`이 메모리에만 있어 서버 재시작 후 롤백 불가. 랜덤워크 σ(1.2%/3.6%)가 실변동성(7.3~8.4%)보다 낮아 미측정. 폭설 배치의 재학습 데이터는 "최신 업로드의 마지막 41행"이라 배치와 무관한 기간일 수 있다 — 시연 시 업로드 순서 준수 필요. 스켈레톤의 "RMSE > 2,700이면 재학습"에서 바뀐 판정 체계라 팀 결정 필요로 기록되어 있다.

**어떻게 확인했나.** 브랜치 `exp/drift-anomaly`에서 in-process 실험(스크래치 `exp4*.py`, `scan_ft.py`, `sim_policy.py`, `sim2.py`), 브랜치 `exp/clean-cancellation`에서 임시 워크트리 + `MODEL_SOURCE=mlflow uvicorn ... --port 8011` 실서버 요청(스크래치 `sim_clean.py`, `sim_rmse.py`, `sim_290.py`, `sim_500.py`), `feat/e-batch-departures`에서 `TestClient`로 실제 `/predict/batch-test` 왕복(배치 10개 × 2조건) + `python -m unittest discover -s tests` 8개 통과 + 임시 서버(포트 8010, `MODEL_SOURCE=local`) 실왕복 대조.

**결과(핵심 수치).** 보간 데이터 실서버 한 바퀴: 정상 `ok`(899/−241) → 드리프트 −1,041 재학습 게이트 1,338 **v2**(16.7초) → 확정 `structure_drift` → 반등 +686 게이트 902 **v3** → 롤백 배치 −1,008 게이트 실패 2,742 → `[ROLLBACK]` v2 복귀 → 폭설 `structure_drift`. 출발 여객 고정 vs 실제: 배치 10개 중 3개에서 판정이 바뀌고, 전체 CSV 21일 윈도우 995개 기준 10.3% 변동(bias 차이 평균 +13, 범위 −375~+634). 80/20 분할 RMSE: 실제 출발 2,347 / 고정 2,682 / 전일값 2,734. 전체 수치는 3장 §B.

**협업 조율.**
- A (선행 필요 → 해소): TODO 1 `_load_from_mlflow()` 이식, `reset_cache()` 추가 — `retrain_trigger`가 승격·롤백 뒤 호출.
- C (완료): `FINE_TUNE_EPOCHS` 10 → 3 (실험 6: 3/10/30 epoch → 게이트 실패 3/8/16회). 남은 것: 승격 시 이전 Production Archived 처리(`archive_existing_versions=True` 권장).
- E (선행 필요 → 해소): 결항일(도착 < 25,000) 보간 + CSV·시연 컷 재생성, CLAUDE.md 상수 표 갱신 — bias 500은 보간 데이터 기준(결항일 포함 데이터에서는 1,500이 맞았음).
- D2: `drift_check.status` 5종(`ok`/`anomaly`/`structure_drift`/`retrain_triggered`/`rolled_back`) 색 구분 요청 — PR #14로 반영됨.

### 2.4 대시보드 (프론트엔드) — D1·D2

**무엇을 구현했나.** 운영 담당자가 보는 화면 하나(`serving_app/static/index.html`). 위쪽은 "내일 도착 여객 예측 N명 · 등급"(D1), 아래쪽은 운영 체계가 돌아가는 것을 보여주는 배치 전송·재학습 로그·모델 버전(D2).

- [D1] 예측 카드(`#forecast-card`): 예측값(천 단위 구분, 명), 혼잡 등급 배지, 기준일(입력 마지막 날) → 예측일, `model_version`·`model_registry_version`, `/predict` 왕복 ms(`performance.now()`), 최근 20일 추이(라이브러리 없이 div 막대 20개, 마우스 오버 시 날짜·값), 새로고침 버튼. 상태 4종(업로드 전 / 로딩 / 성공 / 실패)이며 예측 실패가 다른 카드 초기화를 막지 않는다. 흐름: 페이지 로드·새로고침·업로드 성공 → `GET /data/status` → `exists=false`면 안내 → `recent` 20행을 그대로 정수 `sequence`로 `POST /predict` → 등급 판정(`CONGESTION_HIGH` 40,177 초과 혼잡 / `CONGESTION_LOW` 34,962 미만 여유 / 그 사이 보통 — `data/README.md` 사분위 Q3/Q1과 동일) → 카드 표시. 실패는 422(detail 요약)/기타 4xx·5xx(상태 코드 + 원문)/네트워크 오류로 구분. "내일"은 실제 내일이 아니라 입력 마지막 날 다음날이라 기준일·예측일을 함께 표시한다. `model_version`은 mlflow면 항상 `production`이라 버전 변화는 `model_registry_version`으로 보이므로 둘 다 표시.
- [D2] CSV 배치 전송: 브라우저에서 파일 검증(UTF-8 BOM, CRLF, 따옴표, 열 순서 변경 지원, 최소 41일, 날짜 연속성, 필수 열, 0 이상 정수) 후 `arrivals`와 `departures`(CSV `Departures` 열)를 `batch-test`로 전송. 랜덤워크 배치는 `arrivals`만 보내 서버 고정값 37,000 유지. 전송 중 버튼 비활성화·재진입 방지.
- [D2] 판정 표시: `ok`/`anomaly`/`structure_drift`/`retrain_triggered`/`rolled_back` 5종 구분, 알 수 없는 상태를 정상으로 표시하지 않음.
- [D2] 재학습 로그 카드: 파일 선택(기본 `aiops.log`), 표시 중 5초 주기 조회 + 수동 새로고침, `[WARN]`/`[ROLLBACK]` 주의·`[INFO]` 진행·`[OK]` 성공·`[ERROR]` 오류 색 구분, 로그 원문은 HTML로 해석하지 않음. 조회 8초 시간 초과, 실패 시 오래된 값을 현재 값으로 표시하지 않음.
- [D2] Production 버전 카드: `GET /monitor/versions`의 등록 버전·학습 검증 RMSE·생성 시각(Unix 밀리초·ISO 모두 한국 시간 변환)·서버 식별자 분리 표시, 성공한 조회 간 버전 변화 표시, `model_source=local`은 Registry 미조회로, `registry_error`는 조회 실패로, `stale=null`은 일치 미확인으로 표시.
- [D2] (D1 작성) 두 배치 나란히 비교: `#cmp-summary`(RMSE·bias 가로 막대 + 기준 점선 2,700/±500) + 왼쪽·오른쪽 두 칸(타일, 실제 vs 예측 곡선, 오차 막대 — 인라인 SVG만). 랜덤 정상 → 왼쪽, 랜덤 드리프트 → 오른쪽, CSV → 라디오로 선택. 칸 제목은 출처와 전송 시각, 두 칸의 y축 범위 동일. 결과는 페이지 메모리에만 있다(새로고침 시 빈 칸).
- [D2] (D1 작성) 라이트 테마(`:root` 변수, 글자 대비 text 16.2:1·muted 6.6:1·faint 4.8:1, 차트 coral 3.5:1·indigo 4.9:1·teal 4.6:1), 파이프라인 "드리프트 감지" 설명을 `RMSE vs 임계치` → `bias vs ±500`으로 수정, `BIAS_THRESHOLD = 500` 복제 상수 추가.

**어떻게 확인했나.** curl 실호출(`/predict` 20개 200 / 19개 422), node `vm` + DOM 스텁으로 실제 서버 대상 `loadForecast()`·`sendBatch()`·`parseBatchCsv()` 실행, Python 독립 계산과 수치 대조, `node --check`, `python -m compileall`, 헤드리스 Chrome 화면 확인, 브라우저 실측(카드 ms, v1→v3 전환), Node 임시 점검 스크립트(대체 응답)로 CSV 23개·버전/로그 30개·화면 53개 점검.

**결과(핵심 수치).** 브라우저 카드 v1 39,981명 보통 44 ms → v3 40,501명 혼잡 187 ms(새 모델 lazy 로드 포함). `/predict` 왕복 local 1.8297초 → 0.0153초, mlflow 2.5564초 → 0.0161초. 등급 경계 34,961 여유 / 34,962 보통 / 40,177 보통 / 40,178 혼잡. 비교 차트 수치 35개 항목이 서버 판정·독립 계산과 일치(RMSE·bias 소수 10자리). 정상 CSV 실배치 RMSE 898.53·bias −241.15 `ok`, 확정 CSV `structure_drift`(전체 5,924, 제외 3,169/+199). 재학습 전후 같은 입력 31131.23(v1) → 30622.54(v2). 전체 수치는 3장 §D.

**협업 조율.**
- C: 최신 main의 `registry_error`·`serving_registry_version`·`stale` 비교 구현 확인 — 화면에서 조회 실패·모델 없음·서빙 번호 구분.
- B (완료): `BatchTestRequest`에 선택 `departures` — CSV 배치가 함께 전송.
- A/C: 최신 main 공통 테스트 6개 중 2개 실패(`serving_run_id` KeyError, run_id 없는 경우 `stale=None` 기대 vs 실제 False) — 팀 API 계약과 테스트 정합성 확인 요청. D2 범위 밖 코드는 수정하지 않음.
- E: `CLAUDE.md`·`PROJECT_PLAN.md`에 "main은 10 epoch/팀 합의 전"이 남아 있으나 현재 코드는 3 epoch이고 보간 데이터도 병합됨, `API_SPEC.md`에 완료 API가 여전히 TODO/설계안 — 문서 충돌 정리 요청. 파이프라인 공통 상수의 `RMSE vs 임계치` 문구·`MODEL_SOURCE=mlflow 기준` 문구 정리 요청. D1 현황의 recent 미구현 표기도 최신 C 코드와 다름(이번 0번 표 갱신으로 해소).
- [D1] A: 모델 파일 없을 때 `/predict`가 사유 없는 텍스트 500 — 사유 담긴 JSON 오류(예: 503 + detail) 검토 요청.
- [D1] D2·B: 파이프라인 "드리프트 감지" 문구를 `bias vs ±500`으로 수정 — B는 표기만 확인 요청.
- [D2] (D1 작성) B: `check_and_trigger()`에 동시 실행 방지가 없어 `/predict/batch-test`가 겹치면 같은 데이터로 두 번 재학습·승격된다(트러블슈팅 v6·v7). 화면은 #14로 막았지만 `simulate_drift.py`·curl은 겹칠 수 있다 — 재학습 구간 잠금(`threading.Lock` 등) 검토 요청.
- [D2] (D1 작성) E: `BIAS_THRESHOLD = 500` 복제 상수를 CLAUDE.md "대시보드 상수 복제" 행에 반영 요청.

### 2.5 데이터·상수·통합 — E

**무엇을 구현했나.** 부품이 하나의 파이프라인으로 이어지는지, 상수가 한 곳에서만 바뀌는지 본다. 전체 흐름: ① 한국공항공사 일별 통계 → `scripts/prepare_jeju_data.py` → `data/jeju_airport_arrivals.csv` → 대시보드 업로드(C) → ② baseline → MLflow 학습 → 게이트 2,700명 → Production(C) → ③ `MODEL_SOURCE=mlflow` → `/predict`(A) → 예측·혼잡 카드(D1) → ④ 배치 전송(D2) → `batch-test` → 판정 → `[WARN]` → `fine_tune` → 게이트 → 승격 `[OK]`(B·C) → ⑤ 캐시 비움(A) → 새 버전으로 `/predict` → 버전 표기(D2). 경계: 학습 코드와 서빙 코드는 MLflow Registry(`models:/Airport_Arrivals_Predictor/Production`)로만 만나고, 모니터링은 `recent_predictions`(프로세스 메모리)와 `logs/aiops.log`(파일)로 상태를 남긴다. 서버 재시작이면 둘 다 초기화된다(로그 파일은 남음).

- 학습 데이터: 제주공항 일별 여객 2023-01-01~2025-10-31, 1,035일. **도착 25,000명 미만 18일(2023-01-24 결측 포함)을 양옆 가장 가까운 정상일 평균으로 보간**(`CANCEL_ARRIVALS = 25,000`). 결항일이 스케일러를 압축하고 MSE를 지배해 모델이 20일 이동평균으로 퇴화하던 것이 풀린다(평상시 RMSE 2,415 → 1,807, 예측 표준편차 445 → 1,898). 시도했다 버린 것: 요일·공휴일·연휴 피처(2,344 → 2,695로 악화), 학습 기간 축소, 로그 변환 등 데이터 변형 A~G 중 보간(B)만 효과.
- 시연 데이터: 이상치 시연 `data/jeju_drift_batch_41rows.csv`(폭설 결항일 17,093·6,088명 포함, raw), 드리프트 시연 컷 `data/jeju_demo_drift_*.csv` 9개(학습 ≤2024-06-30, 업로드 컷 3개, 배치 5종 — 배치는 raw, 학습 컷은 보간본), 수준 이동 컷 등. 순서와 확인값은 `data/README.md`.
- `batch-test`에 `departures` 선택 전달 구현(CSV 배치는 실제 출발 여객, 랜덤워크는 37,000 고정 유지).

**상수 표** (한 곳에서만 바꾸는 값 — 바꾸면 CLAUDE.md 표, `index.html` 복제 상수, ③ 운영 설계를 같은 PR에서):

| 항목 | 값 | 위치 |
|---|---|---|
| 배포 게이트 `RMSE_GATE` | 2,700명 | `serving_app/train_and_register.py`, `scripts/train_baseline_v1.py` |
| 드리프트 임계값 `RMSE_THRESHOLD` | 2,700명 (구조 드리프트 알림 전용) | `serving_app/monitoring/drift_detector.py` |
| 판정 윈도우 `WINDOW_SIZE` | 21건 | `serving_app/monitoring/drift_detector.py` |
| 드리프트 bias 기준 `BIAS_THRESHOLD` | 500명 (`exp/drift-anomaly`에선 1,500) | `serving_app/monitoring/drift_detector.py` |
| 이상치 기준 `ANOMALY_THRESHOLD` | 하루 오차 10,000명 | `serving_app/monitoring/drift_detector.py` |
| fine-tune epoch `FINE_TUNE_EPOCHS` | 3 | `serving_app/train_and_register.py` |
| 입력 시퀀스 길이 `SEQ_LEN` | 20일 | `data/features.py` |
| 결항일 보간 기준 `CANCEL_ARRIVALS` | 25,000명 | `scripts/prepare_jeju_data.py` |
| 시뮬레이션 고정 출발 여객 `SIMULATED_DEPARTURES` | 37,000명 (`departures` 없을 때만) | `serving_app/routers/predict.py` |
| MLflow 모델 이름 | `Airport_Arrivals_Predictor` | `serving_app/model_loader.py`, `serving_app/train_and_register.py` |

기초 통계(보간본 기준, 괄호는 raw): 평균/중앙값 37,425 / 37,822명(37,218 / 37,820), 표준편차 3,809명(4,413), 최소/최대 25,168 / 46,954명(raw 최소 1,042), 사분위 Q1/Q3 35,049 / 40,177명, 일별 변화율 표준편차 7.3%(raw 19.6%, 결항 제외 8.4%), 전일값 복사 RMSE 2,523명(raw 3,600). 출처: `data/README.md`.

**어떻게 확인했나.** `python scripts/prepare_jeju_data.py`(18일 보간, 1,035행 재생성). 통합 데모는 임시 워크트리에서 `MODEL_SOURCE=mlflow uvicorn serving_app.main:app --port 8011` → `/data/upload` → baseline → `train_and_register.py` → `/predict`·`/predict/batch-test` 순서 호출(스크래치 `e2e_clean.py`). 사전 실험은 임시 복사본(본 레포 코드 아님)에서 1회.

**팀 결정 대기.** 보간 규칙·bias 500·fine-tune 3 epoch의 `main` 반영(코드는 이미 main에 병합됨 — 문서상 합의 기록 필요), `batch-test`에 `departures` 전달 여부(`docs/proposal/03_operations_design.md` 4번). 결정되면 원자료에 날짜와 결론을 적는다.

## 3. 측정 결과 모음

원자료의 모든 측정값. 값·단위는 기록 그대로이며, 측정 시각이 없는 행은 비워 두었다(변경 기록의 시각은 4장). "미측정"도 그대로 남겼다.

### §A 서빙 API (작성자 윤동현/A — macOS arm64, Python 3.12.13, TensorFlow 2.21.0, 포트 8000, 1회 측정)

| 지표 | 값 | 측정 조건 |
|---|---|---|
| local/lazy 시작 / 첫 요청 / 두 번째 요청 (초) | 0.2241142499842681 / 2.0358974580012728 / 0.01348475000122562 | 1회 측정 |
| local/lazy `/predict` | `{"predicted_arrivals": 42087.61, "model_version": "v1-local", "model_registry_version": null}` | 원자료 마지막 20행, HTTP 200 |
| mlflow/lazy 시작 / 첫 요청 / 두 번째 요청 (초) | 0.19868695898912847 / 3.511652999994112 / 0.016238417010754347 | 1회 측정 |
| mlflow/lazy `/predict` | `{"predicted_arrivals": 39772.69, "model_version": "production", "model_registry_version": "1"}` | 원자료 마지막 20행, HTTP 200 |
| mlflow/eager 시작 / 첫 요청 / 두 번째 요청 (초) | 2.153921208024258 / 0.09605337501852773 / 0.014345583011163399 | 1회 측정 |
| mlflow/eager `/predict` | `{"predicted_arrivals": 39772.69, "model_version": "production", "model_registry_version": "1"}` | 원자료 마지막 20행, HTTP 200 |
| 입력 오류 | 19행·21행·음수·departures 누락 모두 422 (4모드 × 4종) | local/lazy, mlflow/lazy, mlflow/eager, 별도 registry 모드 |
| 캐시 전환 | 승격 직후 번호 1 → reset 후 2 → 롤백+reset 후 1 | 복제 SQLite DB, 같은 v1 artifact로 v2 등록. 실제 재학습/성능 개선 검증 아님 |
| 캐시·로더 테스트 | 8개 통과 | 버전 2/11 숫자 비교, 고정 scaler, Production 없음, 실패 후 재시도, 동시 첫 요청 1회 로드, 로딩 중 reset, eager 캐시, 반복 reset |
| A→C 모델 식별·B 판정 창 회귀 | 신규 6개 + 기존 8개 통과 | `python -m unittest discover -s tests -v`, `logs/verify_a_unit.py` |
| 실제 재학습/조회 통합 | v1→v2→v3→v2 롤백, 단계별 `stale=false`; 의도적 Registry/캐시 불일치 `true` → 복구 `false`. 같은 20일 입력 예측 33,339.53 → 32,677.49 → 32,114.58 → 32,677.49. 게이트 RMSE 1,961.28 → 1,338.33 → 902.29 → 2,741.53(실패) | 별도 작업 폴더·DB, 실제 HTTP 8000 |
| `/health` | lazy 초기 false → 예측 후 true, eager 초기 true. 모두 `status: ok` | |

### §C 학습·배포·버전 조회 (작성자 유경모/C — PC 1대, 별도 표기 없으면 1회)

| 지표 | 값 | 측정 조건 |
|---|---|---|
| Day1 baseline RMSE | 2,571명 | `python scripts/train_baseline_v1.py` |
| Day2 MLflow 학습 RMSE | 2,267명 (2266.94) | `python serving_app/train_and_register.py` |
| 다른 PC에서의 게이트 통과 여부 | 미측정 | |
| 학습 소요 시간 (100 epoch) | 미측정 | |
| 컨테이너 빌드 시간 / 기동 성공 여부 | (아래 개별 행 참고 — 초기 기록은 미측정) | |
| `GET /monitor/versions` 응답 예시 | `{"model_name":"Airport_Arrivals_Predictor","model_source":"mlflow","tracking_uri":null,"production":{"version":2,"run_id":"51d68db9...","rmse":2363.4960823793504,"created_at":1790835940274},"registry_error":null,"serving_version":"production","serving_registry_version":2,"stale":false}` | 컨테이너에서 `curl localhost:8000/monitor/versions`, A·E 머지본 |
| `GET /data/status`의 `recent` | 20건, 2025-10-12 ~ 2025-10-31, 오래된 순, arrivals/departures 모두 int | `curl localhost:8000/data/status` |
| `recent`를 그대로 `/predict`에 전달 | HTTP 200, `{"predicted_arrivals":39772.69,"model_version":"production"}` | `recent`를 `sequence`로 변환해 POST |
| B 머지 뒤 C API 동작 | `/monitor/versions`, `recent`, `recent → /predict` 모두 정상 | `origin/main`(B 포함) 위 리베이스, `MODEL_SOURCE=mlflow` 포트 8000 |
| fine-tune 3 epoch 전체 루프 | v1(rmse 2267) → 드리프트 배치 `bias=+2015` 감지 → 재학습 `[OK] new_rmse=1840` v2 승격. 예측 39772.69 → 41283.94, `/monitor/versions`도 v2 추종 | mlruns 초기화 후 1회, `reset_cache()` 임시 삽입(커밋 안 함) |
| 배치별 판정 (윈도우 격리, 배치마다 서버 재기동) | normal `ok` rmse 2280 bias 315 / falsealarm `retrain_triggered` / drift `retrain_triggered` | 각 배치 전 서버 재기동으로 `recent_predictions` 비움 |
| 윈도우를 안 비웠을 때 | 같은 세 배치 연속 주입 시 normal이 `retrain_triggered`, drift가 `ok`로 뒤집힘 | 앞 배치 예측이 윈도우에 남아 섞임 (가이드 부록1의 7번) |
| E의 수준 이동 시연(PR #17) main 재현 | ① v1 rmse 2363.50 ② 정상 배치 rmse 1409.99 bias +76.14 `ok` ③ 912행 업로드 ④ 드리프트 배치 → 게이트 1892.76 v2 ⑤ 재전송 → 게이트 1423.24 v3 ⑥ 재전송 rmse 1619.16 bias +239.10 `ok` — README 값(2,363 / 1,410 / +76 / 1,893 / 1,423 / 1,619 / +239)과 전부 일치, 소수점까지 재현 | `main` 위 컨테이너 `up --build`, 배치마다 재기동 없이 연속. 시연 공식 경로 |
| 컨테이너 전체 시연 실측 (기획서 ⑥ 증빙) | 학습 974행(≤2025-08-31) 컨테이너 재학습 `[GATE PASSED] rmse=2094` v2 → 승격 직후 `stale: true`(레지스트리 v2/서버 v1) → 재기동 후 false → 정상 배치 `ok` rmse 2183.06 bias −116.79 → 전체 데이터 업로드 → 드리프트 배치 `retrain_triggered` promoted true rmse 2098.47 v3. `data/README.md` 첫 표(2,183 / −117 / 3,490 / −587 / 2,098)와 정수까지 일치. aiops.log: `[WARN] rmse=3490 bias=-587` → `[INFO] retrain triggered` → `[OK] new_rmse=2098 v3`, 감지→승격 3.7초 | `--force-recreate` 깨끗한 컨테이너 |
| 이미지 빌드 (`--no-cache`) | 8단계, 561줄. pip 35.5초 + 빌드 중 학습·등록 32.6초(`[GATE PASSED] rmse=2363 -> v1`) + export 38.2초. 최종 3.26GB | `docker compose build --no-cache` |
| `stale`이 true가 되는 조건 (정정) | 자동 재학습 흐름에서는 true가 뜨지 않는다 — `retrain_trigger`가 승격·롤백 직후 `reset_cache()`를 불러(45행·97행) 캐시가 비면 null → 다음 예측에서 false. 시연 9단계 전 구간에서 false 아니면 null뿐 | 승격이 서버 밖에서 일어날 때만 true (MLflow UI 수동 승격, 별도 배치, 다중 인스턴스) — 컨테이너 안 별도 `train_and_register.py` 실행으로 재현. 운영에서 캐시가 썩는 경로 |
| `stale` 판정, A의 `run_id`(PR #9) 기준 | `stale_basis="run_id"`로 3단계 동작: ① v1 run `d97b47e5` / 서버 동일 / false ② v2 run `f2b9ea8b` 승격, 서버 run `d97b47e5` → true ③ 재기동 → 둘 다 `f2b9ea8b` / false | A·B·C·E 머지 `main`(`56bf712`) 컨테이너, 승격 로그 `[GATE PASSED] rmse=2363 -> v2 promoted` |
| `stale` 판정 실동작 (부록1 6번 재현) | ① 기동 직후 v1/1/false ② 컨테이너 안 재학습 v2 승격·서버 유지 → v2/1/true, `/predict` `model_registry_version: "1"` ③ `docker compose restart` → v2/2/false, `/predict` `"2"` | `docker compose exec serving-app python serving_app/train_and_register.py` → `[GATE PASSED] rmse=2363 -> v2 promoted` |
| 예측값만으로는 stale을 못 본다 | 위 세 단계 `predicted_arrivals` 전부 39980.96 동일 (seed 42 + 같은 데이터라 v1·v2 가중치 동일) — `model_registry_version`·`stale` 필드가 있어야 어긋남이 보임 | 같은 20일 `sequence` 세 번 호출 |
| 컨테이너 기동 (A 머지본, 기본값) | 정상. `MODEL_SOURCE=mlflow` + eager로 바로 뜸. `/monitor/versions` Production v1 rmse 2363.50 | `docker compose up --build -d`. rmse 2266.94 → 2363.50 상승은 E의 결항일 보간 데이터(PR #7) 반영 때문 |
| 컨테이너 `/data/status` (E 보간본) | 1,035행, `min_arrivals` 1042 → 25168 | 보간 전 1,042 |
| 컨테이너 기동 (A TODO 1 이전, 기본값 mlflow) | 기동 실패 — `_load_from_mlflow()` `NotImplementedError` → `Application startup failed. Exiting.` | A TODO 1 병합 전 기록 |
| 컨테이너 기동 (`MODEL_SOURCE=local`) | 정상. `/health` `{"status":"ok","model_loaded":true,"loading_mode":"eager"}` | compose 환경변수 통로 추가 후 |
| 컨테이너 `/monitor/versions` (local) | `model_source: "local"`, `serving_version: "v1-local"`, `production: null`, `stale: false` | 로컬 모델은 비교 대상 없어 false |
| 컨테이너 `/monitor/versions` (mlflow, 임시 패치) | `production: {version: 1, rmse: 2266.940748540207}` | `_load_from_mlflow` 임시 구현 빌드 (커밋 안 함) |
| 컨테이너 `/data/status` | 1,035행, 2023-01-01 ~ 2025-10-31, `recent` 20건 (마지막 2025-10-31 arrivals 42323 / departures 42950) | 빌드 중 시드된 `build_seed.csv` |
| 컨테이너 `recent → /predict` 왕복 | HTTP 200, `{"predicted_arrivals":42096.13,"model_version":"v1-local"}` | 로컬 모델 (MLflow 경로 39772.69와 다른 가중치) |
| `/monitor/versions` 응답 시간 (트래킹 서버 무응답) | MLflow 기본값(120초+재시도 7회) 848.03초(14분 8초) → 재시도 1회 10.68초 → 재시도 없음 5.70초. 연결 거부 0.54초 | `http://10.255.255.1:9999` in-process 호출 각 1회. 조회 하나가 sync 핸들러의 스레드풀 슬롯을 물고 있음 |
| 오타난 sqlite 트래킹 URI | `production: null`, `registry_error: null` — 조회가 "성공"하며 0건 (MLflow가 그 이름으로 빈 DB 생성). `tracking_uri` 값으로만 구분 | `MLFLOW_TRACKING_URI=sqlite:////tmp/oops_typo_check.db` |
| Day1 baseline RMSE (PC 2대째) | 2,945명 | 다른 PC에서는 2,571명과 2,191명 |
| Day2 MLflow 학습 RMSE (PC 2대째) | 2,267명 (2266.94) | seed 42 고정이라 다른 PC와 같은 값 |

### §B-1 모니터링 실험 (브랜치 `exp/drift-anomaly`, PC 1대, 1회, in-process, 출발 여객 고정 37,000. bias = 실제 − 예측)

| 지표 | 값 | 측정 조건 |
|---|---|---|
| v1 학습 (~2024-06-30, 547행) 게이트 RMSE | 2,353 (baseline 2,365~2,922, 실행마다 다름) | `train_baseline_v1.py`, `train_and_register.py` |
| 정상 배치 2024-04-15~05-25 (v1) | RMSE 2,374 / bias +252 → `ok` | |
| 드리프트 배치 2025-02-20~04-01 (v1) | RMSE 4,321 / bias −3,056 → fine-tuning 게이트 2,491 → v2 승격 (14초) | 업로드 ~2025-04-01 |
| 드리프트 배치 (v2) | RMSE 3,073 / bias −309 | 재학습이 치우침 제거 |
| 다음 윈도우 2025-04-02~05-12 (v2) | RMSE 3,918 / bias +2,394 → 재학습 게이트 실패 3,769 → 미확정 승격이므로 롤백 → v1. v1로는 3,126 / −396 | 재학습 모델 자체는 이후 3개월 2,023 / 2,773 / 2,366으로 좋았으나 5일 검증(05-08~12) 3,769. 같은 5일에서 v1 3,732, v2 3,149 — 5일 표본은 우열을 못 가림 |
| 2025-05-13~06-22, 07-22~08-31 (v1) | 2,009 / −476, 2,383 / +715 → `ok` | |
| 오탐 배치 2025-09-01~10-11 (v1) | 이상치 1일(10-11, −11,306) + 나머지 4,381 / +1,318 → `structure_drift` | 재학습 안 함 |
| 폭설 배치 (v1) | 이상치 3일(02-04 −12,587, 02-05 −10,940, 02-07 −29,616) + 나머지 4,323 / −3,636 → fine-tuning 게이트 실패 3,865 → v1 유지 | 재학습 데이터 = 최신 업로드(~05-12) 마지막 41행 |
| 롤백 없이 v2에서 재재학습 | 게이트 실패 3,769 → v2에 갇힘 (이후 전 구간 bias +2,300~3,500) | |
| 재학습 윈도우 41/62/90/120/180행 (v1) | after 배치 bias +2,394 / +3,432 / +4,633 / +4,506 / +2,365 — 길어도 과적응 | |
| 실험 5 재학습 정책 시뮬레이션 (2025-01-21~08-31 하루씩 전진, 실제 fine-tune, 승격 시 윈도우 초기화, 결항 포함 RMSE) | v1 고정 4,201 / bias>2,000 4,100(재학습 13, 승격 4) / bias>1,500 3,942(9, 6) / bias>1,000 3,943(21, 8) / bias>500 3,993(28, 9) / bias>500·게이트 3,500 4,032(20, 9) / bias>500·게이트 없음 4,166(10, 10) / 21일마다 무조건 3,891(11, 4) / 20일 이동평균 3,651 / 전일값 3,462 | 스크래치 `sim_policy.py`, `sim2.py`. 1,500이 재학습 최소·RMSE 최저, 게이트 없으면 v1 고정 수준으로 악화 |
| 모델 특성 (2025-05~08, 결항 없음) | LSTM 2,398 vs 전일값 2,083 / 7일 MA 2,380 / 상수(학습평균) 2,446 / 20일 MA 2,543. 예측 std 460 vs 실제 2,444. 20일 MA 상관 0.927. 전일 대비 방향 적중 67% (MA 63%, 동전 50%) | 변화량 자기상관 lag1 −0.18(평균 회귀), lag7 +0.36(주간). 수준+약한 주기만 배움, 진폭은 MSE가 줄임 |
| `simulate_drift.py` 랜덤워크 배치 | 미측정 | |

### §B-2 보간 데이터 실측 (브랜치 `exp/clean-cancellation`, 임시 워크트리, `MODEL_SOURCE=mlflow uvicorn ... --port 8011` 실서버, PC 1대, 1회. 배치 입력 raw, 레지스트리 번호는 상대 표기)

| 지표 | 값 | 측정 조건 |
|---|---|---|
| v1 학습 (~2024-06-30, 547행) | baseline 1,999 / MLflow 게이트 1,961 → v1 | 보간 전 2,353 |
| `/predict` (06-11~06-30 입력) | 33,339.53 · 첫 호출 8.8초(lazy) · 재호출 2.1초 | |
| 정상 배치 2024-10-01~11-10 (v1, 학습 밖) | RMSE 899 / bias −241 → `ok` | 보간 전 정상 배치(2024-04-15~05-25)는 보간 모델로 이상치 1일 `anomaly`라 교체 |
| 드리프트 배치 2024-12-12~2025-01-21 (v1) | 이상치 1일(01-09 결항 17,093 vs 29,523) 제외 RMSE 2,656 / bias −1,041 → 재학습 게이트 1,338 → v2 (16.7초) | 업로드 ~2025-01-21 |
| `/predict` 재학습 후 (01-02~01-21 입력) | 30,622.54 (v2) | |
| 확정 배치 2025-01-02~02-11 (v2) | 이상치 1일(02-07) 제외 RMSE 3,169 / bias +199 → `structure_drift` → v2 승격 확정 | |
| 반등 배치 2025-01-16~02-25 (v2) | 이상치 1일(02-07) 제외 RMSE 2,456 / bias +686 → 재학습 게이트 902 → v3 (14.7초) | 업로드 ~2025-02-25 |
| 롤백 배치 2025-02-06~03-18 (v3) | RMSE 3,474 / bias −1,008 → 재학습 게이트 실패 2,742 → `[ROLLBACK]` v3 Archived, v2 Production (14.7초) | 업로드 ~2025-03-18. 2,742 vs 2,700 간발 — 다른 PC 재확인 필요 |
| `/predict` 롤백 후 (02-27~03-18 입력) | 31,309.10 (v2) | |
| 폭설 배치 2025-01-09~02-18 (v2) | 이상치 1일(02-07 6,088 vs 29,242) 제외 RMSE 3,039 / bias −103 → `structure_drift` | |
| 시연 순서 (≤2025-08-31 학습) | v1 게이트 2,094 · 정상 07-22~08-31 2,183 / −117 `ok` · 9~10월 배치 3,490 / −587 → 재학습 2,098 → v2 · 폭설 이상치 1일 제외 2,981 / +214 `structure_drift` | `data/README.md` "시연 순서". 보간 전엔 9~10월 배치가 오탐 사례 |
| 실험 6 보간 모델 트리거 시뮬레이션 (2025-01-01~10-31 하루씩 전진, raw 입력, 출발 37,000, 실제 fine-tune·게이트·롤백, 결항 포함 RMSE) | 재학습 없음 3,038 · bias>1,500/ep10 3,058(3, 2) · bias>2,000·2,500·3,000 재학습 0 · RMSE>2,700 3,039(10, 6) · RMSE>3,000 3,069(6, 6) · bias>290 ep3 3,010(16, 12) / ep10 3,031(21, 13) / ep30 3,092(30, 12) · bias>400 ep3 3,011(13, 11) · bias>500 ep3 3,017(10, 9, 롤백 1) | 스크래치 `sim_clean.py`, `sim_rmse.py`, `sim_290.py`, `sim_500.py`. v1 고정 21일 bias 최대 −1,989(3월). 어느 설정도 9~10월 구조 드리프트는 못 줄여 전체 RMSE 3,000대 유지 |

### §B-3 출발 여객 고정 vs 실제 (2026-10-01, 브랜치 `feat/e-batch-departures`, Mac 1대, 1회, `TestClient` 실왕복, 로컬 baseline 모델 `airport_v1.keras`, 재학습 없이 `assess()` 판정만. 값 = 이상치 제외 RMSE / bias / 이상치 일수 / 드리프트 여부)

| 배치 (`data/`) | `departures` 생략 (37,000 고정) | `departures` 전송 (실제) |
|---|---|---|
| `jeju_demo_batch_normal_41rows.csv` | 2,248 / +605 / 0일 / 드리프트 | 1,892 / +812 / 0일 / 드리프트 |
| `jeju_demo_batch_sep_oct_41rows.csv` | 3,608 / +323 / 0일 / 아님 | 3,195 / +604 / 0일 / **드리프트** |
| `jeju_demo_drift_batch_41rows.csv` | 1,991 / −438 / 2일 / 아님 | 2,749 / −66 / 1일 / 아님 |
| `jeju_demo_drift_batch_after_41rows.csv` | 1,824 / +708 / 2일 / 드리프트 | 1,735 / +413 / 2일 / **아님** |
| `jeju_demo_drift_batch_confirm_41rows.csv` | 2,409 / +530 / 2일 / 드리프트 | 2,189 / +30 / 2일 / **아님** |
| `jeju_demo_drift_batch_normal_41rows.csv` | 1,033 / +446 / 0일 / 아님 | 1,031 / +451 / 0일 / 아님 |
| `jeju_demo_drift_batch_rollback_41rows.csv` | 3,417 / −792 / 0일 / 드리프트 | 2,897 / −1,028 / 0일 / 드리프트 |
| `jeju_demo_shift_batch_drift_41rows.csv` | 2,515 / +2,034 / 0일 / 드리프트 | 2,425 / +1,919 / 0일 / 드리프트 |
| `jeju_demo_shift_batch_normal_41rows.csv` | 1,401 / +163 / 0일 / 아님 | 1,156 / +116 / 0일 / 아님 |
| `jeju_drift_batch_41rows.csv` | 2,101 / −228 / 2일 / 아님 | 2,235 / +28 / 2일 / 아님 |

- 10개 중 3개에서 드리프트 판정이 바뀐다. 같은 모델로 전체 CSV 21일 윈도우 995개를 재면 10.3%에서 바뀐다(bias 차이 평균 +13, 범위 −375 ~ +634).
- 게이트와 같은 80/20 분할(test 2025-04-12~10-31, 203일) RMSE: 실제 출발 2,347 / 37,000 고정 2,682 / 전일값 복사 2,734.
- 길이가 다른 `departures`(41 vs 40)는 422. 시연 순서의 Production 모델(v1 → v2 → …) 기준 값은 미측정.

### §D 대시보드 (D1 youjin09222, D2 — PC 1대)

| 지표 | 값 | 측정 조건 |
|---|---|---|
| [D1] 페이지 로드 → 예측 표시 시간 (카드 `performance.now()` ms) | v1 카드 44 ms / v3 전환 후 첫 새로고침 187 ms (새 모델 lazy 로드 포함) / 이후 새로고침 2회 미측정 | 브라우저 카드 표시값을 화면에서 읽음. `MODEL_SOURCE=mlflow`, main `8f22c4e`. 카드 ms는 `/predict` 왕복(`/data/status` 제외) |
| [D1] 표시된 예측값 / 등급 (`recent` 2025-10-12~10-31) | local 41304.39 → 혼잡 / mlflow 39980.96 → 보통 (카드 39,981명·보통). 브라우저 카드 v1 39,981명 보통 → v3 40,501명 혼잡 | curl POST + node `vm`+DOM 스텁+실서버 `loadForecast()`. main `8f22c4e` 위 rebase, 업로드·모델·`mlflow.db`·`mlruns`·`logs` 초기화 후 보간 CSV 1035행 업로드 → baseline RMSE 2308 → MLflow RMSE 2363 v1. macOS, Python 3.12.13, uvicorn 단일 프로세스, lazy |
| [D1] `/predict` 왕복 (local, curl `time_total`) | 1.829673s (lazy 로드 포함) / 0.015298s / 0.014600s | 기동 후 학습 → 첫 요청부터 3회 연속, 같은 20행 |
| [D1] `/predict` 왕복 (mlflow) | 2.556365s (lazy MLflow 로드 포함) / 0.016117s / 0.015749s | mlflow로 재시작 직후 3회 연속, 로컬 sqlite, v1 |
| [D1] 등급 경계값 (`congestionLevel()`) | 34961 → 여유 / 34962 → 보통 / 40177 → 보통 / 40178 → 혼잡 | index.html `<script>` 추출 후 node `vm` 실행 |
| [D2] (D1 작성) 비교 차트 수치 일치 | 35개 항목 일치: 정상 배치 클라이언트 RMSE·bias = 서버 `drift_check` 533.5709725306 / −281.9296839497 (소수 10자리), 드리프트 배치 2,137 / +1,252 = `aiops.log` 판정값, 타일·공유 축·곡선 21점·오차 막대 21개·요약 막대 좌표 | `sendBatch()` node 실행(DOM 스텁), 실서버 랜덤 배치, Python 독립 계산 대조. main `56bf712`, Production v3 시작 |
| [D2] (D1 작성) 비교 칸 배정 (#14 위) | normal CSV → 왼쪽(1764 / −1025), drift CSV → 오른쪽(1887 / +1061), 랜덤 정상 → 왼쪽 교체·오른쪽 유지 | main `1507bff` + D1, node 실행, 실서버. 로컬 Registry v10 시작이라 세 배치 모두 드리프트 → v11·v12·v13 승격 — E 시연 순서(v1 시작)와 다른 이유는 시작 모델 차이 |
| [D2] 브라우저 CSV 미리보기 | 2025-01-09 ~ 2025-02-18, 41일, 예측 비교 21일 | `jeju_drift_batch_41rows.csv` 실제 선택, 전송 버튼 활성화. 전송은 안 함 |
| [D2] 정상 CSV 실제 배치 | RMSE 898.5326646816719명, bias −241.1471457935515명, `ok`, 예측 21개, HTTP 200, 0.3319초 | `jeju_demo_drift_batch_normal_41rows.csv`, 모델 v1 |
| [D2] CSV 버튼 실제 배치 | 전체 RMSE 5924.090023249845명, `structure_drift`, 이상치 1일; 로그의 이상치 제외 RMSE 3169명 / bias +199명 | `jeju_demo_drift_batch_confirm_41rows.csv`, 모델 v2, 브라우저 선택→전송. 전체 오차와 판정용 제외 오차 구분 |
| [D2] 입력 검증·전송·상태·오류 처리 점검 | 23개 통과 | Node 임시 스크립트(`/tmp/aiops-d2-check.cjs`), 대체 응답 — 실제 품질 수치 아님 |
| [D2] 통합 전 서버 상태 / compileall | `/health` 200, `model_loaded=false`, lazy; compileall 종료 0 | 기존 서버 8000 |
| [D2] 로그 카드 `[WARN]`→`[OK]` 반영 시간 | 미측정 | |
| [D2] 재학습·버전 전환 | v1→v2, 학습 검증 RMSE 1338.3271859414438명 게이트 통과; UI 변화·로그 확인 | 15:21:08~15:21:13 배치 입력 출처 미확인 — 고정 CSV 재현 결과로 간주 안 함. 실제 롤백 미측정 |
| [D2] 재학습 전후 실제 예측 | 같은 입력 31131.23명/번호 1 → 30622.54명/번호 2, HTTP 200 | 응답 시간 0.0186초 / 0.1479초. 반영 확인이며 품질 개선 근거 아님 |
| [D2] 최초 학습·첫 예측 | baseline 2163명(정수 반올림) / MLflow 1961.2788973713987명(v1) / 첫 `/predict` 33339.53명, 1.8543초 | 547일 보간 학습 자료 `jeju_demo_drift_train_until_20240630.csv` |
| [D2] 입력 오류 거부 | 19일 입력 HTTP 422 | 실제 A API |
| [D2] 버전·로그·배치 후 갱신 점검 | 22개 통과, 기존 CSV 23개 재점검 통과 | `/tmp/aiops-d2-monitor-check.cjs`, `/tmp/aiops-d2-csv-regression.cjs` — 대체 응답 |
| [D2] 통합 전 브라우저 연동 | 버전 조회 준비 중 / 로그 기록 없음, 수동·주기 조회 시각 갱신 확인 | 실제 localhost:8000, 실제 모델 시험 아님 |
| [D2] 통합 전 서버 조회 | `/monitor/versions` 404, `/logs/aiops.log` 200·빈 content | C API 미구현 시점, 2026-10-01 |
| [D2] 최신 코드 통합 점검 | 버전·로그 30개 + CSV 23개 통과 | 대체 응답 테스트 |
| [D2] 최신 서버 상태 | `/health` 200 (model_loaded=true, lazy), `/monitor/versions` 200 (v2), `/logs/aiops.log` 200 | `MLFLOW_TRACKING_URI=sqlite:///mlflow.db MODEL_SOURCE=mlflow .venv/bin/python -m uvicorn serving_app.main:app --host 127.0.0.1 --port 8000` |

### §E 사전 실험 (2026-10-01, 임시 복사본, PC 1대, 1회 — 본 레포 코드 아님)

| 지표 | 값 |
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

통합 데모(본 레포 코드, 팀 PC)는 E 섹션 기준 미실행으로 기록 — 단, C 섹션의 main 컨테이너 재현(§C)이 이후 이를 대체 확인했다.

## 4. 구현 경과 (변경 기록 전체, 시간순)

시각이 기록되지 않은 2026-10-01 작업은 시간 기록이 있는 작업 뒤에 영역별로 모았다.

| 시각 | 작성자/영역 | 무엇을 바꿨나 | 확인한 수치·결과와 실행 명령 | 브랜치 / PR |
|---|---|---|---|---|
| 2026-09-30 21:40 | A 초기 작성 | 스켈레톤을 공항 도메인으로 치환한 상태를 기록 | 해당 없음 | main |
| 2026-09-30 21:40 | C 초기 작성 | Day1·Day2 실행 결과 기록 | Day1 RMSE 2,571명 (`python scripts/train_baseline_v1.py`), Day2 RMSE 2,267명 (`python serving_app/train_and_register.py`) | main |
| 2026-09-30 21:40 | B 초기 작성 | TODO 상태 기록 | 해당 없음 | main |
| 2026-09-30 21:40 | D 초기 작성 | 스켈레톤 카드 구성 기록 | 해당 없음 | main |
| 2026-09-30 21:40 | E 초기 작성 | 데이터와 상수 현재 값 기록 | 해당 없음 | main |
| 2026-10-01 13:10 | 유경모/C | `GET /monitor/versions` 신설(`routers/monitor.py`, `main.py` 한 줄), `GET /data/status`에 `recent` 20건 추가 | `/monitor/versions`가 Production v1, rmse 2266.94 반환. `recent`를 그대로 `/predict`에 보내 HTTP 200. Day1 2,945명 / Day2 2,267명 | feat/c-monitor-versions |
| 2026-10-01 14:27:20 | 윤동현/A | 실습 2-1 MLflow 로더, 실제 등록 번호, 캐시 초기화 구현 | `uv run python logs/verify_a_unit.py` 8개 통과; `uv run python logs/verify_a_http.py local/lazy/eager/cache` (모드별 별도 실행)로 HTTP 결과 확인 | feat/a-mlflow-serving |
| 2026-10-01 14:39:52 | 윤동현/A | B PR #3이 병합된 main(de32913)을 A 브랜치에 충돌 없이 반영 | 로더 테스트 8개·compileall 통과. 실제 HTTP MLflow/lazy 200, 잘못된 입력 4종 422. 복제 registry에서 실제 B 승격·롤백 후 A 로드 버전 1→2→1, 승격 없는 게이트 실패 시 캐시 유지, 정상 배치 21개/판정 창 21개 (`logs/ab-integration.json`, `logs/a-b-unit-results.log`, `logs/a-b-http-lazy.log`). 학습 결과는 mock, v2는 v1 artifact 재사용, 정상 배치는 상수 모델이라 재학습 품질 검증 아님 | feat/a-mlflow-serving / PR #5 |
| 2026-10-01 14:50 | 유경모/C | B 머지본 위로 리베이스. `FINE_TUNE_EPOCHS` 10 → 3 (B의 `BIAS_THRESHOLD` 500이 3 epoch 전제) | 충돌 없음. 전체 루프 재학습 전 41283.94 → 후 41547.42. 당시 `reset_cache()` 부재로 승격 시점 `AttributeError` 500 (A 대기) | feat/c-monitor-versions |
| 2026-10-01 15:08 | youjin09222/D1 | index.html에 내일 도착 여객 예측·혼잡 등급 카드 추가 (상태 4종, 20일 막대, 버전 표기, 왕복 ms, FALLBACK_RECENT 분기) | curl `/predict` 20개 200(local 40924.95 v1-local / mlflow 41799.6 production v3), 19개 422 too_short, 경계값 4건 node 확인, compileall·`/health` 200. 브라우저 ms·캡처 미측정 | feat/d1-forecast-card |
| 2026-10-01 15:25:14 | 윤동현/A | C 요청의 run_id 연결 및 연속 배치 진단 | 신규 6개/기존 8개 통과, 실제 학습·HTTP 8000에서 v1→v2→v3→v2 및 stale false/true/false. B/C 런타임 코드·임계값·E 데이터 미변경 | fix/a-serving-run-id |
| 2026-10-01 15:27 | youjin09222/D1 | main `8f22c4e` 위로 rebase, C의 `recent` 연결 확인 후 `FALLBACK_RECENT`·임시 데이터 배지·`resolveRecent()` 제거, 보간 CSV 기준 재측정 | `recent` 20행(정수), curl `/predict` 200(local 41304.39 / mlflow 39980.96 v1), 19개 422, node `loadForecast()` 성공·422·원복, compileall·`/health` 200 | feat/d1-forecast-card |
| 2026-10-01 15:32 | youjin09222/D1 | 측정값에 브라우저 카드 ms·예측값 추가, 증빙 [D1] 5행 정리, D2·B에 파이프라인 문구 요청 | 브라우저 카드 v1 39,981명 보통 44 ms → v3 40,501명 혼잡 187 ms. 버전 1→3은 `logs/aiops.log` 15:29:07 v2·15:29:58 v3 승격 | feat/d1-forecast-card |
| 2026-10-01 16:11 | youjin09222/D1 | D1이 D2·E 영역 작업(팀 공유 후): 라이트 테마, 두 배치 나란히 비교, 파이프라인 문구 `bias vs ±500`, `BIAS_THRESHOLD` | 비교 수치 35개 일치(실서버 + Python 독립 계산), 칸 배정 CSV 2건·랜덤 1건(v11~v13), `node --check` 통과, 외부 리소스 없음 | feat/d-dashboard-compare |
| 2026-10-01 16:40 | youjin09222/D1 | 비교 칸 출처 줄 "출처:" 접두어 제거, 전송 시각 시:분 표시 | `node --check` 통과, 저장된 응답 렌더링 확인 | feat/d-dashboard-compare / #23 |
| 2026-10-01 16:50 | youjin09222/D1 | 비교 영역 방향 표기 제거(칸 제목 = 출처). 버그 수정: 칸 키 변경 후 요약 막대가 안 그려지던 문제(`stats.normal`/`stats.drift` 참조) | `node --check` 통과, 실제 응답 렌더링: 요약 SVG 2개, RMSE 막대 폭 45.3/181.3px = 독립 계산, 헤드리스 Chrome 확인 | feat/d-dashboard-compare / #23 |
| 2026-10-01 17:20 | 유경모/C | 컨테이너 빌드 확인 + 리뷰 반영: `monitor.py` MLflow 타임아웃(5초×1회)·`tracking_uri`·`registry_error` 추가, `_model_cache` 접근 `getattr`, `data.py` `_as_count()`, 실수 커밋된 876KB `does_not_exist_typo.db` 제거 | `docker compose up --build` 성공, 컨테이너 `/health` ok·eager, `/monitor/versions` v1 rmse 2266.94, `/data/status` 1,035행. 무응답 주소 재시도 1회 10.68초 → 없음 5.70초, 연결 거부 0.54초. 오타 sqlite URI는 `tracking_uri`로만 구분 | feat/c-monitor-versions |
| 2026-10-01 17:50 | 유경모/C | `docker-compose.yml`에 `environment` 추가 (`${VAR:-기본값}`) | A TODO 1 이전 기본값(mlflow+eager) 기동 실패 확인. `MODEL_SOURCE=local`로 `/health`·`/monitor/versions`·`/data/status`·`/predict` 전부 200 (예측 42096.13) | feat/c-monitor-versions |
| 2026-10-01 18:20 | 유경모/C | A·E 머지본 위로 리베이스. `stale` 판정을 A의 `LoadedModel.registry_version`에 연결 (`serving_run_id` → `serving_registry_version`) | 컨테이너 부록1 6번 3단계 재현: v1/v1 false → v2 승격 후 v2/v1 true → 재기동 false. 예측값 세 단계 모두 39980.96. 무응답 기준선 848.03초 → 5.70초 | feat/c-monitor-versions |
| 2026-10-01 (시각 미기록) | B 실험 | TODO 2~5 이식, 두 층 판정(이상치/드리프트, bias 조건), 롤백 추가 | §B-1 측정값 표 전체. 스크래치 worktree in-process (`exp4*.py`, `scan_ft.py`) | exp/drift-anomaly |
| 2026-10-01 (시각 미기록) | B 실험 | 판정을 bias만으로, 롤백 조건 "승격 뒤 첫 재학습 게이트 실패", `BIAS_THRESHOLD` 1,500 확정, 승격 시 윈도우 초기화, `high_error` → `structure_drift` | 실험 5 + 전체 루프 재실행 — 정상 ok → 드리프트 v2 → 재학습 실패 롤백 v1 → ok → 구조 드리프트 알림 → 폭설 재학습 실패 유지 | exp/drift-anomaly (당시 main 미반영) |
| 2026-10-01 (시각 미기록) | B 실험 | 학습 데이터 결항일 보간에 맞춰 `BIAS_THRESHOLD` 1,500 → 500, fine-tune 3 epoch. 시연 배치 재선정 | 실험 6 + 서버 경유 전체 루프 실측(§B-2): ok → −1,041 v2 → 확정 → +686 v3 → −1,008 게이트 2,742 실패 롤백 v2 → 폭설 structure_drift. 하루 전진 시뮬레이션과 수치 일치 | exp/clean-cancellation (당시 main 미반영) |
| 2026-10-01 (시각 미기록) | B | B 담당 파일 4개 + B 섹션만 `main`용 PR로 분리. A·C·E 선행 PR 요청 | 코드는 exp/clean-cancellation과 동일, compileall 통과. main에서는 A TODO 1 전이라 `MODEL_SOURCE=mlflow` 루프 미실행 | feat/b-monitoring / PR #3 |
| 2026-10-01 (시각 미기록) | E (A·B 영역 수정, 팀 공유) | `BatchTestRequest`에 선택 `departures` 추가, `batch_test()`가 있으면 실제 출발 여객 사용, 없으면 37,000 고정. 판정 로직·임계값 유지 | `python -m unittest discover -s tests` 8개 통과(신규 2개), compileall 통과, `TestClient` 배치 10개 × 2조건 실측(§B-3), 길이 불일치 422. 임시 서버(8010, local)에서 `/health` 200, shift normal 왕복 일치(1,156 / +116, 생략 시 1,401 / +163). MLflow Production·재학습 루프 미실행 | feat/e-batch-departures |
| 2026-10-01 (시각 미기록) | E 실험 | `prepare_jeju_data.py` 결항일(도착 < 25,000) 보간 추가, CSV·학습 컷 4개 재생성, 드리프트 시연 컷 재선정(배치 raw / 학습 보간본), `falsealarm` → `sep_oct` 개명 | `python scripts/prepare_jeju_data.py` → 18일 보간, 1,035행. 기초 통계 보간본 기준 갱신 | exp/clean-cancellation (이후 PR #7로 main 반영) |
| 2026-10-01 (시각 미기록) | E (D2 영역 수정, 팀 공유) | CSV 배치가 `Departures` 열을 `departures`로 전송(`parseBatchCsv()`, `sendBatch(kind, csv)`), 랜덤워크는 `arrivals`만. 안내 문구 수정 | `node --check` 통과, `parseBatchCsv()` node 실행으로 41행 도착·출발 값 CSV 일치. 브라우저 화면 확인 미실행 | feat/e-batch-departures |
| 2026-10-01 (시각 미기록) | E | 섹션을 담당자별로 재편, 아키텍처·측정값·증빙을 섹션 안으로 이동, 사전 실험 수치 기록 | 해당 없음 | main |
| 2026-10-01 (시각 미기록) | D2 | CSV 선택·검증·배치 전송, 중복 요청 방지, 서버 판정 상태 표시 보완 | 대체 응답 점검 23개 통과, compileall, `/health` 200. 실제 모델 RMSE·재학습·버전 전환 미측정 | feat/d2-dashboard-batch-and-version (PR #14) |
| 2026-10-01 (시각 미기록) | D2 | 모델 버전 카드, 5초 주기 로그 조회·색 구분, 안전한 텍스트 표시, 배치 후 재조회, 잘못된 예측 응답 거부 | 대체 응답 22개 + CSV 회귀 23개 통과. 실서버 버전 404·빈 로그 200 확인 | feat/d2-dashboard-batch-and-version |
| 2026-10-01 (시각 미기록) | D2 | 원격 main `8f22c4e` fast-forward 통합 + D2 변경 복원. 실제 C API 날짜·소스·stale 표시 대응 | 대체 응답 50개, 실제 학습 v1·재학습 v2, `/predict` 등록 번호 1→2, 정상/확정 CSV·UI·로그 확인. 입력 출처 미확인 배치는 별도 표시 | feat/d2-dashboard-batch-and-version |
| 2026-10-01 (시각 미기록) | D2 | PR 준비 중 main `56bf712` 통합. 버전 응답 오류·실제 등록 번호 연결, 측정값 정밀도·표 서식 보완 | 대체 응답 53개, compileall 통과. 공통 테스트 4개 통과/2개 실패(7장 참고). 두 초안 PR로 분리 | feat/d2-dashboard-batch-and-version (PR #16) |
| 2026-10-01 (시각 미기록) | D2 | #16 병합 상태를 main으로 전달하기 위해 최신 main fd760c0 통합. C의 run_id 우선 비교에 맞춰 일치 안내를 특정 번호에 한정하지 않도록 수정 | 백엔드·팀원 변경 보존 | feat/d2-monitor-main |
| 2026-10-01 (시각 미기록) | D2 | main 976b069의 D1 예측 카드와 PR #20 충돌 해결. init에서 D1 예측과 D2 모니터 조회 모두 시작 | 화면 점검 53개 통과, compileall. `/tmp/d2-init-merge-check.cjs`로 D1 함수·기록 보존 확인 | feat/d2-monitor-main |

## 5. 트러블슈팅 (증상 → 원인 → 해결 → 전후 결과)

### 서빙 API (A)

1. `/monitor/versions`가 계속 `stale=null` → A 모델 객체에 `run_id`가 없었음 → 선택한 모델 버전 메타데이터에서 함께 보관 → 실제 HTTP에서 정상 `false`, 불일치 `true` 확인. `null` 자체가 500을 내지는 않는다.
2. 연속 배치 진단: 41행 요청은 예측 21개로 이전 창을 전부 교체하고, 21행 요청은 새 예측 1개라 과거 20개를 유지한다(rolling window) → 테스트 2개로 구분. `data/README.md` 9단계 실측은 서버 재시작 없이 정상→승격→확정→승격→롤백 완료. 모델/학습 CSV가 바뀌므로 배치 순서별 판정 차이만으로 캐시 잔존을 단정하지 않는다.
3. MLflow 로드가 `NotImplementedError` → TODO 1 미구현 → Production 조회·정확한 버전 로드·고정 scaler 결합 → 실제 `/predict` 200, `model_version=production`, `model_registry_version=1`.
4. Production 상태 이름만으로는 버전 전환 구분 불가 → 기존 필드 보존 + 등록 번호 필드 추가.
5. MLflow stage API 폐기 예정 경고 출력 → 수업의 Production stage 방식 유지, 별칭 전환은 범위 밖 (기능 오류 아님).
6. 시작 시간 측정 기준: 프로세스 내부 import 시작부터 Uvicorn startup 완료까지 — OS 프로세스 생성 시간 미포함, 환경·캐시 영향을 받는 1회 값.

### 모니터링·AIOps (B)

1. 드리프트 시연 정상 배치(2024-05-21~06-30)가 v1로 RMSE 2,820 → 임계값 초과 → `train_test_split`이 마지막 20%를 검증으로 떼어 이 구간이 학습에 안 들어감 + 06-29 하루 −7,598 → 학습 구간 안쪽 2024-04-15~05-25(2,374 / +252)로 교체 → 2,820 → 2,374.
2. 재학습 후 다음 윈도우에서 v2가 v1보다 나쁨(3,918 vs 3,126), v2 재재학습은 게이트 실패 → 41행 fine-tuning이 1분기 저점에 과적응 + 게이트 5일 검증이 노이즈 → "승격 뒤 첫 드리프트의 재학습이 게이트 실패하면 이전 버전으로 롤백" 도입 (처음엔 "반대 부호 bias" 조건이었으나 설명이 단순한 쪽으로 교체) → v2 갇힘 → v1 복귀, 이후 윈도우 `ok`.
3. 예측선이 평평하고 v1은 1분기 내내 실제 위, v2는 4월 이후 내내 실제 아래 → 모델이 입력 20일 수준이 아니라 학습 기간 평균 쪽으로 예측(예측 std 460 vs 실제 2,444, 전일값 2,083 < LSTM 2,398, 학습 범위 35,000~40,000 밖 입력에 외삽 안 됨) → 해결 없음(피처·학습 흐름 변경 범위 밖). 기획서 ②의 "전일값 복사 3,600보다 25% 좋아야" 근거는 결항일 포함 값이라 수정 필요.

### 대시보드 (D1·D2)

1. 브라우저 `ERR_CONNECTION_REFUSED` → 서버 실행 명령이 아직 없었음 → `python -m uvicorn serving_app.main:app --port 8000` 실행 → 대시보드 표시·`/health` 200 확인.
2. 기존 UI가 `anomaly`·`structure_drift`·`rolled_back`를 기본 정상 문구로 표시 → B가 추가한 상태 분기 부재 → 문구·파이프라인 분기 추가 → 대체 응답으로 상태별 표시 확인.
3. [D1] 모델 학습 전 `/predict` 500: `curl -X POST localhost:8000/predict -d @req20.json` → 텍스트 500 → 서버 로그 `ValueError: File not found: filepath=serving_app/models/airport_v1.keras` (새 환경이라 로컬 모델 없음) → `curl -F file=@data/jeju_airport_arrivals.csv localhost:8000/data/upload`(1035행) → `python scripts/train_baseline_v1.py`(baseline v1 RMSE = 2517명) → 같은 요청 HTTP 200 `{"predicted_arrivals":40924.95,"model_version":"v1-local","model_registry_version":null}`.
4. [D2] (D1 작성) 배치 2건 동시 전송으로 Production 중복 승격(v6·v7): 2026-10-01 15:53, #14 반영 전 작업본에서 배치 버튼이 연달아 눌려 `POST /predict/batch-test` 2건이 서로 다른 연결(127.0.0.1:53655, :53658)로 겹쳐 1.4초 간격 두 번 승격. `logs/aiops.log` 원문 — `15:53:21,994 [WARNING] [WARN] drift detected - rmse=1451 bias=-892` → `15:53:24,192 [WARNING] [WARN] drift detected - rmse=2246 bias=+1160` → `15:53:27,297 [INFO] [OK] new_rmse=2233 - production promoted: v6` → `15:53:28,672 [INFO] [OK] new_rmse=2233 - v7` → 원인: 전송 중에도 버튼이 눌렸고 서버 `check_and_trigger()`에 동시 실행 방지가 없어 같은 v5·같은 데이터로 각각 fine-tune(둘 다 `new_rmse=2233`) → 해결: D2의 #14가 전송 중 버튼 비활성화·`batchSending` 재진입 방지로 해결(main 머지), 서버 쪽은 B에 요청 → 후: 요청 3건(CSV 2 + 랜덤 1) 순서 전송 시 승격 3회(v11·v12·v13) 겹침 없음. 브라우저 연타 확인은 미측정.
5. [D2] 버전 API 404 → C 담당 API 미구현 시점 → 화면은 `-`와 조회 준비 중 표시, 로그 200/빈 content는 "기록 없음"으로 구분. 통신 실패·지연·파일 전환·로그 내 HTML 문자열은 대체 응답으로 확인.
6. [D2 통합] 기존 서버가 이전 Python 코드를 메모리에 유지 → 병합해도 재시작 전 버전 API 미반영 → 기존 서버 정상 종료 후 MLflow 모드 재기동 → 실제 버전 API 200.
7. [D2 통합] 설계 문서는 생성 시각이 문자열, 실제 C 코드는 Unix 밀리초 → 날짜 `-` 표시 가능 → 두 형식 모두 한국 시간 변환 → 실제 v1/v2 생성 시각 표시 확인.
8. [D2 통합] 파일 선택 자동 조작 중 대기 시간 초과와 별도 배치 실행 관측 — 원인 입력 미확인으로 사용자 조작으로 단정하지 않음. 해당 배치는 RMSE 1729명/bias −1358명, 게이트 1338.3271859414438명으로 v2 승격 — 입력 출처 미확인으로 기록, 고정 CSV 재현 근거로 쓰지 않음. 이후 정확한 파일 입력으로 확정 CSV `structure_drift` 확인.

### 데이터·통합 (E)

1. 시연 확인 스크립트에서 학습 서브프로세스가 0초 만에 종료, 모델 없음 → `subprocess.run(["python", ...])`이 venv가 아닌 시스템 Python(`C:\Python312`, tensorflow 없음)을 잡음 → `sys.executable`로 호출 → 팀 PC에서도 `python`이 venv를 가리키는지 먼저 확인.
2. 워크트리 `mlruns/`를 지워도 레지스트리 버전이 v4부터 이어짐 → 레지스트리는 cwd의 `mlflow.db`(sqlite), `mlruns/`는 아티팩트만 → `mlflow.db`도 함께 삭제 → 시연 PC에서 "v1부터" 보여주려면 `mlflow.db`, `mlruns/`, `data/uploads/*.csv`, `serving_app/models/*.keras|pkl` 모두 비우고 시작.

### 학습·배포 (C)

기록 없음.

## 6. 증빙 목록

| 영역 | 증빙 | 무엇을 보여주나 | 상태 | 파일 | 기록자·시각 |
|---|---|---|---|---|---|
| A | HTTP local/lazy/eager | 예측 결과·시간·health·422 | 실제 실행 로그 확보, UI 캡처 미촬영 | `logs/a-http-local.json`, `logs/a-http-lazy.json`, `logs/a-http-eager.json` | 윤동현/A |
| A | 버전 전환·롤백 후 reset | 같은 프로세스에서 등록 번호 1→2→1 | 실제 HTTP 확인, 재학습 호출 없음 | `logs/a-http-cache.json` | 윤동현/A |
| A | run_id 연결·실제 A/B/C 루프 | 승격·롤백 후 예측/조회 일치, 의도적 불일치 탐지 | 통과 | `logs/run-id-e2e-results.json`, `logs/run-id-e2e.log`, `tests/test_serving_identity.py` | 윤동현/A |
| A | 캐시·로더 테스트 | 경합·실패·재시도 등 8개 | 통과 | `logs/a-unit-results.log` | 윤동현/A |
| C | `train_and_register.py` 실행 로그 | `[GATE PASSED]`와 RMSE | 미촬영 | | |
| C | MLflow UI Registry 화면 | Production 버전 | 미촬영 | | |
| C | 컨테이너 빌드·실행 로그 | Docker로 재현됨 | 미촬영 | | |
| C | 컨테이너의 Swagger 화면 | 컨테이너에서도 같은 API 동작 | 미촬영 | | |
| C | `GET /monitor/versions` 응답 | Production 버전·RMSE | 미촬영 | | |
| B | `simulate_drift.py` 실행 결과 | 정상·드리프트 배치의 `drift_check` | 미촬영 | | |
| B | `logs/aiops.log` | `[WARN]` → `[INFO]` → `[OK]` 순서 | 미촬영 | | |
| B | 재학습 후 `/predict` 응답 | 새 Production 버전 반영 | 미촬영 | | |
| D1 | 업로드 전 카드 | `exists=false` → 안내 문구 | 미촬영 | 제안: `docs/snapshots/d1_01_before_upload.png` | |
| D1 | 예측·혼잡 등급 카드 | 예측값·등급·버전·왕복 ms·20일 추이 | 미촬영 | 제안: `docs/snapshots/d1_02_predict_success.png` | |
| D1 | 422 실패 카드 | 19개 입력 → 422 detail 요약, 다른 카드 정상 | 미촬영 | 제안: `docs/snapshots/d1_03_predict_422.png` | |
| D1 | 모델 전환 전 카드 | `production` / 번호 1, 39,981명 보통, 44 ms | 미촬영 | 제안: `docs/snapshots/d1_04a_version_before.png` | |
| D1 | 모델 전환 후 카드 | `production` / 번호 3, 40,501명 혼잡, 187 ms. 사이에 드리프트 배치로 v2(15:29:07, rmse=889 bias=+813 → 재학습 2230)·v3(15:29:58, rmse=1619 bias=+739 → 재학습 2269) 승격 | 미촬영 | 제안: `docs/snapshots/d1_04b_version_after.png` | |
| D2 | CSV 선택 미리보기 | 파일 기간·행 수·버튼 활성화 | 촬영 | `/tmp/aiops-d2-csv-preview.jpg` (임시 로컬) | D2, 2026-10-01 |
| D2 | CSV 배치 전송 결과 | 확정 CSV의 실제 `structure_drift` 응답 | 촬영·JSON 저장 | `logs/d2-evidence/d2-confirm-live.jpg`, `logs/d2-evidence/d2-confirm-browser.json` | D2, 2026-10-01 |
| D2 | 재학습 로그 패널 | `[WARN]` → `[INFO]` → `[OK]`·구조 변화 알림 | 촬영 | `logs/d2-evidence/d2-logs-live.jpg`, `logs/aiops.log` | D2, 2026-10-01 |
| D2 | Production 조회 준비 상태 | 버전 API 미구현 안내와 빈 값 | 촬영 | `/tmp/aiops-d2-model-status.jpg` (임시 로컬) | D2, 2026-10-01 |
| D2 | Production 버전 표기 | v1→v2 관측 변화·학습 검증 RMSE·생성 시각 | 촬영 | `logs/d2-evidence/d2-version-live.jpg`, `logs/d2-live-integration-20261001.json` | D2, 2026-10-01 |
| E | 통합 데모 한 바퀴 | 업로드 → 예측 → 드리프트 → 재학습 → 버전 변화 | 미촬영 | | |

A의 로그·검증 스크립트(`logs/verify_a_unit.py`, `logs/verify_a_http.py`)는 해당 PC의 Git 제외 경로에 보관하며, 측정표와 PR 본문에 결과를 함께 기록한다(UI 스크린샷으로 간주하지 않음).

## 7. 확인 필요

### 서로 다른 값이 기록된 지표 (발표 전 조건 명기 필요)

| 지표 | 기록된 값들 | 차이의 원인 |
|---|---|---|
| Day1 baseline RMSE | 2,571명(C, PC 1) / 2,945명(C, PC 2) / 2,191명(C, 다른 PC) / 2,517명(D1, 새 환경) / 2,308명(D1, 보간 CSV) / 1,999(B, 547행 보간 컷) / 2,163(D2, 547일 컷) / 2,365~2,922(B, 실행마다 다름) | baseline은 seed 미고정 + 데이터 버전(raw/보간)·학습 컷이 달라 실행·PC마다 다름 |
| Day2 MLflow 게이트 RMSE (전체 데이터) | 2,267명(2266.94, 보간 전) / 2,363.50(보간 후) | E의 결항일 보간 데이터(PR #7) 반영 전후. seed 42라 같은 데이터면 PC가 달라도 동일 |
| 배치 판정 RMSE — README 표 vs API 응답 | 확정 배치 3,169(표, 이상치 제외) vs 5,924(API, 포함) / 폭설 3,039 vs 5,858 | `check_and_trigger` 반환은 이상치 **포함** `rmse`, 판정·로그는 **제외** 값. B가 `rmse_excl_anomalies`를 반환에 추가하거나 발표 대본에 "표 값은 `logs/aiops.log` 기준" 명시 필요 |
| 롤백 시연의 게이트 실패 값 | 2,742 vs 게이트 2,700 — 간발 | 다른 PC에서 재확인 필요 (학습 비결정성으로 통과로 뒤집힐 수 있음) |

### 문서·코드 불일치 (정리 필요)

- ~~`CLAUDE.md`·`PROJECT_PLAN.md` 일부에 "main은 10 epoch / 팀 합의 전" 잔존~~ — `CLAUDE.md` 상수 표는 2026-10-01 PR #3(B)·#6(C)·#7(E) main 반영 완료로 갱신됨 (코드 확인: `FINE_TUNE_EPOCHS=3`, `BIAS_THRESHOLD=500`). `PROJECT_PLAN.md` 잔존 문구는 확인 필요.
- `docs/API_SPEC.md`가 완료된 API를 TODO/설계안으로 표기, `/monitor/versions` 응답 필드도 구현보다 적음 — 프롬프트 5에서 갱신.
- 최신 main 공통 테스트 6개 중 2개 실패: `serving_run_id` KeyError, run_id 없는 경우 `stale=None` 기대 vs 실제 False — A/C API 계약과 테스트 정합성 확인 필요.
- 3월 18일 학습 컷 마지막 값 30,608이 3월 19일 원자료를 쓴 보간(미래 정보 누수) — 시점별 데이터 생성 후 게이트/롤백 재확인 필요.
- 기획서 ②의 "전일값 복사 3,600명 대비 25% 개선" 근거는 결항일 포함 값 — 보간본 기준(전일값 복사 RMSE 2,523명)으로 수정 필요.
- `batch-test`의 `departures` 전달 여부 팀 결정 대기 — 판정이 10개 배치 중 3개에서 바뀌므로 시연 수치에 직접 영향. `data/README.md` 시연 수치는 고정값 조건이라 CSV 배치(departures 포함) 시연 시 재측정 필요.
- 승격 시 기존 Production 버전을 Archived로 내리지 않아 여러 버전이 Production에 남음 — `archive_existing_versions=True` 후속 PR 조율 중.
- `_last_promotion`이 프로세스 메모리에만 있어 서버 재시작 후 롤백 불가 — 알려진 한계로 기록.

### 미측정 항목 (담당자에게 측정 요청)

- 다른 PC에서의 게이트 통과 여부 (C)
- 학습 소요 시간 100 epoch (C)
- `simulate_drift.py` 랜덤워크 배치 실측 (B — σ 설계 문제로 보류, 계단식 생성기 안은 C→D2 이관)
- 시연 순서의 Production 모델(v1 → v2 → …) 기준 `departures` 전달 값 (B/E)
- [D1] v3 전환 후 새로고침 2회 ms, 브라우저 캡처 전부 (D1)
- [D2] 로그 카드 `[WARN]`→`[OK]` 반영 시간, 실제 롤백, 브라우저 연타 확인, 재학습 품질 개선 비교 (D2)
- 증빙 목록의 미촬영 스냅샷 전부 (각 담당 — 프롬프트 6의 촬영 목록 참고)

