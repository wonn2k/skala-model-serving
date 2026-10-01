# ⑥ 동작 화면 스냅샷

> **생성물.** 프롬프트 6이 `docs/code_current.md` 증빙 기록과 `logs/aiops.log`에서 생성했다 (2026-10-01). 손으로 고치지 않는다.
> 원칙: 각 스냅샷은 **요청 전과 후의 상태 변화**를 보여주고, ③ 운영 설계의 어떤 정책을 증명하는지 한 줄로 연결한다. 로그 발췌는 실제 파일 그대로다(줄·시각 무수정).
> 현황: 레포의 `docs/snapshots/` 폴더는 아직 없다. D2가 찍은 캡처 5장은 D2 PC의 `logs/d2-evidence/`·`/tmp`(Git 제외 경로)에 있어 레포 반영이 필요하고, 나머지는 미촬영이다 — 맨 아래 **촬영 목록** 참고.

## 묶음 1 — 모델 버전 전환 (local ↔ MLflow Production)

**증명하는 정책(③):** 서빙은 로컬 파일과 MLflow Production 두 출처를 설정(`MODEL_SOURCE`)으로 전환하며, 응답의 버전 필드로 어느 모델이 답했는지 추적한다.

| # | 전 → 후 | 증빙 상태 |
|---|---|---|
| 1-1 | `MODEL_SOURCE=local`의 `/predict` → `{"predicted_arrivals": 42087.61, "model_version": "v1-local", "model_registry_version": null}` vs `MODEL_SOURCE=mlflow` → `{"predicted_arrivals": 39772.69, "model_version": "production", "model_registry_version": "1"}` — 같은 20행 입력, 가중치가 달라 예측값·버전 표기가 모두 다르다 | 실행 로그 확보(`logs/a-http-local.json`, `logs/a-http-lazy.json`, A PC) · **UI 캡처 미촬영** |
| 1-2 | 승격·롤백 뒤 `reset_cache()` — 같은 프로세스에서 등록 번호 **1 → 2 → 1** | 실제 HTTP 확인(`logs/a-http-cache.json`, A PC) · **UI 캡처 미촬영** |
| 1-3 | [D1] 카드 버전 전환 전 → 후: `production`/번호 1, **39,981명 보통, 44 ms** → `production`/번호 3, **40,501명 혼잡, 187 ms**(새 모델 lazy 로드 포함). 사이에 드리프트 배치로 v2(15:29:07)·v3(15:29:58) 승격 | **미촬영** — 제안 파일명 `docs/snapshots/d1_04a_version_before.png`, `d1_04b_version_after.png` |

## 묶음 2 — 컨테이너 실행

**증명하는 정책(③):** 업로드 → 학습 → 게이트 → 서빙이 어느 PC에서나 `docker compose up --build` 한 번으로 재현된다.

| # | 전 → 후 | 증빙 상태 |
|---|---|---|
| 2-1 | `docker compose build --no-cache`: 8단계·561줄, pip 35.5초 + 빌드 중 학습·등록 32.6초(`[GATE PASSED] rmse=2363 -> v1`) + export 38.2초, 이미지 3.26GB → 기동 후 `/health` `{"status":"ok","model_loaded":true,"loading_mode":"eager"}` | 수치는 `docs/code_current.md` 2번 실측 · **빌드·실행 로그 캡처 미촬영** |
| 2-2 | 컨테이너의 Swagger(`/docs`) — 로컬과 같은 엔드포인트 8개 | **미촬영** |
| 2-3 | 컨테이너 `stale` 3단계: 기동 직후 v1/1/`false` → 컨테이너 안 재학습으로 v2 승격(서버 유지) → v2/1/**`true`**, `/predict`는 여전히 `"1"` → `docker compose restart` → v2/2/`false`, `/predict` `"2"` | `/monitor/versions` 응답 기록 확보(`docs/code_current.md` 2번) · **화면 캡처 미촬영** |

## 묶음 3 — 드리프트 대응 (정상 → 드리프트 → 재학습 → 재배포)

**증명하는 정책(③):** |bias| > 500명이면 감지 → 최근 41행 fine-tuning(3 epoch) → 게이트(2,700명) 재검증 → 통과 시에만 재배포. 통과 못 하면 기존 버전 유지, 미확정 승격 뒤 실패면 롤백.

실제 `logs/aiops.log` 발췌 — 이 PC, 2026-10-01 (감지 `[WARN]` → 재학습 시작 `[INFO]`, 줄 그대로):

```
2026-10-01 17:14:52,538 [WARNING] [WARN] drift detected - rmse=2189 bias=+1753 (excluding 0 anomaly day(s)) - triggering retrain
2026-10-01 17:14:52,875 [INFO] [INFO] retrain triggered (window=last_21_days)
2026-10-01 17:14:56,553 [WARNING] [WARN] anomaly on 6 day(s) - worst actual=56324 predicted=43891 (excluded from drift check)
```

`main` 컨테이너 전체 시연 실측의 3단계(`docs/code_current.md` 2번에 기록된 발췌): `[WARN] rmse=3490 bias=-587` → `[INFO] retrain triggered` → `[OK] new_rmse=2098 v3`, 감지에서 승격까지 **3.7초**.

| # | 전 → 후 | 증빙 상태 |
|---|---|---|
| 3-1 | 정상 배치 → `{"status":"ok"}` (rmse 2183.06 / bias −116.79, 컨테이너 시연) → 드리프트 배치 → `retrain_triggered`, promoted true, rmse 2098.47, v3 | 응답·로그 수치 확보 · **화면 캡처 미촬영** |
| 3-2 | 재학습 전후 같은 입력 `/predict`: 31131.23명/번호 1 → 30622.54명/번호 2 (D2 실측) | JSON 확보(D2 PC) · **캡처는 D2 PC `logs/d2-evidence/`** |
| 3-3 | 롤백: 반등 v3 승격 → 다음 배치 드리프트·재학습 게이트 실패(2,742 > 2,700) → `[ROLLBACK]` v3 Archived·v2 복귀, `/predict` 31,309.10(v2) | 서버 경유 실측 기록(3번 "보간 데이터 실측") · **화면 캡처 미촬영** |
| 3-4 | `simulate_drift.py` 실행 결과 | **미촬영** (σ 설계 문제로 실측 보류 — 시연은 CSV 배치 경로) |

## 묶음 4 — 대시보드

**증명하는 정책(③):** 운영자는 코드 없이 화면에서 예측·혼잡 등급을 보고, 배치 전송 → 감지 → 재학습 → 버전 변화를 눈으로 따라간다.

| # | 전 → 후 | 증빙 상태 |
|---|---|---|
| 4-1 | [D1] 업로드 전 "데이터를 먼저 업로드하세요" → 업로드·학습 후 예측값·혼잡 등급·버전·왕복 ms·20일 추이 카드 | **미촬영** — 제안 `d1_01_before_upload.png`, `d1_02_predict_success.png` |
| 4-2 | [D1] 19개 입력(콘솔 명령) → 422 detail 요약 표시, 다른 카드는 정상 유지 | **미촬영** — 제안 `d1_03_predict_422.png` |
| 4-3 | [D2] CSV 선택 미리보기(기간·행 수·버튼 활성화) → 전송 → 확정 CSV의 실제 `structure_drift` 응답 표시 | **촬영됨** — D2 PC `/tmp/aiops-d2-csv-preview.jpg`, `logs/d2-evidence/d2-confirm-live.jpg` + `d2-confirm-browser.json` (레포 `docs/snapshots/` 반영 필요) |
| 4-4 | [D2] 재학습 로그 패널: 빈 로그 → 배치 후 `[WARN]`→`[INFO]`→`[OK]` 색 구분 표시 | **촬영됨** — D2 PC `logs/d2-evidence/d2-logs-live.jpg` (반영 필요) |
| 4-5 | [D2] Production 버전 카드: v1 → 재학습 후 v2, 학습 검증 RMSE·생성 시각 표시 | **촬영됨** — D2 PC `logs/d2-evidence/d2-version-live.jpg`, `d2-live-integration-20261001.json` (반영 필요) |
| 4-6 | [D2] 버전 API 미구현 시점의 "조회 준비 중" → 구현 후 실제 버전 표시 (전·후 대비) | 전: D2 PC `/tmp/aiops-d2-model-status.jpg` · 후: 4-5와 동일 |

## 촬영 목록 (남은 스냅샷 — 누가, 어떤 명령으로, 어떤 화면)

공통 준비: 시연 PC에서 "v1부터" 보이려면 `mlflow.db`, `mlruns/`, `data/uploads/*.csv`, `serving_app/models/*.keras|pkl`을 비우고 시작한다 (`docs/code_current.md` 5번 트러블슈팅). 캡처는 `docs/snapshots/`에 모은다.

| 담당 | 명령/조작 | 찍을 화면 | 파일명 제안 |
|---|---|---|---|
| A | `MODEL_SOURCE=local uvicorn …` → `/predict` curl, 이어서 `MODEL_SOURCE=mlflow`로 재기동 → 같은 입력 | 두 응답의 `model_version`·`model_registry_version` 차이 (터미널) | `a_01_local_vs_mlflow.png` |
| C | `docker compose -f serving_app/docker-compose.yml up --build` | `[GATE PASSED]` 포함 빌드·기동 로그, `docker ps` | `c_01_container_build.png` |
| C | 컨테이너 기동 후 브라우저 `localhost:8000/docs` | Swagger 엔드포인트 8개 | `c_02_container_swagger.png` |
| C | MLflow UI(`mlflow ui --backend-store-uri sqlite:///mlflow.db`) | Registry의 Production 버전 목록 | `c_03_mlflow_registry.png` |
| C | `curl localhost:8000/monitor/versions` (승격 전/컨테이너 안 재학습 후/재기동 후) | `stale` false → true → false 3단계 | `c_04_stale_3steps.png` |
| B | 대시보드에서 정상 배치 → 드리프트 배치 전송 (`data/README.md` 시연 순서) | `logs/aiops.log`의 `[WARN]` → `[INFO]` → `[OK]` (로그 카드 또는 터미널 `tail -f`) | `b_01_aiops_log_cycle.png` |
| B | 재학습 직후 같은 입력 `/predict` | 승격 전·후 응답의 `model_registry_version` 변화 | `b_02_predict_after_retrain.png` |
| D1 | 초기화 상태로 대시보드 접속 → CSV 업로드 → 학습 | 업로드 전 안내 카드 / 예측·혼잡 등급 카드 | `d1_01_before_upload.png`, `d1_02_predict_success.png` |
| D1 | 콘솔 1회용 fetch 패치(`docs/code_current.md` 4번 현재 상태의 명령) | 422 실패 카드 + 다른 카드 정상 | `d1_03_predict_422.png` |
| D1 | 드리프트 배치로 버전 올린 전·후 새로고침 | 버전 전환 전·후 카드 (39,981 보통 ↔ 40,501 혼잡) | `d1_04a/d1_04b_version_*.png` |
| D2 | 기존 D2 PC 캡처 5장을 레포로 이동 | 4-3 ~ 4-6 | `d2_01`~`d2_05_*.png` (기존 파일명 유지 가능) |
| E | 위 전체를 한 PC에서 이어서 (업로드 → 학습 → 예측 → 드리프트 → 재학습 → 재배포) | 통합 데모 한 바퀴 (단계별 1장씩) | `e_01`~`e_06_demo_*.png` |
