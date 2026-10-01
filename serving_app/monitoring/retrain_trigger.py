"""
Day3: 드리프트 감지 -> fine-tuning 재학습 -> 재배포를 잇는 파이프라인의 핵심 조립 지점.

흐름: 이상 탐지(RMSE>2,700명) -> 알림 -> 최근 3주(21일) 데이터 수집 ->
      Production 가중치에서 이어서 fine-tuning(warm start) -> 게이트 재검증 ->
      Production 재배포 (통과 못하면 기존 버전 유지)

왜 "처음부터 재학습"이 아니라 fine-tuning인가: 최근 3주(21일)만으로 LSTM을
스크래치로 학습시키기엔 샘플이 너무 적어 불안정합니다. 이미 전체 기간 데이터로 학습된
Production 가중치에서 이어서 짧게(10 epoch) 미세조정하는 쪽이 훨씬 안정적입니다.

데이터는 data/uploads/에 업로드된 파일 중 가장 최근 것을 사용합니다(data/storage.py의
latest_upload() - Day2 train_and_register()가 쓰는 것과 같은 소스).
"""
import logging

from serving_app.monitoring.drift_detector import RMSE_THRESHOLD, WINDOW_SIZE, assess

logger = logging.getLogger("aiops")

# 마지막 승격 기록(아직 확정되지 않은 승격). 승격 뒤 첫 판정이 드리프트이고 그 재학습마저 게이트에 실패하면
# 새 버전이 일시적 변화에 과적응한 것으로 보고 이전 버전으로 되돌린다. 승격 뒤 한 윈도우라도 드리프트가 아니면 승격 확정.
# ponytail: 프로세스 메모리에만 저장. 서버 재시작 후에는 롤백 대상을 모른다. 필요하면 MLflow 태그로 옮긴다.
_last_promotion: dict | None = None  # {"prev": 이전 Production 버전, "new": 승격 버전}


def _production_version() -> str | None:
    from mlflow import MlflowClient
    from serving_app.train_and_register import MODEL_NAME

    vs = MlflowClient().get_latest_versions(MODEL_NAME, stages=["Production"])
    return vs[0].version if vs else None


def _rollback(failed_rmse: float) -> dict:
    global _last_promotion
    from mlflow import MlflowClient
    from serving_app import model_loader
    from serving_app.train_and_register import MODEL_NAME

    prev, new = _last_promotion["prev"], _last_promotion["new"]
    c = MlflowClient()
    c.transition_model_version_stage(MODEL_NAME, prev, "Production")
    c.transition_model_version_stage(MODEL_NAME, new, "Archived")
    model_loader.reset_cache()
    logger.warning(
        f"[ROLLBACK] retrain after promotion failed gate (rmse={failed_rmse:.0f}) "
        f"- v{new} archived, v{prev} back to Production"
    )
    _last_promotion = None
    return {"status": "rolled_back", "production_version": prev, "rmse": failed_rmse}


def check_and_trigger(recent_predictions: list[dict]) -> dict:
    global _last_promotion
    a = assess(recent_predictions)
    if not a["drift"] and len(recent_predictions) >= WINDOW_SIZE:
        _last_promotion = None  # 승격 뒤 드리프트 없는 윈도우가 한 번 나오면 승격 확정
    if a["anomalies"]:
        # 하루 오차가 아주 큰 날(결항 등)은 재학습으로 배울 수 없다. 알림만 남기고 판정에서 뺀다.
        worst = max(a["anomalies"], key=lambda p: abs(p["actual"] - p["predicted"]))
        logger.warning(
            f"[WARN] anomaly on {len(a['anomalies'])} day(s) - worst actual={worst['actual']:.0f} "
            f"predicted={worst['predicted']:.0f} (excluded from drift check)"
        )
    if not a["drift"]:
        if a["rmse_excl_anomalies"] > RMSE_THRESHOLD:
            # 오차는 크지만 치우침이 없다 - 연휴·변동 큰 기간. 재학습으로 줄지 않으므로 알림만.
            logger.warning(f"[WARN] high error without bias - rmse={a['rmse_excl_anomalies']:.0f} bias={a['bias_excl_anomalies']:+.0f} (no retrain)")
            status = "high_error"
        else:
            status = "anomaly" if a["anomalies"] else "ok"
        return {"status": status, "rmse": a["rmse"], "bias": a["bias"], "anomaly_days": len(a["anomalies"])}

    logger.warning(
        f"[WARN] drift detected - rmse={a['rmse_excl_anomalies']:.0f} bias={a['bias_excl_anomalies']:+.0f} "
        f"(excluding {len(a['anomalies'])} anomaly day(s)) - triggering retrain"
    )

    #   1) 최근 3주(21일) + 시퀀스 구성용 선행 SEQ_LEN(20)일을 업로드된 최신 CSV에서 조회한다.
    #   2) fine_tune(rows)로 Production 가중치에서 이어서 재학습한다 (처음부터 다시 학습하지 않음).
    #   3) 반환된 결과(dict)의 "promoted" 값으로 게이트 통과 여부를 판단한다.
    # (train_and_register는 import 시점에 tensorflow/mlflow를 불러오므로, 서버 기동을 늦추지 않도록
    #  드리프트가 감지된 시점에만 import한다.)
    from data.features import load_rows, SEQ_LEN
    from data.storage import latest_upload
    from serving_app.train_and_register import fine_tune

    logger.info("[INFO] retrain triggered (window=last_21_days)")
    prev = _production_version()
    rows = load_rows(latest_upload())[-(WINDOW_SIZE + SEQ_LEN):]
    result = fine_tune(rows)
    if result["promoted"]:
        from serving_app import model_loader

        model_loader.reset_cache()  # 다음 /predict부터 새 Production 사용
        _last_promotion = {"prev": prev, "new": result["version"]}
        logger.info(f"[OK] new_rmse={result['rmse']:.0f} - production promoted: Airport_Arrivals_Predictor v{result['version']}")
        return {"status": "retrain_triggered", "promoted": True, "rmse": result["rmse"], "version": result["version"]}
    if _last_promotion:
        # 승격 직후 또 드리프트인데 재학습도 게이트를 못 넘음 - 직전 승격이 일시적 변화에 과적응한 것. 되돌린다.
        return _rollback(result["rmse"])
    return {"status": "retrain_triggered", "promoted": False, "rmse": result["rmse"]}
