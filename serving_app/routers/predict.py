"""
Day1 -> Day3(시뮬레이션 엔드포인트 추가) 확장 파일.

Day1: POST /predict - 최근 SEQ_LEN(20)일 시퀀스로 다음날 도착 여객 수 예측
Day3: POST /predict/batch-test - 드리프트 감지 시뮬레이션 시작점 (scripts/simulate_drift.py 참고)
"""
from fastapi import APIRouter

from data.features import SEQ_LEN
from serving_app import model_loader
from serving_app.schemas import PredictRequest, PredictResponse, BatchTestRequest, BatchTestResponse
from serving_app.monitoring.retrain_trigger import check_and_trigger

router = APIRouter()

# Day3: 최근 예측 기록(actual/predicted)을 쌓아두는 슬라이딩 윈도우.
# monitoring/drift_detector.py의 WINDOW_SIZE(21)만큼만 유지한다.
recent_predictions: list[dict] = []

# 시뮬레이션 배치는 도착 여객 수만 주입하므로, 출발 여객 수는 이 고정값(명)으로 채운다.
# (대상 공항의 일평균 출발 여객 규모 - 실제 데이터 확정 후 평균값으로 갱신)
SIMULATED_DEPARTURES = 37_000


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = model_loader.get_model()
    sequence = [p.model_dump() for p in req.sequence]
    predicted_arrivals = model.predict_one(sequence)
    return PredictResponse(predicted_arrivals=round(predicted_arrivals, 2), model_version=model.version)


@router.post("/predict/batch-test", response_model=BatchTestResponse)
def batch_test(req: BatchTestRequest):
    """
    Day3 드리프트 감지 시뮬레이션 엔드포인트.

    scripts/simulate_drift.py 가 정상/드리프트 배치(SEQ_LEN+N개의 연속 일별 도착 여객 수)를
    이 엔드포인트로 전송합니다.

    TODO(Day3):
      1) req.arrivals 에서 길이 SEQ_LEN짜리 슬라이딩 윈도우를 만들어 각 윈도우 다음의
         실제 도착 여객 수(actual)를 예측(predicted)과 함께 얻으세요.
         (출발 여객 수는 SIMULATED_DEPARTURES 고정값을 사용하면 됩니다 - 실전 피처와 100% 동일하지
          않아도 시뮬레이션 목적에는 충분합니다.)
      2) 예측 결과를 {"predicted": ..., "actual": ...} 형태로 recent_predictions 에 누적하세요.
      3) check_and_trigger(recent_predictions) 를 호출해 드리프트 여부를 확인하세요.
    """
    model = model_loader.get_model()
    predictions: list[float] = []

    # --- 여기부터 TODO ---
    # arrivals = req.arrivals
    # for i in range(len(arrivals) - SEQ_LEN):
    #     window = arrivals[i : i + SEQ_LEN]
    #     sequence = [{"arrivals": a, "departures": SIMULATED_DEPARTURES} for a in window]
    #     pred = model.predict_one(sequence)
    #     actual = arrivals[i + SEQ_LEN]
    #     predictions.append(pred)
    #     recent_predictions.append({"predicted": pred, "actual": actual})
    # recent_predictions[:] = recent_predictions[-21:]  # WINDOW_SIZE 유지
    # --- 여기까지 TODO ---

    drift_check = check_and_trigger(recent_predictions)
    return BatchTestResponse(predictions=predictions, drift_check=drift_check)
