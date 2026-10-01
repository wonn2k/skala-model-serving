"""
Day1 -> Day3(시뮬레이션 엔드포인트 추가) 확장 파일.

Day1: POST /predict - 최근 SEQ_LEN(20)일 시퀀스로 다음날 도착 여객 수 예측
Day3: POST /predict/batch-test - 드리프트 감지 시뮬레이션 시작점 (scripts/simulate_drift.py 참고)
"""
from fastapi import APIRouter

from data.features import SEQ_LEN
from serving_app import model_loader
from serving_app.schemas import PredictRequest, PredictResponse, BatchTestRequest, BatchTestResponse
from serving_app.monitoring.drift_detector import WINDOW_SIZE
from serving_app.monitoring.retrain_trigger import check_and_trigger

router = APIRouter()

# Day3: 최근 예측 기록(actual/predicted)을 쌓아두는 슬라이딩 윈도우.
# monitoring/drift_detector.py의 WINDOW_SIZE(21)만큼만 유지한다.
recent_predictions: list[dict] = []

# 랜덤워크 시뮬레이션 배치는 도착 여객 수만 주입하므로, 출발 여객 수는 이 고정값(명)으로 채운다.
# 요청에 departures가 오면(실제 데이터 CSV 배치) 이 값 대신 그 값을 쓴다.
# (대상 공항의 일평균 출발 여객 규모 - 실제 데이터 확정 후 평균값으로 갱신)
SIMULATED_DEPARTURES = 37_000


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = model_loader.get_model()
    sequence = [p.model_dump() for p in req.sequence]
    predicted_arrivals = model.predict_one(sequence)
    return PredictResponse(
        predicted_arrivals=round(predicted_arrivals, 2),
        model_version=model.version,
        model_registry_version=model.registry_version,
    )


@router.post("/predict/batch-test", response_model=BatchTestResponse)
def batch_test(req: BatchTestRequest):
    """
    Day3 드리프트 감지 시뮬레이션 엔드포인트.

    scripts/simulate_drift.py 가 정상/드리프트 배치(SEQ_LEN+N개의 연속 일별 도착 여객 수)를
    이 엔드포인트로 전송합니다.

      1) req.arrivals 에서 길이 SEQ_LEN짜리 슬라이딩 윈도우를 만들어 각 윈도우 다음의
         실제 도착 여객 수(actual)를 예측(predicted)과 함께 얻는다.
         (출발 여객 수는 req.departures가 오면 같은 날짜의 실제값, 없으면
          SIMULATED_DEPARTURES 고정값 - 학습·/predict와 같은 피처로 판정하려면 함께 보낸다.)
      2) 예측 결과를 {"predicted": ..., "actual": ...} 형태로 recent_predictions 에 누적한다.
      3) check_and_trigger(recent_predictions) 로 드리프트 여부를 확인한다.
    """
    model = model_loader.get_model()
    predictions: list[float] = []

    arrivals = req.arrivals
    departures = req.departures if req.departures is not None else [SIMULATED_DEPARTURES] * len(arrivals)
    for i in range(len(arrivals) - SEQ_LEN):
        window = arrivals[i : i + SEQ_LEN]
        sequence = [{"arrivals": a, "departures": d} for a, d in zip(window, departures[i : i + SEQ_LEN])]
        pred = model.predict_one(sequence)
        actual = arrivals[i + SEQ_LEN]
        predictions.append(pred)
        recent_predictions.append({"predicted": pred, "actual": actual})
    recent_predictions[:] = recent_predictions[-WINDOW_SIZE:]  # WINDOW_SIZE 유지

    drift_check = check_and_trigger(recent_predictions)
    return BatchTestResponse(predictions=predictions, drift_check=drift_check)
