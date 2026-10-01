"""
Day3: RMSE 기반 데이터 드리프트 판정.

판단 기준 - 최근 WINDOW_SIZE(21)건의 (predicted, actual) 쌍으로 RMSE를 계산해
RMSE_THRESHOLD(2,700명)와 비교한다. 너무 짧은 윈도우는 노이즈에 민감하고,
너무 긴 윈도우는 드리프트 반응이 느려진다 - 21은 "최근 3주(21일)" 기준으로 정한 절충점.
(임계값은 배포 게이트 RMSE_GATE와 같은 값으로 두어, "게이트를 못 넘는 수준의 오차 = 드리프트"로 해석한다.)

이상치와 드리프트 분리 - 폭설 결항일처럼 하루 오차가 ANOMALY_THRESHOLD를 넘는 날은 "이상치"로
따로 표시하고 드리프트 판정에서 뺀다. 이상치는 재학습으로 배울 수 없는 외부 충격이라 알림만 남기고,
남은 날들의 RMSE가 임계값을 넘을 때만 드리프트(수요 수준 변화)로 보아 재학습한다.
"""
import math

RMSE_THRESHOLD = 2700.0  # 명 - serving_app/train_and_register.py의 RMSE_GATE와 동일
WINDOW_SIZE = 21  # 최근 21건 기준
ANOMALY_THRESHOLD = 10_000.0  # 명 - 하루 오차가 이 값을 넘으면 이상치(결항 등). 평상시 최대 하루 오차(~8,000)보다 크게
BIAS_THRESHOLD = 1_500.0  # 명 - 평균 오차가 한쪽으로 이만큼 치우치면 수요 수준이 바뀐 것. 1,000/1,500/2,000 시뮬레이션 중 재학습 최소·RMSE 최저 (실험 5)


def compute_rmse(recent_predictions: list[dict]) -> float:
    """
    recent_predictions: [{"predicted": float, "actual": float}, ...]

    RMSE = sqrt( mean( (actual - predicted) ** 2 ) )

    빈 리스트가 들어오면 드리프트가 없다고 간주할 수 있도록 0.0을 반환한다.
    """
    if not recent_predictions:
        return 0.0
    squared_errors = [(p["actual"] - p["predicted"]) ** 2 for p in recent_predictions]
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def compute_bias(recent_predictions: list[dict]) -> float:
    """평균 오차(actual - predicted). 한쪽으로 치우치면 수요 수준이 바뀐 것(드리프트)이다."""
    if not recent_predictions:
        return 0.0
    return sum(p["actual"] - p["predicted"] for p in recent_predictions) / len(recent_predictions)


def assess(recent_predictions: list[dict]) -> dict:
    """최근 윈도우를 이상치 / 드리프트로 나눠 진단한다. 부작용 없음."""
    window = recent_predictions[-WINDOW_SIZE:]
    anomalies = [p for p in window if abs(p["actual"] - p["predicted"]) > ANOMALY_THRESHOLD]
    rest = [p for p in window if p not in anomalies]
    return {
        "rmse": compute_rmse(window),
        "bias": compute_bias(window),
        "anomalies": anomalies,
        "rmse_excl_anomalies": compute_rmse(rest),
        "bias_excl_anomalies": compute_bias(rest),
        # 드리프트 = 이상치를 뺀 오차가 한쪽으로 치우침(bias). RMSE >= |bias|라 RMSE 조건은 따로 두지 않는다.
        # 치우침 없는 큰 오차(RMSE > RMSE_THRESHOLD)는 변동이 큰 기간이지 수준 변화가 아니라 알림만 (retrain_trigger).
        "drift": len(window) >= WINDOW_SIZE and abs(compute_bias(rest)) > BIAS_THRESHOLD,
    }


def is_drift(recent_predictions: list[dict]) -> bool:
    if len(recent_predictions) < WINDOW_SIZE:
        return False  # 아직 판단할 만큼 데이터가 쌓이지 않음
    return assess(recent_predictions)["drift"]
