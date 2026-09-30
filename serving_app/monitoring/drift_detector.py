"""
Day3: RMSE 기반 데이터 드리프트 판정.

판단 기준 - 최근 WINDOW_SIZE(21)건의 (predicted, actual) 쌍으로 RMSE를 계산해
RMSE_THRESHOLD(2,700명)와 비교한다. 너무 짧은 윈도우는 노이즈에 민감하고,
너무 긴 윈도우는 드리프트 반응이 느려진다 - 21은 "최근 3주(21일)" 기준으로 정한 절충점.
(임계값은 배포 게이트 RMSE_GATE와 같은 값으로 두어, "게이트를 못 넘는 수준의 오차 = 드리프트"로 해석한다.)
"""
RMSE_THRESHOLD = 2700.0  # 명 - serving_app/train_and_register.py의 RMSE_GATE와 동일
WINDOW_SIZE = 21  # 최근 21건 기준


def compute_rmse(recent_predictions: list[dict]) -> float:
    """
    recent_predictions: [{"predicted": float, "actual": float}, ...]

    TODO(Day3, 핵심 실습): 아래 수식대로 RMSE를 직접 구현하세요.
        RMSE = sqrt( mean( (actual - predicted) ** 2 ) )

    빈 리스트가 들어오면 드리프트가 없다고 간주할 수 있도록 0.0을 반환하세요.
    """
    raise NotImplementedError("compute_rmse를 구현하세요 (실습 4-1)")


def is_drift(recent_predictions: list[dict]) -> bool:
    if len(recent_predictions) < WINDOW_SIZE:
        return False  # 아직 판단할 만큼 데이터가 쌓이지 않음
    window = recent_predictions[-WINDOW_SIZE:]
    rmse = compute_rmse(window)
    return rmse > RMSE_THRESHOLD
