"""
Day1 -> Day2(MLflow 연동) 확장 파일.

Day1 실습 목표: Lazy Loading vs Eager Loading 두 방식을 직접 구현하고
서버 시작 시간 / 첫 요청 응답 시간을 비교합니다. (44번 슬라이드 결과표 참고)
LSTM은 로컬 pickle 모델보다 로딩 자체가 무거워서, 이 비교가 Day1보다 오히려
더 체감됩니다.

Day2 실습 목표: MODEL_SOURCE=mlflow 로 전환해, 로컬 .keras 파일 대신
MLflow Model Registry의 Production 버전을 로드하도록 확장합니다.
main.py / train_and_register.py 코드는 그대로 두고 이 파일만 손대면 되도록
설계되어 있습니다 - 이것이 "조립 블록" 구조입니다.

스케일러(scaler.pkl)는 Day1~3 내내 동일한 파일을 그대로 재사용합니다
(MODEL_SOURCE와 무관하게 항상 로컬 파일에서 로드) - 정규화 기준이 바뀌면
이미 그 기준으로 학습된 가중치와 어긋나기 때문입니다.

환경변수
    LOADING_MODE = lazy(기본값) | eager
    MODEL_SOURCE = local(기본값, Day1) | mlflow(Day2+)
    MLFLOW_TRACKING_URI = MODEL_SOURCE=mlflow 일 때 필요
"""
import os
import time
from threading import RLock

from data.features import AirportScaler

LOCAL_MODEL_PATH = "serving_app/models/airport_v1.keras"
SCALER_PATH = "serving_app/models/scaler.pkl"
MLFLOW_MODEL_URI = "models:/Airport_Arrivals_Predictor/Production"

_model_cache = None  # Lazy Loading 캐시
_cache_lock = RLock()  # 첫 로딩과 승격 후 초기화가 겹쳐 이전 모델이 다시 캐시되지 않게 한다.


class LoadedModel:
    """local .keras와 mlflow 두 소스를 동일한 인터페이스로 감싸는 래퍼."""

    def __init__(self, keras_model, scaler: AirportScaler, version: str, registry_version: str | None = None):
        self._keras_model = keras_model
        self.scaler = scaler
        self.version = version
        self.registry_version = registry_version

    def predict_one(self, sequence: list[dict]) -> float:
        """
        sequence: [{"arrivals": ..., "departures": ...}, ...] 길이 SEQ_LEN, 오래된 날 -> 최근 날 순서.
        """
        import numpy as np

        scaled = [self.scaler.transform_point(p["arrivals"], p["departures"]) for p in sequence]
        x = np.array([scaled], dtype="float32")  # (1, SEQ_LEN, 2)
        pred_scaled = float(self._keras_model.predict(x, verbose=0)[0][0])
        return self.scaler.inverse_arrivals(pred_scaled)


def _load_from_local() -> LoadedModel:
    from tensorflow import keras

    keras_model = keras.models.load_model(LOCAL_MODEL_PATH)
    scaler = AirportScaler.load(SCALER_PATH)
    return LoadedModel(keras_model=keras_model, scaler=scaler, version="v1-local")


def _load_from_mlflow() -> LoadedModel:
    """실습 2-1: Production을 실제 버전으로 해석해 모델과 응답 버전을 일치시킨다."""
    import mlflow.tensorflow
    from mlflow import MlflowClient

    model_uri, stage = MLFLOW_MODEL_URI.rsplit("/", 1)
    model_name = model_uri.removeprefix("models:/")
    versions = MlflowClient().get_latest_versions(model_name, stages=[stage])
    if not versions:
        raise RuntimeError(f"{model_name}에 {stage} 모델이 없습니다. 학습·게이트 통과 후 다시 시도하세요.")
    registry_version = str(max(versions, key=lambda v: int(v.version)).version)
    # 로딩 도중 Production이 바뀌더라도 보고한 번호와 실제 가중치는 같은 버전이어야 한다.
    keras_model = mlflow.tensorflow.load_model(f"{model_uri}/{registry_version}")
    scaler = AirportScaler.load(SCALER_PATH)  # Day1에서 fit한 고정 스케일러를 그대로 사용
    return LoadedModel(keras_model, scaler, version="production", registry_version=registry_version)


def _load_model() -> LoadedModel:
    source = os.getenv("MODEL_SOURCE", "local")
    if source == "mlflow":
        return _load_from_mlflow()
    return _load_from_local()


def load_eager() -> LoadedModel:
    """Eager Loading: 서버 시작 시점에 즉시 모델을 로드한다."""
    start = time.time()
    global _model_cache
    with _cache_lock:
        model = _load_model()
        _model_cache = model
        print(f"[eager] model loaded in {time.time() - start:.3f}s at startup")
        return model


def reset_cache() -> None:
    """B가 승격·롤백 성공 후 호출. 이 프로세스의 다음 예측에서 Production을 다시 읽는다.

    이미 모델을 받은 진행 중 요청은 그 모델로 마친다. 다중 worker의 캐시는 별개다.
    """
    global _model_cache
    with _cache_lock:
        _model_cache = None


def get_model() -> LoadedModel:
    """Lazy Loading: 첫 요청이 들어올 때만 로드하고, 이후에는 캐시를 재사용한다."""
    global _model_cache
    with _cache_lock:
        if _model_cache is None:
            start = time.time()
            _model_cache = _load_model()
            print(f"[lazy] model loaded in {time.time() - start:.3f}s on first request")
        return _model_cache
