"""
공항 다음날 도착 여객 수 예측용 LSTM 아키텍처 (Day1 baseline과 Day2 MLflow 학습이 공유).

3년치(~1,000일) 일별 데이터 + SEQ_LEN(20)을 적용하면 학습 시퀀스가 수백 개 이상 확보되어,
파라미터 대비 샘플 비율이 충분합니다. 그래서 LSTM 3층
(32 -> 32 -> 16, 앞 두 층은 return_sequences=True로 다음 LSTM에 전체 시퀀스를 넘김) +
Dense 1층 구조를 택했습니다 - 이 정도 크기(파라미터 약 1.6만 개)는 CPU로 50 epoch을
학습해도 수십 초~1분 내외면 끝납니다.
"""
from tensorflow import keras

from data.features import SEQ_LEN, N_FEATURES  # (arrivals, departures) + 다음 날 달력 4개


def build_model() -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(SEQ_LEN, N_FEATURES)),
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.LSTM(16),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model
