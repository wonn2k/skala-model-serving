"""
한국공항공사 제주공항 일별 여객 원자료 -> 서빙 앱 업로드 형식(Date,Arrivals,Departures) 변환.

원자료: data/raw/kac_jeju_daily_passengers_20230101_20251031.csv (UTF-8로 재인코딩한 원본)
    컬럼: 운항일자, 도착여객(명), 출발여객(명), 전체여객(명)
출력:   data/jeju_airport_arrivals.csv
    컬럼: Date, Arrivals(=도착여객), Departures(=출발여객)   # 전체여객은 사용하지 않음

처리 규칙:
    - 도착 여객이 CANCEL_ARRIVALS(25,000명) 미만인 날(결측 0 포함, 폭설·강풍 결항일)은 양옆의 정상일 평균으로 보간한다.
      결항일이 이어지면 그 바깥의 정상일을 쓴다.
      이유: 결항일을 두면 (1) 스케일러 min이 1,042로 잡혀 정상 범위 30,000~45,000이 0.63~0.96으로 압축되고
      (2) 결항일 하나의 제곱오차가 평상시 수백 일치보다 커서 모델이 평균만 내는 쪽으로 수렴한다.
      보간 후 같은 코드로 학습하면 평상시 RMSE 2,415 → 1,807, 예측 표준편차 445 → 1,898 (docs/code_current.md 5번).
    - 결항일의 실제 값은 data/raw/ 원자료와 data/jeju_drift_batch_41rows.csv(이상 탐지 시연용)에 남아 있다.
      모니터링은 raw 입력을 받으며, 하루 오차 > 10,000은 이상치로 따로 처리한다 (serving_app/monitoring/drift_detector.py).

실행: python scripts/prepare_jeju_data.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_PATH = "data/raw/kac_jeju_daily_passengers_20230101_20251031.csv"
OUT_PATH = "data/jeju_airport_arrivals.csv"

RAW_DATE, RAW_ARR, RAW_DEP = "운항일자", "도착여객(명)", "출발여객(명)"

CANCEL_ARRIVALS = 25_000  # 이 미만이면 결항일로 보고 보간 (정상 범위 30,000~45,000, 결항일 1,042~24,000)


def interpolate_cancellations(rows, threshold=CANCEL_ARRIVALS):
    """rows: [(date, arr, dep)]. arr < threshold 인 날을 양옆의 가장 가까운 정상일 평균으로 바꾼다."""
    normal = [i for i, (_, arr, _) in enumerate(rows) if arr >= threshold]
    out = []
    for i, (date, arr, dep) in enumerate(rows):
        if arr < threshold:
            prev = max((j for j in normal if j < i), default=None)
            nxt = min((j for j in normal if j > i), default=None)
            nb = [rows[j] for j in (prev, nxt) if j is not None]
            arr, dep = (round(sum(r[1] for r in nb) / len(nb)), round(sum(r[2] for r in nb) / len(nb)))
            print(f"[interpolate] {date}: {rows[i][1]},{rows[i][2]} -> {arr},{dep}")
        out.append((date, arr, dep))
    return out


def main():
    with open(RAW_PATH, encoding="utf-8") as f:
        rows = [(r[RAW_DATE], int(r[RAW_ARR]), int(r[RAW_DEP])) for r in csv.DictReader(f)]

    out = interpolate_cancellations(rows)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Arrivals", "Departures"])
        w.writerows(out)
    print(f"saved {len(out)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
