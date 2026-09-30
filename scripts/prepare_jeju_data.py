"""
한국공항공사 제주공항 일별 여객 원자료 -> 서빙 앱 업로드 형식(Date,Arrivals,Departures) 변환.

원자료: data/raw/kac_jeju_daily_passengers_20230101_20251031.csv (UTF-8로 재인코딩한 원본)
    컬럼: 운항일자, 도착여객(명), 출발여객(명), 전체여객(명)
출력:   data/jeju_airport_arrivals.csv
    컬럼: Date, Arrivals(=도착여객), Departures(=출발여객)   # 전체여객은 사용하지 않음

처리 규칙 (팀 가공본과 동일):
    - 도착·출발이 모두 0인 날(결측, 2023-01-24 1건)은 전날·다음날 평균으로 보간한다.
      (0을 그대로 두면 스케일러 min이 0으로 잡혀 정규화 범위가 왜곡되고, 학습 타깃에도 가짜 급락이 생김)
    - 그 외 값은 손대지 않는다. 폭설 결항 등 실제 급감일(예: 2025-02-07 6,088명)은 실데이터이므로 유지.

실행: python scripts/prepare_jeju_data.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_PATH = "data/raw/kac_jeju_daily_passengers_20230101_20251031.csv"
OUT_PATH = "data/jeju_airport_arrivals.csv"

RAW_DATE, RAW_ARR, RAW_DEP = "운항일자", "도착여객(명)", "출발여객(명)"


def main():
    with open(RAW_PATH, encoding="utf-8") as f:
        rows = [(r[RAW_DATE], int(r[RAW_ARR]), int(r[RAW_DEP])) for r in csv.DictReader(f)]

    out = []
    for i, (date, arr, dep) in enumerate(rows):
        if arr == 0 and dep == 0 and 0 < i < len(rows) - 1:
            arr = round((rows[i - 1][1] + rows[i + 1][1]) / 2)
            dep = round((rows[i - 1][2] + rows[i + 1][2]) / 2)
            print(f"[interpolate] {date}: 0,0 -> {arr},{dep} (이웃 평균)")
        out.append((date, arr, dep))

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Arrivals", "Departures"])
        w.writerows(out)
    print(f"saved {len(out)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
