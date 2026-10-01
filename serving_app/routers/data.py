"""
공항 도착 여객 데이터 업로드 - 데이터를 스크립트로 자동 생성하던 방식을 대체합니다.

/data 폴더는 이 라우터로 업로드된 CSV만 쌓이는 곳입니다(data/uploads/). 여러 번
업로드하면 계속 쌓이고, 학습(train_and_register.py, fine_tune 등)은 항상 가장
최근 파일 하나를 사용합니다(data/storage.py의 latest_upload()).

대시보드(static/index.html)에서 파일을 올리면 이 엔드포인트가 호출됩니다.
"""
import csv
import io
import os
import time

from fastapi import APIRouter, File, HTTPException, UploadFile

from data.features import SEQ_LEN, load_rows
from data.storage import UPLOAD_DIR, latest_upload
from serving_app.monitoring.drift_detector import WINDOW_SIZE

router = APIRouter(prefix="/data")

REQUIRED_COLUMNS = {"Date", "Arrivals", "Departures"}
MIN_ROWS = SEQ_LEN + WINDOW_SIZE  # 시퀀스 구성 + 드리프트 판정 윈도우에 필요한 최소 행 수


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "UTF-8로 인코딩된 CSV 파일만 업로드할 수 있습니다.")

    reader = csv.DictReader(io.StringIO(text))
    if not REQUIRED_COLUMNS.issubset(set(reader.fieldnames or [])):
        raise HTTPException(400, f"CSV에 {sorted(REQUIRED_COLUMNS)} 컬럼이 모두 있어야 합니다.")
    rows = list(reader)
    if len(rows) < MIN_ROWS:
        raise HTTPException(400, f"최소 {MIN_ROWS}행 이상의 데이터가 필요합니다.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, f"airport_{int(time.time())}.csv")
    with open(dest, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    return {"filename": os.path.basename(dest), "rows": len(rows)}


def _as_count(value: float) -> int | None:
    """여객 수를 정수로 바꾼다. 바꿀 수 없으면 None.

    load_rows가 float로 읽어 오는데 CSV에 "nan"이 들어 있으면 float("nan")은 통과하고
    int(nan)에서 ValueError가 난다. 조회 엔드포인트가 그걸로 500이 되면 대시보드가
    아예 안 뜨므로, 그 칸만 null로 두고 나머지는 그대로 보여 준다.
    """
    try:
        return int(round(value))
    except (ValueError, OverflowError, TypeError):
        return None


@router.get("/status")
def status():
    try:
        path = latest_upload()
    except FileNotFoundError:
        return {"exists": False}

    rows = load_rows(path)
    arrivals = [r["Arrivals"] for r in rows]

    # recent: 대시보드가 "내일 예측"을 띄우려면 최근 SEQ_LEN(20)일을 그대로 /predict에
    # 실어 보내야 한다. 프론트가 CSV를 다시 읽지 않아도 되도록 여기서 잘라 준다.
    # 순서는 오래된 날 -> 최근 날. PredictRequest.sequence가 요구하는 순서와 같다.
    recent = [
        {
            "date": r["Date"],
            "arrivals": _as_count(r["Arrivals"]),
            "departures": _as_count(r["Departures"]),
        }
        for r in rows[-SEQ_LEN:]
    ]

    return {
        "exists": True,
        "filename": os.path.basename(path),
        "rows": len(rows),
        "start_date": rows[0]["Date"],
        "end_date": rows[-1]["Date"],
        "min_arrivals": min(arrivals),
        "max_arrivals": max(arrivals),
        "recent": recent,
    }
