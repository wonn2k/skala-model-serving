"""
Day2~3: 지금 무엇이 서비스되고 있는지 보여주는 조회 전용 엔드포인트.

왜 필요한가
    배포 담당이 실제로 자주 하는 일은 학습이 아니라 조회다. 지금 Production이 몇 번인지,
    어떤 지표로 올라갔는지 알아야 롤백 판단이 된다.

    그리고 레지스트리와 서빙이 서로 다른 말을 할 수 있다. 재학습이 성공해 v2가 Production이
    되어도, 서버는 메모리에 들고 있던 v1으로 계속 응답한다(model_loader._model_cache).
    로그에는 [OK] promoted가 찍혀 있어서 로그만 보면 반영된 줄 안다.
    그래서 이 엔드포인트는 "레지스트리가 뭐라고 하는가"와 "서버가 실제로 뭘 쓰고 있는가"를
    나란히 돌려준다.

    다만 지금은 둘이 어긋났는지 자동으로 판정할 수 없다. LoadedModel이 version 문자열
    ("production")만 들고 있고 어느 run에서 왔는지는 모르기 때문이다. 판정하려면
    model_loader가 run_id를 함께 보관해야 하는데 그 파일은 A의 소유라, code_current.md의
    "다른 영역에 요청"에 적어 두었다. 그때까지는 serving_run_id를 null로 돌려준다.

확인 방법
    MODEL_SOURCE=mlflow uvicorn serving_app.main:app --port 8077
    curl localhost:8077/monitor/versions
"""
import os

from fastapi import APIRouter

from serving_app import model_loader

router = APIRouter(prefix="/monitor")

MODEL_NAME = "Airport_Arrivals_Predictor"


def _registry_production() -> dict | None:
    """MLflow Registry에서 Production 단계 버전을 찾는다.

    MLflow가 설치돼 있지 않거나 등록된 모델이 없으면 None. 조회 실패로 서빙이
    멈추면 안 되므로 예외를 밖으로 내보내지 않는다.
    """
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return None

    try:
        client = MlflowClient()
        versions = [
            v for v in client.search_model_versions(f"name='{MODEL_NAME}'")
            if getattr(v, "current_stage", None) == "Production"
        ]
        if not versions:
            return None
        latest = max(versions, key=lambda v: int(v.version))

        rmse = None
        try:
            rmse = client.get_run(latest.run_id).data.metrics.get("rmse")
        except Exception:
            pass

        return {
            "version": latest.version,
            "run_id": latest.run_id,
            "rmse": rmse,
            "created_at": getattr(latest, "creation_timestamp", None),
        }
    except Exception:
        return None


@router.get("/versions")
def versions():
    """지금 서비스 중인 모델과 레지스트리 상태를 함께 돌려준다."""
    source = os.getenv("MODEL_SOURCE", "local")

    # 서버가 실제로 들고 있는 것. 아직 로드 전이면(lazy 첫 요청 전) None이다.
    cached = model_loader._model_cache
    serving_version = cached.version if cached is not None else None

    if source != "mlflow":
        return {
            "model_name": MODEL_NAME,
            "model_source": source,
            "production": None,
            "serving_version": serving_version,
            "serving_run_id": None,
            "stale": False,  # 로컬 모델은 레지스트리와 비교할 대상이 없다
        }

    production = _registry_production()

    # 서버가 어느 run의 모델을 들고 있는지. LoadedModel이 아직 run_id를 보관하지 않아
    # 지금은 항상 None이다. A가 추가하면 production["run_id"]와 비교해 stale을 판정할 수 있다.
    serving_run_id = getattr(cached, "run_id", None) if cached is not None else None

    stale = None  # 판정 불가. True/False가 아니라 null로 둬서 "모른다"를 구분한다
    if production is not None and serving_run_id is not None:
        stale = serving_run_id != production["run_id"]

    return {
        "model_name": MODEL_NAME,
        "model_source": source,
        "production": production,
        "serving_version": serving_version,
        "serving_run_id": serving_run_id,
        "stale": stale,
    }
