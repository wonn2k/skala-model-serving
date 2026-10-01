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

    처음에는 둘이 어긋났는지 판정할 수 없었다. LoadedModel이 version 문자열("production")만
    들고 있어서 몇 번 버전인지 몰랐기 때문이다. A가 LoadedModel.registry_version을 추가해
    줘서(PR #5) 이제 레지스트리의 Production 번호와 직접 비교해 stale을 판정한다.
    이것이 수업 가이드 부록1의 6번(승격했는데 서버는 옛 모델로 응답) 재현 지점이다.

조회 실패를 숨기지 않는다
    처음엔 실패를 모두 production: null로 뭉갰는데, 그러면 네 가지 상황이 똑같이 보인다.
    (1) 아직 등록된 모델이 없음  (2) 트래킹 URI 오타  (3) 트래킹 서버가 죽음  (4) mlflow 미설치.
    (1)은 정상이고 (2)(3)은 장애다. 구분이 안 되면 "Production이 없네"로 읽고 멀쩡한 모델을
    롤백하는 판단을 할 수 있다. 그래서 tracking_uri와 registry_error를 함께 돌려준다.

    둘이 잡는 범위가 다르다. registry_error는 조회가 예외로 실패한 (3)(4)를 잡는다.
    (2)는 registry_error로 안 잡힌다 - sqlite URI를 오타 내면 MLflow가 그 이름으로 빈 DB를
    새로 만들어 조회가 "성공"하고 0건을 돌려주기 때문이다(실측: production null,
    registry_error null). 그래서 tracking_uri를 같이 노출한다. 기대한 저장소를 보고 있는지는
    사람이 그 값을 보고 판단한다.

타임아웃
    트래킹 서버가 응답하지 않을 때 MLflow 기본값(요청 120초 + 재시도 7회)으로는 이 응답이
    몇 분씩 막힌다. 이 핸들러는 sync def라 그동안 AnyIO 스레드풀 슬롯(기본 40개)을 하나
    물고 있어 /predict와 /health까지 느려질 수 있다. 그래서 모듈 로드 시점에 짧은 기본값을
    심는다. setdefault라서 운영에서 환경변수로 올려 잡으면 그 값이 이긴다.
    응답 없는 주소로 실측: 재시도 1회 10.68초 -> 재시도 없음 5초대. 연결이 거부되는
    경우는 원래 빨라서 0.54초였다.

확인 방법
    MODEL_SOURCE=mlflow uvicorn serving_app.main:app --port 8077
    curl localhost:8077/monitor/versions
"""
import os

from fastapi import APIRouter

from serving_app import model_loader

# MLflow는 이 두 값을 요청할 때마다 환경변수에서 읽는다. 조회용 엔드포인트가 서빙을
# 붙잡고 있지 않도록 짧게 잡는다 (기본 120초 + 재시도 7회 -> 5초 + 재시도 없음).
# 조회 전용이라 재시도 가치가 낮다. 실패하면 registry_error를 보고 새로 고치면 된다.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")

router = APIRouter(prefix="/monitor")

MODEL_NAME = "Airport_Arrivals_Predictor"


def _registry_production() -> tuple[dict | None, str | None]:
    """MLflow Registry에서 Production 단계 버전을 찾는다.

    돌려주는 값은 (production, error) 쌍이다.
        (dict, None)  Production 버전을 찾음
        (None, None)  조회는 됐지만 Production이 없음 - 정상 상태
        (None, str)   조회 자체가 실패 - 장애. 메시지로 원인을 구분한다

    조회 실패로 서빙이 멈추면 안 되므로 예외를 밖으로 내보내지는 않는다.
    """
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return None, "mlflow가 설치돼 있지 않습니다"

    try:
        client = MlflowClient()
        versions = [
            v for v in client.search_model_versions(f"name='{MODEL_NAME}'")
            if getattr(v, "current_stage", None) == "Production"
        ]
    except Exception as e:
        # 트래킹 URI 오타, 서버 다운, 권한 등. 타입과 메시지를 그대로 노출해야
        # "모델이 없다"와 구분된다.
        return None, f"{type(e).__name__}: {e}"

    if not versions:
        return None, None

    latest = max(versions, key=lambda v: int(v.version))

    rmse = None
    try:
        rmse = client.get_run(latest.run_id).data.metrics.get("rmse")
    except Exception:
        # 지표를 못 읽어도 버전 정보는 쓸 수 있으므로 rmse만 비운다.
        pass

    return {
        "version": latest.version,
        "run_id": latest.run_id,
        "rmse": rmse,
        "created_at": getattr(latest, "creation_timestamp", None),
    }, None


@router.get("/versions")
def versions():
    """지금 서비스 중인 모델과 레지스트리 상태를 함께 돌려준다."""
    source = os.getenv("MODEL_SOURCE", "local")

    # 서버가 실제로 들고 있는 것. 아직 로드 전이면(lazy 첫 요청 전) None이다.
    # _model_cache는 A의 파일에 있는 내부 이름이라 getattr로 감싼다. 이름이 바뀌어도
    # 조회 엔드포인트가 500으로 죽지 않고 null을 돌려주게 한다.
    cached = getattr(model_loader, "_model_cache", None)
    serving_version = getattr(cached, "version", None) if cached is not None else None

    if source != "mlflow":
        return {
            "model_name": MODEL_NAME,
            "model_source": source,
            "tracking_uri": None,
            "production": None,
            "registry_error": None,
            "serving_version": serving_version,
            "serving_registry_version": None,
            "stale": False,  # 로컬 모델은 레지스트리와 비교할 대상이 없다
        }

    production, registry_error = _registry_production()

    # 서버가 들고 있는 모델의 레지스트리 번호. lazy 모드에서 첫 요청 전이면 None이다.
    serving_registry_version = (
        getattr(cached, "registry_version", None) if cached is not None else None
    )

    # stale 판정. 비교할 양쪽이 다 있을 때만 True/False를 내고, 하나라도 없으면 null로 둬서
    # "어긋났다"와 "모른다"를 구분한다. 번호는 문자열로 올 수 있어 str로 맞춰 비교한다.
    stale = None
    if production is not None and serving_registry_version is not None:
        stale = str(serving_registry_version) != str(production["version"])

    return {
        "model_name": MODEL_NAME,
        "model_source": source,
        # 어느 저장소를 본 결과인지. 오타난 URI로 빈 DB를 보고 있는 경우를 눈으로 잡는다.
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
        "production": production,
        "registry_error": registry_error,
        "serving_version": serving_version,
        "serving_registry_version": serving_registry_version,
        "stale": stale,
    }
