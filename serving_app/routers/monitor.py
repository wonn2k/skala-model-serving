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
    들고 있어서 몇 번 버전인지 몰랐기 때문이다. A가 registry_version(PR #5)과 run_id(PR #9)를
    추가해 줘서 이제 실제로 판정한다.

    stale이 true가 되는 경우와 안 되는 경우
        이 서버 자신이 재학습해서 승격한 경우에는 true가 뜨지 않는다. retrain_trigger가
        승격·롤백 직후 model_loader.reset_cache()를 부르기 때문에 캐시가 비고, 다음 예측이
        새 버전을 읽는다(그 사이에는 비교 대상이 없어 null이다). 즉 자동 재학습 흐름은
        이미 어긋남을 스스로 막는다.
        true가 뜨는 것은 승격이 이 서버 밖에서 일어났을 때다. MLflow UI에서 손으로 승격했거나,
        별도 배치·다른 인스턴스가 승격했거나, 여러 서버를 띄워 둔 경우다. 실제 운영에서
        캐시가 썩는 건 이쪽이고, 이 엔드포인트가 필요한 이유도 이쪽이다.
        실측: 컨테이너 안에서 train_and_register.py를 따로 실행해 v2를 승격시키자
        production v2 / serving v1 / stale true가 떴고, 재기동 후 false로 돌아왔다.

    판정은 run_id를 먼저 본다. 번호보다 좁은 식별자라서, 같은 번호가 다른 run을 가리키게
    다시 등록된 경우까지 잡는다. run_id가 없으면(로더가 아직 안 채웠거나 옛 버전) 번호로
    비교한다. 양쪽 중 하나라도 없으면 null로 둬서 "어긋났다"와 "모른다"를 구분한다.

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
            "serving_run_id": None,
            "stale": False,  # 로컬 모델은 레지스트리와 비교할 대상이 없다
            "stale_basis": None,
        }

    production, registry_error = _registry_production()

    # 서버가 들고 있는 모델의 신원. lazy 모드에서 첫 요청 전이면 둘 다 None이다.
    serving_registry_version = (
        getattr(cached, "registry_version", None) if cached is not None else None
    )
    serving_run_id = getattr(cached, "run_id", None) if cached is not None else None

    # stale 판정. run_id가 양쪽에 다 있으면 그걸로 끝난다.
    #
    # run_id가 없을 때는 번호로 비교하는데, 한쪽으로만 결론을 낸다.
    #   번호가 다르다  -> 분명히 다른 모델이다. true로 단정해도 된다.
    #   번호가 같다    -> 같다고 단정할 수 없다. 같은 번호가 다른 run을 가리키게
    #                     다시 등록됐을 수 있다. 확인할 방법이 없으므로 null로 둔다.
    #
    # 전에는 번호가 같으면 false를 돌려줬는데 그건 "확인했고 문제 없다"는 뜻이라
    # 근거 없는 안심을 준다. 이 엔드포인트는 조용히 어긋난 걸 잡으려고 만든 것이라
    # 모르는 걸 모른다고 하는 쪽이 맞다 (A의 tests/test_serving_identity.py 기준).
    stale = None
    stale_basis = None
    if production is not None:
        if serving_run_id is not None and production["run_id"] is not None:
            stale = serving_run_id != production["run_id"]
            stale_basis = "run_id"
        elif serving_registry_version is not None:
            if str(serving_registry_version) != str(production["version"]):
                stale = True
                stale_basis = "registry_version"

    return {
        "model_name": MODEL_NAME,
        "model_source": source,
        # 어느 저장소를 본 결과인지. 오타난 URI로 빈 DB를 보고 있는 경우를 눈으로 잡는다.
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
        "production": production,
        "registry_error": registry_error,
        "serving_version": serving_version,
        "serving_registry_version": serving_registry_version,
        "serving_run_id": serving_run_id,
        "stale": stale,
        # 무엇을 근거로 판정했는지. null이면 판정하지 않았다는 뜻이다.
        "stale_basis": stale_basis,
    }
