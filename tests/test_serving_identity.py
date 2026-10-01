"""A 로더와 C 버전 조회, B 배치 판정 창의 연결 회귀 검사.

실행: python -m unittest discover -s tests -v
모델 로딩/학습은 대체하며 실제 Registry는 변경하지 않는다.
"""
import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from serving_app import model_loader as loader
from serving_app.routers import monitor, predict
from serving_app.schemas import BatchTestRequest


@contextmanager
def registry_state(production):
    # C 내부 helper의 반환 형식 대신 외부 MLflow 경계를 대체한다.
    with patch("mlflow.tracking.MlflowClient") as client:
        client.return_value.search_model_versions.side_effect = lambda *_: [
            SimpleNamespace(**production, current_stage="Production", creation_timestamp=0)
        ]
        client.return_value.get_run.return_value = SimpleNamespace(
            data=SimpleNamespace(metrics={"rmse": 1.})
        )
        yield client


class ServingIdentityTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"MODEL_SOURCE": "mlflow"})
        self.environment.start()
        loader.reset_cache()

    def tearDown(self):
        loader.reset_cache()
        self.environment.stop()

    def test_identity_stays_with_loaded_version_when_production_changes(self):
        versions = [SimpleNamespace(version="2", run_id="run-2"),
                    SimpleNamespace(version="11", run_id="run-11")]

        def load_weights(uri):
            self.assertEqual(uri, "models:/Airport_Arrivals_Predictor/11")
            versions[:] = [SimpleNamespace(version="12", run_id="run-12")]
            return "weights-11"

        with patch("mlflow.MlflowClient") as client, \
                patch("mlflow.tensorflow.load_model", side_effect=load_weights), \
                patch.object(loader.AirportScaler, "load", return_value="fixed-scaler"):
            client.return_value.get_latest_versions.return_value = versions
            model = loader.get_model()
        self.assertEqual((model.registry_version, model.run_id), ("11", "run-11"))
        self.assertEqual(model._keras_model, "weights-11")

    def test_monitor_detects_promotion_and_clears_stale_after_reset_and_rollback(self):
        old = loader.LoadedModel(None, None, "production", "1", "run-1")
        new = loader.LoadedModel(None, None, "production", "2", "run-2")
        production = {"version": "1", "run_id": "run-1"}
        with registry_state(production), \
                patch.object(loader, "_load_model", side_effect=[old, new, old]) as load:
            self.assertIsNone(monitor.versions()["stale"])
            load.assert_not_called()  # 조회는 lazy 모델을 강제로 로드하지 않는다.
            loader.get_model()
            self.assertFalse(monitor.versions()["stale"])
            production.update(version="2", run_id="run-2")
            self.assertTrue(monitor.versions()["stale"])
            loader.reset_cache()
            self.assertIsNone(monitor.versions()["stale"])
            loader.get_model()
            self.assertFalse(monitor.versions()["stale"])
            self.assertEqual(monitor.versions()["serving_run_id"], "run-2")
            production.update(version="1", run_id="run-1")
            self.assertTrue(monitor.versions()["stale"])
            loader.reset_cache()
            loader.get_model()
            self.assertFalse(monitor.versions()["stale"])

    def test_local_model_has_no_run_and_does_not_query_registry(self):
        model = loader.LoadedModel(None, None, "v1-local")
        self.assertIsNone(model.run_id)
        with patch.dict(os.environ, {"MODEL_SOURCE": "local"}), \
                patch.object(loader, "_load_model", return_value=model), \
                patch("mlflow.tracking.MlflowClient") as registry:
            loader.get_model()
            self.assertFalse(monitor.versions()["stale"])
            registry.assert_not_called()

    def test_registration_without_run_remains_unknown(self):
        with patch("mlflow.MlflowClient") as client, \
                patch("mlflow.tensorflow.load_model", return_value=None), \
                patch.object(loader.AirportScaler, "load", return_value=None):
            client.return_value.get_latest_versions.return_value = [SimpleNamespace(version="1")]
            self.assertIsNone(loader.get_model().run_id)
        with registry_state({"version": "1", "run_id": "other"}):
            self.assertIsNone(monitor.versions()["stale"])


class BatchWindowTests(unittest.TestCase):
    def setUp(self):
        self.original = list(predict.recent_predictions)
        predict.recent_predictions[:] = [{"predicted": 1., "actual": 1.}] * 21

    def tearDown(self):
        predict.recent_predictions[:] = self.original

    def send(self, count):
        model = SimpleNamespace(predict_one=lambda sequence: 37000.)
        with patch.object(loader, "get_model", return_value=model), \
                patch.object(predict, "check_and_trigger", return_value={}) as check:
            response = predict.batch_test(BatchTestRequest(arrivals=[37000.] * count))
            return response, list(check.call_args.args[0])

    def test_41_rows_replace_all_previous_predictions(self):
        response, window = self.send(41)
        self.assertEqual(len(response.predictions), 21)
        self.assertEqual(len(window), 21)
        self.assertTrue(all(p["actual"] == 37000. for p in window))

    def test_short_batch_intentionally_keeps_rolling_history(self):
        response, window = self.send(21)
        self.assertEqual(len(response.predictions), 1)
        self.assertEqual(sum(p["actual"] == 1. for p in window), 20)
        self.assertEqual(window[-1]["actual"], 37000.)


if __name__ == "__main__":
    unittest.main()
