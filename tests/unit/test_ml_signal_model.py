"""Tests unitaires pour SignalModel."""

import pytest

import trading.ml.signal_model as signal_model_module
from trading.ml.signal_model import SignalModel


pytestmark = pytest.mark.unit


class TestSignalModel:
    def test_untrained_fresh_instance(self, tmp_path, monkeypatch):
        monkeypatch.setattr(signal_model_module, "_MODEL_PATH", tmp_path / "model.joblib")
        monkeypatch.setattr(signal_model_module, "_META_PATH", tmp_path / "model.meta.json")
        model = SignalModel()
        assert model.trained is False

    def test_predict_untrained(self, tmp_path, monkeypatch):
        monkeypatch.setattr(signal_model_module, "_MODEL_PATH", tmp_path / "model.joblib")
        monkeypatch.setattr(signal_model_module, "_META_PATH", tmp_path / "model.meta.json")
        model = SignalModel()
        result = model.predict(ticker="AAPL")
        assert result == {"action": "HOLD", "confidence": 0.0, "model_trained": False}

    @pytest.mark.slow
    def test_train_insufficient_data(self, tmp_path, monkeypatch):
        monkeypatch.setattr(signal_model_module, "_MODEL_PATH", tmp_path / "model.joblib")
        monkeypatch.setattr(signal_model_module, "_META_PATH", tmp_path / "model.meta.json")
        model = SignalModel()
        result = model.train()
        assert result == {"trained": False, "reason": "insufficient_data"}

    def test_info(self, tmp_path, monkeypatch):
        monkeypatch.setattr(signal_model_module, "_MODEL_PATH", tmp_path / "model.joblib")
        monkeypatch.setattr(signal_model_module, "_META_PATH", tmp_path / "model.meta.json")
        model = SignalModel()
        info = model.info()
        assert set(info.keys()) == {"trained", "feature_cols", "model_path", "meta_path"}
