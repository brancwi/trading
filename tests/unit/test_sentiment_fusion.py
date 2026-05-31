"""Tests unitaires pour FusionModel."""

import json

import numpy as np
import pytest

import trading.sentiment.fusion_model as fusion_module
from trading.sentiment.fusion_model import FusionModel, _label_to_score


pytestmark = pytest.mark.unit


class TestLabelToScore:
    def test_positive(self):
        assert _label_to_score("positive") == 1.0

    def test_negative(self):
        assert _label_to_score("negative") == -1.0

    def test_neutral(self):
        assert _label_to_score("neutral") == 0.0


class TestFusionModel:
    def test_untrained_fresh_instance(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fusion_module, "_WEIGHTS_PATH", tmp_path / "fusion.weights.json")
        model = FusionModel()
        assert model.trained is False

    def test_predict_untrained_mean(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fusion_module, "_WEIGHTS_PATH", tmp_path / "fusion.weights.json")
        model = FusionModel()
        result = model.predict(roberta=0.8, modern=0.2, qwen=None, lexical=None)
        assert result == 0.5

    def test_predict_untrained_no_scores(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fusion_module, "_WEIGHTS_PATH", tmp_path / "fusion.weights.json")
        model = FusionModel()
        result = model.predict(roberta=None, modern=None, qwen=None, lexical=None)
        assert result == 0.0

    def test_predict_clamps_result(self, tmp_path, monkeypatch):
        weights_file = tmp_path / "fusion.weights.json"
        weights_file.write_text(
            json.dumps(
                {
                    "weights": [10.0, 0.0, 0.0, 0.0, 0.0],
                    "trained": True,
                    "features": ["roberta", "modern", "qwen", "lexical"],
                }
            )
        )
        monkeypatch.setattr(fusion_module, "_WEIGHTS_PATH", weights_file)
        model = FusionModel()
        assert model.trained is True

        # dot([1, 0, 0, 0, 1], [10, 0, 0, 0, 0]) = 10 → clamped to 1.0
        result = model.predict(roberta=1.0, modern=0.0, qwen=0.0, lexical=0.0)
        assert result == 1.0

        # dot([-1, 0, 0, 0, 1], [10, 0, 0, 0, 0]) = -10 → clamped to -1.0
        result = model.predict(roberta=-1.0, modern=0.0, qwen=0.0, lexical=0.0)
        assert result == -1.0

    def test_info(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fusion_module, "_WEIGHTS_PATH", tmp_path / "fusion.weights.json")
        model = FusionModel()
        info = model.info()
        assert set(info.keys()) == {"trained", "features", "weights", "weights_file"}
