"""Sentiment Engine - analyse multi-modèles GPU."""

import json
import logging
import threading
from typing import Any

from sqlalchemy.orm import Session

from trading.core.config import get_settings
from trading.core.models import News, SentimentScore, Signal, SignalAction

logger = logging.getLogger(__name__)
settings = get_settings()

# Lazy import des modèles pour éviter le chargement au démarrage API
_transformers: Any = None
_torch: Any = None
_model_lock = threading.Lock()


def _load_transformers():
    global _transformers, _torch
    if _transformers is None:
        import transformers as tr
        import torch as to
        _transformers = tr
        _torch = to
    return _transformers, _torch


class SentimentAnalyzer:
    """Analyseur FinancialBERT + RoBERTa avec fusion pondérée."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.device = settings.ml_device if settings.ml_device != "cpu" else "cpu"
        self.finbert_name = settings.ml_model_finbert
        self.roberta_name = settings.ml_model_roberta
        self.tokenizer_fb = None
        self.model_fb = None
        self.tokenizer_rb = None
        self.model_rb = None
        self._initialized = True

    def load_models(self):
        """Charge les modèles en VRAM (appel explicite requis, thread-safe)."""
        if self.model_fb is not None:
            return
        with _model_lock:
            if self.model_fb is not None:
                return
            transformers, torch = _load_transformers()
            logger.info("Chargement FinancialBERT...")
            self.tokenizer_fb = transformers.AutoTokenizer.from_pretrained(self.finbert_name)
            self.model_fb = transformers.AutoModelForSequenceClassification.from_pretrained(
                self.finbert_name
            ).to(self.device).eval()
            logger.info("Chargement RoBERTa...")
            self.tokenizer_rb = transformers.AutoTokenizer.from_pretrained(self.roberta_name)
            self.model_rb = transformers.AutoModelForSequenceClassification.from_pretrained(
                self.roberta_name
            ).to(self.device).eval()
            logger.info("Modèles chargés en VRAM")

    def _infer(self, text: str, tokenizer, model) -> float:
        transformers, torch = _load_transformers()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # Hypothèse: 0=négatif, 1=neutre, 2=positif pour la plupart des modèles sentiment
        if probs.shape[-1] == 3:
            score = (probs[0][2] - probs[0][0]).item()
        else:
            score = probs[0][1].item() * 2 - 1
        return float(score)

    def analyze_text(self, text: str) -> dict[str, float]:
        """Analyse un texte et retourne les scores."""
        self.load_models()
        finbert = self._infer(text, self.tokenizer_fb, self.model_fb)
        roberta = self._infer(text, self.tokenizer_rb, self.model_rb)
        combined = 0.7 * finbert + 0.3 * roberta
        confidence = 1.0 - abs(finbert - roberta)
        anomaly = abs(finbert - roberta) > 0.5
        return {
            "finbert": round(finbert, 4),
            "roberta": round(roberta, 4),
            "combined": round(combined, 4),
            "confidence": round(confidence, 4),
            "anomaly": anomaly,
        }

    def process_unprocessed_news(self, db: Session) -> int:
        """Analyse toutes les news non traitées et génère des signaux."""
        self.load_models()
        news_items = db.query(News).filter(News.processed == 0).all()
        count = 0
        for item in news_items:
            text = f"{item.title}. {item.description or ''}"
            result = self.analyze_text(text)
            sentiment = SentimentScore(
                news_id=item.id,
                ticker=item.ticker,
                finbert_score=result["finbert"],
                roberta_score=result["roberta"],
                combined_score=result["combined"],
                confidence=result["confidence"],
                anomaly_flag=int(result["anomaly"]),
            )
            db.add(sentiment)
            item.processed = 1
            # Génération automatique de signal si fort sentiment
            if abs(result["combined"]) >= 0.5 and result["confidence"] >= 0.6:
                action = (
                    SignalAction.STRONG_BUY if result["combined"] > 0.6
                    else SignalAction.BUY if result["combined"] > 0
                    else SignalAction.STRONG_SELL if result["combined"] < -0.6
                    else SignalAction.SELL
                )
                signal = Signal(
                    ticker=item.ticker,
                    action=action.value,
                    sentiment=result["combined"],
                    strength=min(abs(result["combined"]) * 1.4, 1.0),
                    confidence=result["confidence"],
                    source="sentiment_engine",
                )
                db.add(signal)
                count += 1
        db.commit()
        logger.info(f"{len(news_items)} news analysées, {count} signaux générés")
        return count
