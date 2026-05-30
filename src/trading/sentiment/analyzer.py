"""Sentiment Engine v2 — analyse multi-modèles à 4 tiers avec arbitre LLM."""

import json
import logging
import threading
from typing import Any

from sqlalchemy.orm import Session

from trading.core.config import get_settings
from trading.core.models import News, SentimentScore, Signal, SignalAction
from trading.sentiment.cloud_fallback import CloudFallback
from trading.sentiment.lexical_rules import apply_lexical_rules, extract_financial_keywords

logger = logging.getLogger(__name__)
settings = get_settings()

# Lazy import des librairies lourdes
_transformers: Any = None
_torch: Any = None
_model_lock = threading.Lock()


def _load_ml_libs():
    global _transformers, _torch
    if _transformers is None:
        import torch as to
        import transformers as tr
        _transformers = tr
        _torch = to
    return _transformers, _torch


def _label_to_score(label: str) -> float:
    label = label.lower().strip()
    if "positive" in label:
        return 1.0
    if "negative" in label:
        return -1.0
    return 0.0


class SentimentAnalyzerV2:
    """Analyseur à 4 tiers : lexical → RoBERTa + ModernFinBERT → Qwen → Cloud."""

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

        # Modèles
        self.roberta_name = settings.ml_model_roberta
        self.modern_name = settings.ml_model_modern
        self.qwen_name = settings.ml_model_qwen

        # Tokenizers & modèles
        self._tk_roberta = None
        self._md_roberta = None
        self._tk_modern = None
        self._md_modern = None
        self._tk_qwen = None
        self._md_qwen = None

        # Mappings labels → index (détectés au chargement)
        self._label_map_roberta: dict[str, int] = {}
        self._label_map_modern: dict[str, int] = {}

        # Fallback cloud
        api_key = settings.openai_api_key or settings.anthropic_api_key
        self._cloud = CloudFallback(
            api_key=api_key or None,
            provider=settings.ml_cloud_provider,
            model=settings.ml_cloud_model,
        )

        self._initialized = True

    # ------------------------------------------------------------------
    # Chargement des modèles
    # ------------------------------------------------------------------

    def load_models(self):
        """Charge tous les modèles en VRAM (thread-safe, lazy)."""
        if self._md_roberta is not None:
            return

        with _model_lock:
            if self._md_roberta is not None:
                return

            tr, torch = _load_ml_libs()
            dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

            logger.info("[SentimentV2] Chargement DistilRoBERTa-financial …")
            self._tk_roberta = tr.AutoTokenizer.from_pretrained(self.roberta_name)
            self._md_roberta = tr.AutoModelForSequenceClassification.from_pretrained(
                self.roberta_name
            ).to(self.device).eval()
            self._label_map_roberta = {
                v.lower(): k for k, v in self._md_roberta.config.id2label.items()
            }

            logger.info("[SentimentV2] Chargement ModernFinBERT …")
            self._tk_modern = tr.AutoTokenizer.from_pretrained(self.modern_name)
            self._md_modern = tr.AutoModelForSequenceClassification.from_pretrained(
                self.modern_name
            ).to(self.device).eval()
            self._label_map_modern = {
                v.lower(): k for k, v in self._md_modern.config.id2label.items()
            }

            if settings.ml_enable_qwen:
                logger.info("[SentimentV2] Chargement Qwen3-0.6B (arbitre) …")
                self._tk_qwen = tr.AutoTokenizer.from_pretrained(
                    self.qwen_name, trust_remote_code=True
                )
                self._md_qwen = tr.AutoModelForCausalLM.from_pretrained(
                    self.qwen_name,
                    trust_remote_code=True,
                    dtype=dtype,
                    device_map="auto",
                ).eval()
                logger.info("[SentimentV2] Qwen chargé (bfloat16=%s)", dtype == torch.bfloat16)

            logger.info("[SentimentV2] Tous les modèles sont en VRAM")

    # ------------------------------------------------------------------
    # Inférence individuelle
    # ------------------------------------------------------------------

    def _infer_classifier(self, text: str, tokenizer, model, label_map: dict) -> float:
        """Inférence sur un modèle de classification (RoBERTa / ModernFinBERT)."""
        tr, torch = _load_ml_libs()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]

        # Construire le score : P(positive) - P(negative)
        pos_idx = label_map.get("positive", label_map.get("bullish", 2))
        neg_idx = label_map.get("negative", label_map.get("bearish", 0))
        score = (probs[pos_idx] - probs[neg_idx]).item()
        return float(score)

    def _infer_roberta(self, text: str) -> float:
        return self._infer_classifier(text, self._tk_roberta, self._md_roberta, self._label_map_roberta)

    def _infer_modern(self, text: str) -> float:
        return self._infer_classifier(text, self._tk_modern, self._md_modern, self._label_map_modern)

    def _infer_qwen(self, text: str) -> tuple[float, float]:
        """Inférence Qwen3-0.6B causal. Retourne (score, confidence)."""
        tr, torch = _load_ml_libs()
        system = "Classify the financial sentiment as positive, neutral, or negative."
        prompt = self._tk_qwen.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        inputs = self._tk_qwen([prompt], return_tensors="pt").to(self._md_qwen.device)

        with torch.no_grad():
            outputs = self._md_qwen.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Extraire le token généré
        gen_token_id = outputs.sequences[0, -1].item()
        gen_text = self._tk_qwen.decode([gen_token_id], skip_special_tokens=True).strip().lower()
        score = _label_to_score(gen_text)

        # Confiance = proba du token généré
        token_scores = outputs.scores[0][0]
        token_probs = torch.nn.functional.softmax(token_scores, dim=-1)
        confidence = token_probs[gen_token_id].item()

        logger.debug(f"Qwen arbitre → '{gen_text}' (score={score:.2f}, conf={confidence:.2f})")
        return score, confidence

    # ------------------------------------------------------------------
    # Analyse principale
    # ------------------------------------------------------------------

    def analyze_text(self, text: str) -> dict:
        """Pipeline complet à 4 tiers.

        Retourne un dict avec tous les scores, le score final, les flags.
        """
        self.load_models()
        result: dict[str, Any] = {
            "lexical_triggered": False,
            "lexical_score": None,
            "lexical_rule": None,
            "roberta_score": None,
            "modern_score": None,
            "qwen_score": None,
            "qwen_confidence": None,
            "cloud_score": None,
            "cloud_confidence": None,
            "combined": 0.0,
            "confidence": 0.0,
            "divergence": 0.0,
            "anomaly": False,
            "keywords": "",
        }

        # ---- Tier 0 : règles lexicales ----
        if settings.ml_lexical_override:
            lex = apply_lexical_rules(text)
            if lex.triggered:
                result["lexical_triggered"] = True
                result["lexical_score"] = lex.score_override
                result["lexical_rule"] = lex.rule_name
                result["combined"] = lex.score_override
                result["confidence"] = lex.confidence
                result["keywords"] = ",".join(extract_financial_keywords(text))
                logger.info(f"Lexical override: {lex.rule_name}/{lex.matched_keyword} → {lex.score_override:.2f}")
                return result

        # ---- Tier 1 & 2 : RoBERTa + ModernFinBERT (sequentiel pour eviter conflits CUDA) ----
        roberta_score = self._infer_roberta(text)
        modern_score = self._infer_modern(text)

        result["roberta_score"] = round(roberta_score, 4)
        result["modern_score"] = round(modern_score, 4)
        divergence = abs(roberta_score - modern_score)
        result["divergence"] = round(divergence, 4)

        # ---- Pas de divergence → fusion simple 50/50 ----
        if divergence <= settings.ml_divergence_threshold:
            combined = 0.5 * roberta_score + 0.5 * modern_score
            confidence = 1.0 - divergence  # plus la divergence est faible, plus on est confiant
            result["combined"] = round(combined, 4)
            result["confidence"] = round(confidence, 4)
            result["keywords"] = ",".join(extract_financial_keywords(text))
            return result

        # ---- Tier 3 : divergence → Qwen arbitre ----
        logger.info(f"Divergence detected ({divergence:.2f}) → calling Qwen arbitre")
        result["anomaly"] = True
        qwen_score, qwen_conf = self._infer_qwen(text)
        result["qwen_score"] = round(qwen_score, 4)
        result["qwen_confidence"] = round(qwen_conf, 4)

        # Si Qwen est confiant, on lui fait confiance
        if qwen_conf >= 0.5:
            combined = qwen_score
            confidence = qwen_conf * (1.0 - divergence * 0.5)
            result["combined"] = round(combined, 4)
            result["confidence"] = round(confidence, 4)
            result["keywords"] = ",".join(extract_financial_keywords(text))
            return result

        # ---- Tier 4 : Qwen incertain → fallback cloud ----
        if settings.ml_enable_cloud_fallback and self._cloud.enabled:
            logger.info("Qwen uncertain → calling cloud fallback")
            cloud_res = self._cloud.analyze(text)
            if cloud_res:
                result["cloud_score"] = round(cloud_res["score"], 4)
                result["cloud_confidence"] = round(cloud_res["confidence"], 4)
                combined = cloud_res["score"]
                confidence = cloud_res["confidence"] * 0.9  # pénalité légère cloud
                result["combined"] = round(combined, 4)
                result["confidence"] = round(confidence, 4)
                result["keywords"] = ",".join(extract_financial_keywords(text))
                return result

        # ---- Dernier recours : moyenne pondérée avec pénalité ----
        combined = 0.5 * roberta_score + 0.5 * modern_score
        confidence = max(0.3, 1.0 - divergence)  # confiance réduite
        result["combined"] = round(combined, 4)
        result["confidence"] = round(confidence, 4)
        result["keywords"] = ",".join(extract_financial_keywords(text))
        return result

    # ------------------------------------------------------------------
    # Traitement batch de news
    # ------------------------------------------------------------------

    def process_unprocessed_news(self, db: Session) -> int:
        """Analyse toutes les news non traitées et génère des signaux."""
        self.load_models()
        news_items = db.query(News).filter(News.processed == 0).all()
        signal_count = 0

        for item in news_items:
            text = f"{item.title}. {item.description or ''}"
            result = self.analyze_text(text)

            sentiment = SentimentScore(
                news_id=item.id,
                ticker=item.ticker,
                finbert_score=result.get("finbert_score"),
                roberta_score=result["roberta_score"],
                modern_score=result["modern_score"],
                qwen_score=result["qwen_score"],
                cloud_score=result["cloud_score"],
                lexical_score=result["lexical_score"],
                lexical_rule=result["lexical_rule"],
                combined_score=result["combined"],
                confidence=result["confidence"],
                divergence=result["divergence"],
                keywords=result["keywords"],
                anomaly_flag=int(result["anomaly"]),
                qwen_arbitrated=int(result["qwen_score"] is not None and result["anomaly"]),
                cloud_fallback_used=int(result["cloud_score"] is not None),
            )
            db.add(sentiment)
            item.processed = 1

            # Génération automatique de signal si fort sentiment
            if abs(result["combined"]) >= settings.ml_signal_threshold and result["confidence"] >= settings.ml_confidence_threshold:
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
                    source="sentiment_engine_v2",
                )
                db.add(signal)
                signal_count += 1

        db.commit()
        logger.info(f"{len(news_items)} news analysées, {signal_count} signaux générés")
        return signal_count


# Compatibilité ascendante — alias pour l'import existant
SentimentAnalyzer = SentimentAnalyzerV2
