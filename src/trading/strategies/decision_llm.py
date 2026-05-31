"""DecisionLLM — Couche décision pilotée par LLM local (Qwen) ou cloud (DeepSeek).

Architecture séquentielle : le modèle local est chargé à la demande puis déchargé
pour libérer la VRAM. Cela permet d'utiliser un modèle plus gros (ex: Qwen3-8B)
sans conflit avec les modèles de sentiment déjà chargés.

Modes:
  - local  : Qwen2.5-3B (défaut) ou Qwen3-8B si VRAM suffisante
  - cloud  : DeepSeek-V3 via API (fallback si local indisponible)
  - hybrid : local par défaut, cloud en fallback

Usage:
    from trading.strategies.decision_llm import DecisionLLM
    llm = DecisionLLM()
    decisions = llm.decide(portfolio, signals, prices)
"""

import json
import logging
import os
import time
from typing import Any

import httpx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from trading.core.config import get_settings
from trading.core.models import Portfolio, Signal
from trading.ml.features import FeatureEngine
from trading.monitoring.decorator import trace_llm_call
from trading.monitoring.message_logger import MessageLogger
from trading.sentiment.token_tracker import TokenTracker, TokenUsage

logger = logging.getLogger(__name__)
settings = get_settings()

# Configuration quantization 4-bit pour tenir dans ~2-5 GB VRAM
_BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

_DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
_DEEPSEEK_MODEL = "deepseek-chat"  # DeepSeek-V3 (rapide et précis)


class DecisionLLM:
    """Portfolio Manager LLM — local (Qwen) ou cloud (DeepSeek)."""

    _local_model = None
    _local_tokenizer = None

    def __init__(self):
        self.model_name = getattr(settings, "decision_llm_model", "Qwen/Qwen2.5-3B-Instruct")
        # En dev → local (Qwen), en staging/prod → cloud (DeepSeek)
        self.use_cloud = settings.use_cloud_llm or getattr(settings, "decision_llm_use_cloud", False)
        self.deepseek_key = getattr(settings, "deepseek_api_key", "")
        self.device = settings.ml_device if settings.ml_device != "cpu" else "cpu"
        self.feature_engine = FeatureEngine()
        self.token_tracker = TokenTracker()
        self._http = httpx.Client(timeout=60.0)
        logger.info(
            "[DecisionLLM] env=%s | use_cloud=%s | model=%s",
            settings.environment, self.use_cloud, self.model_name
        )

    # ------------------------------------------------------------------
    # Local model — chargement / déchargement séquentiel
    # ------------------------------------------------------------------

    def _load_local(self) -> None:
        """Charge le modèle local (lazy, une seule fois en mémoire)."""
        if DecisionLLM._local_model is not None:
            return
        logger.info("[DecisionLLM] Chargement local de %s ...", self.model_name)
        try:
            tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            mod = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=_BNB_CONFIG,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            DecisionLLM._local_tokenizer = tok
            DecisionLLM._local_model = mod
            allocated = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
            logger.info("[DecisionLLM] Modèle local chargé — VRAM: %.1f MB", allocated)
        except Exception as e:
            logger.error("[DecisionLLM] Échec chargement local: %s", e)
            raise

    def _unload_local(self) -> None:
        """Décharge le modèle local et libère la VRAM."""
        if DecisionLLM._local_model is not None:
            del DecisionLLM._local_model
            del DecisionLLM._local_tokenizer
            DecisionLLM._local_model = None
            DecisionLLM._local_tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[DecisionLLM] Modèle local déchargé — VRAM libérée")

    # ------------------------------------------------------------------
    # Cloud — DeepSeek API (compatible OpenAI)
    # ------------------------------------------------------------------

    @trace_llm_call(model="deepseek-chat", provider="deepseek", backend="cloud", triggered_by="decision_llm")
    def _call_deepseek(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Appelle l'API DeepSeek et retourne le JSON parsé."""
        if not self.deepseek_key:
            raise RuntimeError("DeepSeek API key manquante")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload = {
            "model": _DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1024,
            "top_p": 0.9,
            "response_format": {"type": "json_object"},
        }

        resp = self._http.post(
            _DEEPSEEK_URL,
            headers={
                "Authorization": f"Bearer {self.deepseek_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        cost_usd = (input_tokens * 0.14 + output_tokens * 0.28) / 1_000_000  # DeepSeek-V3 pricing

        self.token_tracker.record(TokenUsage(
            model=_DEEPSEEK_MODEL,
            provider="deepseek",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost_usd,
        ))

        logger.info(
            "[DecisionLLM] DeepSeek call — input=%d, output=%d, cost=$%.6f",
            input_tokens, output_tokens, cost_usd,
        )

        content = data["choices"][0]["message"]["content"]
        return self._extract_json(content)

    # ------------------------------------------------------------------
    # Local inference
    # ------------------------------------------------------------------

    @trace_llm_call(provider="local", backend="local", triggered_by="decision_llm")
    def _call_local(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Appelle le LLM local et retourne le JSON parsé."""
        self._load_local()
        assert DecisionLLM._local_tokenizer is not None
        assert DecisionLLM._local_model is not None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = DecisionLLM._local_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = DecisionLLM._local_tokenizer([text], return_tensors="pt").to(
            DecisionLLM._local_model.device
        )
        input_tokens = model_inputs["input_ids"].shape[1]

        with torch.no_grad():
            generated_ids = DecisionLLM._local_model.generate(
                **model_inputs,
                max_new_tokens=1024,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=DecisionLLM._local_tokenizer.eos_token_id,
            )

        output_tokens = generated_ids.shape[1] - input_tokens
        generated_text = DecisionLLM._local_tokenizer.decode(
            generated_ids[0][input_tokens:], skip_special_tokens=True
        )

        self.token_tracker.record(TokenUsage(
            model=self.model_name,
            provider="local",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=0.0,
        ))

        logger.info(
            "[DecisionLLM] Local call — input=%d, output=%d",
            input_tokens, output_tokens,
        )

        return self._extract_json(generated_text)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extrait le JSON de la réponse du LLM."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
            logger.error("[DecisionLLM] JSON invalide: %s", text[:500])
            return {"decisions": [], "hold_cash": True, "reasoning": "Parse error"}

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _aggregate_signals(self, signals: list[Signal]) -> dict[str, dict[str, Any]]:
        """Agrège les signaux par ticker (moyenne pondérée par confidence²)."""
        from collections import defaultdict
        groups: dict[str, list[Signal]] = defaultdict(list)
        for sig in signals:
            groups[sig.ticker].append(sig)

        aggregated: dict[str, dict[str, Any]] = {}
        for ticker, sigs in groups.items():
            sentiments = [s.sentiment for s in sigs]
            confidences = [s.confidence for s in sigs]
            weights = [c ** 2 for c in confidences]
            total_weight = sum(weights)
            avg_sentiment = (
                sum(s * w for s, w in zip(sentiments, weights)) / total_weight
                if total_weight else 0.0
            )
            avg_confidence = sum(confidences) / len(confidences)
            actions = [s.action for s in sigs]
            dominant_action = max(set(actions), key=actions.count)
            aggregated[ticker] = {
                "ticker": ticker,
                "signal_count": len(sigs),
                "avg_sentiment": round(avg_sentiment, 4),
                "avg_confidence": round(avg_confidence, 4),
                "dominant_action": dominant_action,
                "actions_distribution": {a: actions.count(a) for a in set(actions)},
            }
        return aggregated

    def _build_prompt(
        self,
        portfolio: Portfolio,
        aggregated: dict[str, dict[str, Any]],
        prices: dict[str, float],
    ) -> tuple[str, str]:
        """Construit le prompt système + utilisateur."""
        positions_json = []
        for pos in portfolio.positions:
            positions_json.append({
                "ticker": pos.ticker,
                "quantity": round(pos.quantity, 2),
                "avg_entry": round(pos.avg_entry_price, 2),
                "current_price": round(pos.current_price or 0, 2),
                "unrealized_pnl": round(pos.unrealized_pnl or 0, 2),
            })

        tech_json = {}
        for ticker in aggregated:
            if ticker in prices:
                tech = self.feature_engine.compute(ticker)
                if tech:
                    tech_json[ticker] = {
                        k: round(v, 2) if v is not None else None
                        for k, v in tech.items()
                    }

        system = (
            "Tu es un gestionnaire de portfolio day-trading expert. "
            "Tu reçois des signaux de sentiment agrégés, l'état du portfolio, "
            "les prix de marché et les indicateurs techniques. "
            "Tu dois décider quels tickers acheter et avec quel montant. "
            "Réponds UNIQUEMENT en JSON valide, sans markdown."
        )

        user = f"""### CONTEXTE PORTFOLIO
- Cash disponible: ${portfolio.cash_available:.2f}
- Cash minimum à conserver: $100
- Montant max par trade: ${portfolio.max_trade_amount or 500:.2f}
- Positions actuelles: {json.dumps(positions_json, ensure_ascii=False)}

### SIGNAUX AGRÉGÉS
{json.dumps(list(aggregated.values()), indent=2, ensure_ascii=False)}

### PRIX DE MARCHÉ
{json.dumps(prices, indent=2, ensure_ascii=False)}

### INDICATEURS TECHNIQUES
{json.dumps(tech_json, indent=2, ensure_ascii=False)}

### RÈGLES
1. Ne jamais dépasser le cash disponible.
2. Respecter le montant max par trade.
3. Garder au moins $100 de cash.
4. Ne pas racheter un ticker déjà en position (sauf si très forte opportunité).
5. Privilégier la diversification.
6. Si le sentiment est négatif ou la confiance faible, ne pas acheter.

### FORMAT DE RÉPONSE (JSON uniquement)
{{
  "decisions": [
    {{
      "ticker": "AAPL",
      "action": "BUY",
      "amount": 500.0,
      "confidence": 0.92,
      "reason": "Fort sentiment positif (0.85) avec RSI neutre (45). Bonne diversification."
    }}
  ],
  "hold_cash": false,
  "reasoning": "J'ai choisi AAPL et NVDA car..."
}}"""
        return system, user

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def decide(
        self,
        portfolio: Portfolio,
        signals: list[Signal],
        prices: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Prend les décisions d'allocation.

        Retourne: [{"ticker": "AAPL", "action": "BUY", "amount": 500.0,
                    "confidence": 0.92, "reason": "..."}, ...]
        """
        if not signals:
            logger.info("[DecisionLLM] Aucun signal — pas de décision")
            return []

        # Log incoming signals
        MessageLogger.log(
            channel="system",
            source="decision_llm.decide",
            metadata={
                "portfolio_id": portfolio.id,
                "signal_count": len(signals),
                "tickers": list({s.ticker for s in signals}),
                "price_count": len(prices),
            },
        )

        aggregated = self._aggregate_signals(signals)
        system_prompt, user_prompt = self._build_prompt(portfolio, aggregated, prices)

        start = time.perf_counter()

        # ── Choix du backend : cloud demandé ? ──
        if self.use_cloud:
            try:
                result = self._call_deepseek(system_prompt, user_prompt)
                backend = "deepseek"
            except Exception as e:
                logger.warning("[DecisionLLM] DeepSeek failed (%s) — fallback local", e)
                result = self._call_local(system_prompt, user_prompt)
                backend = "local_fallback"
        else:
            try:
                result = self._call_local(system_prompt, user_prompt)
                backend = "local"
            except Exception as e:
                if self.deepseek_key:
                    logger.warning("[DecisionLLM] Local failed (%s) — fallback DeepSeek", e)
                    result = self._call_deepseek(system_prompt, user_prompt)
                    backend = "deepseek_fallback"
                else:
                    raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        decisions = result.get("decisions", [])
        reasoning = result.get("reasoning", "")
        hold_cash = result.get("hold_cash", False)

        logger.info("[DecisionLLM] %s | %d décisions, hold_cash=%s", backend, len(decisions), hold_cash)
        logger.info("[DecisionLLM] Reasoning: %s", reasoning)

        # Validation
        valid: list[dict[str, Any]] = []
        for d in decisions:
            ticker = d.get("ticker", "")
            amount = d.get("amount", 0.0)
            action = d.get("action", "")
            if action not in ("BUY", "STRONG_BUY"):
                continue
            if ticker not in prices:
                continue
            if amount <= 0:
                continue
            valid.append(d)

        # Log outgoing decisions
        MessageLogger.log(
            channel="system",
            source=f"decision_llm.decide.{backend}",
            metadata={
                "portfolio_id": portfolio.id,
                "duration_ms": duration_ms,
                "decisions_count": len(decisions),
                "valid_decisions_count": len(valid),
                "hold_cash": hold_cash,
                "backend": backend,
            },
        )

        return valid
