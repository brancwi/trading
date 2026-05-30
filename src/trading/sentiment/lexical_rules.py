"""Règles métier lexicales — override du sentiment par mots-clés financiers."""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LexicalMatch:
    """Résultat d'une règle lexicale."""

    triggered: bool
    score_override: float | None = None
    confidence: float = 1.0
    rule_name: str = ""
    matched_keyword: str = ""


# ============================================================
# Dictionnaires de mots-clés financiers avec force de sentiment
# ============================================================

STRONG_POSITIVE = {
    "fda approves": 0.95,
    "fda approval": 0.95,
    "record profit": 0.90,
    "record earnings": 0.90,
    "all-time high": 0.90,
    "breakthrough": 0.85,
    "exceeds expectations": 0.85,
    "beats estimates": 0.85,
    "strong buy": 0.90,
    "upgrade": 0.70,
    "partnership": 0.65,
    "merger": 0.60,
    "acquisition": 0.55,
    "dividend increase": 0.70,
    "share buyback": 0.65,
    "bullish": 0.75,
    "surge": 0.80,
    "soar": 0.85,
    "rally": 0.70,
    "outperform": 0.75,
}

STRONG_NEGATIVE = {
    "sec investigation": -0.95,
    "sec probe": -0.95,
    "bankruptcy": -0.95,
    "layoffs": -0.80,
    "misses expectations": -0.85,
    "misses estimates": -0.85,
    "earnings shortfall": -0.85,
    "profit warning": -0.85,
    "downgrade": -0.70,
    "sell rating": -0.80,
    "bearish": -0.75,
    "plunge": -0.85,
    "crash": -0.95,
    "collapse": -0.90,
    "tank": -0.80,
    "nosedive": -0.85,
    "underperform": -0.70,
    "recession": -0.75,
    "inflation surge": -0.70,
    "supply chain disruption": -0.65,
}

MODERATE_POSITIVE = {
    "growth": 0.45,
    "expansion": 0.40,
    "new contract": 0.50,
    "revenue increase": 0.45,
    "positive outlook": 0.50,
    "guidance raised": 0.55,
}

MODERATE_NEGATIVE = {
    "decline": -0.45,
    "contraction": -0.40,
    "loss": -0.50,
    "debt": -0.35,
    "negative outlook": -0.50,
    "guidance lowered": -0.55,
    "delay": -0.40,
    "recall": -0.50,
}

# Multiplicateurs contextuels — amplifient/diminuent le score
# quand un mot-clé contextuel est présent avec un mot-clé principal
CONTEXT_MULTIPLIERS = {
    "unexpectedly": 1.3,
    "surprisingly": 1.3,
    "significantly": 1.2,
    "substantially": 1.2,
    "slightly": 0.7,
    "marginally": 0.6,
    "reportedly": 0.8,
    "allegedly": 0.7,
    "rumored": 0.6,
}

ALL_RULES = {
    "strong_positive": STRONG_POSITIVE,
    "strong_negative": STRONG_NEGATIVE,
    "moderate_positive": MODERATE_POSITIVE,
    "moderate_negative": MODERATE_NEGATIVE,
}


def _normalize(text: str) -> str:
    """Normalise le texte pour la recherche de mots-clés."""
    return text.lower().strip()


def apply_lexical_rules(text: str) -> LexicalMatch:
    """Applique les règles lexicales sur un texte.

    Retourne un LexicalMatch avec score_override si un mot-clé fort est détecté.
    La règle la plus forte (valeur absolue max) l'emporte en cas de conflit.
    """
    normalized = _normalize(text)
    best_match: tuple[float, str, str] | None = None

    for rule_name, keyword_dict in ALL_RULES.items():
        for keyword, base_score in keyword_dict.items():
            if keyword in normalized:
                # Appliquer les multiplicateurs contextuels
                multiplier = 1.0
                for ctx_word, ctx_mult in CONTEXT_MULTIPLIERS.items():
                    if ctx_word in normalized:
                        multiplier = max(multiplier, ctx_mult)

                adjusted_score = max(-1.0, min(1.0, base_score * multiplier))
                if best_match is None or abs(adjusted_score) > abs(best_match[0]):
                    best_match = (adjusted_score, rule_name, keyword)

    if best_match is None:
        return LexicalMatch(triggered=False)

    score, rule_name, keyword = best_match
    # Confiance proportionnelle à la force du signal
    confidence = min(0.99, 0.7 + abs(score) * 0.3)

    logger.debug(f"Lexical rule triggered: {rule_name}/{keyword} → score={score:.3f}")
    return LexicalMatch(
        triggered=True,
        score_override=score,
        confidence=confidence,
        rule_name=rule_name,
        matched_keyword=keyword,
    )


def extract_financial_keywords(text: str) -> list[str]:
    """Extrait les mots-clés financiers détectés dans le texte (pour le debugging)."""
    normalized = _normalize(text)
    found = []
    for keyword_dict in ALL_RULES.values():
        for keyword in keyword_dict:
            if keyword in normalized:
                found.append(keyword)
    return found
