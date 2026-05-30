"""Token tracking and cost estimation for sentiment arbitration models."""

from dataclasses import dataclass, field

# Tarifs au million de tokens (USD) — mis à jour 2025-06
# source : OpenAI / Anthropic pricing public
CLOUD_PRICING = {
    "openai": {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    },
    "anthropic": {
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    },
}


def get_pricing(provider: str, model: str) -> dict[str, float]:
    """Retourne les tarifs (input, output) par million de tokens pour un provider/modèle."""
    provider = provider.lower()
    model = model.lower()
    if provider in CLOUD_PRICING:
        for m, rates in CLOUD_PRICING[provider].items():
            if model == m.lower():
                return rates
    # Fallback : tarif moyen si modèle inconnu
    return {"input": 2.00, "output": 8.00}


@dataclass
class TokenUsage:
    """Métriques de consommation tokens pour une inférence."""

    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = "local"  # "local" | "openai" | "anthropic"
    model: str = ""
    estimated_cost_usd: float = field(default=0.0, repr=False)

    def __post_init__(self):
        self.compute_cost()

    def compute_cost(self):
        """Recalcule le coût estimé à partir des tarifs."""
        if self.provider == "local":
            self.estimated_cost_usd = 0.0
            return
        rates = get_pricing(self.provider, self.model)
        cost = (
            self.input_tokens * rates["input"] / 1_000_000
            + self.output_tokens * rates["output"] / 1_000_000
        )
        self.estimated_cost_usd = round(cost, 6)

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "provider": self.provider,
            "model": self.model,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


class TokenTracker:
    """Tracker global de consommation tokens (singleton par session)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._reset()
        return cls._instance

    def _reset(self):
        self.total_input = 0
        self.total_output = 0
        self.total_estimated_cost = 0.0
        self.call_count = 0
        self.provider_breakdown: dict[str, dict] = {}

    def record(self, usage: TokenUsage):
        """Enregistre une utilisation de tokens."""
        self.total_input += usage.input_tokens
        self.total_output += usage.output_tokens
        self.total_estimated_cost += usage.estimated_cost_usd
        self.call_count += 1

        key = f"{usage.provider}/{usage.model}"
        if key not in self.provider_breakdown:
            self.provider_breakdown[key] = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost_usd": 0.0,
            }
        self.provider_breakdown[key]["calls"] += 1
        self.provider_breakdown[key]["input_tokens"] += usage.input_tokens
        self.provider_breakdown[key]["output_tokens"] += usage.output_tokens
        self.provider_breakdown[key]["estimated_cost_usd"] += usage.estimated_cost_usd

    def summary(self) -> dict:
        """Retourne un résumé de la consommation totale."""
        return {
            "calls": self.call_count,
            "total_input_tokens": self.total_input,
            "total_output_tokens": self.total_output,
            "total_estimated_cost_usd": round(self.total_estimated_cost, 6),
            "by_provider": self.provider_breakdown,
        }
