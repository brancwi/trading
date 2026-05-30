"""Fallback cloud — GPT-4 / Claude pour les cas difficiles."""

import json
import logging
import os

import httpx

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a financial sentiment classifier. "
    "Respond ONLY with a JSON object: {\"sentiment\": \"positive|neutral|negative\", "
    "\"confidence\": 0.0-1.0, \"reason\": \"one sentence\"}. "
    "No other text."
)

DEFAULT_USER_TEMPLATE = (
    "Classify the financial sentiment of this text as positive, neutral, or negative:\n\n{text}"
)

# Mapping labels → score numérique
LABEL_SCORES = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}


class CloudFallback:
    """Appel API cloud (OpenAI, Anthropic, ou compatible) pour sentiment difficile."""

    def __init__(
        self,
        api_key: str | None = None,
        provider: str = "openai",
        model: str | None = None,
        timeout: float = 10.0,
    ):
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.timeout = timeout
        self.enabled = bool(self.api_key)

        if self.provider == "openai":
            self.base_url = "https://api.openai.com/v1/chat/completions"
            self.model = model or "gpt-4o-mini"
        elif self.provider == "anthropic":
            self.base_url = "https://api.anthropic.com/v1/messages"
            self.model = model or "claude-3-haiku-20240307"
        else:
            self.base_url = provider  # URL directe pour provider custom
            self.model = model or "gpt-4o-mini"

    def _build_payload(self, text: str) -> dict:
        """Construit le payload selon le provider."""
        if self.provider == "openai":
            return {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": DEFAULT_USER_TEMPLATE.format(text=text)},
                ],
                "temperature": 0.0,
                "max_tokens": 100,
                "response_format": {"type": "json_object"},
            }
        elif self.provider == "anthropic":
            return {
                "model": self.model,
                "max_tokens": 100,
                "temperature": 0.0,
                "system": DEFAULT_SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": DEFAULT_USER_TEMPLATE.format(text=text)},
                ],
            }
        else:
            # Generic OpenAI-compatible
            return {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": DEFAULT_USER_TEMPLATE.format(text=text)},
                ],
                "temperature": 0.0,
                "max_tokens": 100,
            }

    def _build_headers(self) -> dict:
        """Construit les headers selon le provider."""
        if self.provider == "anthropic":
            return {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        return {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

    def _parse_response(self, response_data: dict) -> dict | None:
        """Extrait le JSON de la réponse API."""
        try:
            if self.provider == "anthropic":
                content = response_data["content"][0]["text"]
            else:
                content = response_data["choices"][0]["message"]["content"]

            parsed = json.loads(content)
            label = parsed.get("sentiment", "neutral").lower().strip()
            confidence = float(parsed.get("confidence", 0.5))
            reason = parsed.get("reason", "")
            score = LABEL_SCORES.get(label, 0.0)

            return {
                "score": score,
                "label": label,
                "confidence": min(1.0, max(0.0, confidence)),
                "reason": reason,
            }
        except Exception as e:
            logger.warning(f"Failed to parse cloud response: {e}")
            return None

    def analyze(self, text: str) -> dict | None:
        """Analyse un texte via API cloud. Retourne dict ou None si échec/indisponible."""
        if not self.enabled:
            return None

        payload = self._build_payload(text)
        headers = self._build_headers()

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(self.base_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                result = self._parse_response(data)
                if result:
                    logger.info(
                        f"Cloud fallback [{self.provider}/{self.model}] → "
                        f"{result['label']} (conf={result['confidence']:.2f})"
                    )
                return result
        except httpx.HTTPStatusError as e:
            logger.error(f"Cloud API HTTP error {e.response.status_code}: {e.response.text}")
        except httpx.TimeoutException:
            logger.error("Cloud API timeout")
        except Exception as e:
            logger.error(f"Cloud API error: {e}")

        return None
