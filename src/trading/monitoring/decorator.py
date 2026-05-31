"""Decorator for automatic LLM call tracing.

Usage:
    @trace_llm_call(model="Qwen2.5-3B", provider="local")
    def _call_local(self, system_prompt: str, user_prompt: str) -> dict:
        ...

The decorator captures:
  - duration_ms
  - input_tokens / output_tokens (from the returned dict or kwargs)
  - cost_usd
  - provider_info (system_fingerprint, finish_reason, etc.)
  - error messages on exception
  - prompt/response hashes for audit
"""

import hashlib
import json
import logging
import time
from functools import wraps
from typing import Any, Callable

from trading.monitoring.service import MonitorService

logger = logging.getLogger(__name__)


def trace_llm_call(
    model: str | None = None,
    provider: str | None = None,
    backend: str = "local",
    triggered_by: str = "llm_inference",
):
    """Decorator that automatically logs every LLM call to the monitoring DB.

    Args:
        model: Model name (e.g. "Qwen2.5-3B-Instruct", "deepseek-chat").
               If None, tries to read from kwargs / self.
        provider: Provider name (e.g. "local", "deepseek", "openai").
        backend: "local" | "cloud" | "hybrid".
        triggered_by: Human-readable origin (e.g. "decision_llm", "sentiment_qwen").
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            error_msg = None
            result = None

            # Try to infer model/provider from instance attributes
            _model = model
            _provider = provider
            _portfolio_id = kwargs.get("portfolio_id")

            if args and hasattr(args[0], "model_name"):
                _model = _model or getattr(args[0], "model_name", "unknown")
            elif args and hasattr(args[0], "model"):
                _model = _model or getattr(args[0], "model", "unknown")
            if args and hasattr(args[0], "deepseek_key"):
                # heuristic: if deepseek_key is set, provider might be deepseek
                _provider = _provider or (
                    "deepseek" if getattr(args[0], "deepseek_key", "") else "local"
                )
            elif args and hasattr(args[0], "provider"):
                _provider = _provider or getattr(args[0], "provider", "unknown")

            # Hash prompts if present in kwargs
            prompt_hash = None
            response_hash = None
            for key in ("system_prompt", "user_prompt", "prompt", "text"):
                if key in kwargs and kwargs[key]:
                    prompt_hash = hashlib.sha256(str(kwargs[key]).encode()).hexdigest()[:16]
                    break

            try:
                result = func(*args, **kwargs)

                # Attempt to extract token/cost info from result
                input_tokens = 0
                output_tokens = 0
                cost_usd = 0.0
                provider_info = {}

                # Unwrap tuple/list if the last element looks like usage info
                _result = result
                if isinstance(result, (tuple, list)) and result:
                    _result = result[-1]

                if isinstance(_result, dict):
                    # DeepSeek / OpenAI style response embedding
                    input_tokens = _result.get("input_tokens", _result.get("usage", {}).get("prompt_tokens", 0))
                    output_tokens = _result.get("output_tokens", _result.get("usage", {}).get("completion_tokens", 0))
                    cost_usd = _result.get("cost_usd", 0.0)
                    provider_info = {
                        k: v
                        for k, v in _result.items()
                        if k in ("system_fingerprint", "model", "finish_reason", "id")
                    }
                    # Hash the textual response if present
                    if "content" in _result and _result["content"]:
                        response_hash = hashlib.sha256(str(_result["content"]).encode()).hexdigest()[:16]
                    elif "text" in _result and _result["text"]:
                        response_hash = hashlib.sha256(str(_result["text"]).encode()).hexdigest()[:16]
                elif hasattr(_result, "input_tokens") and hasattr(_result, "output_tokens"):
                    # TokenUsage dataclass or similar
                    input_tokens = getattr(_result, "input_tokens", 0)
                    output_tokens = getattr(_result, "output_tokens", 0)
                    cost_usd = getattr(_result, "cost_usd", 0.0)

                duration_ms = round((time.perf_counter() - start) * 1000, 2)

                MonitorService().log_llm_call(
                    function_name=func.__qualname__,
                    model=_model or "unknown",
                    provider=_provider or "unknown",
                    backend=backend,
                    input_tokens=int(input_tokens) if input_tokens else 0,
                    output_tokens=int(output_tokens) if output_tokens else 0,
                    cost_usd=float(cost_usd) if cost_usd else 0.0,
                    duration_ms=duration_ms,
                    prompt_hash=prompt_hash,
                    response_hash=response_hash,
                    provider_info=provider_info,
                    triggered_by=triggered_by,
                    portfolio_id=_portfolio_id,
                )

                return result

            except Exception as exc:
                duration_ms = round((time.perf_counter() - start) * 1000, 2)
                error_msg = str(exc)[:500]

                MonitorService().log_llm_call(
                    function_name=func.__qualname__,
                    model=_model or "unknown",
                    provider=_provider or "unknown",
                    backend=backend,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    duration_ms=duration_ms,
                    prompt_hash=prompt_hash,
                    error_message=error_msg,
                    triggered_by=triggered_by,
                    portfolio_id=_portfolio_id,
                )
                raise

        return wrapper
    return decorator
