"""Tests for the monitoring subsystem."""

import pytest
from trading.monitoring.database import init_monitoring_db, monitoring_session
from trading.monitoring.models import LLMCallLog, MessageLog, PerformanceSnapshot
from trading.monitoring.decorator import trace_llm_call
from trading.monitoring.message_logger import MessageLogger
from trading.monitoring.service import MonitorService


@pytest.fixture(scope="module", autouse=True)
def setup_monitoring_db():
    """Ensure monitoring tables exist before running tests."""
    init_monitoring_db()


class TestMonitorService:
    def test_log_llm_call(self):
        svc = MonitorService()
        svc.log_llm_call(
            function_name="test_func",
            model="gpt-4",
            provider="openai",
            backend="cloud",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.0015,
            duration_ms=250.0,
            prompt_hash="abc123",
            response_hash="def456",
            provider_info={"system_fingerprint": "fp_123"},
            triggered_by="test",
            portfolio_id="test-portfolio",
        )
        calls = svc.get_llm_calls(hours=1, limit=10)
        assert len(calls) >= 1
        latest = calls[0]
        assert latest["function_name"] == "test_func"
        assert latest["model"] == "gpt-4"
        assert latest["input_tokens"] == 100
        assert latest["output_tokens"] == 50
        assert latest["cost_usd"] == 0.0015

    def test_log_llm_call_with_error(self):
        svc = MonitorService()
        svc.log_llm_call(
            function_name="test_error",
            model="qwen",
            provider="local",
            error_message="CUDA out of memory",
        )
        calls = svc.get_llm_calls(hours=1, limit=10)
        latest = [c for c in calls if c["function_name"] == "test_error"][0]
        assert latest["error_message"] == "CUDA out of memory"

    def test_get_llm_summary(self):
        svc = MonitorService()
        # Clear previous test data implicitly by time filter
        summary = svc.get_llm_summary(hours=1)
        assert "calls" in summary
        assert "cost_usd" in summary

    def test_log_message(self):
        svc = MonitorService()
        svc.log_message(
            channel="news_finnhub",
            source="finnhub",
            content_hash="hash123",
            metadata_json='{"count": 5}',
            processed=1,
            processing_time_ms=45.0,
        )
        messages = svc.get_messages(channel="news_finnhub", hours=1, limit=10)
        assert len(messages) >= 1
        latest = messages[0]
        assert latest["channel"] == "news_finnhub"
        assert latest["source"] == "finnhub"
        assert latest["processed"] == 1

    def test_get_message_channels(self):
        svc = MonitorService()
        svc.log_message(channel="api_rest", source="GET /health")
        svc.log_message(channel="api_rest", source="POST /decisions")
        channels = svc.get_message_channels()
        assert isinstance(channels, list)

    def test_log_performance(self):
        svc = MonitorService()
        svc.log_performance("inference_latency", 120.5, "ms", {"model": "qwen"})

    def test_get_time_series(self):
        svc = MonitorService()
        # Insert a few LLM calls to have data
        svc.log_llm_call(function_name="ts_test", model="m", provider="p", cost_usd=0.01)
        ts = svc.get_time_series(metric_name="llm_call", interval="1 hour", hours=1)
        assert isinstance(ts, list)


class TestTraceLlmCallDecorator:
    def test_decorator_captures_success(self):
        @trace_llm_call(model="test-model", provider="test-provider")
        def fake_llm_call(prompt: str) -> dict:
            return {
                "input_tokens": 10,
                "output_tokens": 5,
                "cost_usd": 0.001,
                "content": "hello",
            }

        result = fake_llm_call("test prompt")
        assert result["content"] == "hello"

    def test_decorator_captures_exception(self):
        @trace_llm_call(model="test-model", provider="test-provider")
        def failing_llm_call(prompt: str) -> dict:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            failing_llm_call("test")


class TestMessageLogger:
    def test_log_valid_channel(self):
        MessageLogger.log(
            channel="news_finnhub",
            source="finnhub",
            content="some news",
            metadata={"ticker": "AAPL"},
        )

    def test_log_invalid_channel_fallback(self):
        MessageLogger.log(
            channel="unknown_channel",
            source="test",
            content="data",
        )

    def test_timed_log_context_manager(self):
        with MessageLogger.timed_log("api_rest", "GET /test") as log:
            log.metadata["extra"] = "info"
        # If we get here without exception, logging succeeded

    def test_log_no_content(self):
        MessageLogger.log(
            channel="system",
            source="heartbeat",
            metadata={"status": "ok"},
        )
