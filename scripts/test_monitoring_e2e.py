"""Test end-to-end: sentiment analysis + monitoring DB writes."""

import torch
from trading.sentiment.analyzer import SentimentAnalyzerV2
from trading.core.database import db_session
from trading.core.config import get_settings
from trading.core.models import TokenUsageLog, AuditLog, MonitoringMetric

settings = get_settings()
settings.ml_lexical_override = False

torch.cuda.empty_cache()

analyzer = SentimentAnalyzerV2()
analyzer.load_models()

# Texte ambigu qui force Qwen
text = "The stock surged on FDA approval news but the company also announced massive layoffs and missed revenue targets."
res = analyzer.analyze_text(text)
print("Analyse:", res["combined"], "anomaly=", res["anomaly"], "qwen=", res["qwen_score"])

# Verifier en DB
with db_session() as db:
    token_count = db.query(TokenUsageLog).count()
    audit_count = db.query(AuditLog).count()
    metric_count = db.query(MonitoringMetric).count()

    print(f"\nDB counts: token_usage={token_count}, audit={audit_count}, metrics={metric_count}")

    if token_count > 0:
        latest = db.query(TokenUsageLog).order_by(TokenUsageLog.timestamp.desc()).first()
        print(f"Latest token usage: {latest.provider}/{latest.model}, tokens={latest.input_tokens}+{latest.output_tokens}, cost=${latest.cost_usd:.6f}")
