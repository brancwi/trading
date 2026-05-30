"""Test end-to-end du token tracking."""

import torch
from trading.sentiment.analyzer import SentimentAnalyzerV2
from trading.sentiment.token_tracker import TokenTracker
from trading.core.config import get_settings

settings = get_settings()
settings.ml_lexical_override = False  # Désactiver pour forcer Qwen

# Vider GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()

analyzer = SentimentAnalyzerV2()
analyzer.load_models()

tracker = TokenTracker()

# Test 1: texte simple (pas de divergence -> pas de Qwen)
print("--- Test simple (fusion directe) ---")
res1 = analyzer.analyze_text("The firm delivered solid quarterly results and raised guidance.")
print(f"  combined={res1['combined']:.3f}, tokens={res1['input_tokens']}+{res1['output_tokens']}, cost=${res1['estimated_cost_usd']:.6f}")

# Test 2: texte ambigu qui a diverge precedemment
print("\n--- Test ambigu (Qwen arbitre) ---")
text2 = "The stock surged on FDA approval news but the company also announced massive layoffs and missed revenue targets."
res2 = analyzer.analyze_text(text2)
print(f"  combined={res2['combined']:.3f}, anomaly={res2['anomaly']}, qwen={res2['qwen_score']}")
print(f"  tokens={res2['input_tokens']}+{res2['output_tokens']}, est_cost_cloud=${res2['estimated_cost_usd']:.6f}")

# Test 3: texte neutre
print("\n--- Test neutre ---")
res3 = analyzer.analyze_text("The company filed its quarterly 10-Q report with standard disclosures.")
print(f"  combined={res3['combined']:.3f}, tokens={res3['input_tokens']}+{res3['output_tokens']}, cost=${res3['estimated_cost_usd']:.6f}")

# Resume tracker
print("\n=== Resume TokenTracker ===")
summary = tracker.summary()
print(f"  Appels Qwen arbitrés: {summary['calls']}")
print(f"  Tokens input total: {summary['total_input_tokens']}")
print(f"  Tokens output total: {summary['total_output_tokens']}")
print(f"  Cout estimé cloud total: ${summary['total_estimated_cost_usd']:.6f}")
print(f"  By provider: {summary['by_provider']}")
