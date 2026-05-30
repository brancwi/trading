#!/usr/bin/env python3
"""Exporte la table sentiment_scores en dataset ML (JSONL / CSV).

Usage:
    uv run python scripts/export_sentiment_dataset.py --format jsonl --output dataset.jsonl
    uv run python scripts/export_sentiment_dataset.py --format csv --output dataset.csv --only-labeled
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading.core.database import db_session
from trading.core.models import SentimentScore


def build_row(s: SentimentScore) -> dict:
    """Construit un dict plat à partir d'un SentimentScore."""
    return {
        "id": s.id,
        "timestamp": s.timestamp.isoformat() if s.timestamp else None,
        "ticker": s.ticker,
        "input_text": s.input_text,
        "predicted_label": s.predicted_label,
        "human_label": s.human_label,
        "roberta_score": s.roberta_score,
        "modern_score": s.modern_score,
        "qwen_score": s.qwen_score,
        "cloud_score": s.cloud_score,
        "lexical_score": s.lexical_score,
        "lexical_rule": s.lexical_rule,
        "combined_score": s.combined_score,
        "confidence": s.confidence,
        "divergence": s.divergence,
        "anomaly_flag": s.anomaly_flag,
        "qwen_arbitrated": s.qwen_arbitrated,
        "cloud_fallback_used": s.cloud_fallback_used,
        "keywords": s.keywords,
        "input_tokens": s.input_tokens,
        "output_tokens": s.output_tokens,
        "estimated_cost_usd": s.estimated_cost_usd,
        "pipeline_config_json": s.pipeline_config_json,
        "model_versions_json": s.model_versions_json,
    }


def export_jsonl(rows: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_csv(rows: list[dict], path: str) -> None:
    if not rows:
        print("Aucune ligne à exporter.")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export sentiment_scores → dataset ML")
    parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    parser.add_argument("--output", required=True, help="Chemin du fichier de sortie")
    parser.add_argument("--only-labeled", action="store_true", help="Exporter uniquement les lignes avec human_label")
    parser.add_argument("--limit", type=int, default=0, help="Limiter le nombre de lignes (0 = illimité)")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Filtrer confidence >= valeur")
    args = parser.parse_args()

    with db_session() as db:
        query = db.query(SentimentScore)
        if args.only_labeled:
            query = query.filter(SentimentScore.human_label.isnot(None))
        if args.min_confidence > 0:
            query = query.filter(SentimentScore.confidence >= args.min_confidence)
        query = query.order_by(SentimentScore.timestamp.desc())
        if args.limit > 0:
            query = query.limit(args.limit)

        scores = query.all()
        rows = [build_row(s) for s in scores]

    if args.format == "jsonl":
        export_jsonl(rows, args.output)
    else:
        export_csv(rows, args.output)

    print(f"✅ {len(rows)} lignes exportées → {args.output}")
    if args.only_labeled:
        print(f"   (filtré: uniquement les lignes avec human_label)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
