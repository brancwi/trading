#!/usr/bin/env python3
"""Entraîne le modèle de fusion apprenante sur l'historique annoté.

Usage:
    uv run python scripts/train_fusion_model.py
    uv run python scripts/train_fusion_model.py --info
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading.sentiment.fusion_model import FusionModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Entraîne la fusion apprenante")
    parser.add_argument("--info", action="store_true", help="Affiche les infos du modèle sans entraîner")
    args = parser.parse_args()

    fm = FusionModel()

    if args.info:
        print(json.dumps(fm.info(), indent=2))
        return 0

    print("[FusionModel] Entraînement en cours...")
    ok = fm.train()
    if ok:
        print(f"✅ Entraînement réussi!")
        print(f"   Poids: {fm.weights.tolist()}")
        print(f"   Fichier: {fm.info()['weights_file']}")
    else:
        print("❌ Entraînement échoué (pas assez de données annotées)")
        print("   Pour annoter des données, mettez à jour human_label dans sentiment_scores")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
