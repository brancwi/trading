# Documentation Trading Engine v1.0

## Sommaire

Ce répertoire regroupe les analyses, rapports de recherche et documentations techniques du projet.

---

## 📊 Rapports d'analyse

### [deep-research-report-models.md](./deep-research-report-models.md)
**Analyse comparative des modèles de sentiment financier.**

- Shortlist des modèles candidats : FinBERT, DistilRoBERTa, ModernFinBERT, Qwen3-0.6B
- Tableau comparatif (taille, VRAM, latence, accuracy sur Financial PhraseBank)
- Recommandations d'implémentation (fine-tuning, quantization, batching, fusion pondérée)
- Plan de tests A/B et mesures de fiabilité

> **Statut** : Implémenté — Sentiment Engine v2 avec 4 tiers (lexical → RoBERTa + ModernFinBERT → Qwen → Cloud)

---

### [deep-research-report-training.md](./deep-research-report-training.md)
**Stratégies d'entraînement de modèles décisionnels achat/vente.**

- Sources de données historiques (FNSPID, Alpha Vantage, Finnhub, Kaggle)
- Génération de labels buy/sell à partir des rendements futurs
- Gestion des événements corporates (dividendes, splits)
- Approches : random forest, réseaux de neurones, reinforcement learning

> **Statut** : Recherche complète — à implémenter dans la phase v4.3 (modèle de décision ML)

---

### [deep-research-report-histo-data.md](./deep-research-report-histo-data.md)
**Collecte et préparation des données historiques de marché.**

- API gratuites vs commerciales (Yahoo Finance, Alpha Vantage, Bloomberg, Refinitiv)
- Formats de données (CSV, Parquet, JSON)
- Alignement temporel prix ↔ news
- Base de données recommandées (SQLite/Postgres)

> **Statut** : Recherche complète — ingestion Finnhub + Alpha Vantage déjà opérationnelle

---

## 🏗️ Architecture du projet

Pour la documentation technique du code (structure, API, flows, déploiement), voir :

- **[`../AGENTS.md`](../AGENTS.md)** — Bible du projet pour les sessions IA
- **[`../.kimi/skills/prefect-trading/SKILL.md`](../.kimi/skills/prefect-trading/SKILL.md)** — Connaissances Prefect spécifiques au projet

---

## 🗺️ Feuille de route

| Phase | Objectif | Documentation |
|-------|----------|---------------|
| **v4.1** ✅ | Architecture event-driven + 3 stratégies | AGENTS.md |
| **v4.2** ✅ | Sentiment Engine multi-tier + token tracking | `deep-research-report-models.md` |
| **v4.3** 🔄 | Modèle de décision ML (entraînement sur historique) | `deep-research-report-training.md` |
| **v4.4** 📋 | Collecte données historiques enrichies | `deep-research-report-histo-data.md` |

---

*Dernière mise à jour : 2026-05-30*
