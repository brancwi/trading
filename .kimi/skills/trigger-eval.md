# Évaluation des descriptions de triggering

## Skill : prefect-mastery

### Should trigger (doit s'activer)

| # | Query | Confiance |
|---|-------|-----------|
| 1 | "Comment orchestrer des workflows Python ?" | Haute |
| 2 | "Je veux remplacer Airflow par quelque chose de plus simple" | Haute |
| 3 | "Scheduling de tâches Python avec retry" | Haute |
| 4 | "Pipeline ETL avec caching" | Haute |
| 5 | "Exécuter des tâches en parallèle en Python" | Haute |
| 6 | "Event-driven workflow avec webhooks" | Haute |
| 7 | "@flow et @task en Python" | Haute |
| 8 | "Déployer un workflow sur Docker avec Prefect" | Haute |
| 9 | "Human-in-the-loop approval workflow" | Haute |
| 10 | "Task runner Python avec UI" | Haute |
| 11 | "Comment faire du cron en Python proprement" | Haute |
| 12 | "Gérer retry et cache sur mes scripts" | Haute |
| 13 | "Prefect serve() vs deploy()" | Haute |
| 14 | "work pool et worker Prefect" | Haute |
| 15 | "emit_event et automation" | Haute |

### Should NOT trigger (ne doit PAS s'activer)

| # | Query | Raison |
|---|-------|--------|
| 1 | "Comment configurer FastAPI ?" | Pas d'orchestration |
| 2 | "Optimiser une requête SQL" | Pas de workflow |
| 3 | "Créer un Dockerfile pour mon app" | Docker général |
| 4 | "Configurer Kubernetes" | K8s général |
| 5 | "Trading, simulation, portfolio" | → prefect-trading |
| 6 | "FinBERT, sentiment analysis" | → prefect-trading |
| 7 | "Hermes, command bus" | → prefect-trading |
| 8 | "Metabase dashboard" | Pas de Prefect |
| 9 | "Simple script Python sans scheduling" | Pas besoin d'orchestration |
| 10 | "Comment faire une boucle for" | Trop basique |

---

## Skill : prefect-trading

### Should trigger (doit s'activer)

| # | Query | Confiance |
|---|-------|-----------|
| 1 | "Ajouter une stratégie au système de trading" | Haute |
| 2 | "Modifier le flow de sentiment analysis" | Haute |
| 3 | "Le listener ne reçoit plus les prix" | Haute |
| 4 | "Hermes ne peut pas liquider le portfolio" | Haute |
| 5 | "Nouveau signal FinBERT" | Haute |
| 6 | "Backtest sur la stratégie rotation" | Haute |
| 7 | "Configurer les clés API Finnhub" | Haute |
| 8 | "Telegram ne notifie plus les trades" | Haute |
| 9 | "Ajouter un portefeuille ninja" | Haute |
| 10 | "Event-driven trading flow" | Haute |
| 11 | "Simulation de trading algorithmique" | Haute |
| 12 | "Mon bot de trading ne tourne plus" | Haute |
| 13 | "Table signals dans SQLite" | Haute |
| 14 | "Metabase dashboard trading" | Haute |
| 15 | "Command bus pour ordres Hermes" | Haute |

### Should NOT trigger (ne doit PAS s'activer)

| # | Query | Raison |
|---|-------|--------|
| 1 | "Prefect en général" | → prefect-mastery |
| 2 | "Comment faire un workflow Prefect ?" | → prefect-mastery |
| 3 | "FastAPI sans lien trading" | Pas ce projet |
| 4 | "Analyse financière sans code" | Pas technique |
| 5 | "Trading sur un autre projet" | Pas ce répertoire |
| 6 | "Docker général" | Pas lié |
| 7 | "Kubernetes général" | Pas lié |
| 8 | "SQL général" | Pas lié |
| 9 | "Python basique" | Trop basique |
| 10 | "Investissement boursier" | Pas algorithmique |
