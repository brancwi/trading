#!/bin/sh

echo "Démarrage Prefect server..."
export PREFECT_SERVER_API_HOST=0.0.0.0
export PREFECT_SERVER_API_PORT=4200
export PREFECT_HOME=/data
prefect server start > /tmp/server.log 2>&1 &
echo "Serveur démarré en arrière-plan"

# Attendre que le serveur soit prêt
echo "Attente serveur..."
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  if python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:4200/api/health', timeout=2)" 2>/dev/null; then
    echo "Serveur prêt!"
    break
  fi
  echo "Tentative $i..."
  sleep 2
done

echo "Démarrage runner avec serve()..."
export PREFECT_API_URL=http://localhost:4200/api
python -m trading.flows.serve
