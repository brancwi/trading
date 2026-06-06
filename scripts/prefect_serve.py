import asyncio
import subprocess
import time
import sys

# Démarrer le serveur en sous-processus
print("Démarrage Prefect server...")
server_proc = subprocess.Popen(
    ["prefect", "server", "start", "--host", "0.0.0.0", "--port", "4200"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Attendre que le serveur soit prêt
print("Attente serveur...")
for i in range(30):
    time.sleep(2)
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:4200/api/health", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                print("Serveur prêt!")
                break
    except Exception:
        print(f"Tentative {i+1}...")
else:
    print("Timeout serveur")
    server_proc.terminate()
    sys.exit(1)

# Déployer les flows
print("Déploiement des flows...")
subprocess.run([sys.executable, "-m", "trading.flows.deploy"], check=True)

# Démarrer le worker
print("Démarrage worker...")
worker_proc = subprocess.Popen(
    ["prefect", "worker", "start", "--pool", "default", "--limit", "5"],
    env={**os.environ, "PREFECT_API_URL": "http://localhost:4200/api"},
)

# Attendre indéfiniment
try:
    while True:
        time.sleep(1)
        if server_proc.poll() is not None:
            print("Serveur arrêté")
            break
        if worker_proc.poll() is not None:
            print("Worker arrêté")
            break
except KeyboardInterrupt:
    pass
finally:
    server_proc.terminate()
    worker_proc.terminate()
