import os
import sys

# Configurer l'URL API
os.environ["PREFECT_API_URL"] = "http://localhost:4200/api"

from prefect.server.api.server import create_app
import uvicorn

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4200)
