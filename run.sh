#!/usr/bin/env bash
# Una sola instrucción: Redis, Phoenix, la API y el worker.
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Se creó .env desde .env.example. Cargá la clave del proveedor antes de usar la API."
fi
docker compose up --build
