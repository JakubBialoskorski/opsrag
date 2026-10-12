#!/usr/bin/env bash
# Stops and removes all opsrag containers. The search index and downloaded models are kept,
# so ./start.sh is quick next time. Use --purge to delete them as well.
set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "--purge" ]; then
  read -r -p "Delete the search index, downloaded docs and models stored in Docker? [y/N] " answer
  if [ "$answer" != y ] && [ "$answer" != Y ]; then
    echo "Nothing deleted."
    exit 0
  fi
  docker compose --profile '*' down --volumes --remove-orphans
  echo "Stopped and deleted all data. ./start.sh will set everything up from scratch."
else
  docker compose --profile '*' down --remove-orphans
  echo "Stopped. Data is kept, ./start.sh brings it back."
fi
