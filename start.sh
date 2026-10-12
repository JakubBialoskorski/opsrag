#!/usr/bin/env bash
# Starts opsrag: containers, models and the search index. Safe to run again,
# it only does what is still missing.
#   ./start.sh                     everything (Kubernetes docs take the longest)
#   ./start.sh --quick             only the small Helm and Prometheus docs, to try it out fast
set -euo pipefail

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mError:\033[0m %s\n' "$*" >&2; exit 1; }
warn() { printf '\n\033[33mWarning:\033[0m %s\n' "$*" >&2; }

setup_args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) setup_args=(--source helm --source prometheus) ;;
    -h|--help) sed -n '2,5s/^# \{0,1\}//p' "$0"; exit 0 ;;
    *) fail "unknown option $1 (use --quick or --help)" ;;
  esac
  shift
done
cd "$(dirname "$0")"

command -v docker >/dev/null || fail "Docker is not installed. Get it from https://docs.docker.com/get-docker/"
docker info >/dev/null 2>&1 || fail "Docker is installed but not running. Start it and run ./start.sh again."
docker compose version >/dev/null 2>&1 || fail "'docker compose' is missing, update Docker to a recent version."

# Where Ollama runs is decided once and saved in .env (delete .env to decide again).
# macOS: natively, because Docker on a Mac can't use the GPU. Linux: in a container.
if [ ! -f .env ]; then
  mode="${OLLAMA_MODE:-$([ "$(uname)" = Darwin ] && echo host || echo docker)}"
  {
    echo "# Written by start.sh. See .env.example for all settings."
    if [ "$mode" = docker ]; then
      echo "COMPOSE_PROFILES=ollama"
      echo "DOCKER_OLLAMA_URL=http://ollama:11434"
      if docker info --format '{{json .Runtimes}}' | grep -q nvidia; then
        echo "COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml"
      fi
    fi
  } > .env
  say "Ollama will run: $([ "$mode" = docker ] && echo "in Docker" || echo "on this machine")"
fi

if ! grep -q '^COMPOSE_PROFILES=.*ollama' .env && ! curl -sf http://localhost:11434/api/version >/dev/null; then
  if [ "$(uname)" != Darwin ]; then
    fail "Ollama is not running on this machine.
  Set OLLAMA_HOST=0.0.0.0 for the Ollama service, so containers can reach it.
  Or run Ollama in Docker instead: rm .env && OLLAMA_MODE=docker ./start.sh"
  elif command -v ollama >/dev/null; then
    fail "Ollama is installed but not running. Start it in another terminal, leave it open
  and run ./start.sh again:
    ollama serve"
  else
    fail "Ollama is not installed. Install it, start it in another terminal, leave it open
  and run ./start.sh again:
    brew install ollama && ollama serve      (or https://ollama.com/download)"
  fi
fi

# Ollama on this machine can be any version, the answer checks in CI run on the one in docker-compose.yml
if ! grep -q '^COMPOSE_PROFILES=.*ollama' .env; then
  tested="$(docker compose --profile ollama config --images ollama | sed 's/^[^:]*:\([^@]*\).*/\1/')"
  running="$(curl -sf http://localhost:11434/api/version | sed -n 's/.*"version": *"\([^"]*\)".*/\1/p' || true)"
  if [ "$running" != "$tested" ]; then
    warn "Ollama ${running:-?} runs on this machine, but the answers were tested on $tested,
  so they can differ a bit. Ollama $tested: https://github.com/ollama/ollama/releases/tag/v$tested"
  fi
fi

port="$(sed -n 's/^OPSRAG_PORT=//p' .env)"
port="${port:-8000}"

say "Starting containers"
# Use the prebuilt image when it's published, otherwise build it from this repo's code
echo "Downloading the app image (about 400 MB)..."
build=()
if ! docker compose pull --quiet app 2>/dev/null; then
  echo "No prebuilt image, building it from this repo (a few minutes the first time)."
  build=(--build)
fi
docker compose up -d --wait ${build[@]+"${build[@]}"}

say "Downloading models and building the search index"
echo "Meanwhile the web UI already works at http://localhost:$port and shows the progress."
docker compose exec -T app opsrag setup "${setup_args[@]+"${setup_args[@]}"}"

say "Ready"
echo "  On this machine:   http://localhost:$port"
echo "  Stop everything:   ./stop.sh"
