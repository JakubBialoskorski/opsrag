# Day-to-day commands. They run inside the app container, so the host only needs Docker.
.PHONY: up down ask ingest update update-models stats eval logs psql lint fmt test serve

APP := docker compose exec app opsrag

up:
	./start.sh

down:
	./stop.sh

# make ask Q="how do I roll back a deployment?"
ask:
	@$(APP) ask "$(Q)"

# make ingest SOURCE=helm to index a single source
ingest:
	$(APP) ingest $(if $(SOURCE),--source $(SOURCE))

# Bump docs to their newest versions and re-index only what changed.
# Lock file updates run as root only to be able to write the mounted lock files.
update:
	docker compose exec -u root app opsrag update
	$(APP) ingest

# Check for newer model versions; asks before downloading, tests before pinning
update-models:
	docker compose exec -u root app opsrag update-models

stats:
	$(APP) stats

eval:
	$(APP) eval

logs:
	docker compose logs -f app

psql:
	docker compose exec db psql -U opsrag -d opsrag

# Development, needs uv on the host
lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest -q
	node --test tests/*.test.mjs

# Web UI from source, outside Docker
serve:
	uv run opsrag serve
