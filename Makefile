.PHONY: help setup dev build up up-gpu down logs ps clean publish-streams

CMP := docker compose

help:
	@echo "Football Stadium Sentiment"
	@echo "  make setup    poetry install (local, no Docker) + .env + dirs"
	@echo "  make dev      run Dagster locally (no Docker)"
	@echo "  make build    build the stack images (needs SSH agent for the framework repo)"
	@echo "  make up       start the full stack (minio/influx/grafana/mediamtx/dagster/inference)"
	@echo "  make up-gpu   start the stack with the GPU overlay"
	@echo "  make down     stop the stack"
	@echo "  make logs     tail stack logs"
	@echo "  make clean    remove local runtime artifacts"
	@echo ""
	@echo "Ports: dagster :3000  inference :8000  grafana :3001  minio :9001  influx :8086  rtsp :8554"

setup:
	poetry install
	@test -f .env || cp .env.example .env
	@mkdir -p dagster_home storage mlruns
	@echo "Drop audio under audio_input/sentiment/CrowdAudioDataport/<label>/*.wav"

dev:
	AUDIO_OPS_TASKS_DIR=$$PWD/tasks DAGSTER_HOME=$$PWD/dagster_home \
	  poetry run dagster dev -w workspace.yaml

build:
	@test -f .env || cp .env.example .env
	DOCKER_BUILDKIT=1 $(CMP) build --ssh default

up:
	@test -f .env || cp .env.example .env
	$(CMP) up -d

up-gpu:
	$(CMP) -f docker-compose.yml -f docker-compose.gpu.yml up -d

down:
	$(CMP) down

logs:
	$(CMP) logs -f --tail=100

ps:
	$(CMP) ps

clean:
	rm -rf dagster_home/* storage/* mlruns/*

# Publish 4 demo RTSP streams from local files (put match*.mp4 in ./demo_clips/).
publish-streams:
	@for i in 1 2 3 4; do \
	  ffmpeg -nostdin -re -stream_loop -1 -i demo_clips/match$$i.mp4 -an -c:a aac \
	    -f rtsp rtsp://localhost:8554/stream$$i > /dev/null 2>&1 & \
	  echo "publishing stream$$i"; \
	done
