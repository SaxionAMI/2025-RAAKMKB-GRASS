# Football stadium sentiment (GRASS). `make help` lists the commands.
# Local targets use the .venv created by `make setup`, so no need to activate it.
.PHONY: help setup train predict serve dev test list build up up-gpu restart down logs ps status clean publish-streams

BIN  := $(if $(wildcard .venv/bin/python),.venv/bin/,)
TASK := sentiment

# Only this project's task, not the framework's bundled example.
export AUDIO_OPS_DISABLE_BUILTIN_TASKS ?= true

help:
	@echo "Football stadium sentiment (GRASS)"
	@echo ""
	@echo "On your machine:"
	@echo "  make setup                 one-time setup: .venv, PyTorch, framework, .env"
	@echo "  make train                 features, splits, training, evaluation, batch scoring"
	@echo "  make predict FILE=x.wav    score one or more files with the trained model"
	@echo "  make serve                 inference API on http://localhost:8000/docs"
	@echo "  make dev                   Dagster UI on http://localhost:3000"
	@echo "  make test                  check that the task works with the installed framework"
	@echo ""
	@echo "Docker stack (MinIO, InfluxDB, Grafana, MediaMTX, Dagster, inference):"
	@echo "  make up | up-gpu           build if needed, start, wait until ready"
	@echo "  make restart               restart the inference service (after training)"
	@echo "  make down | logs | status  stop, follow logs, show status"
	@echo "  make build                 rebuild the image (after changing the framework version)"
	@echo "  make publish-streams       publish demo_clips/match1..4.mp4 as RTSP stream1..4"

setup:
	bash scripts/setup.sh

train:
	$(BIN)audio-ops run $(TASK)

predict:
	@test -n "$(FILE)" || (echo "usage: make predict FILE=clip.wav" && exit 1)
	$(BIN)audio-ops predict $(TASK) $(FILE)

serve:
	$(BIN)audio-ops serve

dev:
	$(BIN)audio-ops dev

test:
	$(BIN)audio-ops validate $(TASK)
	$(BIN)python -m pytest -q

list:
	$(BIN)audio-ops list

build:
	bash scripts/stack.sh build

up:
	bash scripts/stack.sh up

up-gpu:
	bash scripts/stack.sh up --gpu

restart:
	bash scripts/stack.sh restart

down:
	bash scripts/stack.sh down

status ps:
	bash scripts/stack.sh status

logs:
	docker compose logs -f --tail=100

clean:
	rm -rf dagster_home/* storage/* mlruns/*

# Publish 4 demo RTSP streams from local files (put match1..4.mp4 in ./demo_clips/).
publish-streams:
	@command -v ffmpeg >/dev/null || (echo "ffmpeg is not installed" && exit 1)
	@for i in 1 2 3 4; do \
	  if [ -f demo_clips/match$$i.mp4 ]; then \
	    ffmpeg -nostdin -re -stream_loop -1 -i demo_clips/match$$i.mp4 -vn -c:a aac \
	      -f rtsp rtsp://localhost:8554/stream$$i > /dev/null 2>&1 & \
	    echo "publishing stream$$i from demo_clips/match$$i.mp4"; \
	  else echo "skipping stream$$i: demo_clips/match$$i.mp4 not found"; fi; \
	done
	@echo "Stop them with: pkill -f 'rtsp://localhost:8554/stream'"
