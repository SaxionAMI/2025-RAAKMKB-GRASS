.PHONY: help setup dev train clean

help:
	@echo "Football Stadium Sentiment"
	@echo "  make setup   poetry install + .env + runtime dirs"
	@echo "  make dev     run Dagster UI (http://localhost:3000)"
	@echo "  make clean   remove local runtime artifacts"

setup:
	poetry install
	@test -f .env || cp .env.example .env
	@mkdir -p dagster_home storage mlruns
	@echo "Drop audio under audio_input/sentiment/CrowdAudioDataport/<label>/*.wav"

dev:
	AUDIO_OPS_TASKS_DIR=$$PWD/tasks DAGSTER_HOME=$$PWD/dagster_home \
	  poetry run dagster dev -w workspace.yaml

clean:
	rm -rf dagster_home/* storage/* mlruns/*
