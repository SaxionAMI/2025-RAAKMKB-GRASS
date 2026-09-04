# syntax=docker/dockerfile:1.4
#
# Shared base image. The heavy framework install (torch + CUDA + dagster…) runs
# ONCE here; both the dagster and inference services reuse it instead of each
# rebuilding the whole ~4 GB stack. Project tasks are MOUNTED at runtime
# (AUDIO_OPS_TASKS_DIR=/app/tasks), so this image stays task-agnostic.
#
# GPU is kept (nvidia-smi works in WSL). To run on GPU use the gpu overlay.
#
# Build (needs SSH agent for the private framework repo):
#   eval "$(ssh-agent -s)" && ssh-add
#   DOCKER_BUILDKIT=1 docker compose build --ssh default
#
FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libsndfile1 ffmpeg curl openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Trust github for the SSH-based install
RUN mkdir -p -m 0700 /root/.ssh && ssh-keyscan github.com >> /root/.ssh/known_hosts

ARG FRAMEWORK_REF=v0.1.0
# --mount=type=ssh   : uses your forwarded SSH key to read the private repo
# --mount=type=cache : keeps downloaded wheels between builds -> fast rebuilds
RUN --mount=type=ssh --mount=type=cache,target=/root/.cache/pip \
    pip install "git+ssh://git@github.com/SaxionAMI/2026-AMI-AudioOpsFramework.git@${FRAMEWORK_REF}"

WORKDIR /app
ENV AUDIO_OPS_TASKS_DIR=/app/tasks
EXPOSE 3000 8000

# Named stages (both are just the base) so compose can build a single shared image.
FROM base AS dagster
FROM base AS inference