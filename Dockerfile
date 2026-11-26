# syntax=docker/dockerfile:1.4
#
# One image for both the dagster and the inference service. The framework and
# its heavy dependencies (torch, dagster, fastapi) are installed once here. The
# project's tasks are mounted at runtime (AUDIO_OPS_TASKS_DIR=/app/tasks), so the
# image does not need rebuilding when the task changes.
#
# Build:
#   docker compose build
# While the framework repository is private, build over SSH instead:
#   eval "$(ssh-agent -s)" && ssh-add
#   DOCKER_BUILDKIT=1 docker compose build --ssh default \
#     --build-arg FRAMEWORK_URL=git+ssh://git@github.com/SaxionAMI/2026-AMI-AudioOpsFramework.git
#
# By default torch comes from PyPI, which on Linux is the CUDA build: the same
# image runs on CPU and, with the GPU compose overlay, on an NVIDIA GPU. For a
# much smaller CPU-only image set TORCH_INDEX=https://download.pytorch.org/whl/cpu
# (in .env or with --build-arg).
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libsndfile1 ffmpeg curl openssh-client \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p -m 0700 /root/.ssh && ssh-keyscan github.com >> /root/.ssh/known_hosts

ARG TORCH_INDEX=
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision ${TORCH_INDEX:+--index-url "$TORCH_INDEX"}

ARG FRAMEWORK_URL=git+https://github.com/SaxionAMI/2026-AMI-AudioOpsFramework.git
ARG FRAMEWORK_REF=v1.2.0
RUN --mount=type=ssh --mount=type=cache,target=/root/.cache/pip \
    pip install "audio-ops-framework[train,serve,s3,metrics] @ ${FRAMEWORK_URL}@${FRAMEWORK_REF}"

WORKDIR /app
ENV AUDIO_OPS_TASKS_DIR=/app/tasks \
    AUDIO_OPS_DISABLE_BUILTIN_TASKS=true \
    DAGSTER_HOME=/app/dagster_home
EXPOSE 3000 8000
