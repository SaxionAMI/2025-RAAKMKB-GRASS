#!/usr/bin/env bash
# Build and run the Docker stack (MinIO, InfluxDB, Grafana, MediaMTX, Dagster
# and the live inference service) with checks that explain what is wrong.
#
#   ./scripts/stack.sh build          build the image (works out GitHub access itself)
#   ./scripts/stack.sh up [--gpu]     start everything and wait until it is ready
#   ./scripts/stack.sh restart        restart the inference service (after training)
#   ./scripts/stack.sh down           stop everything (data is kept)
#   ./scripts/stack.sh status         containers and the inference health check
#
# The Makefile wraps these: make build, make up, make up-gpu, make restart, ...

set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="grass/app:local"
WAIT="${STACK_WAIT:-180}"     # seconds to wait for the inference service

if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'; else B=; G=; Y=; R=; N=; fi
step() { printf '\n%s==> %s%s\n' "$B" "$*" "$N"; }
ok()   { printf '%s  ok%s %s\n' "$G" "$N" "$*"; }
warn() { printf '%s  warning:%s %s\n' "$Y" "$N" "$*"; }
die()  { printf '\n%serror:%s %s\n' "$R" "$N" "$*" >&2; exit 1; }

export GIT_TERMINAL_PROMPT=0 GCM_INTERACTIVE=never
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new"

COMPOSE=(docker compose)

preflight() {
  step "Checking Docker"
  command -v docker >/dev/null 2>&1 || die "Docker is not installed. Install Docker Desktop (Windows/macOS)
       or Docker Engine (Linux): https://docs.docker.com/get-docker/"
  docker info >/dev/null 2>&1 || die "Docker is installed but not running, or you have no permission to use it.
       Start Docker Desktop, or on Linux: sudo systemctl start docker, and add yourself
       to the docker group: sudo usermod -aG docker \$USER (then log out and in)."
  docker compose version >/dev/null 2>&1 || die "Docker Compose v2 ('docker compose') is missing. Update Docker,
       or install the compose plugin: https://docs.docker.com/compose/install/"
  ok "$(docker compose version --short 2>/dev/null | sed 's/^/compose /')"
  if [ ! -f .env ]; then cp .env.example .env; ok ".env created from .env.example"; fi
}

framework_ref() {
  sed -n 's/.*AudioOpsFramework\.git@\([^"]*\)".*/\1/p' pyproject.toml | head -n1
}

cmd_build() {
  preflight
  local ref https_url ssh_url
  ref="$(framework_ref)"
  [ -n "$ref" ] || die "Could not find the framework version (git tag) in pyproject.toml."
  # FRAMEWORK_REPO: another HTTPS URL for the framework, for example a fork.
  https_url="${FRAMEWORK_REPO:-https://github.com/SaxionAMI/2026-AMI-AudioOpsFramework.git}"
  ssh_url="ssh://git@github.com/SaxionAMI/2026-AMI-AudioOpsFramework.git"
  command -v git >/dev/null 2>&1 || die "git is not installed (sudo apt install git, or brew install git)."

  step "Building $IMAGE with framework $ref"
  if git ls-remote --exit-code "$https_url" >/dev/null 2>&1; then
    ok "framework repository is reachable over HTTPS"
    "${COMPOSE[@]}" build --build-arg FRAMEWORK_REF="$ref" --build-arg FRAMEWORK_URL="git+$https_url"
  else
    echo "  the framework repository is not public (or not reachable); trying SSH"
    command -v ssh-agent >/dev/null 2>&1 || die "No SSH client found. Install it (sudo apt install openssh-client),
       or wait until the framework repository is public."
    if [ -z "${SSH_AUTH_SOCK:-}" ] || ! ssh-add -l >/dev/null 2>&1; then
      echo "  starting an SSH agent and adding your key (you may be asked for its passphrase)"
      eval "$(ssh-agent -s)" >/dev/null
      ssh-add || die "Could not add an SSH key. Create one and add it to GitHub:
       https://docs.github.com/en/authentication/connecting-to-github-with-ssh"
    fi
    GIT_SSH_COMMAND="ssh $SSH_OPTS" git ls-remote --exit-code "$ssh_url" >/dev/null 2>&1 \
      || die "Your SSH key can't read SaxionAMI/2026-AMI-AudioOpsFramework. Check that
       'ssh -T git@github.com' greets you by name and that you have access to the repository."
    ok "framework repository is reachable over SSH"
    "${COMPOSE[@]}" build --ssh default \
      --build-arg FRAMEWORK_REF="$ref" \
      --build-arg FRAMEWORK_URL="git+$ssh_url"
  fi
  ok "image built"
}

wait_for_inference() {
  step "Waiting for the inference service (up to ${WAIT}s)"
  local i=0 health=""
  while [ "$i" -lt "$WAIT" ]; do
    if health="$(curl -fsS http://localhost:8000/health 2>/dev/null)"; then
      ok "$health"
      case "$health" in
        *'"models":[]'*) warn "no model loaded yet. Train first (make train, or sentiment_full_training_job
       in Dagster at http://localhost:3000), then run: make restart" ;;
      esac
      return 0
    fi
    sleep 3; i=$((i + 3))
  done
  warn "the inference service did not answer on http://localhost:8000/health. Its last log lines:"
  "${COMPOSE[@]}" logs --tail=40 inference || true
  return 1
}

cmd_up() {
  local gpu=0
  [ "${1:-}" = "--gpu" ] && gpu=1
  preflight
  if [ "$gpu" = 1 ]; then
    COMPOSE+=(-f docker-compose.yml -f docker-compose.gpu.yml)
    docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia \
      || die "Docker has no NVIDIA runtime. Install the NVIDIA Container Toolkit
       (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
       or start without GPU: make up"
    ok "NVIDIA runtime available"
  fi
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "  the image does not exist yet, building it first"
    cmd_build
  fi
  if grep -q '^INFLUXDB_TOKEN=dev-token-change-me' .env; then
    warn "INFLUXDB_TOKEN is the default demo value; change it in .env before using this beyond your own machine.
       (InfluxDB keeps the token it was first started with: after changing it, run
       'docker compose down -v' once to recreate the database. This deletes stored metrics.)"
  fi
  [ -f storage/sentiment/checkpoints/best_model.pt ] || warn "no trained model in storage/sentiment/ yet"

  step "Starting the stack"
  "${COMPOSE[@]}" up -d
  local ready=1
  wait_for_inference || ready=0
  if [ "$ready" = 1 ]; then printf '\n%s%sThe stack is running.%s\n' "$G" "$B" "$N"
  else printf '\n%s%sThe containers started, but the inference service is not answering yet.%s
It can take a few minutes the first time. Check with: make status, or make logs.\n' "$Y" "$B" "$N"; fi
  cat <<EOF
    Dagster     http://localhost:3000     training and batch scoring
    Inference   http://localhost:8000/docs
    Grafana     http://localhost:3001     login from .env (default admin / admin)
    MinIO       http://localhost:9001     login from .env (default minioadmin / minioadmin)
    RTSP        rtsp://localhost:8554/stream1 ... (make publish-streams)

Stop with: make down    Logs: make logs
EOF
  [ "$ready" = 1 ]
}

cmd_restart() { preflight; "${COMPOSE[@]}" restart inference; wait_for_inference; }
cmd_down()    { preflight; "${COMPOSE[@]}" down; ok "stopped (volumes and ./storage are kept)"; }
cmd_status()  {
  preflight
  "${COMPOSE[@]}" ps
  curl -fsS http://localhost:8000/health 2>/dev/null && echo || warn "inference service not reachable on :8000"
}

case "${1:-}" in
  build)   cmd_build ;;
  up)      shift; cmd_up "${1:-}" ;;
  restart) cmd_restart ;;
  down)    cmd_down ;;
  status)  cmd_status ;;
  *) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
