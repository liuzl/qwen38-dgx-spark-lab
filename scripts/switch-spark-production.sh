#!/usr/bin/env bash
# Gated switch of the Spark native-LoRA service to a new vLLM image, with rollback.
#
#   1. stop the running $CONTAINER, rename it to $BACKUP_CONTAINER, restart=no
#   2. start $NEW_IMAGE under the same name/port via serve-native-lora.sh
#      (same aliases and flags; health check included)
#   3. wait for /health, then run gate-spark.sh; any failure removes the new
#      container and restores the backup under the original name
#
# Required: the serve-native-lora.sh variables (MODEL_DIR DRAFT_DIR ADAPTER_DIR),
#   NEW_IMAGE, NEW_CACHE_DIR, IMAGE_FIXTURE.
# Optional: EXPECT_OLD_IMAGE (abort unless production runs it), REFERENCE_LABEL
#   (gate run to compare against), PREFIX_CACHE_RETENTION_MODE (env for v0.28.0),
#   STARTUP_TIMEOUT (default 1200 s).
#
# usage: switch-spark-production.sh [--dry-run]
set -euo pipefail
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${NEW_IMAGE:?}" "${NEW_CACHE_DIR:?}" "${IMAGE_FIXTURE:?}"
CONTAINER="${CONTAINER:-qwen38-vllm-native-lora}"
PORT="${PORT:-18102}"
BACKUP_CONTAINER="${BACKUP_CONTAINER:-$CONTAINER-before-switch-$(date -u +%Y%m%d)}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1200}"
DRY="${1:-}"
log() { echo "[$(date -u +%FT%TZ)] $*"; }

docker image inspect "$NEW_IMAGE" >/dev/null
current_image="$(docker inspect -f '{{.Config.Image}}' "$CONTAINER")"
if [[ -n "${EXPECT_OLD_IMAGE:-}" && "$current_image" != "$EXPECT_OLD_IMAGE" ]]; then
  echo "$CONTAINER runs $current_image, expected $EXPECT_OLD_IMAGE; aborting" >&2; exit 1
fi
if docker inspect "$BACKUP_CONTAINER" >/dev/null 2>&1; then
  echo "$BACKUP_CONTAINER already exists; aborting" >&2; exit 1
fi
log "plan: $CONTAINER ($current_image) -> $NEW_IMAGE, backup $BACKUP_CONTAINER, cache $NEW_CACHE_DIR"
[[ "$DRY" == --dry-run ]] && { log "dry run: no changes made"; exit 0; }

rollback() {
  log "ROLLBACK: restoring $BACKUP_CONTAINER as $CONTAINER"
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker rename "$BACKUP_CONTAINER" "$CONTAINER"
  docker update --restart unless-stopped "$CONTAINER" >/dev/null
  docker start "$CONTAINER" >/dev/null
  log "rollback container started"
}

log "window start"
docker stop "$CONTAINER" >/dev/null
docker rename "$CONTAINER" "$BACKUP_CONTAINER"
docker update --restart no "$BACKUP_CONTAINER" >/dev/null
trap rollback ERR

IMAGE="$NEW_IMAGE" CACHE_DIR="$NEW_CACHE_DIR" CONTAINER="$CONTAINER" PORT="$PORT" "$SCRIPTS/serve-native-lora.sh"

deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  [[ "$(docker inspect -f '{{.State.Status}}' "$CONTAINER")" == running ]] || { log "new container exited"; false; }
  ((SECONDS < deadline)) || { log "not healthy within ${STARTUP_TIMEOUT}s"; false; }
  sleep 15
done
log "new image healthy; running gates"

if "$SCRIPTS/gate-spark.sh" "switch-$(date -u +%Y%m%dT%H%M)" "http://127.0.0.1:$PORT" "${REFERENCE_LABEL:-}"; then
  trap - ERR
  log "ALL GATES PASSED. Rollback copy kept as $BACKUP_CONTAINER (restart=no)."
else
  log "gates failed"; false
fi
