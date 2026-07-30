#!/usr/bin/env bash
# Start one PROTEUS stack per usable card and stay in the foreground.
#
# Hive expects the miner process to live for as long as the miner is "on": when
# this script exits, the rig is shown as stopped. The work itself runs in Docker,
# so after bringing the stacks up we tail their logs, which both keeps us alive
# and feeds the log viewer in the web UI.
cd /hive/miners/custom/proteus || exit 1
. h-manifest.conf
. "$CUSTOM_CONFIG_FILENAME" 2>/dev/null

LOG="${CUSTOM_LOG_BASENAME}.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== proteus $CUSTOM_VERSION starting $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

REPO=/hive/miners/custom/proteus/src

# The stack itself lives in the public repo. Clone once, then fast-forward on
# each start so a rig picks up fixes without the owner reinstalling the package.
if [[ -d "$REPO/.git" ]]; then
  git -C "$REPO" pull --ff-only --quiet 2>/dev/null && echo "repo: updated"
else
  echo "repo: cloning"
  git clone --depth 1 --quiet https://github.com/PROTEUS-COMPUTE/proteus-miner "$REPO" || {
    echo "FATAL: cannot clone the miner repository"; exit 1; }
fi

# Hand the flight sheet through to the launcher. rig.sh does the part that has to
# happen on the machine itself: pin one card per stack, size the model from its
# VRAM, and stop adding cards once measured host memory says no more fit.
export WALLET_NAME RPC RELAY RESERVE_MB
export WALLET_HOTKEY_PREFIX="$HOTKEY_PREFIX"
[[ -n "$GPUS"  ]] && export GPUS
[[ -n "$MODEL" ]] && export MODEL_OVERRIDE="$MODEL"
[[ "${MAX_CARDS:-0}" -gt 0 ]] && export MAX_CARDS

cd "$REPO" || exit 1
chmod +x rig.sh 2>/dev/null

if ! ./rig.sh; then
  echo "FATAL: no stack could be started, see the lines above"
  exit 1
fi

# Record which cards ended up running so h-stats.sh reports the right set.
docker ps --format '{{.Names}}' \
  | grep -oP '^proteus-vllm-\K[0-9]+' | sort -n > /run/proteus-gpus 2>/dev/null

echo "=== proteus running, following miner logs ==="

# Foreground for the agent, and a useful log for the operator. Following every
# miner container interleaves them with a per-card prefix.
mapfile -t NAMES < <(docker ps --format '{{.Names}}' | grep '^proteus-miner-')
if [[ "${#NAMES[@]}" -eq 0 ]]; then
  echo "FATAL: no miner container is running"
  exit 1
fi
for n in "${NAMES[@]}"; do
  ( docker logs -f --tail 20 "$n" 2>&1 | sed "s/^/[${n#proteus-miner-}] /" ) &
done
wait
