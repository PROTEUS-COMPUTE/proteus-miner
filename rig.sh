#!/usr/bin/env bash
# PROTEUS multi-GPU launcher.
#
# One stack per card: each GPU gets its own vLLM, its own miner neuron, its own
# hotkey and therefore its own uid and its own share of the emission. Running one
# miner across a rig would earn a single share; running N earns N.
#
#   ./rig.sh                 public IP, one stack per usable card
#   RELAY=1 ./rig.sh         behind CGNAT, tunnel each axon through the relay
#   GPUS=0,2,5 ./rig.sh      only these cards
#   ./rig.sh --plan          print the plan and exit, start nothing
#   ./rig.sh --down          stop every stack this script started
#
# How many cards actually start is measured, not assumed. Host RAM is the binding
# constraint on a rig: a containerised vLLM can hold several GB of system memory
# on top of its VRAM, and the figure depends on the image, the model and the
# machine. So the first stack is started alone, its real footprint is measured on
# THIS host, and that measurement decides how many more will fit. A card that
# cannot be fed is not started: an OOM twenty minutes into a model download is a
# worse outcome than a card left idle with a clear reason.
set -uo pipefail

RELAY="${RELAY:-0}"
RESERVE_MB="${RESERVE_MB:-2048}"   # host RAM kept for the OS and everything else
SETTLE_S="${SETTLE_S:-20}"         # grace period before measuring a stack
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-900}"

log()  { printf '%s  %s\n' "$(date -u +%H:%M:%S)" "$*"; }
warn() { printf '%s  WARN  %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die()  { printf '%s  ERROR %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight ---

command -v docker >/dev/null || die "docker not found. On Hiveon see the README, it is not installed by default."
docker compose version >/dev/null 2>&1 || die "docker compose v2 not found (the 'docker-compose' v1 binary will not do)."
command -v nvidia-smi >/dev/null || die "nvidia-smi not found. Install the NVIDIA driver first."

# A flight sheet is copied from the docs, so the docs' placeholder gets copied
# with it. Caught here rather than handed to vLLM, which would fail much later
# with a message about a model nobody meant to ask for.
case "${MODEL_OVERRIDE:-}" in
  *'<'*|*'>'*) die "MODEL is still the documentation placeholder (${MODEL_OVERRIDE}). Put a real Hugging Face id such as Qwen/Qwen2.5-7B-Instruct-AWQ, or remove the MODEL line entirely and each card is sized from its own VRAM." ;;
esac

# Pull first, and SAY SO. This image is a few hundred MB and is absent on a fresh
# Hiveon host, so folding the download into the check below (whose output has to
# be silenced, it is a probe) left the operator staring at a frozen screen for
# minutes with no way to tell a slow download from a hang.
CUDA_PROBE=nvidia/cuda:12.4.0-base-ubuntu22.04
if ! docker image inspect "$CUDA_PROBE" >/dev/null 2>&1; then
  echo "gpu check: downloading $CUDA_PROBE (a few hundred MB, once)"
  docker pull "$CUDA_PROBE" || die "cannot download $CUDA_PROBE. Check the network and the disk."
fi

echo "gpu check: asking docker for the cards..."
if ! docker run --rm --gpus all "$CUDA_PROBE" nvidia-smi -L >/dev/null 2>&1; then
  die "docker cannot reach the GPUs. Install nvidia-container-toolkit and run: nvidia-ctk runtime configure --runtime=docker && systemctl restart docker"
fi

# ------------------------------------------------------------------ inputs ---

# index,VRAM(MiB),name for every card, then the subset we were asked for.
mapfile -t ALL_GPUS < <(nvidia-smi --query-gpu=index,memory.total,name --format=csv,noheader,nounits)
[ "${#ALL_GPUS[@]}" -gt 0 ] || die "nvidia-smi reported no GPU."

declare -a GPU_IDX GPU_VRAM GPU_NAME
for row in "${ALL_GPUS[@]}"; do
  IFS=, read -r idx vram name <<<"$row"
  idx="${idx// /}"; vram="${vram// /}"; name="$(echo "$name" | sed 's/^ *//;s/ *$//;s/^NVIDIA //')"
  if [ -n "${GPUS:-}" ] && [[ ",${GPUS// /}," != *",$idx,"* ]]; then continue; fi
  GPU_IDX+=("$idx"); GPU_VRAM+=("$vram"); GPU_NAME+=("$name")
done
[ "${#GPU_IDX[@]}" -gt 0 ] || die "GPUS='${GPUS:-}' selected no card."

mem_available_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
disk_free_gb()     { df -BG --output=avail /var/lib/docker 2>/dev/null | tail -1 | tr -dc '0-9'; }

# A card runs the largest model its VRAM can hold with headroom for the KV cache,
# unless MODEL_OVERRIDE forces one on every card. The override still respects the
# 8 GB floor: a card that cannot serve anything is skipped either way.
model_for_vram() {
  local mb="$1"
  [ "$mb" -lt 7000 ] && { echo ""; return; }
  if [ -n "${MODEL_OVERRIDE:-}" ]; then echo "$MODEL_OVERRIDE"; return; fi
  # (a MODEL still carrying the doc placeholder is rejected in the preflight)
  if   [ "$mb" -ge 22000 ]; then echo "Qwen/Qwen2.5-14B-Instruct-AWQ"
  elif [ "$mb" -ge 15000 ]; then echo "Qwen/Qwen2.5-7B-Instruct-AWQ"
  else echo "Qwen/Qwen2.5-3B-Instruct-AWQ"; fi
}

# The pinned vLLM build predates the RTX 50 series and carries no sm_120 kernels.
image_for_gpu() {
  case "${1^^}" in
    *5050*|*5060*|*5070*|*5080*|*5090*) echo "vllm/vllm-openai:v0.26.0-cu129" ;;
    *) echo "${VLLM_IMAGE:-vllm/vllm-openai:v0.6.6.post1}" ;;
  esac
}

# ------------------------------------------------------------------- plan -----

AVAIL_MB="$(mem_available_mb)"
DISK_GB="$(disk_free_gb)"

log "host        $(nproc) cores, ${AVAIL_MB} MB RAM available, ${DISK_GB:-?} GB free for images"
log "gpus        ${#GPU_IDX[@]} selected"
for i in "${!GPU_IDX[@]}"; do
  m="$(model_for_vram "${GPU_VRAM[$i]}")"
  printf '              [%s] %-28s %5s MB  %s\n' \
    "${GPU_IDX[$i]}" "${GPU_NAME[$i]}" "${GPU_VRAM[$i]}" "${m:-UNSUPPORTED, needs 8 GB VRAM}"
done

if [ "${1:-}" = "--plan" ]; then
  log "plan only, nothing started"
  exit 0
fi

if [ "${1:-}" = "--down" ]; then
  for idx in "${GPU_IDX[@]}"; do
    GPU_ID="$idx" docker compose -p "proteus-gpu$idx" --profile relay down 2>/dev/null
    log "gpu $idx     stopped"
  done
  exit 0
fi

# ------------------------------------------------------------------ launch ----

# Refresh the miner image before starting anything. `docker compose up` reuses a
# :latest tag that is already on disk without ever asking the registry, so a rig
# could restart for weeks and keep the build it downloaded on day one. Every card
# shares this image, so one pull covers the whole rig. A failed pull is not a
# reason to stop: the local image still runs.
MINER_IMAGE=ghcr.io/proteus-compute/proteus-miner:latest
echo "miner image: refreshing $MINER_IMAGE"
if docker pull "$MINER_IMAGE" >/dev/null 2>&1; then
  echo "  up to date"
else
  docker image inspect "$MINER_IMAGE" >/dev/null 2>&1 \
    && echo "  registry unreachable, keeping the image already on this machine" \
    || die "cannot download $MINER_IMAGE and there is no local copy. Check the network."
fi

export RELAY_HOST="${RELAY_HOST:-89.116.27.24}"
export RELAY_TOKEN="${RELAY_TOKEN:-0705c2dabde90ccef3b472e8689e1b17ea75e9114afa5aa8}"
export RPC="${RPC:-wss://rpc.proteus-agent.com}"
export GPU_UTIL="${GPU_UTIL:-0.90}"
export WALLET_NAME="${WALLET_NAME:-miner}"
# Rigs are RAM-bound, not throughput-bound. Eager mode skips CUDA graph capture,
# which is where a containerised vLLM balloons; the deadline the router enforces
# leaves ample room for the slower path.
export VLLM_EXTRA="${VLLM_EXTRA:---enforce-eager}"
export SWAP_SPACE="${SWAP_SPACE:-0}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-20}"

relay_port_for() {  # FNV-1a over the ss58, same mapping as run.sh and the desktop app
  python3 -c "
h=2166136261
for b in '$1'.encode():
    h ^= b; h = (h*16777619) & 0xffffffff
print(20000 + h % 5000)"
}

wait_healthy() {  # $1 = gpu id, returns non-zero on timeout
  local idx="$1" waited=0
  while [ "$waited" -lt "$HEALTH_TIMEOUT_S" ]; do
    if docker exec "proteus-vllm-$idx" \
         python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3)" \
         >/dev/null 2>&1; then
      return 0
    fi
    if [ "$(docker inspect -f '{{.State.Running}}' "proteus-vllm-$idx" 2>/dev/null)" != "true" ]; then
      return 2
    fi
    sleep 10; waited=$((waited + 10))
  done
  return 1
}

stack_rss_mb() {  # host RAM actually held by one stack
  local idx="$1" total=0 v
  for c in "proteus-vllm-$idx" "proteus-miner-$idx"; do
    v="$(docker stats --no-stream --format '{{.MemUsage}}' "$c" 2>/dev/null | awk '{print $1}')"
    case "$v" in
      *GiB) v="$(awk -v x="${v%GiB}" 'BEGIN{print int(x*1024)}')" ;;
      *MiB) v="${v%MiB}"; v="${v%.*}" ;;
      *) v=0 ;;
    esac
    total=$((total + v))
  done
  echo "$total"
}

started=0; skipped=0; measured_mb=0

for i in "${!GPU_IDX[@]}"; do
  idx="${GPU_IDX[$i]}"; vram="${GPU_VRAM[$i]}"; name="${GPU_NAME[$i]}"
  model="$(model_for_vram "$vram")"

  if [ -z "$model" ]; then
    warn "gpu $idx     skipped, ${vram} MB VRAM is below the 8 GB minimum"
    skipped=$((skipped + 1)); continue
  fi

  if [ "${MAX_CARDS:-0}" -gt 0 ] && [ "$started" -ge "${MAX_CARDS}" ]; then
    warn "gpu $idx     skipped, MAX_CARDS=${MAX_CARDS} reached"
    skipped=$((skipped + 1)); continue
  fi

  # Budget check. The first card is a leap of faith by definition: nothing has
  # been measured yet, so we only require that some RAM is free. Every card after
  # it is checked against what the previous one really took.
  avail="$(mem_available_mb)"
  if [ "$started" -gt 0 ]; then
    need=$((measured_mb + RESERVE_MB))
    if [ "$avail" -lt "$need" ]; then
      warn "gpu $idx     skipped, ${avail} MB RAM available and a stack needs ~${measured_mb} MB plus ${RESERVE_MB} MB reserved"
      skipped=$((skipped + 1)); continue
    fi
  elif [ "$avail" -lt $((RESERVE_MB + 2048)) ]; then
    die "only ${avail} MB RAM available, not enough for a single stack"
  fi

  hotkey="${WALLET_HOTKEY_PREFIX:-expert}$((idx + 1))"
  hk="$HOME/.bittensor/wallets/$WALLET_NAME/hotkeys/$hotkey"
  if [ ! -f "$hk" ]; then
    warn "gpu $idx     skipped, hotkey '$hotkey' does not exist (btcli wallet new-hotkey --wallet.name $WALLET_NAME --wallet.hotkey $hotkey)"
    skipped=$((skipped + 1)); continue
  fi

  export GPU_ID="$idx"
  export MODEL="$model"
  export WALLET_HOTKEY="$hotkey"
  export AXON_PORT=$((8091 + idx))
  export VLLM_IMAGE="$(image_for_gpu "$name")"
  export RELAY_EXTERNAL=""

  profile=()
  if [ "$RELAY" = "1" ]; then
    ss58="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['ss58Address'])" "$hk")"
    export RELAY_PORT="$(relay_port_for "$ss58")"
    export RELAY_EXTERNAL="--axon.external_ip $RELAY_HOST --axon.external_port $RELAY_PORT"
    profile=(--profile relay)
    log "gpu $idx     $name, $model, hotkey $hotkey, axon $RELAY_HOST:$RELAY_PORT"
  else
    log "gpu $idx     $name, $model, hotkey $hotkey, axon :$AXON_PORT"
  fi

  if ! docker compose -p "proteus-gpu$idx" "${profile[@]}" up -d >/dev/null 2>&1; then
    warn "gpu $idx     failed to start, see: docker compose -p proteus-gpu$idx logs"
    skipped=$((skipped + 1)); continue
  fi

  log "gpu $idx     waiting for vLLM to load the model"
  case "$(wait_healthy "$idx"; echo $?)" in
    0) : ;;
    2) warn "gpu $idx     vLLM exited during startup: docker logs proteus-vllm-$idx"
       docker compose -p "proteus-gpu$idx" "${profile[@]}" down >/dev/null 2>&1
       skipped=$((skipped + 1)); continue ;;
    *) warn "gpu $idx     vLLM did not become healthy within ${HEALTH_TIMEOUT_S}s, leaving it running"
       ;;
  esac

  sleep "$SETTLE_S"
  rss="$(stack_rss_mb "$idx")"
  # Calibrate on the first stack, then track the worst case seen so far.
  [ "$rss" -gt "$measured_mb" ] && measured_mb="$rss"
  started=$((started + 1))
  log "gpu $idx     ready, holding ${rss} MB of host RAM"
done

# ------------------------------------------------------------------ report ----

echo
log "started     $started stack(s), skipped $skipped"
[ "$measured_mb" -gt 0 ] && log "footprint   ~${measured_mb} MB host RAM per stack, measured on this host"
log "ram left    $(mem_available_mb) MB"
echo
if [ "$started" -gt 0 ]; then
  log "logs        docker compose -p proteus-gpu<id> logs -f miner"
  log "stop        ./rig.sh --down"
fi
[ "$started" -gt 0 ] || exit 1
