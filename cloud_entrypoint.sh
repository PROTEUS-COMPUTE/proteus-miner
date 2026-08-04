#!/bin/bash
# One-container startup for a rented GPU: register, serve, tunnel, mine.
#
# The Linux launcher (run.sh) drives docker compose. A pod has no docker, so the
# same sequence happens here in one process tree. Everything the operator has to
# supply is PROTEUS_OWNER; the rest is read off the machine.
set -euo pipefail

PY=/opt/proteus/bin/python          # bittensor's venv, never the image's python
VLLM_PORT=8000
AXON_PORT="${AXON_PORT:-8091}"
RELAY_HOST="${RELAY_HOST:-89.116.27.24}"
RELAY_TOKEN="${RELAY_TOKEN:-0705c2dabde90ccef3b472e8689e1b17ea75e9114afa5aa8}"

say() { echo "[proteus] $*"; }

# --- the card, read here because only this container can see it --------------
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)"
GPU_NAME_AUTO="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^NVIDIA //' || true)"
export GPU_NAME="${GPU_NAME-$GPU_NAME_AUTO}"
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)"

# ONE card. This image runs a single vLLM and a single neuron, unlike rig.sh
# which brings up one full stack per card on a Hiveon box.
#
# Refusing rather than warning, because the failure is expensive and invisible:
# a four-card pod would be billed four times and mine once, everything would
# look healthy, and nobody reads a startup log after the first minute. Rent a
# single-GPU instance instead. The override exists for someone who knowingly
# wants one card of a bigger machine.
if [ "${GPU_COUNT:-1}" -gt 1 ] && [ "${PROTEUS_ALLOW_IDLE_GPUS:-0}" != "1" ]; then
  echo ""
  echo "  This machine has $GPU_COUNT GPUs and this image only uses ONE."
  echo "  You would pay for $GPU_COUNT cards and mine with 1."
  echo ""
  echo "  Rent a single-GPU instance instead. Multi-card hosts are served by"
  echo "  rig.sh, which starts one full stack per card."
  echo "  To proceed anyway, set PROTEUS_ALLOW_IDLE_GPUS=1."
  echo ""
  exit 2
fi

# Same thresholds as rig.sh, which were recalibrated on measured latency and not
# on model size: the 3B is SLOWER in practice than the 7B on the same card,
# because it is the model the slowest machines run.
pick_model() {
  if   [ "${VRAM_MB:-0}" -ge 22000 ]; then echo "Qwen/Qwen2.5-14B-Instruct-AWQ"
  elif [ "${VRAM_MB:-0}" -ge 12000 ]; then echo "Qwen/Qwen2.5-7B-Instruct-AWQ"
  else echo "Qwen/Qwen2.5-3B-Instruct-AWQ"; fi
}
MODEL="${MODEL:-$(pick_model)}"

say "gpu    ${GPU_NAME:-unknown} (${VRAM_MB} MB)"
say "model  $MODEL"

# --- 1. identity on chain ----------------------------------------------------
# Refuses to go further without a valid owner address, so a misconfigured pod
# stops before it starts billing for work nobody can claim.
"$PY" /app/cloud_register.py

HK_FILE="$HOME/.bittensor/wallets/${WALLET_NAME:-miner}/hotkeys/${WALLET_HOTKEY:-expert1}"
SS58="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['ss58Address'])" "$HK_FILE")"

# --- 2. the relay ------------------------------------------------------------
# Deliberately the default here, not a fallback. Both providers document that a
# pod's public port changes on every reset, so an axon published from their
# values is stale after the first restart and the router quietly stops reaching
# it. This port is derived from the hotkey and never moves.
RELAY_PORT="$("$PY" -c "
h=2166136261
for b in '$SS58'.encode():
    h ^= b; h = (h*16777619) & 0xffffffff
print(20000 + h % 5000)")"
export RELAY_HOST RELAY_TOKEN RELAY_PORT AXON_PORT
export MINER_HOST=127.0.0.1          # single container: the axon is local
say "relay  $RELAY_HOST:$RELAY_PORT"
frpc -c /app/frpc.toml &
FRPC_PID=$!

# --- 3. inference ------------------------------------------------------------
say "starting vllm, the first run downloads the weights"
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 127.0.0.1 --port "$VLLM_PORT" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --max-num-seqs "${MAX_NUM_SEQS:-20}" \
  --swap-space "${SWAP_SPACE:-0}" \
  ${VLLM_EXTRA:-} &
VLLM_PID=$!

# A dead vllm must not leave the neuron answering nothing: the router would keep
# querying and scoring zero, which looks exactly like a network problem.
cleanup() { kill "$VLLM_PID" "$FRPC_PID" 2>/dev/null || true; }
trap cleanup EXIT

say "waiting for the model to load"
for _ in $(seq 1 240); do          # weights can take many minutes on a cold pod
  if curl -sf "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then say "vllm died during startup"; exit 1; fi
  sleep 10
done
curl -sf "http://127.0.0.1:$VLLM_PORT/health" >/dev/null || { say "vllm never became ready"; exit 1; }
say "model ready"

# --- 4. mine -----------------------------------------------------------------
export VLLM_HOST="http://127.0.0.1:$VLLM_PORT"
# The neuron names the model in every request it sends to vLLM, so the two have
# to agree. docker-compose.yml passes this alongside VLLM_HOST; leaving it out
# here made the miner ask for the image's default, Meta-Llama-3.1-8B, of a
# server holding Qwen. vLLM answers 404 on EVERY request and the node scores
# zero while looking perfectly healthy from the outside.
export MODEL_NAME="$MODEL"
exec "$PY" /app/neurons/miner.py \
  --netuid "${NETUID:-1}" \
  --subtensor.chain_endpoint "${SUBTENSOR_NETWORK:-wss://rpc.proteus-agent.com}" \
  --wallet.name "${WALLET_NAME:-miner}" \
  --wallet.hotkey "${WALLET_HOTKEY:-expert1}" \
  --axon.port "$AXON_PORT" \
  --expert.backend vllm \
  --axon.external_ip "$RELAY_HOST" \
  --axon.external_port "$RELAY_PORT"
