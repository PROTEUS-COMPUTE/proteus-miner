#!/usr/bin/env bash
# PROTEUS miner launcher (Linux, Docker).
#   ./run.sh            run on a public-IP machine (cloud / dedicated). No relay.
#   RELAY=1 ./run.sh    home GPU behind CGNAT/NAT: tunnel the axon through the
#                       public relay so validators can reach it. No inbound port
#                       to open on your box.
#
# Config via env or a .env file (see .env.example): MODEL, WALLET_NAME,
# WALLET_HOTKEY, GPU_UTIL, RELAY.
set -euo pipefail

# The header above promises a .env file, and docker compose does read it, but
# this script never did: anything set there was honoured inside the containers
# and ignored by the port derivation right here. Values already exported by the
# caller still win, same precedence compose applies.
if [ -f .env ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue;; esac
    key=${line%%=*}
    [ "$key" = "$line" ] && continue
    case "$key" in *[!A-Za-z0-9_]*) continue;; esac
    value=${line#*=}
    value=${value%$'\r'}
    eval ": \${$key:=\$value}" && export "$key"
  done < .env
fi

# shared relay endpoint (public); the token only gates tunnel creation, the
# relay just forwards to public axons.
export RELAY_HOST="${RELAY_HOST:-89.116.27.24}"
export RELAY_TOKEN="${RELAY_TOKEN:-0705c2dabde90ccef3b472e8689e1b17ea75e9114afa5aa8}"
export RPC="${RPC:-wss://rpc.proteus-agent.com}"
export MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct-AWQ}"
export GPU_UTIL="${GPU_UTIL:-0.90}"
export WALLET_NAME="${WALLET_NAME:-miner}"
export WALLET_HOTKEY="${WALLET_HOTKEY:-expert1}"
export RELAY_EXTERNAL=""

RELAY="${RELAY:-0}"

# Single-GPU host: card 0, the default axon port. rig.sh overrides both per card.
export GPU_ID="${GPU_ID:-0}"
export AXON_PORT="${AXON_PORT:-8091}"

# The card model, read here because the miner container cannot see the hardware.
# It is reported to the router next to the model served, so the network can
# publish what a given GPU actually delivers rather than an estimate. Export
# GPU_NAME= yourself to keep it private; the field then stays empty and nothing
# else is affected.
export GPU_NAME="${GPU_NAME-$(nvidia-smi --query-gpu=name --format=csv,noheader \
  -i "$GPU_ID" 2>/dev/null | head -1)}"

# Containers used to be named without a card suffix. After an update compose
# creates the suffixed ones and leaves the old pair running, still holding the
# GPU and the axon port, which looks exactly like a node that starts but is never
# queried. Retire them first.
for legacy in proteus-vllm proteus-miner proteus-frpc; do
  if [ -n "$(docker ps -aq -f "name=^${legacy}$" 2>/dev/null)" ]; then
    echo "removing pre-update container $legacy"
    docker rm -f "$legacy" >/dev/null 2>&1
  fi
done

# Get the miner image without depending on the registry being reachable.
# `docker compose up` aborts on a failed pull, which would stop a miner dead for
# a reason that has nothing to do with their machine. The repository carries the
# Dockerfile, so building locally is always an option and produces the same
# image; it just costs a couple of minutes the first time.
ensure_miner_image() {
  if docker image inspect ghcr.io/proteus-compute/proteus-miner:latest >/dev/null 2>&1; then
    # Already here, so refresh it: fixes ship in this image and a miner that
    # never re-pulls keeps whatever it downloaded the first day. A failed
    # refresh is not a reason to stop: the local image still works.
    echo "refreshing the miner image..."
    if docker compose pull miner >/dev/null 2>&1; then
      echo "  up to date"
    else
      echo "  registry unreachable, keeping the image already on this machine"
    fi
    return
  fi
  echo "fetching the miner image..."
  if docker compose pull miner >/dev/null 2>&1; then
    echo "  pulled from ghcr.io"
  else
    echo "  registry unreachable, building it here instead (a few minutes, once)"
    docker compose build miner
  fi
}

if [ "$RELAY" = "1" ]; then
  HK="$HOME/.bittensor/wallets/$WALLET_NAME/hotkeys/$WALLET_HOTKEY"
  if [ ! -f "$HK" ]; then
    echo "hotkey not found: $HK"
    echo "create your wallet first (see README step 1)."
    exit 1
  fi
  # ss58 of the hotkey, then a stable public port (FNV-1a into 20000-25000),
  # the SAME scheme the desktop app uses so a hotkey always maps to one port.
  #
  # 5000 ports for a growing network means two hotkeys eventually hash to the
  # same one. The relay gives that port to whoever connects first and refuses
  # the other with "port already used", who then mines into the void with a
  # tunnel that never opens. Setting RELAY_PORT in .env is the way out: it is
  # honoured here and published on chain, so the axon and the tunnel agree.
  SS58="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['ss58Address'])" "$HK")"
  if [ -n "${RELAY_PORT:-}" ]; then
    case "$RELAY_PORT" in
      ''|*[!0-9]*) echo "RELAY_PORT must be a number, got '$RELAY_PORT'"; exit 1;;
    esac
    if [ "$RELAY_PORT" -lt 20000 ] || [ "$RELAY_PORT" -gt 25000 ]; then
      echo "RELAY_PORT must be between 20000 and 25000, got $RELAY_PORT"
      exit 1
    fi
    echo "relay port forced to $RELAY_PORT (from your environment)"
  else
    RELAY_PORT="$(python3 -c "
h=2166136261
for b in '$SS58'.encode():
    h ^= b; h = (h*16777619) & 0xffffffff
print(20000 + h % 5000)")"
  fi
  export RELAY_PORT
  export RELAY_EXTERNAL="--axon.external_ip $RELAY_HOST --axon.external_port $RELAY_PORT"
  echo "relay ON  ->  axon published at $RELAY_HOST:$RELAY_PORT  (hotkey $SS58)"
  ensure_miner_image
  docker compose --profile relay up -d
  # nc only proves the relay is listening, which stays true with a dead axon
  # behind it. Asking the axon to answer is the check that means something: 401
  # is a live axon refusing an unsigned request, 000 is an empty tunnel.
  echo "verify from another machine (401 = reachable, 000 = empty tunnel):"
  echo "  curl -s -o /dev/null -w '%{http_code}\\n' --max-time 8 -X POST http://$RELAY_HOST:$RELAY_PORT/Synapse -d '{}'"
else
  echo "relay OFF ->  serving axon directly on this host:8091 (needs a public IP)"
  ensure_miner_image
  docker compose up -d
fi

echo "logs:  docker compose logs -f miner"
