#!/usr/bin/env bash
# PROTEUS miner setup. Nvidia GPU (CUDA) required - there is no CPU mining.
set -e

echo "== PROTEUS miner setup =="

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found. PROTEUS mining requires an Nvidia GPU with CUDA."
fi

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo ""
echo "Done. Next steps:"
echo "  1. btcli wallet new-coldkey --wallet.name miner    (or regen-coldkey to reuse your web wallet)"
echo "  2. btcli wallet new-hotkey  --wallet.name miner --wallet.hotkey expert1"
echo "  3. btcli subnets register --netuid 1 --network wss://rpc.proteus-agent.com --wallet.name miner --wallet.hotkey expert1"
echo "  4. python neurons/miner.py --netuid 1 --subtensor.chain_endpoint wss://rpc.proteus-agent.com --wallet.name miner --wallet.hotkey expert1 --axon.port 8091"
