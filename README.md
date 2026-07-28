# PROTEUS miner

![NVIDIA GPU only](https://img.shields.io/badge/NVIDIA-GPU%20only-76B900?style=for-the-badge&logo=nvidia&logoColor=white) &nbsp; ![proof of useful work](https://img.shields.io/badge/proof_of_useful_work-1a1a1a?style=for-the-badge) &nbsp; ![earn $PRTS](https://img.shields.io/badge/earn-%24PRTS-9FFF00?style=for-the-badge&labelColor=1a1a1a)

Run a PROTEUS **expert** (miner) or **router** (validator) neuron and earn **$PRTS** for useful GPU compute.

PROTEUS is a sovereign Layer-1 chain (a fork of subtensor, mono-token era). It is a decentralized mixture of experts: experts serve inference, the router routes each request to the best expert and scores the answer, and $PRTS emission lands automatically on-chain, proportional to the quality of the work. This is proof of useful work, not proof of hash.

> [!TIP]
> **Early miner incentive.** PROTEUS sets aside part of the owner's share (18% of emissions) to reward the **first GPUs to join** the network. An on-chain **experts leaderboard** ranks contributors by the quality of their compute, so the earliest, strongest experts are recognized and rewarded first. Get in before the crowd.

> **Mainnet is live.** The chain is producing blocks and the public RPC is `wss://rpc.proteus-agent.com`. Follow the steps below to register an expert and start earning.

## Requirements

- **An Nvidia GPU with CUDA.** PROTEUS mining is GPU-only. There is no CPU mining: the task is heavy enough that only Nvidia GPUs are competitive.
- Python 3.10 - 3.12
- Linux (or WSL2 on Windows)

## Install

```bash
git clone https://github.com/PROTEUS-COMPUTE/proteus-miner
cd proteus-miner
pip install -r requirements.txt
```

Or the one-liner:

```bash
curl -sSL https://raw.githubusercontent.com/PROTEUS-COMPUTE/proteus-miner/main/install.sh | bash
```

## 1. Create a wallet

```bash
btcli wallet new-coldkey --wallet.name miner
btcli wallet new-hotkey  --wallet.name miner --wallet.hotkey expert1
```

Keep your coldkey mnemonic safe and offline. It controls your funds.

**Already have a wallet on the web wallet?** Reuse the same account: copy its 12-word phrase from https://app.proteus-agent.com/wallet and regenerate it here as your coldkey, then add a hotkey.

```bash
btcli wallet regen-coldkey --wallet.name miner   # paste your 12 words
btcli wallet new-hotkey    --wallet.name miner --wallet.hotkey expert1
btcli wallet list                                # the address must match your web wallet
```

## 2. Register on the network

First check that your miner can reach the chain:

```bash
btcli subnets list --network wss://rpc.proteus-agent.com
```

Then register your hotkey on the subnet. There are two ways in, and if you are
arriving with an empty wallet you want the first one.

**Proof of work (free).** You pay with compute instead of $PRTS, so it works from
a wallet with a zero balance. This is the normal path for a new miner.

```bash
btcli subnets pow-register --netuid 1 \
  --network wss://rpc.proteus-agent.com \
  --wallet.name miner --wallet.hotkey expert1
```

**Recycle (costs $PRTS).** Instant, but it burns a small amount from your
coldkey, currently around 0.02 $PRTS. Only useful once you already hold some.

```bash
btcli subnets register --netuid 1 \
  --network wss://rpc.proteus-agent.com \
  --wallet.name miner --wallet.hotkey expert1
```

Either way the transaction fee itself is zero. The Windows app uses proof of work
automatically, nothing to choose.

## 3. Run your expert (Docker, recommended)

The compose stack starts vLLM (GPU) + the miner together. You need Docker, the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
and the wallet from step 1.

```bash
cp .env.example .env      # set MODEL, WALLET_NAME, WALLET_HOTKEY
./run.sh                  # public-IP machine (cloud / dedicated server)
```

### Behind a home router / CGNAT

Most home connections sit behind NAT or carrier-grade NAT and **cannot open an
inbound port**, so the router could never reach your axon directly. Run with the
relay instead: your miner opens an **outbound** tunnel to the public relay, and
publishes the relay endpoint on-chain as its axon. Nothing to open on your box.

```bash
RELAY=1 ./run.sh
```

The launcher derives a stable public port from your hotkey and prints it, e.g.
`axon published at 89.116.27.24:2xxxx`. Verify from another machine with
`nc -zv 89.116.27.24 <port>`. Not sure whether you need it? If `./run.sh` works
and the dashboard shows your expert being queried, you don't. If your box is
behind CGNAT, use `RELAY=1`.

Logs: `docker compose logs -f miner`. Stop: `docker compose down`.

## 3b. Run your expert (bare metal)

If you prefer running Python directly (you manage vLLM yourself, see below):

```bash
python neurons/miner.py --netuid 1 \
  --subtensor.chain_endpoint wss://rpc.proteus-agent.com \
  --wallet.name miner --wallet.hotkey expert1 \
  --axon.port 8091
```

On a public-IP machine this is enough. Behind NAT/CGNAT, prefer the Docker relay
path above, or publish a reachable endpoint yourself with
`--axon.external_ip <ip> --axon.external_port <port>`.

From there the router queries your expert, scores its answers, and $PRTS emission lands on your hotkey every tempo. Track your expert on the dashboard at https://proteus-agent.com.

## Run a router (validator)

```bash
python neurons/validator.py --netuid 1 \
  --subtensor.chain_endpoint wss://rpc.proteus-agent.com \
  --wallet.name validator --wallet.hotkey default
```

## Inference backend

Experts serve model inference on an Nvidia GPU via **vLLM** (CUDA). This is the
only production-supported backend. Install the vLLM build that matches your
CUDA version, then start the server before the miner:

```bash
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3.1-8B-Instruct
```

The miner connects to `http://localhost:8000` by default. Override with the
`VLLM_HOST` and `MODEL_NAME` environment variables.

```bash
VLLM_HOST=http://localhost:8000 MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct \
  python neurons/miner.py --netuid 1 ...
```

### Dev / smoke-test backend (ollama)

For local development only, you can point the miner at a local
[Ollama](https://ollama.com) instance with `--expert.backend ollama`:

```bash
python neurons/miner.py --expert.backend ollama --netuid 1 ...
```

This lets you run the full reward loop end-to-end without an Nvidia GPU. It is
**dev / smoke-test only**: ollama on CPU is far too slow to be competitive on
mainnet, the latency factor will tank your reward, and you will earn ~0. Do not
use it as a production path.

## License

MIT.
