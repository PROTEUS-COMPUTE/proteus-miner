# PROTEUS miner

![NVIDIA GPU only](https://img.shields.io/badge/NVIDIA-GPU%20only-76B900?style=for-the-badge&logo=nvidia&logoColor=white) &nbsp; ![proof of useful work](https://img.shields.io/badge/proof_of_useful_work-1a1a1a?style=for-the-badge) &nbsp; ![earn $PRTS](https://img.shields.io/badge/earn-%24PRTS-9FFF00?style=for-the-badge&labelColor=1a1a1a)

Run a PROTEUS **expert** (miner) or **router** (validator) neuron and earn **$PRTS** for useful GPU compute.

PROTEUS is a sovereign Layer-1 chain (a fork of subtensor, mono-token era). It is a decentralized mixture of experts: experts serve inference, the router routes each request to the best expert and scores the answer, and $PRTS emission lands automatically on-chain, proportional to the quality of the work. This is proof of useful work, not proof of hash.

> [!TIP]
> **Early miner incentive.** PROTEUS sets aside part of the owner's share (18% of emissions) to reward the **first GPUs to join** the network. An on-chain **experts leaderboard** ranks contributors by the quality of their compute, so the earliest, strongest experts are recognized and rewarded first. Get in before the crowd.

> **Mainnet is live.** The chain is producing blocks and the public RPC is `wss://rpc.proteus-agent.com`. Follow the steps below to register an expert and start earning.

## Contents

- [Requirements](#requirements)
- [Install](#install)
- [1. Create a wallet](#1-create-a-wallet)
- [2. Register on the network](#2-register-on-the-network)
- [3. Run your expert (Docker, recommended)](#3-run-your-expert-docker-recommended)
  - [Behind a home router / CGNAT](#behind-a-home-router--cgnat)
- [3b. Run your expert (bare metal)](#3b-run-your-expert-bare-metal)
- [4. Multi-GPU hosts and mining rigs](#4-multi-gpu-hosts-and-mining-rigs)
  - [How many cards will actually start](#how-many-cards-will-actually-start)
  - [What it configures for you](#what-it-configures-for-you)
  - [Preparing a Hiveon host](#preparing-a-hiveon-host)
  - [Hotkeys](#hotkeys)
- [Run a router (validator)](#run-a-router-validator)
- [Inference backend](#inference-backend)
  - [Dev / smoke-test backend (ollama)](#dev--smoke-test-backend-ollama)
- [License](#license)

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
`axon published at 89.116.27.24:2xxxx`.

To check that you are really reachable, ask your axon to answer:

```bash
curl -s -o /dev/null -w '%{http_code}\n' --max-time 8 \
  -X POST http://89.116.27.24:<your-port>/Synapse -d '{}'
```

`401` is the answer you want: your axon is alive and correctly refusing an
unsigned request. `000` means the port accepts TCP but nothing is behind it,
so the router cannot reach you and you will not be scored.

Do **not** use `nc -zv` for this. It only proves the relay is listening, which
stays true even when your own node is down.

Not sure whether you need the relay? If `./run.sh` works and the dashboard shows
your expert being queried, you don't. If your box is behind CGNAT, use `RELAY=1`.

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

## 4. Multi-GPU hosts and mining rigs

<img src="docs/img/hiveon.png" alt="Hiveon" height="72" align="left" hspace="20" vspace="4">

**One GPU is one expert.** Each card runs its own vLLM and its own neuron, under
its own hotkey, and therefore holds its own uid and earns its own share of the
emission. A single miner spread across a rig would earn one share; eight cards
running eight stacks earn eight.

<br clear="left">

`rig.sh` starts one stack per usable card:

```bash
./rig.sh --plan           # inspect the host, print the plan, start nothing
./rig.sh                  # public IP
RELAY=1 ./rig.sh          # behind CGNAT
GPUS=0,2,5 ./rig.sh       # a subset
./rig.sh --down           # stop every stack
```

### How many cards will actually start

Host RAM is the binding constraint on a rig, not VRAM. A containerised vLLM holds
several GB of system memory on top of the weights it puts in VRAM, and the exact
figure depends on the image, the model and the machine, so it cannot be assumed.

`rig.sh` measures it instead. The first stack is started alone; once vLLM answers
`/health`, its real resident memory on this host is measured, and that number
decides how many further cards can be fed. A card that would not fit is not
started, and the reason is printed. Nothing is launched on a promise.

```
09:14:02  host        16 cores, 61440 MB RAM available, 780 GB free for images
09:14:02  gpus        6 selected
            [0] GeForce RTX 4070 SUPER        12282 MB  Qwen/Qwen2.5-7B-Instruct-AWQ
            [1] GeForce RTX 4070 SUPER        12282 MB  Qwen/Qwen2.5-7B-Instruct-AWQ
            ...
09:14:03  gpu 0       RTX 4070 SUPER, Qwen/Qwen2.5-7B-Instruct-AWQ, hotkey expert1, axon 89.116.27.24:21344
09:14:03  gpu 0       waiting for vLLM to load the model
09:17:41  gpu 0       ready, holding 4118 MB of host RAM
...
09:31:08  started     6 stack(s), skipped 0
09:31:08  footprint   ~4210 MB host RAM per stack, measured on this host
09:31:08  ram left    35832 MB
```

A card is skipped, never silently dropped, when it has under 8 GB of VRAM, when
its hotkey does not exist, or when the remaining RAM cannot cover another stack.

### What it configures for you

| | |
|---|---|
| model | chosen per card from its VRAM: 3B under 15 GB, 7B under 22 GB, 14B above |
| vLLM image | the build carrying kernels for that card, including RTX 50-series |
| GPU pinning | `device_ids` per stack, so stacks never contend for a card |
| hotkey | `expert<N+1>` for GPU `N`, override with `WALLET_HOTKEY_PREFIX` |
| axon port | `8091 + N`, and a distinct relay port derived from each hotkey |
| model cache | one shared Docker volume, so the weights are downloaded once |
| memory | eager mode and no swap reservation, which is what keeps a stack near 4 GB instead of 16 |

### Preparing a Hiveon host

Hiveon is Debian based, but it ships neither Docker nor the container toolkit:

```bash
apt update && apt install -y docker.io
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt update && apt install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
```

**Do not let any guide update your NVIDIA driver.** Hiveon manages its own, and
replacing it breaks the rest of the rig. The toolkit works with the driver already
installed.

Two host limits decide whether a rig is worth it, and both are printed by
`./rig.sh --plan` before anything is downloaded:

- **RAM.** Rigs are commonly built with 4-8 GB for eight cards, which is enough
  for one or two stacks, not eight. This is the usual reason a rig serves fewer
  cards than it holds.
- **Disk.** Rigs often boot from a 32-64 GB drive. The vLLM image alone is several
  GB. The model cache is shared between stacks, so it is paid once, not per card.

PCIe risers are not a concern here. Weights are loaded once and inference stays on
the card, so x1 links cost startup time and nothing after that.

### Hotkeys

One hotkey per card, all under the same coldkey:

```bash
for i in 1 2 3 4 5 6; do
  btcli wallet new-hotkey --wallet.name miner --wallet.hotkey expert$i
  btcli subnets pow-register --netuid 1 --network wss://rpc.proteus-agent.com \
    --wallet.name miner --wallet.hotkey expert$i
done
```

Registration is rate limited on-chain to twelve per 360-block interval, roughly 72
minutes. Past that the chain returns `Custom error: 5` and you retry in the next
interval.

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
