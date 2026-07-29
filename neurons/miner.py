# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 PROTEUS Compute

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import typing
import urllib.error
import urllib.request

import bittensor as bt

import proteus

from proteus.base.miner import BaseMinerNeuron
from proteus.protocol import InferenceSynapse
from proteus.miner.expert import ExpertEngine


class Miner(BaseMinerNeuron):
    """PROTEUS expert neuron (miner).

    Wires up the axon, receives an InferenceSynapse from the router, calls the
    ExpertEngine for real inference on an Nvidia GPU (vLLM / CUDA), and returns
    the completion + commit hash. Mining is GPU-only by design.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        backend = getattr(self.config.expert, "backend", "vllm")
        self.engine = ExpertEngine(backend=backend)
        bt.logging.info(f"PROTEUS expert initialized: backend={backend}")

    async def forward(
        self, synapse: InferenceSynapse
    ) -> InferenceSynapse:
        """Handle an inference request from the router."""
        result = self.engine.infer(
            prompt=synapse.prompt,
            max_tokens=synapse.max_tokens,
            deadline_ms=synapse.deadline_ms,
        )
        synapse.completion = result["completion"]
        synapse.model_used = result["model_used"]
        synapse.tokens_generated = result["tokens_generated"]
        synapse.response_hash = synapse.compute_hash()
        return synapse

    async def blacklist(
        self, synapse: InferenceSynapse
    ) -> typing.Tuple[bool, str]:
        """Only accept registered validators with enough stake."""

        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: InferenceSynapse) -> float:
        """Prioritize high-stake validators."""
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return 0.0

        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )
        priority = float(
            self.metagraph.S[caller_uid]
        )
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


def _self_check(miner) -> None:
    """Ask our own published axon to answer, from the outside.

    Every silent failure this project has hit looked healthy from the inside: the
    process is up, vLLM serves, the tunnel is registered, and the node is simply
    unreachable. The address published on-chain is the only one the router will
    ever use, so it is the only one worth testing. A live axon rejects an unsigned
    request with 401; anything else means no request will ever arrive.
    """
    ip = getattr(miner.axon, "external_ip", None)
    port = getattr(miner.axon, "external_port", None)
    if not ip or not port:
        return
    url = f"http://{ip}:{port}/InferenceSynapse"
    try:
        req = urllib.request.Request(url, data=b"{}", method="POST")
        with urllib.request.urlopen(req, timeout=8) as r:
            code = r.status
    except urllib.error.HTTPError as e:
        code = e.code  # 401/403 mean something answered, which is what we want
    except Exception as e:
        bt.logging.error(
            f"UNREACHABLE: nothing answers at {ip}:{port} ({type(e).__name__}). "
            f"This is the address the router uses, so you will score zero. "
            f"Behind NAT? use RELAY=1 ./run.sh. Already relaying? the tunnel is up "
            f"but your axon is not answering behind it."
        )
        return
    if code in (401, 403):
        bt.logging.info(f"reachable: axon answers at {ip}:{port}")
    else:
        bt.logging.warning(f"axon at {ip}:{port} answered {code}, expected 401")


if __name__ == "__main__":
    with Miner() as miner:
        step = 0
        while True:
            # ponytail: no scheduler, a counter on the existing loop is enough.
            if step % 60 == 0:  # at start, then every ~5 minutes
                _self_check(miner)
            bt.logging.info(f"Miner running... {time.time()}")
            step += 1
            time.sleep(5)
