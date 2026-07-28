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


if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)
