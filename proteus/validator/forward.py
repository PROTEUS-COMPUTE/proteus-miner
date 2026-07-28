# The MIT License (MIT)
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

"""PROTEUS Router epoch loop (validator).

Orchestration: generate/collect requests -> route -> query the experts ->
score -> update scores -> (the neuron then pushes the weights on-chain).

This file is called by neurons/validator.py. It assumes base classes from the
Bittensor template (self.dendrite, self.metagraph, self.update_scores...).
"""

from __future__ import annotations

import os
import time
import secrets

import numpy as np
import bittensor as bt

from proteus.protocol import InferenceSynapse
from proteus.validator.reward import compute_reward
from proteus.validator.router import MoERouter
from proteus.utils.golden import GoldenSet
from proteus.utils.uids import check_uid_availability


EMA_ALPHA = float(os.getenv("EMA_ALPHA", "0.1"))
DEADLINE_MS = int(os.getenv("DEADLINE_MS", "9000"))


async def forward(self):
    """One iteration of the validator loop.

    `self` is the template's BaseValidatorNeuron (provides dendrite, metagraph, scores...).
    """
    if not hasattr(self, "router"):
        self.router = MoERouter()
        self.golden = GoldenSet()

    # 1) Always pull from the golden set so reference is never None.
    #    Without a reference, quality_score returns 0 (no signal = no proof = no pay).
    prompt, reference = self.golden.sample()

    # 2) route: domain -> experts
    domain = self.router.classify(prompt)
    available = _available_uids(self)
    selected = self.router.select_experts(domain, available)
    if not selected:
        bt.logging.warning("No expert available.")
        time.sleep(self.config.neuron.timeout)
        return

    # 3) query the experts (in parallel, with a deadline)
    synapse = InferenceSynapse(
        prompt=prompt, domain=domain, deadline_ms=DEADLINE_MS, nonce=_new_nonce()
    )
    axons = [self.metagraph.axons[uid] for uid in selected]

    t0 = time.time()
    responses = await self.dendrite(
        axons=axons,
        synapse=synapse,
        deserialize=False,
        timeout=DEADLINE_MS / 1000.0,
    )
    latency_ms = (time.time() - t0) * 1000.0

    # 4) score each response
    peer_completions = [
        r.completion for r in responses if r is not None and r.completion
    ]
    rewards_list = []
    for uid, resp in zip(selected, responses):
        availability = _availability(self, uid)
        others = [c for c in peer_completions if c != (resp.completion if resp else None)]

        r = compute_reward(
            response=resp if resp is not None else InferenceSynapse(prompt=prompt),
            latency_ms=latency_ms,
            availability=availability,
            reference=reference,
            peer_completions=others,
            judge_fn=getattr(self, "judge_fn", None),
        )

        # 5) update the scores (per domain + the template's global score)
        self.router.update(domain, uid, r, alpha=EMA_ALPHA)
        rewards_list.append(r)

        bt.logging.info(f"uid={uid} domain={domain} reward={r:.3f}")

    # inject the smoothed rewards into the template's score tensor
    if rewards_list:
        self.update_scores(np.array(rewards_list, dtype=np.float32), selected)

    time.sleep(self.config.neuron.timeout)


# --------- helpers ---------

def _available_uids(self) -> list:
    """UIDs of servable experts (active axon), excluding self."""
    out = []
    for uid in range(self.metagraph.n.item()):
        if uid == self.uid:
            continue
        if check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        ):
            out.append(uid)
    return out


def _availability(self, uid: int) -> float:
    """Rate of recent valid responses from the expert. TODO: sliding window."""
    return 1.0


def _new_nonce() -> str:
    return secrets.token_hex(8)
