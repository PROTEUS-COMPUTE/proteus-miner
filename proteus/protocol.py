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

"""PROTEUS protocol (Synapse).

The communication contract between the router (validator) and the experts (miners).
This defines what travels over the network: an inference request with a latency
budget, and the expert's completion + commit hash.
"""

from __future__ import annotations

import hashlib
import typing

import bittensor as bt


class InferenceSynapse(bt.Synapse):
    """An inference request routed to an expert.

    The router fills the input fields; the expert fills the output fields.
    """

    # ---------- INPUT (filled by the router) ----------
    prompt: str
    model_hint: typing.Optional[str] = None
    domain: typing.Optional[str] = None
    max_tokens: int = 512
    deadline_ms: int = 4000
    nonce: typing.Optional[str] = None

    # ---------- OUTPUT (filled by the expert) ----------
    completion: typing.Optional[str] = None
    model_used: typing.Optional[str] = None
    tokens_generated: int = 0
    response_hash: typing.Optional[str] = None
    # Declared by the miner from the GPU_NAME environment variable, empty when
    # it is unset. Diagnostic only, like `model_used`: it never reaches a score,
    # and a miner is free to leave it blank or put anything in it.
    #
    # It exists so the network can answer "what does a 4090 actually deliver on
    # a 14B", which today nobody can, ourselves included. The router already
    # logs the model and the generation time; the card is the missing third of
    # that triplet, and without it every figure published per GPU is a guess.
    gpu: typing.Optional[str] = None

    def compute_hash(self) -> str:
        """Hash of the response for the commit-reveal scheme."""
        payload = f"{self.completion or ''}{self.nonce or ''}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def deserialize(self) -> typing.Optional[str]:
        """Bittensor calls deserialize() on the validator side."""
        return self.completion


class CommitRevealSynapse(bt.Synapse):
    """Second pass to reveal the response after committing the hashes.

    Step 1: InferenceSynapse returns only response_hash.
    Step 2: CommitRevealSynapse requests the reveal (completion + nonce) and we verify the hash.
    Reserved for a future hardening pass; not used in the initial loop.
    """

    request_id: str
    completion: typing.Optional[str] = None
    nonce: typing.Optional[str] = None

    def deserialize(self) -> typing.Optional[str]:
        return self.completion
