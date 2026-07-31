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

import os
import socket
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
        # The router states its deadline on every request, so the self-check
        # below never has to guess it. It was guessing, from an env default that
        # did not match production, and told operators they were too slow when
        # they were comfortably inside the real budget.
        self.router_deadline_ms = synapse.deadline_ms
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
    # TCP only, no HTTP. The axon treats any path as a synapse name, so probing it
    # over HTTP logs an error in the miner's own logs whatever path we pick. A plain
    # connect is enough now that frpc health-checks the tunnel: the relay closes the
    # public port when the local axon stops answering, so an open port means a live
    # axon rather than just a live tunnel.
    try:
        with socket.create_connection((ip, port), timeout=8):
            pass
    except Exception as e:
        bt.logging.error(
            f"UNREACHABLE: nothing accepts connections at {ip}:{port} "
            f"({type(e).__name__}). This is the address the router uses, so you "
            f"will score zero. Behind NAT? use RELAY=1 ./run.sh. Already relaying? "
            f"the tunnel is up but your axon is not answering behind it."
        )
        return
    bt.logging.info(f"reachable: {ip}:{port} accepts connections")


# The router states its deadline on every request it sends, and that is the only
# number worth measuring against. It is NOT guessed from a default here: the
# default in the code and the value running in production were different, and a
# check that invents the threshold tells an operator to shrink a model that was
# never too big. Until a real request has arrived we simply do not claim one.
# The probe's own ceiling, deliberately well above whatever the deadline is: the
# point is to MEASURE a slow engine, not to time out on it. Timing out at the
# deadline would report "produced nothing" for a box that merely needs a smaller
# model, and those are two different repairs. It also bounds how long an engine
# that accepts connections without ever answering can hold this loop (the axon
# serves on its own thread, so no request is dropped meanwhile).
PROBE_CEILING_FLOOR_MS = 30000


def _probe_ceiling(miner) -> int:
    """Three times the router's deadline, never under 30 s."""
    known = getattr(miner, "router_deadline_ms", 0) or 0
    return max(PROBE_CEILING_FLOOR_MS, int(known) * 3)
PROBE_PROMPT = "In one short paragraph, explain what a GPU does."
PROBE_MAX_TOKENS = 256


def _inference_check(miner) -> None:
    """Generate a real answer locally, and time it against the router's deadline.

    `_self_check` above proves the axon accepts connections. It cannot prove the
    node can answer: the axon rejects an unknown route instantly, without ever
    reaching the engine. So a miner whose vLLM is dead passes the reachability
    check and still scores zero, sees "running" everywhere, and has no way to
    know. Measured on 2026-07-31: 65 registered miners were queried and answered
    nothing, every one of them reachable.

    This goes through `ExpertEngine.infer`, the same call the router's request
    lands on, so what it measures is what the router will experience. One short
    generation every five minutes is well under a percent of a serving card.
    """
    engine = getattr(miner, "engine", None)
    if engine is None:
        return
    # infer() swallows backend errors and returns an empty completion, so a dead
    # engine reports as "no answer" rather than taking the neuron down with it.
    # Belt and braces anyway: this is a diagnostic, and a diagnostic that can
    # crash the neuron would cost the operator the very earnings it exists to
    # protect.
    try:
        out = engine.infer(
            prompt=PROBE_PROMPT,
            max_tokens=PROBE_MAX_TOKENS,
            deadline_ms=_probe_ceiling(miner),
        )
    except Exception as e:  # noqa: BLE001
        bt.logging.warning(f"inference check could not run ({type(e).__name__})")
        return
    ms = out.get("latency_ms", 0.0)
    tokens = out.get("tokens_generated", 0)

    if not out.get("completion"):
        # A model still loading is the normal reason at startup and can take
        # minutes, so the first miss is a warning. A second one in a row is not
        # a cold start any more.
        streak = getattr(miner, "_infer_fail_streak", 0) + 1
        miner._infer_fail_streak = streak
        if streak == 1:
            bt.logging.warning(
                f"inference check: no answer after {ms / 1000:.0f}s. If you just "
                f"started, the model is probably still loading and the next check "
                f"in ~5 min will say so."
            )
        else:
            bt.logging.error(
                f"NOT SERVING: your engine produced nothing, {streak} checks in a "
                f"row. The router reaches you and gets silence, so you score zero "
                f"whatever else looks fine. Check that vLLM is up and holding the "
                f"model: docker logs proteus-vllm --tail 30"
            )
        return

    miner._infer_fail_streak = 0
    deadline = getattr(miner, "router_deadline_ms", None)
    if not deadline:
        # No request has arrived yet, so we do not know the budget and will not
        # invent one. The measurement is still worth printing.
        bt.logging.info(
            f"engine ok: {tokens} words in {ms / 1000:.1f}s. No request from the "
            f"router yet, so no deadline to compare against."
        )
        return
    if ms > deadline:
        bt.logging.error(
            f"TOO SLOW: {tokens} words in {ms / 1000:.1f}s, and the router stops "
            f"waiting at {deadline / 1000:.0f}s. It reaches you and you answer, "
            f"but always late, which scores the same as silence. A smaller model "
            f"on this card is the usual fix."
        )
        return
    bt.logging.info(
        f"serving: {tokens} words in {ms / 1000:.1f}s "
        f"(router deadline {deadline / 1000:.0f}s)"
    )


if __name__ == "__main__":
    with Miner() as miner:
        step = 0
        while True:
            # ponytail: no scheduler, a counter on the existing loop is enough.
            if step % 60 == 0:  # at start, then every ~5 minutes
                _self_check(miner)
                _inference_check(miner)
            bt.logging.info(f"Miner running... {time.time()}")
            step += 1
            time.sleep(5)
