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

"""PROTEUS Router (MoE gating).

Classifies each request into a domain, then selects the top-k experts by
historical score on that domain, with a little random exploration. This is the
"gating network" of the mixture of experts.
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
import typing


TOP_K = int(os.getenv("TOP_K", "3"))
EXPLORATION_EPS = float(os.getenv("EXPLORATION_EPS", "0.15"))

DOMAINS = ["code", "resume", "qa", "creative", "general"]


class MoERouter:
    def __init__(self, embed_fn: typing.Optional[typing.Callable[[str], list]] = None):
        self.embed_fn = embed_fn
        # score_ema[domain][uid] -> float
        self.score_ema: typing.Dict[str, typing.Dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def classify(self, prompt: str) -> str:
        """Classify the request into a domain.

        TODO: replace with embed(prompt) + nearest domain cluster.
        Keyword heuristic to bootstrap.
        """
        p = prompt.lower()
        if any(k in p for k in ("def ", "function", "python", "bug", "code", "```")):
            return "code"
        if any(k in p for k in ("resume", "summar", "tl;dr")):
            return "resume"
        if p.strip().endswith("?") or p.startswith(("what", "how", "why", "when", "who")):
            return "qa"
        if any(k in p for k in ("write", "create", "imagine", "story", "poem")):
            return "creative"
        return "general"

    def select_experts(self, domain: str, available_uids: typing.List[int]) -> typing.List[int]:
        """Top-k by score on the domain + epsilon-exploration."""
        if not available_uids:
            return []

        ranked = sorted(
            available_uids,
            key=lambda uid: self.score_ema[domain].get(uid, 0.0),
            reverse=True,
        )
        selected = ranked[:TOP_K]

        pool = [u for u in available_uids if u not in selected]
        if pool and random.random() < EXPLORATION_EPS:
            selected.append(random.choice(pool))

        return selected

    def update(self, domain: str, uid: int, reward: float, alpha: float = 0.1) -> None:
        """Update the EMA of the expert's score on this domain."""
        prev = self.score_ema[domain].get(uid, 0.0)
        self.score_ema[domain][uid] = alpha * reward + (1 - alpha) * prev

    def global_scores(self, uids: typing.List[int]) -> typing.Dict[int, float]:
        """Aggregated score per expert (mean across domains) for set_weights()."""
        out: typing.Dict[int, float] = {}
        for uid in uids:
            vals = [self.score_ema[d].get(uid, 0.0) for d in self.score_ema]
            out[uid] = sum(vals) / len(vals) if vals else 0.0
        return out
