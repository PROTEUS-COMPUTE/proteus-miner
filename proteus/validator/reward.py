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

"""PROTEUS Reward function (THE CORE).

reward = quality^w_q * latency_gate * availability^w_a

The GPU-only constraint is enforced here, not by hardware detection: a CPU
cannot generate MIN_TOKENS of quality content under DEADLINE_MS. The latency
factor is a hard gate (not a gradient), so a faster but smaller model gains
no advantage over a slower but better one.

CRITICAL RULE: every field on the synapse that is filled by the miner
(completion, tokens_generated, model_used, response_hash) is a DECLARATION,
not a proof. The router never scores on a declaration. It recomputes
everything from completion text alone:
  - token count: len(completion.split()), NOT synapse.tokens_generated
  - quality: cosine embedding of completion vs reference, NOT synapse fields

See 03-INCENTIVE-MECHANISM.md for the full theory.
"""

from __future__ import annotations

import os
import typing
from dataclasses import dataclass

from proteus.protocol import InferenceSynapse


W_QUALITY = float(os.getenv("REWARD_W_QUALITY", "1.0"))
W_AVAILABILITY = float(os.getenv("REWARD_W_AVAILABILITY", "0.3"))
DEADLINE_MS = int(os.getenv("DEADLINE_MS", "9000"))
MIN_TOKENS = int(os.getenv("MIN_TOKENS", "300"))
COSINE_THRESHOLD = float(os.getenv("COSINE_THRESHOLD", "0.35"))
MAX_COMPLETION_CHARS = int(os.getenv("MAX_COMPLETION_CHARS", "8000"))
# keyword coverage: ramp, not a cliff. Below KW_FLOOR the answer is generic /
# a cheat -> crushed to 0.1. Above KW_FULL it is fully on-topic -> 1.0. In
# between, a linear ramp so an honest answer near the boundary is not hard-zeroed
# (that binary cliff was what made the honest miner's score flicker).
KW_FLOOR = float(os.getenv("KW_FLOOR", "0.15"))
KW_FULL = float(os.getenv("KW_FULL", "0.50"))

_EMBED_MODEL = None
_EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
_EMBED_MAX_SEQ = 256


@dataclass
class Scored:
    uid: int
    reward: float
    quality: float
    latency_factor: float
    availability: float


def latency_factor(latency_ms: float, deadline_ms: int = DEADLINE_MS) -> float:
    """Hard gate: 1.0 if under the deadline, 0.0 if over.

    This is NOT a gradient. A faster model gains no advantage. The deadline
    is calibrated so that only a GPU can generate MIN_TOKENS of quality
    content in time. A CPU will blow past it and get 0.
    """
    if latency_ms >= deadline_ms:
        return 0.0
    return 1.0


def quality_score(
    response: InferenceSynapse,
    reference: typing.Optional[str],
    peer_completions: typing.List[str],
    judge_fn: typing.Optional[typing.Callable[[str, str], float]] = None,
) -> float:
    """Response quality in [0, 1], combining several signals (anti-gaming).

    - token floor       : below MIN_TOKENS the answer is too short to be
                          real work, quality = 0 (kills instant-cheat bots).
                          Token count is RECOMPUTED by the router from the
                          completion text, never trusted from the synapse.
    - golden/reference  : cosine similarity to the ground truth via embeddings
    - judge model       : an LLM judge scores the response
    - cross consensus   : cosine similarity to the best peer completion

    No signal = no proof of work = quality 0. The fallback is NEVER 0.5.
    """
    if not response.completion:
        return 0.0

    completion = response.completion[:MAX_COMPLETION_CHARS]

    n_tokens = len(completion.split())
    if n_tokens < MIN_TOKENS:
        return 0.0

    if reference is None:
        return 0.0

    signals: typing.List[typing.Tuple[float, float]] = []

    signals.append((_semantic_similarity(completion, reference), 2.0))

    if judge_fn is not None:
        signals.append((float(judge_fn(response.prompt, completion)), 1.5))

    if peer_completions:
        consensus = max(
            _semantic_similarity(completion, peer[:MAX_COMPLETION_CHARS])
            for peer in peer_completions
        )
        signals.append((consensus, 1.0))

    if not signals:
        return 0.0

    num = sum(v * w for v, w in signals)
    den = sum(w for _, w in signals)
    return max(0.0, min(1.0, num / den))


def compute_reward(
    response: InferenceSynapse,
    latency_ms: float,
    availability: float,
    reference: typing.Optional[str] = None,
    peer_completions: typing.Optional[typing.List[str]] = None,
    judge_fn: typing.Optional[typing.Callable[[str, str], float]] = None,
) -> float:
    q = quality_score(response, reference, peer_completions or [], judge_fn)
    lf = latency_factor(latency_ms)
    av = max(0.0, min(1.0, availability))
    return (q ** W_QUALITY) * lf * (av ** W_AVAILABILITY)


def _get_embed_model():
    """Lazily load the embedding model. Crashes loudly if it fails.

    No silent fallback to Jaccard: that would reintroduce the cheat in
    production. The router runs on CPU; all-MiniLM-L6-v2 is ~90 MB and
    takes a few ms per sentence.
    """
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(_EMBED_MODEL_NAME)
    return _EMBED_MODEL


def _semantic_similarity(a: str, b: str) -> float:
    """Cosine similarity between sentence embeddings, gated by keyword coverage.

    Uses all-MiniLM-L6-v2 (sentence-transformers). Runs on CPU in a few ms.
    The router needs NO GPU for this; only the miner needs one.

    all-MiniLM-L6-v2 has a 256-token window. Both texts are chunked into
    256-token segments, embedded separately, and the embeddings are
    mean-pooled. This prevents silent truncation of long references.

    Two gates prevent generic ramble from scoring:
    1. Cosine threshold: below COSINE_THRESHOLD the texts are not semantically
       related. Kept well below the honest expert's typical score to give margin.
    2. Keyword coverage: a ramp on the share of the reference's technical
       keywords present in the response. This is what actually kills the
       cheater (near-zero coverage -> crushed), not the cosine.
    """
    cos = _cosine_embedding(a, b)
    if cos < COSINE_THRESHOLD:
        return 0.0
    kw_factor = _keyword_coverage(a, b)
    return cos * kw_factor


def _cosine_embedding(a: str, b: str) -> float:
    """Cosine similarity with chunked mean-pooling for long texts."""
    model = _get_embed_model()
    import numpy as np

    emb_a = _embed_chunked(model, a)
    emb_b = _embed_chunked(model, b)
    cos = np.dot(emb_a, emb_b) / (
        np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-9
    )
    return float(max(0.0, min(1.0, cos)))


def _embed_chunked(model, text: str):
    """Embed a long text by chunking into _EMBED_MAX_SEQ-token segments
    and mean-pooling the embeddings. Prevents silent truncation."""
    import numpy as np

    words = text.split()
    if len(words) <= _EMBED_MAX_SEQ:
        return model.encode([text], convert_to_numpy=True)[0]

    chunks = []
    for i in range(0, len(words), _EMBED_MAX_SEQ):
        chunk = " ".join(words[i : i + _EMBED_MAX_SEQ])
        chunks.append(chunk)

    embs = model.encode(chunks, convert_to_numpy=True)
    return np.mean(embs, axis=0)


_STOPWORDS = frozenset(
    "the a an and or but in on at to for of is are was were be been being "
    "this that these those it its as by with from has have had not no do does "
    "did done will would could should may might can must shall if then else "
    "when where why how what which who whom whose all any each every some "
    "more most less least very much many few into about above below over "
    "under again further once here there both each other such only own same "
    "so than too very just also".split()
)


def _keyword_coverage(response: str, reference: str) -> float:
    """Returns a multiplier in [0.1, 1.0] based on how many reference keywords
    appear in the response, as a RAMP (not a binary cliff).

    - ratio <= KW_FLOOR (0.15): 0.1 (crush generic / cheat responses)
    - ratio >= KW_FULL  (0.50): 1.0 (fully on-topic)
    - in between: linear. An honest answer near the boundary keeps a partial
      score instead of being hard-zeroed. A cheater with ~0 coverage is still
      crushed, so this does not weaken the anti-cheat.
    """
    ref_words = {
        w for w in reference.lower().split()
        if len(w) > 5 and w not in _STOPWORDS and w.isalpha()
    }
    if not ref_words:
        return 1.0
    resp_lower = response.lower()
    hits = sum(1 for kw in ref_words if kw in resp_lower)
    ratio = hits / len(ref_words)
    if ratio <= KW_FLOOR:
        return 0.1
    if ratio >= KW_FULL:
        return 1.0
    return 0.1 + 0.9 * (ratio - KW_FLOOR) / (KW_FULL - KW_FLOOR)
