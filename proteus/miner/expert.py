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

"""PROTEUS Expert (model manager on the miner side).

Decoupled from the neuron: the neuron handles the chain and the axon; the
ExpertEngine handles inference. Backend is swappable, but PROTEUS mining is
Nvidia GPU-only by design: the default and only production-supported backend is
vLLM on CUDA. An ollama backend is provided for local development and smoke
testing of the reward loop only; it is NOT competitive on mainnet and will
earn ~0.

Run vLLM separately, e.g.:
    python -m vllm.entrypoints.openai.api_server --model <MODEL_NAME>
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
import urllib.error

import bittensor as bt

VLLM_HOST = os.getenv("VLLM_HOST", "http://localhost:8000")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
# HTTP timeout in seconds. 0 = derive from the request deadline (recommended:
# there is no point generating past the deadline, the router scores it 0 anyway).
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "0"))

# Reasoning ("thinking") models (qwen3, deepseek-r1, QwQ...) emit a private
# <think>...</think> trace that is NOT the answer. If it leaks into the
# completion the router scores it as garbage; if the model puts everything in
# the trace the answer comes back empty. We disable thinking where the backend
# supports it AND strip any leaked trace, so these (very common, strong) models
# still earn instead of silently scoring 0.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning traces and surrounding whitespace.

    Also drops a dangling '<think>' with no closing tag (some models truncate
    mid-trace): everything from that tag on is reasoning, not answer.
    """
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    lead = text.lower().find("<think>")
    if lead != -1:
        text = text[:lead]
    return text.strip()


class ExpertEngine:
    """Loads one or more models and runs inference within a latency budget.

    backend="vllm" (default): Nvidia GPU / CUDA via the OpenAI-compatible endpoint.
    backend="ollama": local dev / smoke-test only, NOT competitive on mainnet.
    """

    def __init__(self, backend: str = "vllm", model: str = MODEL_NAME):
        self.backend = backend
        self.model = model
        bt.logging.info(
            f"ExpertEngine: backend={backend} model={model}"
            + (
                " (dev / smoke-test only, not competitive on mainnet)"
                if backend == "ollama"
                else ""
            )
        )

    def infer(self, prompt: str, max_tokens: int = 512, deadline_ms: int = 4000) -> dict:
        """Returns {completion, model_used, tokens_generated, latency_ms}.

        On any backend error the completion is empty (the router will score 0)
        so the neuron never crashes on an inference failure.
        """
        t0 = time.time()
        # no point waiting past the deadline: the router scores a late answer 0.
        timeout_s = REQUEST_TIMEOUT_S or max(5.0, deadline_ms / 1000.0)
        completion = ""
        try:
            if self.backend == "vllm":
                completion = self._vllm(prompt, max_tokens, timeout_s)
            elif self.backend == "ollama":
                completion = self._ollama(prompt, max_tokens, timeout_s)
            else:
                bt.logging.error(f"Unknown backend: {self.backend}")
        except Exception as e:
            bt.logging.error(f"Inference failed (backend={self.backend}): {e}")

        completion = _strip_thinking(completion)
        latency_ms = (time.time() - t0) * 1000.0
        return {
            "completion": completion,
            "model_used": self.model,
            "tokens_generated": len(completion.split()) if completion else 0,
            "latency_ms": latency_ms,
        }

    def _vllm(self, prompt: str, max_tokens: int, timeout_s: float) -> str:
        """vLLM on an Nvidia GPU (CUDA), via its OpenAI-compatible endpoint.

        The CHAT endpoint first, because it applies the model's own template and
        that is what keeps a reasoning model's thinking out of the answer. Fed a
        bare prompt through /v1/completions instead, Qwen3 answers and then keeps
        going out loud: "Okay, so I need to explain what a GPU does. Let me start
        by recalling what I know." Untagged, so `_strip_thinking` cannot catch it,
        and the router scores the monologue as part of the answer. Observed on a
        real miner on 2026-07-31, burning all 512 tokens every request.

        Falls back to the raw completions call if chat is unavailable: a base
        model with no chat template must keep working rather than go silent.
        """
        text = self._vllm_chat(prompt, max_tokens, timeout_s)
        return self._vllm_completions(prompt, max_tokens, timeout_s) if text is None else text

    def _vllm_chat(self, prompt: str, max_tokens: int, timeout_s: float):
        """Returns the answer, or None if this model cannot be chatted with."""
        req = urllib.request.Request(
            f"{VLLM_HOST}/v1/chat/completions",
            data=json.dumps({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": False,
                # Qwen3 and friends read this from their template. Templates that
                # do not know it simply ignore it.
                "chat_template_kwargs": {"enable_thinking": False},
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                data = json.loads(r.read())
            return data["choices"][0]["message"].get("content", "")
        except Exception as e:  # noqa: BLE001
            bt.logging.warning(
                f"chat endpoint unavailable ({type(e).__name__}), using raw completions"
            )
            return None

    def _vllm_completions(self, prompt: str, max_tokens: int, timeout_s: float) -> str:
        req = urllib.request.Request(
            f"{VLLM_HOST}/v1/completions",
            data=json.dumps({
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stream": False,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            data = json.loads(r.read())
            return data["choices"][0].get("text", "")

    def _ollama(self, prompt: str, max_tokens: int, timeout_s: float) -> str:
        """Local Ollama. dev / smoke-test only, NOT competitive on mainnet."""
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "think": False,  # answer, not reasoning trace (ignored by non-thinking models)
                "options": {"num_predict": max_tokens},
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return json.loads(r.read()).get("response", "")


if __name__ == "__main__":
    # self-check for the thinking-trace stripper (run: python -m proteus.miner.expert)
    assert _strip_thinking("<think>reasoning here</think>The answer.") == "The answer."
    assert _strip_thinking("Answer <think>a</think> and <think>b</think> end") == "Answer  and  end"
    assert _strip_thinking("Good answer with no trace.") == "Good answer with no trace."
    assert _strip_thinking("visible<think>truncated trace, no close tag") == "visible"
    assert _strip_thinking("<think>only reasoning, no answer</think>") == ""
    assert _strip_thinking("") == ""
    print("expert._strip_thinking self-check OK")
