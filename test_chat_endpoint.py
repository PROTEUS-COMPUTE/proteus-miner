"""L'expert doit passer par le gabarit du modele, sans jamais devenir muet.

Le passage a /v1/chat/completions est ce qui empeche un modele a raisonnement de
livrer son monologue comme reponse. Mais un modele de base n'a pas de gabarit de
conversation: s'il echouait la, un mineur qui fonctionnait deviendrait silencieux.
On exige donc les deux: le chat en premier, et le repli qui marche.

Run: python test_chat_endpoint.py
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import sys
import types

if "bittensor" not in sys.modules:
    _log = types.SimpleNamespace(
        info=lambda *a: None, warning=lambda *a: None,
        error=lambda *a: None, debug=lambda *a: None, trace=lambda *a: None)
    sys.modules["bittensor"] = types.SimpleNamespace(logging=_log)

MODE = {"chat": "ok"}
SEEN = []


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append((self.path, body))
        if self.path.endswith("/chat/completions"):
            if MODE["chat"] in ("404", "400"):
                self.send_response(int(MODE["chat"])); self.end_headers(); return
            if MODE["chat"] == "slow400":
                # answers, but only after eating most of the deadline
                time.sleep(1.2)
                self.send_response(400); self.end_headers(); return
            if MODE["chat"] == "hang":
                # the busy engine: never answers inside the budget
                time.sleep(3.0)
            payload = {"choices": [{"message": {"content": "answer from chat"}}]}
        else:
            payload = {"choices": [{"text": "answer from completions"}]}
        raw = json.dumps(payload).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except OSError:
            pass  # the client gave up first, which is the point of the hang test

    def log_message(self, *a):
        pass


# threading: the hang test holds a request open, and the next test must not queue
# behind it
srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()

import os  # noqa: E402
os.environ["VLLM_HOST"] = f"http://127.0.0.1:{srv.server_address[1]}"

import importlib.util  # noqa: E402
import pathlib  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "expert_under_test", pathlib.Path(__file__).with_name("proteus") / "miner" / "expert.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
engine = _m.ExpertEngine(backend="vllm", model="test-model")


def test_chat_endpoint_is_used_first():
    SEEN.clear(); MODE["chat"] = "ok"
    out = engine.infer("q", max_tokens=64, deadline_ms=5000)
    assert out["completion"] == "answer from chat", out
    assert SEEN[0][0].endswith("/chat/completions"), SEEN[0][0]


def test_thinking_is_disabled_in_the_request():
    """C'est le reglage qui empeche le monologue chez Qwen3 et consorts."""
    SEEN.clear(); MODE["chat"] = "ok"
    engine.infer("q", max_tokens=64, deadline_ms=5000)
    body = SEEN[0][1]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}, body
    assert body["messages"] == [{"role": "user", "content": "q"}], body


def test_a_model_without_a_chat_template_still_answers():
    """Un modele de base ne doit pas devenir muet a cause de ce changement."""
    SEEN.clear(); MODE["chat"] = "404"
    out = engine.infer("q", max_tokens=64, deadline_ms=5000)
    assert out["completion"] == "answer from completions", out
    assert len(SEEN) == 2 and SEEN[1][0].endswith("/v1/completions"), SEEN


def test_a_400_is_the_real_no_template_answer():
    """vLLM repond 400, pas 404, quand le modele n'a pas de gabarit."""
    SEEN.clear(); MODE["chat"] = "400"
    out = engine.infer("q", max_tokens=64, deadline_ms=5000)
    assert out["completion"] == "answer from completions", out
    assert len(SEEN) == 2 and SEEN[1][0].endswith("/v1/completions"), SEEN


def test_a_busy_engine_is_not_asked_twice():
    """Le defaut du 2026-08-01: un timeout declenchait le repli.

    Le routeur arrete d'attendre a l'echeance. Redemander le budget complet ne
    peut donc rien rapporter, et ca remet en file exactement le travail qui
    avait deja mis le moteur en retard. Un timeout doit couter UNE echeance.
    """
    SEEN.clear(); MODE["chat"] = "hang"
    keep = _m.REQUEST_TIMEOUT_S
    _m.REQUEST_TIMEOUT_S = 1.0
    try:
        t0 = time.time()
        out = engine.infer("q", max_tokens=64, deadline_ms=1000)
        elapsed = time.time() - t0
    finally:
        _m.REQUEST_TIMEOUT_S = keep
    assert out["completion"] == "", out
    assert len(SEEN) == 1, f"the completions endpoint was called too: {SEEN}"
    assert elapsed < 2.0, f"spent {elapsed:.1f}s on a 1s deadline"


def test_the_fallback_only_gets_the_time_that_is_left():
    """Sinon deux tentatives de 18 s tiennent dans une echeance de 18 s."""
    SEEN.clear(); MODE["chat"] = "400"
    seen_timeout = []
    real = _m.ExpertEngine._vllm_completions

    def spy(self, prompt, max_tokens, timeout_s):
        seen_timeout.append(timeout_s)
        return real(self, prompt, max_tokens, timeout_s)

    _m.ExpertEngine._vllm_completions = spy
    try:
        engine.infer("q", max_tokens=64, deadline_ms=8000)
    finally:
        _m.ExpertEngine._vllm_completions = real
    assert seen_timeout and seen_timeout[0] < 8.0, seen_timeout


def test_no_retry_when_the_deadline_is_already_spent():
    """Un 400 qui arrive trop tard ne laisse pas de quoi reessayer utilement."""
    SEEN.clear(); MODE["chat"] = "slow400"
    keep = _m.REQUEST_TIMEOUT_S
    _m.REQUEST_TIMEOUT_S = 2.0
    try:
        out = engine.infer("q", max_tokens=64, deadline_ms=2000)
    finally:
        _m.REQUEST_TIMEOUT_S = keep
    assert out["completion"] == "", out
    assert len(SEEN) == 1, f"retried with {_m.MIN_FALLBACK_S}s or less left: {SEEN}"


def test_tagged_thinking_is_still_stripped():
    """Le filtre a balises reste la deuxieme ligne de defense."""
    assert _m._strip_thinking("<think>trace</think>The answer.") == "The answer."


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
