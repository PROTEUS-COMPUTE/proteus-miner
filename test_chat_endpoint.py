"""L'expert doit passer par le gabarit du modele, sans jamais devenir muet.

Le passage a /v1/chat/completions est ce qui empeche un modele a raisonnement de
livrer son monologue comme reponse. Mais un modele de base n'a pas de gabarit de
conversation: s'il echouait la, un mineur qui fonctionnait deviendrait silencieux.
On exige donc les deux: le chat en premier, et le repli qui marche.

Run: python test_chat_endpoint.py
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

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
            if MODE["chat"] == "404":
                self.send_response(404); self.end_headers(); return
            payload = {"choices": [{"message": {"content": "answer from chat"}}]}
        else:
            payload = {"choices": [{"text": "answer from completions"}]}
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


srv = HTTPServer(("127.0.0.1", 0), H)
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


def test_tagged_thinking_is_still_stripped():
    """Le filtre a balises reste la deuxieme ligne de defense."""
    assert _m._strip_thinking("<think>trace</think>The answer.") == "The answer."


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
