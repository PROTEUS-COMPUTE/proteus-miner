"""The self-check must tell a dead engine from a slow one.

Both score zero, but the repair is not the same: one is "vLLM is not running",
the other is "this model is too big for this card". Reporting either as the
other sends the operator to the wrong place, which is how 65 miners spent a day
serving nothing while every indicator they had said fine.

Run: python test_inference_check.py
"""
import sys
import types

# bittensor is heavy and irrelevant here: the module only needs bt.logging, so
# stub it before importing the neuron rather than pulling the whole SDK in.
_records = []


class _Log:
    @staticmethod
    def info(m): _records.append(("info", m))

    @staticmethod
    def warning(m): _records.append(("warning", m))

    @staticmethod
    def error(m): _records.append(("error", m))

    @staticmethod
    def trace(m): _records.append(("trace", m))


sys.modules.setdefault("bittensor", types.SimpleNamespace(logging=_Log))

# Import the function under test without importing the neuron's dependencies:
# exec only the block we need, so the test stays free of the bittensor stack.
import re
import pathlib

src = pathlib.Path(__file__).with_name("neurons").joinpath("miner.py").read_text()
start = src.index("# The router scores a late answer zero")
end = src.index('if __name__ == "__main__":')
ns = {"os": __import__("os"), "bt": types.SimpleNamespace(logging=_Log)}
exec(compile(src[start:end], "miner.py", "exec"), ns)  # noqa: S102
_inference_check = ns["_inference_check"]
DEADLINE = ns["ROUTER_DEADLINE_MS"]


class FakeMiner:
    def __init__(self, completion, latency_ms, tokens=120):
        self.engine = types.SimpleNamespace(
            infer=lambda **kw: {
                "completion": completion,
                "latency_ms": latency_ms,
                "tokens_generated": tokens,
                "model_used": "test",
            }
        )
        self.last_kwargs = None


def run(miner):
    _records.clear()
    _inference_check(miner)
    return _records[:]


def test_healthy_is_quiet():
    out = run(FakeMiner("a real answer", DEADLINE - 3000))
    assert [lvl for lvl, _ in out] == ["info"], out
    assert "serving" in out[0][1]


def test_slow_engine_is_reported_as_slow_not_as_dead():
    out = run(FakeMiner("a real answer", DEADLINE + 5000))
    assert out[0][0] == "error", out
    assert "TOO SLOW" in out[0][1]
    # the operator must not be sent looking for a stopped container
    assert "NOT SERVING" not in out[0][1]


def test_first_silence_is_a_warning_because_the_model_may_be_loading():
    m = FakeMiner("", 12000)
    out = run(m)
    assert out[0][0] == "warning", out
    assert "still loading" in out[0][1]


def test_second_silence_in_a_row_escalates():
    m = FakeMiner("", 12000)
    run(m)
    out = run(m)
    assert out[0][0] == "error", out
    assert "NOT SERVING" in out[0][1]


def test_one_good_answer_clears_the_streak():
    m = FakeMiner("", 12000)
    run(m)
    m.engine.infer = lambda **kw: {
        "completion": "back", "latency_ms": 1000, "tokens_generated": 10}
    run(m)
    m.engine.infer = lambda **kw: {
        "completion": "", "latency_ms": 12000, "tokens_generated": 0}
    out = run(m)
    # streak was reset, so this is a first miss again: warning, not error
    assert out[0][0] == "warning", out


def test_probe_ceiling_is_above_the_router_deadline():
    """Timing out at the deadline would make every slow miner look dead."""
    assert ns["PROBE_CEILING_MS"] > DEADLINE * 2


def test_a_missing_engine_never_raises():
    m = types.SimpleNamespace()
    _records.clear()
    _inference_check(m)  # must not raise
    assert _records == []


def test_an_engine_that_raises_never_takes_the_neuron_down():
    """A diagnostic that can crash the miner would cost the very earnings it guards."""
    def boom(**kw):
        raise RuntimeError("engine exploded")
    m = FakeMiner("x", 100)
    m.engine.infer = boom
    out = run(m)
    assert out[0][0] == "warning", out
    assert "could not run" in out[0][1]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
