"""patches/0450: an omitted seed is a fresh draw. No GPU and no installed TensorFold.

The image applies every patch in order, then falls back to ``patch -p1`` when ``git apply`` cannot see the
submodule gitdir. This rebuilds that server module and checks the sampling rule the HTTP app actually runs.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRIOR = (
    "0002-glm-tool-call-parser.patch",
    "0090-glm-request-knobs.patch",
    "0110-glm-session-cache.patch",
    "0150-glm-mia-wins.patch",
    "0160-glm-openai-compat.patch",
    "0210-glm-prompt-tokens.patch",
)
FRESH = ROOT / "patches" / "0450-glm-fresh-seed.patch"


def _sampling_cls():
    try:
        from tensorfold.engine.exact_sampling import Sampling
        return Sampling
    except ImportError:
        @dataclass(frozen=True)
        class Sampling:
            seed: int
            temperature: float = 1.0
            top_k: int = 20
            top_p: float = 0.95

        name = "tensorfold.engine.exact_sampling"
        for parent in ("tensorfold", "tensorfold.engine"):
            sys.modules.setdefault(parent, types.ModuleType(parent))
        mod = types.ModuleType(name)
        mod.Sampling = Sampling
        sys.modules[name] = mod
        return Sampling


def _git_apply(root: Path, patch: Path, dest: Path) -> None:
    subprocess.run(
        ["git", "apply", f"--include={dest.relative_to(root)}", str(patch)],
        cwd=root, check=True,
    )


def _patched_source(tmp: Path) -> str:
    """server.py as the image has it after 0210, then 0450 via the same ``patch -p1`` fallback Docker uses."""

    root = tmp / "tree"
    dest = root / "src/tensorfold/cuda/server.py"
    dest.parent.mkdir(parents=True)
    dest.write_bytes((ROOT / "vendor/TensorFold/src/tensorfold/cuda/server.py").read_bytes())
    for name in PRIOR:
        _git_apply(root, ROOT / "patches" / name, dest)
    before = dest.read_text()
    subprocess.run(
        ["patch", "-p1", "--forward", "--batch"],
        cwd=root, input=FRESH.read_bytes(), check=True,
    )
    via_patch = dest.read_text()
    dest.write_text(before)
    _git_apply(root, FRESH, dest)
    assert dest.read_text() == via_patch
    return via_patch


def _load(source: str):
    _sampling_cls()
    spec = importlib.util.spec_from_loader("server_fresh_seed", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(source, "server.py", "exec"), mod.__dict__)
    return mod


class _Enc:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _Tok:
    def encode(self, text: str, add_special_tokens: bool = False):
        return _Enc([ord(c) for c in text[:8]] or [1])

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return " ".join(str(i) for i in ids)


class _Engine:
    def __init__(self) -> None:
        self.eos: tuple[int, ...] = ()
        self.seen = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft: bool = True):
        self.seen.append(sampling)
        on_tokens([11])
        return {"prefill_s": 0.01, "cached": 0}


def _app(mod):
    app = mod.App.__new__(mod.App)
    app.tok = _Tok()
    app.sampling = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
    app.max_tokens = 16
    app.lock = threading.Lock()
    app.engine = _Engine()
    app.default_thinking = False
    return app


def _pack(seed: int) -> int:
    """The job-header split (batchplan.encode_header / the rank-0 generate header)."""

    seed = int(seed) & 0xFFFFFFFFFFFFFFFF
    lo, hi, top = seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF, seed >> 62
    return (top << 62) | (hi << 31) | lo


def _json_number(seed: int) -> int:
    """What a JSON number keeps: an IEEE-754 double, then back to int."""

    return int(struct.unpack("<d", struct.pack("<d", float(seed)))[0])


def test_patch_file_names_the_rule():
    text = FRESH.read_text()
    added = [line[1:] for line in text.splitlines() if line.startswith("+") and not line.startswith("+++")]
    blob = "\n".join(added)
    assert "secrets.randbits(53)" in blob
    assert "seed_for" not in blob
    assert 'stats["seed"]' in blob


def test_omitted_seed_is_fresh_and_explicit_seed_replays(tmp_path, monkeypatch):
    mod = _load(_patched_source(tmp_path))
    app = _app(mod)
    drawn = iter([111, 222, 333])
    seen_bits = []

    def fake_bits(n: int) -> int:
        seen_bits.append(n)
        return next(drawn)

    monkeypatch.setattr("secrets.randbits", fake_bits)

    first = app.sampling_for({"temperature": 1.0}, [1, 2, 3])
    second = app.sampling_for({}, [9, 9, 9])
    assert seen_bits == [53, 53]
    assert first.seed == 111 and second.seed == 222
    assert first.temperature == 1.0 and second.top_k == 20 and second.top_p == 0.95

    explicit = app.sampling_for({"seed": 7, "temperature": 1, "top_k": 4, "top_p": 0.5}, [1, 2, 3])
    again = app.sampling_for({"seed": "7", "temperature": 1}, [4, 5])
    zero = app.sampling_for({"seed": 0, "temperature": 1}, [1, 2, 3])
    assert explicit.seed == 7 and explicit.top_k == 4 and explicit.top_p == 0.5
    assert again.seed == 7
    assert zero.seed == 0
    assert seen_bits == [53, 53]

    assert app.sampling_for({"temperature": 0, "seed": 7}, [1]) is None
    assert app.sampling_for({"temperature": 0.0}, [1]) is None
    assert app.sampling_for({"seed": None, "temperature": -1}, [1]) is None
    assert seen_bits == [53, 53]


def test_run_echoes_the_seed_the_engine_used(tmp_path, monkeypatch):
    mod = _load(_patched_source(tmp_path))
    app = _app(mod)
    seq = iter([123456789, 987654321])
    monkeypatch.setattr("secrets.randbits", lambda n: next(seq))

    omitted = app.run({"prompt": "same", "max_tokens": 4, "temperature": 1}, False, lambda delta: True)
    replay = app.run({"prompt": "same", "max_tokens": 4, "temperature": 1, "seed": omitted["stats"]["seed"]},
                     False, lambda delta: True)
    other = app.run({"prompt": "same", "max_tokens": 4, "seed": 5}, False, lambda delta: True)
    greedy = app.run({"prompt": "same", "max_tokens": 4, "temperature": 0, "seed": 5}, False, lambda delta: True)

    assert omitted["stats"]["seed"] == 123456789
    assert app.engine.seen[0].seed == 123456789
    assert replay["stats"]["seed"] == 123456789
    assert app.engine.seen[1].seed == 123456789
    assert other["stats"]["seed"] == 5
    assert "seed" not in greedy["stats"]
    assert app.engine.seen[3] is None
    source = (tmp_path / "tree/src/tensorfold/cuda/server.py").read_text()
    assert '"tensorfold": result["stats"]' in source
    assert 'end["tensorfold"] = result["stats"]' in source


def test_fresh_seeds_survive_the_job_header_and_json():
    limit = 2**53 - 1
    for seed in (0, 1, 2**31 - 1, 2**31, 2**52, limit):
        assert _pack(seed) == seed
        assert _json_number(seed) == seed
        assert json.loads(json.dumps({"seed": seed}))["seed"] == seed
    # An explicit seed wider than the fresh draw still round-trips the header. JSON may not.
    wide = 2**63 - 1
    assert _pack(wide) == wide
    assert _json_number(limit + 2) != limit + 2


@pytest.mark.skipif(not FRESH.is_file(), reason="patch missing")
def test_patched_source_does_not_hash_the_prompt(tmp_path):
    source = _patched_source(tmp_path)
    assert "seed_for" not in source
    assert "secrets.randbits(53)" in source
