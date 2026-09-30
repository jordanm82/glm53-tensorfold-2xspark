"""patches/0420: BF16 o_proj transplant. No torch and no safetensors module.

The rules live in the patch (the image applies it onto TensorFold). This loads that module from the
patch and checks the real donor header plus a fake rank-local copy.
"""

from __future__ import annotations

import importlib.util
import json
import os
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Not a path in git. Set ABLIT_DONOR_HOST to the local donor safetensors to check its header.
DONOR = Path(os.environ["ABLIT_DONOR_HOST"]) if os.environ.get("ABLIT_DONOR_HOST") else None
WIDE = {15, 19, 23, 27, 31, 35, 39, 43, 45}


def _ablit_source_from_0420() -> str:
    text = (ROOT / "patches" / "0420-glm-ablit-transplant.patch").read_text()
    start = text.index("+++ i/src/tensorfold/families/glm5_next/cuda/ablit.py\n")
    rest = text[start:].splitlines()[1:]  # drop the +++ line; next is the hunk header
    assert rest[0].startswith("@@")
    body = []
    for line in rest[1:]:
        if line.startswith("diff --git ") or line.startswith("--- ") or line.startswith("+++ "):
            break
        if line.startswith("\\"):
            continue
        assert line.startswith("+"), line
        body.append(line[1:])
    return "\n".join(body) + "\n"


def load_ablit():
    """0420's module with 0430 and 0440 applied: the ablit.py the image actually runs."""
    import subprocess
    import tempfile

    src = _ablit_source_from_0420()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dest = root / "src/tensorfold/families/glm5_next/cuda/ablit.py"
        dest.parent.mkdir(parents=True)
        dest.write_text(src)
        for name in ("0430-glm-ablit-q4mse-skip.patch", "0440-glm-ablit-head-minmax.patch"):
            patch = ROOT / "patches" / name
            subprocess.run(
                ["git", "apply", f"--include={dest.relative_to(root)}", str(patch)],
                cwd=root, check=True,
            )
        src = dest.read_text()
    spec = importlib.util.spec_from_loader("ablit_under_test", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, "ablit.py", "exec"), mod.__dict__)
    return mod


ablit = load_ablit()


def bf16(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", value))[0] >> 16
    return bits.to_bytes(2, "little")


def pack(rows: list[list[float]]) -> bytes:
    return b"".join(bf16(v) for row in rows for v in row)


def write_safetensors(path: Path, tensors: dict[str, tuple[list[int], bytes]]) -> None:
    header = {}
    blobs = []
    offset = 0
    for name, (shape, data) in tensors.items():
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"".join(blobs))


def test_patch_hooks_the_loader():
    text = (ROOT / "patches" / "0420-glm-ablit-transplant.patch").read_text()
    assert "ablit.begin(" in text and "ablit.maybe_transplant(" in text and "ablit.finish()" in text
    assert 'key["ablit"] = ablit.cache_extra()' in text
    assert "q4/q4mse would quantize" in text


def test_classify_mtp_is_the_layer_past_the_stack():
    edit = set(range(15, 45))
    assert ablit.classify("layers.14.self_attn.o_proj.weight", 45, edit) == ("guard", 14)
    assert ablit.classify("layers.15.self_attn.o_proj.weight", 45, edit) == ("edit", 15)
    assert ablit.classify("layers.44.self_attn.o_proj.weight", 45, edit) == ("edit", 44)
    # num_hidden_layers == 45, so layers.45 is the MTP block, not a main layer
    assert ablit.classify("layers.45.self_attn.o_proj.weight", 45, edit) == ("mtp", 45)
    assert ablit.classify("layers.45.mtp_block.self_attn.o_proj.weight", 45, edit) == ("mtp", 45)
    assert ablit.classify("layers.3.input_layernorm.weight", 45, edit) == ("other", None)


def test_column_half_matches_the_rank_split():
    full = pack([[1, 2, 3, 4], [5, 6, 7, 8]])
    left, shape_l = ablit.column_half(full, [2, 4], 2, 0, 2)
    right, shape_r = ablit.column_half(full, [2, 4], 2, 1, 2)
    assert shape_l == shape_r == [2, 2]
    assert left == pack([[1, 2], [5, 6]])
    assert right == pack([[3, 4], [7, 8]])
    assert left + right != full  # row-major halves are not a flat cut of the blob
    with pytest.raises(ablit.AblitError):
        ablit.column_half(full, [2, 3], 2, 0, 2)
    with pytest.raises(ablit.AblitError):
        ablit.column_half(full, [2, 4], 2, 2, 2)


def test_apply_store_edits_only_the_requested_layers():
    # n_layers=4 so the guard set is {0, 3} and layer 4 is MTP. Edit 1 and 2.
    full = {1: (pack([[1, 2, 3, 4], [5, 6, 7, 8]]), [2, 4]),
            2: (pack([[9, 8, 7, 6], [5, 4, 3, 2]]), [2, 4])}
    stock = pack([[0, 0], [0, 0]])
    names = {
        0: "layers.0.self_attn.o_proj.weight",
        1: "layers.1.self_attn.o_proj.weight",
        2: "layers.2.self_attn.o_proj.weight",
        3: "layers.3.self_attn.o_proj.weight",
    }
    mtp = "layers.4.self_attn.o_proj.weight"
    block = "layers.4.mtp_block.self_attn.o_proj.weight"
    other = "layers.1.mlp.down_proj.weight"
    store = {names[i]: stock for i in range(4)}
    store[mtp] = stock
    store[block] = stock
    store[other] = b"untouched"
    shapes = {n: [2, 2] for n in store if n != other}
    before = dict(store)
    report = ablit.apply_store(store, shapes, n_layers=4, edit=[1, 2], donor=full, rank=0, world=2)
    assert store[names[1]] == pack([[1, 2], [5, 6]])
    assert store[names[2]] == pack([[9, 8], [5, 4]])
    for keep in (names[0], names[3], mtp, block, other):
        assert store[keep] == before[keep]
    assert report["edited"] == [1, 2] and report["mtp"] is False
    assert report["mean_rel_l2"] == 0.0
    assert report["guarded"] == [0, 3]
    line = ablit.format_log(report, spec="1-2", rank=0, pre_mean=report["pre_mean"])
    assert "mtp=False" in line and "mean rel_l2=0.0000" in line and "donor_layer_45=absent" in line
    # rank 1 takes the other columns and still leaves the anchors alone
    store1 = {names[i]: stock for i in range(4)}
    store1[mtp] = stock
    ablit.apply_store(store1, {n: [2, 2] for n in store1}, n_layers=4, edit=[1, 2], donor=full, rank=1, world=2)
    assert store1[names[1]] == pack([[3, 4], [7, 8]])
    assert store1[names[0]] == stock and store1[mtp] == stock


def test_finalize_refuses_a_moved_anchor_or_mtp():
    edit = {1}
    good = [
        {"kind": "guard", "layer": 0, "changed": False},
        {"kind": "edit", "layer": 1, "post_rel": 0.0},
        {"kind": "mtp", "name": "layers.2.self_attn.o_proj.weight", "changed": False},
    ]
    assert ablit.finalize(good, edit=edit, n_layers=2, donor_has_45=True)["mtp"] is False
    moved = [dict(good[0], changed=True), good[1], good[2]]
    with pytest.raises(ablit.AblitError, match="stay stock"):
        ablit.finalize(moved, edit=edit, n_layers=2, donor_has_45=False)
    mtp = [good[0], good[1], dict(good[2], changed=True)]
    with pytest.raises(ablit.AblitError, match="MTP"):
        ablit.finalize(mtp, edit=edit, n_layers=2, donor_has_45=False)
    sloppy = [good[0], {"kind": "edit", "layer": 1, "post_rel": 0.5}, good[2]]
    with pytest.raises(ablit.AblitError, match="post-copy"):
        ablit.finalize(sloppy, edit=edit, n_layers=2, donor_has_45=False)


def test_q4mse_skip_patch_keeps_edit_oproj_bf16():
    text = (ROOT / "patches" / "0430-glm-ablit-q4mse-skip.patch").read_text()
    assert "keeps_bf16" in text and "make_b16(t(weight))" in text
    assert "nonexpert(t(weight))" in text
    assert "expected bf16, q4, or q4mse" in text
    assert 'o_proj": "edit-bf16"' in text


def test_head_minmax_patch_skips_clip_search_on_lm_head_only():
    text = (ROOT / "patches" / "0440-glm-ablit-head-minmax.patch").read_text()
    assert 'if NONEXPERT == "q4mse":' in text
    assert "quantize4(hw.to(torch.bfloat16), mse=False)" in text
    assert "head = nonexpert(hw)" in text
    assert '"lm_head": "minmax"' in text
    added = [line[1:] for line in text.splitlines() if line.startswith("+") and not line.startswith("+++")]
    # The shared quantizer stays on the mse path. This patch must not rewrite it or the o_proj skip.
    assert not any("mse=NONEXPERT" in line for line in added)
    assert not any("keeps_bf16" in line for line in added)


def test_begin_allows_q4mse_and_refuses_mtp_range_and_a_missing_donor(monkeypatch, tmp_path):
    monkeypatch.setenv("GLM53_TF_ABLIT", "1")
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "nope")
    monkeypatch.setenv("GLM53_TF_ABLIT_DONOR", str(tmp_path / "missing.safetensors"))
    with pytest.raises(ablit.AblitError, match="nope"):
        ablit.cache_extra()
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4mse")
    with pytest.raises(ablit.AblitError, match="missing"):
        ablit.cache_extra()
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "bf16")
    with pytest.raises(ablit.AblitError, match="missing"):
        ablit.begin(rank=0, world=2, n_layers=45)
    path = tmp_path / "donor.safetensors"
    payload = pack([[1, 2, 3, 4]])
    write_safetensors(path, {ablit.donor_key(45): ([1, 4], payload), ablit.donor_key(15): ([1, 4], payload)})
    monkeypatch.setenv("GLM53_TF_ABLIT_DONOR", str(path))
    monkeypatch.setenv("GLM53_TF_ABLIT_LAYERS", "15-45")
    with pytest.raises(ablit.AblitError, match="MTP"):
        ablit.begin(rank=0, world=2, n_layers=45)
    monkeypatch.setenv("GLM53_TF_ABLIT_LAYERS", "15-44")
    with pytest.raises(ablit.AblitError, match="no o_proj"):
        ablit.begin(rank=0, world=2, n_layers=45)
    monkeypatch.setenv("GLM53_TF_ABLIT", "0")
    assert ablit.cache_extra() == {"on": False}
    ablit.begin(rank=0, world=2, n_layers=45)  # off: does not read the donor
    assert ablit.finish() is None
    assert ablit.keeps_bf16("layers.15.self_attn.o_proj.weight") is False


def test_q4mse_keeps_only_the_edit_layers(monkeypatch, tmp_path):
    payload = pack([[1, 2, 3, 4]])
    tensors = {ablit.donor_key(i): ([1, 4], payload) for i in range(15, 46)}
    path = tmp_path / "donor.safetensors"
    write_safetensors(path, tensors)
    monkeypatch.setenv("GLM53_TF_ABLIT", "1")
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4mse")
    monkeypatch.setenv("GLM53_TF_ABLIT_DONOR", str(path))
    monkeypatch.setenv("GLM53_TF_ABLIT_LAYERS", "15-44")
    extra = ablit.cache_extra()
    assert extra["nonexpert"] == "q4mse" and extra["o_proj"] == "edit-bf16"
    assert extra["lm_head"] == "minmax"
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4")
    assert ablit.cache_extra()["lm_head"] == "q4"
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "bf16")
    assert ablit.cache_extra()["lm_head"] == "bf16"
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4mse")
    ablit.begin(rank=0, world=2, n_layers=45)
    try:
        assert ablit.keeps_bf16("layers.15.self_attn.o_proj.weight")
        assert ablit.keeps_bf16("layers.44.self_attn.o_proj.weight")
        assert not ablit.keeps_bf16("layers.14.self_attn.o_proj.weight")
        assert not ablit.keeps_bf16("layers.45.self_attn.o_proj.weight")
        assert not ablit.keeps_bf16("layers.45.mtp_block.self_attn.o_proj.weight")
        assert not ablit.keeps_bf16("layers.20.mlp.down_proj.weight")
        monkeypatch.setenv("GLM53_TF_NONEXPERT", "bf16")
        assert not ablit.keeps_bf16("layers.15.self_attn.o_proj.weight")
    finally:
        monkeypatch.setenv("GLM53_TF_ABLIT", "0")
        ablit.begin(rank=0, world=2, n_layers=45)


def test_real_donor_header_and_begin(monkeypatch):
    if DONOR is None or not DONOR.is_file():
        pytest.skip("set ABLIT_DONOR_HOST to the local donor safetensors to check its header")
    header, _base = ablit.read_header(DONOR)
    layers = ablit.donor_layers_from_header(header)
    assert set(layers) == set(range(15, 46))
    assert not any("mtp" in key for key in header)
    for idx, info in layers.items():
        assert info["dtype"] == "BF16" and info["shape"][0] == 4096
        assert info["shape"][1] == (16384 if idx in WIDE else 8192)
    monkeypatch.setenv("GLM53_TF_ABLIT", "1")
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "bf16")
    monkeypatch.setenv("GLM53_TF_ABLIT_DONOR", str(DONOR))
    monkeypatch.delenv("GLM53_TF_ABLIT_LAYERS", raising=False)
    ablit.begin(rank=1, world=2, n_layers=45)
    try:
        assert ablit._session is not None
        assert ablit._session.edit == set(range(15, 45))
        assert 45 in ablit._session.index and 45 not in ablit._session.edit
        assert ablit._session.rank == 1
    finally:
        monkeypatch.setenv("GLM53_TF_ABLIT", "0")
        ablit.begin(rank=0, world=2, n_layers=45)
