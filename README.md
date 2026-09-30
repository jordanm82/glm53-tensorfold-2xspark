# GLM-5.3-Flash EXL3 on TensorFold, 2× DGX Spark

This fork serves **Mia's TR3 EXL3 checkpoint** on [TensorFold](https://github.com/ashhart/TensorFold), with a
load-time abliteration transplant, across two NVIDIA DGX Sparks (GB10, tensor parallel 2, CX7). OpenAI-compatible
API, no key. The measured profile is **524,288 tokens** of context with **FP8 latent KV**, one sequence at a time.
Full recipe: [`docs/MIA-512K.md`](docs/MIA-512K.md). Copy-paste config:
[`config/mia-512k.env.example`](config/mia-512k.env.example).

> **Work in progress.** One pair of Sparks, one boot, one shot per cell. Not a 1M-context acceptance test, and
> not the imported Jayleaton stack further down this page.

SPDX-License-Identifier: Apache-2.0 (this project's own code, scripts, benchmarks and docs; see [Licensing](#licensing)).

The weights are not in this repo. This work uses ShapleyMcg, created by Brandon M. Music
(<https://github.com/brandonmmusic-max/shapleymcg>). ShapleyMcg is licensed under the ShapleyMcg License v1.0,
an attribution-required source-available license that grants no rights to the person known as "0xSero."
Use without that attribution is unlicensed. The base model is `zai-org/GLM-5.3-Flash` (MIT).

## This serve (2026-09-29)

| | |
| --- | --- |
| Weights | [`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw) @ `25a44fdbf16862a46b7cc9921142c6c81350af2f` (byte-identical mirror of [`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) @ `5ab363a8dcf6405955fd5f99671e01a1c9fb124b`). Not pre-abliterated. |
| Abliteration | Load-time transplant of BF16 `self_attn.o_proj` from a Dealign donor, **layers 15–44 only**. Layer 45 is in the donor and is left stock (it is the MTP block). `mtp=False`. Post-copy `mean rel_l2=0.0000`. |
| Drafter | [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) @ `dc77ff1c99eeb2df044ee3d4f0094eb033fee410` (CC BY-NC-ND 4.0, not bundled), k=7, threshold depth |
| Engine | TensorFold **0.3.4** (`vendor/TensorFold` @ `2f8e514`) plus `patches/` through **0420**. Image `glm53-tensorfold:dev` @ `sha256:cdcc670ac30458d27aa2bd7f2e0e1a67d8785646071a0289bc5778288dda2bc7`, built 2026-09-29 from `nvcr.io/nvidia/pytorch:26.07-py3`. Not TensorFold 0.3.7. |
| KV | Latent FP8. Engine log: `7.4 KB a token a rank (524296 slots, 3.72 GB)`. Requested `CONTEXT=524288`. |
| Batch | `GLM53_TF_BATCH=2` requested. The engine served **1** sequence: 3.96 GB a sequence at that cache, with 4 GB kept free. |
| Link | NCCL on the crossed CX7. RoCE off. Non-experts **bf16** (not q4mse). Graphs on. |
| API | `GLM-5.3-Flash-EXL3` on port **8888**. First start ready in **970 s**. |

TensorFold prints, on every EXL3 launch, that the MLX checkpoint is tested more. That line is not this boot.
The process loaded the Mia snapshot above.

Steady decode on the Explain and Code prompts is about **34–53 tok/s**. A 3,556-token prefill is about
**263–317 tok/s** (about 12–14 s). Every measured request returned HTTP 200. Receipts:
[`results/mia-512k/`](results/mia-512k/README.md).

## How this build is made

### Weights

Routed experts are EXL3 (trellis + suh + svh + mcg, 4-bit). Attention, the shared expert, dense layers and
the head stay BF16 in the checkpoint (`num_hidden_layers=45`, `kv_lora_rank=512`, one MTP layer). Mia's AI Lab
re-hosts Brandon M. Music's TR3 snapshot so the Spark recipe still has a fetch target. This repo does not
re-quantize those BF16 matrices: `GLM53_TF_NONEXPERT=bf16`. Patch 0001's `q4mse` path is a different model
(and the transplant refuses it, because it would quantize the tensors just copied).

### Abliteration

Nothing on disk is rewritten. `patches/0420` copies donor columns into `o_proj` while the rank's slice of the
checkpoint is on CPU, before any 4-bit quant and before graphs. The donor is the full unsharded tensor; `o_proj`
is a column split of the last axis, so rank `r` of 2 takes columns `[r*half:(r+1)*half]`.

The donor is the published Dealign edit: BF16 `model.language_model.layers.{15–45}.self_attn.o_proj.weight`
from [`dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4`](https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4),
the `dealign-oproj-transplant` method described with
[drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock](https://huggingface.co/drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock).
Thirty-one tensors. Wide layers `{15,19,23,27,31,35,39,43,45}` are `[4096, 16384]`; the others are `[4096, 8192]`.
No key is named `mtp`. The file is not in git. Both nodes mount the same host path at `/ablit/donor.safetensors`.

Layers **0–14** stay the Mia checkpoint (the load fails if they change). Layers **15–44** are replaced.
`layers.45` is present in the donor and is **not** applied: this engine's `num_hidden_layers` is 45, so
`layers.45` is the MTP block. The log says `mtp=False`. Do not extend the range to 45 to "match" the donor.

Projection orthogonalization is not used. On this model that direction was measured as noise; the byte copy is
the edit. The acceptance line from this boot (rank 0; rank 1's pre-copy distance was 0.1269):

Both ranks logged `method=transplant layers=15-44 mtp=False mean rel_l2=0.0000 donor_layer_45=present_not_applied`,
with `edited` equal to 15 through 44 and `guarded` equal to 0 through 14. Rank 0's `pre_rel_l2` was 0.1280;
rank 1's was 0.1269. `mean rel_l2` is the distance after the copy. `pre_rel_l2` is the distance before it.

### TensorFold config

`scripts/serve.sh` defaults to `config/prod.env` (the imported 1M / q4mse / RoCE profile). This serve does not.
Start it with `CONFIG=config/mia-512k.env`. The example uses placeholders; fill the ssh target, the head's
address on the CX7 link, the two Hugging Face caches, and the donor path. Interface names below are the crossed
cable that actually came up. Same-named HCAs are not on one subnet; listing every HCA makes NCCL spin.

| Knob | This boot | Not this boot |
| --- | --- | --- |
| `CONTEXT` | `524288` (engine allocated 524296 slots) | 1,048,576; per-head KV (`FORCE_CONTEXT` stays unset) |
| `GLM53_TF_LATENT_KV` / `GLM53_TF_KV_DTYPE` | `1` / `fp8` | bf16 latent, or the 390 KB/token per-head cache |
| `GLM53_TF_NONEXPERT` | `bf16` | `q4mse` |
| `GLM53_TF_ABLIT` / `_LAYERS` | `1` / `15-44` | off, or layer 45 |
| `GLM53_TF_COMM_BACKEND` | `nccl` | `roce` |
| `GLM53_TF_AUTO_FDRAFTS` / `GLM53_TF_DEPTH` | `7` / `threshold` | cost-derived depth |
| `GLM53_TF_LONGCTX_GRAPHS` | `1` | |
| `GLM53_TF_BATCH` | `2` requested, **1** served (4 GB reserve, 3.96 GB a 512k slot) | the 4-request 1M pool |
| Head link | `enp1s0f0np0` / `rocep1s0f0` | the same name on both nodes |
| Worker link | `enp1s0f1np1` / `rocep1s0f1` | |
| Also off | fat MoE, KDA BF16 large-M, KV pool, session quota, prefix share, decode overlap | |

That table is the 2026-09-29 bf16 boot, which allocated one 512k slot and served one sequence. The live profile has since set `GLM53_TF_NONEXPERT=q4mse` and `GLM53_TF_KV_POOL_TOKENS=524544` (patches/0290). The pool restart kept the same image and the same prepared folders. Rank 0 logged one 3.72 GB latent-FP8 pool, 524544 tokens in pages of 256, shared by 2 slots (0.21 GB of extra slot state; each slot can still grow to 524296 tokens). A request reserves prompt + max_tokens + 64 tokens of pages. Session quota, prefix share, and the imported 1,048,576-token pool stay off.

Per-head KV is about 390 KB a token a rank, about 195 GiB at 512k. That does not fit in a 121 GiB GB10 on two
ranks or three. Latent FP8 is what makes 512k fit (about 3.7 GB a rank). The launcher is two ranks. A third
Spark is not part of this build.

The worker's Hugging Face cache has to be readable as the worker user. On this pair it is an NFS mount exported
from the head's **CX7** address. The management network is the wrong interface (the copy crawls), and a Docker
volume the worker user cannot read fails preflight.

`GLM53_TF_EFFORT_FIELD` is unset, so a top-level OpenAI `reasoning_effort` is ignored. Thinking and effort go
through `chat_template_kwargs`. The server default is thinking off. The template writes `Reasoning Effort: Low`
or `High` when that is the effort, and `Max` otherwise. Requests that omit `top_k` still sample with `top_k` 20
(`temperature` 1.0, `top_p` 0.95 from the checkpoint, then the server default). `temperature` 0 is greedy.

## Speed (2026-09-29)

One sequence, drafts on, `max_tokens` 256. Each prompt shape was warmed once at 24 tokens; those rows are in
the JSONL and not in the tables. `cached_tokens` was 0 on every row, including the second pass over the same
3,556-token passage, so the long-prefill numbers are not cache hits.

Decode tok/s = `completion_tokens / tensorfold.decode_s`. Prefill tok/s = `(prompt_tokens - cached_tokens) / prefill_s`.
Tok/round = `(completion_tokens - 1) / rounds` (the first token comes from prefill). A `*` marks fewer than 16
output tokens: that rate is one short step, not throughput. Explain and Code are the decode measurements. The
long passage is the prefill measurement. Short-prompt prefill tok/s is the same kind of noise.

Prompts: capital of France (city only); explain binary search; a Python merge of two sorted lists; a passage
whose first sentence names archive token `184729`, then one filler sentence repeated 220 times, then "reply
with the number only."

### Greedy (`temperature` 0)

Finished 2026-09-29 19:14 UTC. [`results/mia-512k/greedy-20260929.jsonl`](results/mia-512k/greedy-20260929.jsonl).

| Prompt | Thinking | In | Out | Prefill tok/s | Decode tok/s | Tok/round | Wall s | Finish |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Short question | Off | 20 | 2 | 129.2 | 24.6* | 1.0 | 0.24 | stop |
| Short question | Low | 26 | 3 | 149.5 | 34.01* | 2.0 | 0.267 | stop |
| Short question | High | 26 | 3 | 148.7 | 33.98* | 2.0 | 0.267 | stop |
| Short question | Max | 26 | 23 | 147.6 | 34.87 | 3.14 | 0.84 | stop |
| Explain | Off | 24 | 256 | 138.9 | 37.16 | 3.45 | 7.067 | length |
| Explain | Low | 30 | 239 | 95.3 | 37.17 | 3.4 | 6.749 | stop |
| Explain | High | 30 | 256 | 156.7 | 35.28 | 3.45 | 7.452 | length |
| Explain | Max | 30 | 256 | 154.5 | 42.49 | 3.98 | 6.224 | length |
| Code | Off | 29 | 217 | 153.0 | 48.17 | 4.5 | 4.699 | stop |
| Code | Low | 35 | 143 | 104.1 | 53.4 | 5.26 | 3.018 | stop |
| Code | High | 35 | 178 | 164.1 | 51.77 | 5.06 | 3.656 | stop |
| Code | Max | 35 | 256 | 164.4 | 45.29 | 4.25 | 5.87 | length |
| Long passage | Off | 3556 | 4 | 300.8 | 41.97* | 3.0 | 11.924 | stop |
| Long passage | Low | 3562 | 5 | 286.5 | 21.73* | 2.0 | 12.674 | stop |
| Long passage | High | 3562 | 5 | 284.7 | 23.06* | 2.0 | 12.737 | stop |
| Long passage | Max | 3562 | 30 | 262.6 | 42.93 | 4.14 | 14.272 | stop |

Explain Low's prefill (95.3 tok/s, `prefill_s` 0.3149) is a short-prompt blip, not a slower prefill mode.
Max on Explain and Code used the whole 256-token cap inside the think block and returned no answer
(`content` empty, 918 and 1,032 reasoning characters). Low and High wrote no `reasoning_content` on the short,
explain, and long prompts. Code High wrote 70 characters of it. Answers that were checked: `Paris`, and
`184729` on every long-passage row.

### Sampled (`temperature` 1.0, `top_p` 0.95, `top_k` 20)

Finished 2026-09-29 19:20 UTC. [`results/mia-512k/sampled-20260929.jsonl`](results/mia-512k/sampled-20260929.jsonl).
Same prompts and cap.

| Prompt | Thinking | In | Out | Prefill tok/s | Decode tok/s | Tok/round | Wall s | Finish |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Short question | Off | 20 | 2 | 96.2 | 18.48* | 1.0 | 0.32 | stop |
| Short question | Low | 26 | 3 | 112.7 | 29.76* | 2.0 | 0.335 | stop |
| Short question | High | 26 | 3 | 104.7 | 23.26* | 2.0 | 0.381 | stop |
| Short question | Max | 26 | 19 | 104.9 | 35.43 | 4.5 | 0.788 | stop |
| Explain | Off | 24 | 256 | 98.3 | 34.73 | 3.15 | 7.62 | length |
| Explain | Low | 30 | 256 | 156.1 | 36.57 | 3.19 | 7.196 | length |
| Explain | High | 30 | 256 | 156.2 | 33.96 | 3.04 | 7.734 | length |
| Explain | Max | 30 | 256 | 153.7 | 36.09 | 3.11 | 7.292 | length |
| Code | Off | 29 | 256 | 153.0 | 34.82 | 3.64 | 7.546 | length |
| Code | Low | 35 | 143 | 163.5 | 52.85 | 5.07 | 2.924 | stop |
| Code | High | 35 | 159 | 164.5 | 52.68 | 5.1 | 3.235 | stop |
| Code | Max | 35 | 256 | 165.6 | 42.8 | 3.86 | 6.197 | length |
| Long passage | Off | 3556 | 4 | 317.4 | 46.19* | 3.0 | 11.294 | stop |
| Long passage | Low | 3562 | 5 | 264.5 | 28.15* | 2.0 | 13.65 | stop |
| Long passage | High | 3562 | 5 | 281.9 | 18.87* | 1.33 | 12.903 | stop |
| Long passage | Max | 3562 | 28 | 285.1 | 42.51 | 3.86 | 13.158 | stop |

Sampled steady decode stays in the same band. The one clear drop is Code with thinking off: greedy 48.17 tok/s
(217 tokens, stop, 4.5 tok/round) versus sampled 34.82 tok/s (hit the 256 cap, 3.64 tok/round). A few tok/s
either way on the other cells is one-shot noise. `Paris` and `184729` still came back. Max on Explain and Code
again finished inside the think block with an empty answer. Code High wrote 64 characters of reasoning; the
other Low and High rows wrote none.

These numbers are **not** the RigMark or W10 tables below. Those used other weights (`neko-legends` @ `07135ec0`),
`q4mse` non-experts, RoCE, a 4-request pool, and other prompts.

## Contents

- [This serve](#this-serve-2026-09-29)
- [How this build is made](#how-this-build-is-made)
- [Speed](#speed-2026-09-29)
- [Imported stack](#imported-stack-jayleaton-other-weights) (neko-legends, q4mse, 1M pool, RoCE, port 8000)
- [Licensing](#licensing)
- [Credits](#credits)

## Imported stack (Jayleaton, other weights)

The rest of this README, through [Credits](#credits), is the imported text from
[jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark)
([Jay Leaton](https://x.com/jayleaton)). It documents **his** measured stack: `neko-legends` weights, `q4mse`
non-experts, a 4-request pool over 1,048,576 latent-FP8 tokens, RoCE, port 8000. Those sections are not the
Mia transplant serve above. Read them for the engine this fork started from, not as this machine's current config.
Agents: [`AGENTS.md`](AGENTS.md) is that imported procedure; the running profile is
[`docs/MIA-512K.md`](docs/MIA-512K.md).

## Quickstart

This runs the **imported** production config (`config/prod.env.example`: 4 concurrent requests, a 1,048,576-token
context, neko-legends weights). It is not [`config/mia-512k.env.example`](config/mia-512k.env.example). AI coding
agents following that imported path: [`AGENTS.md`](AGENTS.md).

**Prerequisites** (details in [Requirements](#requirements)): two DGX Sparks cabled CX7 to CX7 with an IP address on
the link on each; Docker with the NVIDIA runtime on both; passwordless `ssh` from the head node to the worker (and
passwordless `sudo -n` on both for the memory gate); the weights in each node's Hugging Face cache (gated repo:
request access first):

```bash
hf download neko-legends/GLM-5.3-Flash-Uncensored-EXL3 --revision 07135ec082f8f11f7a71e4244a4e5167a0f96277   # both nodes
hf download incoai/GLM-5.3-Flash-DFlash2 --revision 7d74cdd881ed7e32c31175984a67823127b66cfe   # optional drafter, CC BY-NC-ND 4.0
```

**On the head node:**

```bash
git clone --recurse-submodules https://github.com/jayleaton/glm53-tensorfold-spark
cd glm53-tensorfold-spark
cp config/prod.env.example config/prod.env
$EDITOR config/prod.env
```

Fill in these four fields and change nothing else:

| Field | Value |
| --- | --- |
| `WORKER_SSH` | ssh target of the worker, e.g. `user@<worker CX7 address>` |
| `HEAD_IP` | the head's IPv4 address on the CX7 link (`ip -br addr show <netdev>`) |
| `HEAD_HF` / `WORKER_HF` | absolute path of each node's `~/.cache/huggingface` |

Check `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` against `ibdev2netdev` (the usual Spark names are preset), and set
`DRAFTER=` empty if you did not download the drafter.

```bash
scripts/serve.sh build       # build the image here, copy it to the worker
scripts/serve.sh preflight   # checks both nodes; fix what it reports
scripts/serve.sh start       # first start: 8+ min (compiles kernels, loads, writes prepared weights); later ~25-40 s
```

`serve.sh` reads `config/prod.env` by default; no `CONFIG=` needed.

**Verify:**

```bash
scripts/serve.sh status                 # both containers Up, /v1/models and /health answer
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "GLM-5.3-Flash-EXL3", "max_tokens": 1024,
  "messages": [{"role": "user", "content": "Write a Python function that reverses a linked list."}]}'
```

**Clients:** base URL `http://127.0.0.1:8000/v1`, model `GLM-5.3-Flash-EXL3`, any API key. Set the context window
to **1,048,576** and the maximum output to **32,768** (Continue, Cline, Open WebUI and others:
[client settings](docs/TRYING.md#client-settings)). opencode (`~/.config/opencode/opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "glm-tf": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "GLM-5.3-Flash (TensorFold)",
      "options": { "baseURL": "http://127.0.0.1:8000/v1" },
      "models": {
        "GLM-5.3-Flash-EXL3": {
          "name": "GLM-5.3-Flash",
          "tool_call": true,
          "reasoning": true,
          "limit": { "context": 1048576, "output": 32768 }
        }
      }
    }
  }
}
```

Without `limit.context` opencode never compacts a long session. The API has no authentication and binds to
`127.0.0.1`; put a reverse proxy with auth in front of it before exposing it. Stop with `scripts/serve.sh stop`.
Long prompts refused or cut short: [Context smaller than expected](docs/TRYING.md#10-context-smaller-than-expected).

## RigMark baseline (2026-09-29)

[RigMark](https://github.com/alexellis/rigmark) standard suite, unmodified settings (receipt and details in
[`results/rigmark/`](results/rigmark/README.md)). **A baseline: we expect to improve it over the coming days.**

| RigMark (median) | TensorFold | vLLM TP2 (Alex Ellis, published) |
|---|---:|---:|
| Code decode tok/s | **68.6** | 42.6-44.0 |
| Prose decode tok/s | **43.2** | 18.9-22.2 |
| Structured decode tok/s | **88.2** | 54.6-64.9 |
| C1 / C2 / C4 aggregate tok/s | **54.3 / 65.5 / 82.2** | 31.2-31.6 / 42.0-42.8 / 61.1-66.1 |
| Cold prefill 64K tok/s | 1,621 | **1,905-1,922** |
| Immediate replay 64K tok/s | 6,304 | **11,364-11,464** |

Different weights and drafter policy than Alex's runs; see the notes in [`results/rigmark/`](results/rigmark/README.md).

## Benchmarks

The numbers in this section are on the **imported** checkpoint `neko-legends/GLM-5.3-Flash-Uncensored-EXL3` @
`07135ec0`, on one pair of DGX Sparks (GB10, TP=2), 2026-09-27 to 2026-09-29. They are not the Mia TR3 transplant
serve at the top of this file. The production numbers are from the last test window (W10, 2026-09-29). Full tables
and methodology: [`docs/RESULTS.md`](docs/RESULTS.md); raw JSON and the window scripts in [`results/`](results/).

### (a) Ours vs the vLLM production kit, same weights, same client

The vLLM column is [Reederey87's kit](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) @ `8e443d6` (a fork
of [MiaAI-Lab's kit](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)) as we ran it in production on
this pair: 1M context, FP8 KV, DFlash2 k=7 with adaptive k, fused EXL3 MoE kernels, **the same abliterated weights**.
Both stacks were measured with the same client (`bench/glmbench.py`), one stack at a time, nothing else on the GPUs.

Two TensorFold configurations:

| Config | File | What |
| --- | --- | --- |
| **Production** (4 requests, shared 1M-token pool) | [`config/prod.env.example`](config/prod.env.example) | 4 concurrent requests sharing one **1,048,576-token FP8 latent KV pool** (each request up to 1M tokens while the pool has room), sessions inside the batch slots plus an NVMe session tier, shared system-prompt reuse, row-split prefill in 4,096-row chunks for a lone request, the RoCE all-gather, 16-row verify windows |
| Single-stream (earlier, 2026-09-28) | [`config/prod-single.env.example`](config/prod-single.env.example) | one request at a time, 524,288-token context, bf16 latent KV, fast/lean prefill in 8192-row chunks, session store |

**Decode**, tok/s, single stream, thinking off (decode excludes prefill and time to first token). Production: W10
load R16, median of 3 (R16 is the production config without 0370's decode overlap, which adds another ~1-2%);
the other columns: median of 5.

| Cell | vLLM kit | **TF production** | vs vLLM | TF single-stream (09-28) |
| --- | ---: | ---: | ---: | ---: |
| tf code, sampled (T=1), 64 tok | 35.2 | **42.3** | 1.20x | 40.5 |
| tf chat, sampled (T=1), 64 tok | 27.1 | **41.0** | 1.51x | 37.5 |
| tf code, greedy, 64 tok | 41.9 | **77.6** | 1.85x | 65.8 |
| tf chat, greedy, 64 tok | 22.8 | **44.6** | 1.96x | 44.7 |
| kit hashmap (prose), 200 tok | 30.0 | **53.2** | 1.77x | 50.4 |
| kit structured, 200 tok | 72.7 | **100.6** | 1.38x | 98.1 |
| kit essay, 200 tok | 26.1 | **44.9** | 1.72x | 44.2 |
| tweet sequence, 512 tok | 67.5 | **93.0** | 1.38x | 93.3 |
| tweet code, 512 tok | 42.4 | **66.7** | 1.57x | 62.3 |
| tweet json, 512 tok | 50.4 | **74.8** | 1.48x | 71.7 |
| edit (rename / comments / print-to-log), 1024 tok | not run | **111.2 / 94.7 / 115.5** (final config: 115.3 / 97.1 / 116.5) | - | 81.8 / 78.2 / 82.6 |

The edit cells (the model rewrites a file it was given) benefit from prompt-lookup drafts (patch 0020) and, since
W10, from verify windows of up to 16 rows (patch 0380: +16-27% on these cells, every reply hash unchanged).

**Prefill and time to first token** (cold, unique prompt, no cache hit; kernels already compiled):

| Prompt | vLLM kit | **TF production** (alone) | TF single-stream (09-28) |
| --- | ---: | ---: | ---: |
| ~7k tokens | 1,340 tok/s | - | 1,162 tok/s |
| ~24.5k-28k tokens | 1,448 tok/s at 28k (TTFT 19.4 s) | **~1,607 tok/s** at 24.5k (1,614 / 1,602; TTFT ~13.4 s) | 1,266 tok/s at 28k (TTFT 22.2 s) |
| ~98k-112k tokens | - | **~1,600 tok/s** at 98k (1,577-1,606) | 1,238 tok/s at 112k (TTFT 90.5 s) |
| 314k tokens (needle, after the stress run) | - | 1,376 tok/s, found | - |
| TF vs vLLM at ~28k | | **~1.11x** (24.5k vs vLLM's 28k) | 0.87x |

**Multi-turn, concurrency, boot, quality**:

| | vLLM kit | **TF production** | TF single-stream (09-28) |
| --- | --- | --- | --- |
| context | 1M in one context | **4 concurrent requests sharing a 1,048,576-token KV pool, each request up to 1M** | 1 x 524k |
| concurrent streams 1 / 4, aggregate tok/s (median of 5) | batches up to 4; Reederey87 publishes 63.4-66.3 warm at 4 in flight | 52.1 / **78.1** (5 mixed prompts; 4-stream reps 70-80 across windows) | one at a time (queued) |
| switch back to a stored ~39k-token session | - | **0.42-0.45 s** from the NVMe session tier instead of a 31 s cold prefill, also after a server restart | 0.4-1.0 s (RAM store) |
| resume a 314k-token conversation | - | 0.19 s (314,240 tokens cached) | - |
| 4 new sessions at once over one ~18k-token system prompt (subagent burst) | - | **28.2 s** wall instead of 72.4 s (shared-prefix reuse, patch 0310) | - |
| decoders during a long prefill | - | longest decode gap 3.9 s (4 x ~250k stress) | queued behind it |
| restart to ready (prepared weight folders, patch 0140) | - | **22-23 s** | 34-37 s (490 s from the raw checkpoint) |
| drafted == serial, byte-identical (10 cases) | n/a | 10/10; batched == alone 4/4 | 10/10 |
| MMLU-200 (greedy, thinking off) / refusals (10 prompts) | - / 0/10 | **88.0%** / 0/10 | 89.5% / 0/10 |
| needle retrieval | - | found at 314k (cold and resumed); earlier at 358k | 10/10 at 28k |
| memory stress: 4 conversations grown to ~250k each, then a 32k turn beside 3 decoders | - | no OOM, no request errors; worst MemAvailable 10.49 / 9.36 GiB (head / worker) | - |
| RoCE all-gather instead of NCCL (patch 0230/0350, W9 A/B) | - | decode +10.8% (1 stream) / +4.3% (4 streams) median, transcripts byte-identical | - |

"Drafted == serial" means every drafted reply is the same bytes as the one-token-a-round reply of the same engine
and weights. "Batched == alone": a request served next to 3 others returns the same bytes as served alone. The
TF production column is W10's final config (`FIN`) unless the row says otherwise; the session-tier row is W4 and the
system-prompt row W8 (both features unchanged since).

### (b) MiaAI-Lab's and Reederey87's published numbers (different weights and settings)

These are the kits' own published figures, copied from their READMEs on 2026-09-28. They use the **base** (not
abliterated) weights `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (MiaAI-Lab serves the byte-identical mirror
`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`), their own clients (sparkDash, `tests/bench_decode.py`) and their own
prompts, so they are **not** directly comparable to table (a).

[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks):

| Date | Settings | Result |
| --- | --- | --- |
| 2026-09-07 | cold prefill, E3 grouped MoE; 900k context, util 0.86, MNBT 7168, DFlash2 k=7, 4 seqs, thinking off | 1,492 / 1,554 / 1,428 / 1,587 / 1,562 / 1,517 tok/s at ~8k / 16k / 32k / 64k / 128k / 256k (TTFT 32k: 23.0 s, 128k: 84.0 s) |
| 2026-08-28 | decode, structured + code prompts, DFlash2 k=7, temp 0, thinking off, 400 tok, 1M context | x1: 62.9 tok/s (TTFT 719 ms); x2: 51.7 a stream, 103.3 aggregate; x4: 37.1 a stream, 146.5 aggregate |
| 2026-09-17 | decode, prose, adaptive k (EMA) + dense FP8 + cooperative MoE, 850k context | x1: 36.1 a stream; x4: 19.4 a stream, 75.3 aggregate |
| 2026-09-21 | custom qualification run (their "OFF" arm, 850k) | structured 78.6, code 53.5, prose 33.2 tok/s; cold 32k TTFT 27.7 s, cold 100.7k TTFT 85.6 s |

[Reederey87/glm53-flash-exl3-2x-dgx-spark](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) (production
stack of 2026-09-20, 1M context):

| Metric | Published |
| --- | --- |
| prose decode (hashmap) / hard essay | ~33 tok/s (median 33.03) / ~26 tok/s (median 25.92) |
| structured decode | ~74 tok/s (median 74.19, acceptance 1.0) |
| cold prefill (2026-09-09 stack) | ~1,454 tok/s at 60k, ~1,408 tok/s at 240k |
| 4 in-flight, warm aggregate | 63.4-66.3 tok/s |

How to read the two tables together:

- **Our numbers are with an abliterated model.** The abliterated checkpoint keeps attention, the shared expert, the
  dense layers and the head in BF16; we re-quantize those to 4 bits at load (`q4mse`, patch 0001), which costs a
  little accuracy (13 of 200 MMLU answers change) and still reads more than a natively 4-bit layout. The MTP head
  and the DFlash2 drafter were trained against the base model, so draft acceptance on abliterated weights is likely
  lower. We expect base GLM-5.3-Flash weights (for example the MLX 4-bit checkpoint TensorFold's own recipe uses) to
  net further improvements; that is an expectation, not a measurement.
- On our pair, the vLLM kit on the abliterated weights measured hashmap 30.0 / structured 72.7 / essay 26.1 tok/s,
  close to Reederey87's published 33 / 74 / 26 on base weights.
- Cross-kit comparisons also differ in context length, KV format, drafter settings, prompts, clients and dates.

## Real-agent use

This has been used in [opencode](https://opencode.ai) for real agent workflows (multi-file edits, tool calls, long
sessions) and performed well with thinking at **high** reasoning effort, the default in the production configs
(`GLM53_TF_DEFAULT_EFFORT=high`).

| Check | Result |
| --- | --- |
| opencode tool-call harness (`bench/toolcall_harness.py`: opencode's tool set, 21 cases x 10 runs, streamed, T 0, thinking off) | **200/210 passed, 0 corrupted** calls (same with FP8 prefill on or off) |
| same harness, thinking on, single-stream production load (21 x 5) | 95/105 passed, 0 corrupted |
| same harness through the HTTPS reverse proxy in front of the API (21 x 2) | 38/42 passed, 0 corrupted |
| real-model API checks after each switch | `/v1/models`; a thinking reply returns `reasoning_content` + the answer (finish `stop`); a tool call returns `get_weather({"city":"Paris"})` with finish `tool_calls`; streamed == non-streamed; `stop` strings streamed and not |

"Corrupted" means a tool call with leaked GLM markup (`<arg_key>`, `<tool_call>`, `</think>`) or unparseable
arguments. Every failure is a case where the model made a different, reasonable call than the one the case
expects: `edit_file` reads the file before editing it (a `read` call where the case expects `edit`), and with
thinking on `multi_turn_chain` takes another step first.

The opencode provider entry is in the [Quickstart](#quickstart). Set `limit.context` to the server's `CONTEXT`
(1048576 for the production config, 524288 for single-stream): without it opencode never compacts a long session,
and it sends `max_tokens` = `limit.output` (capped at 32,000), which counts against the context. In the
production config the four requests share one 1,048,576-token pool: one request can use all of it, but four long
ones together wait for pages or spill idle sessions to the store.

## Requirements

- Two DGX Sparks (GB10, 128 GB unified memory each) connected by a QSFP cable between their ConnectX-7 ports, with
  the link configured (an IP address on one CX7 netdev per node; the RDMA device visible in `ibv_devices`).
- Docker with the NVIDIA Container Toolkit on both nodes (stock DGX OS has both).
- Passwordless `ssh` from the head node to the worker, as a user that can run `docker` there. The production
  config's memory gate also drops page caches with `sudo -n` on both nodes.
- The weights in each node's Hugging Face cache, same revision on both (the repo is gated: request access on its
  model card first):

  ```bash
  hf download neko-legends/GLM-5.3-Flash-Uncensored-EXL3 --revision 07135ec082f8f11f7a71e4244a4e5167a0f96277
  ```

- Optional: the DFlash2 drafter `incoai/GLM-5.3-Flash-DFlash2` (revision `7d74cdd`), in both caches. It is
  **CC BY-NC-ND 4.0 (non-commercial only)**; this repo never ships it. Without it, drafting uses the checkpoint's own
  MTP layer.
- Disk: the checkpoint, plus ~83 GB a node for the prepared weight folder (fast restarts) and up to 64 GiB a node
  for the NVMe session tier (`GLM53_TF_SESSION_DISK_GIB`).
- Network access at build time to pull `nvcr.io/nvidia/pytorch:26.07-py3`. At run time the container is offline.

## Ways to run it

The production config is the default and the one to run. [`docs/TRYING.md`](docs/TRYING.md) also covers:
single-stream long context (524k), 4 x 256k batch (FP8 KV), the 32k debugging baseline (`config/minimal.env.example`),
per-request knob A/B (`tf_knobs`), draft policies (`model@policy`), reasoning effort, sessions, fast boot, how to
benchmark (`glmbench`, `multiturn`, `quality`, `toolcall_harness`), how to check exactness, how to roll back, what to
check when the context is smaller than expected, and client settings (opencode, Continue, Cline, Open WebUI).

## Knobs

Every engine change is a patch in [`patches/`](patches/) with its own `GLM53_TF_*` knob, off by default (upstream
behaviour) unless stated. [`docs/PATCHES.md`](docs/PATCHES.md) documents each knob and why each patch keeps the
output exact; [`docs/CHANGES-SUMMARY.md`](docs/CHANGES-SUMMARY.md) lists every patch with its measured gain and
status (on / opt-in / rejected), ordered by impact.

Main groups:

| Area | Patches | Main knobs |
| --- | --- | --- |
| Weights | 0001 | `GLM53_TF_NONEXPERT=q4mse` |
| Drafting | 0010, 0020, 0070, 0071 | `GLM53_TF_AUTO_FDRAFTS=7`, `GLM53_TF_LOOKUP=1`, `GLM53_TF_CALIB`, `GLM53_TF_DEPTH=cost` |
| Drafting / verify | 0380 | `GLM53_TF_MAX_DRAFT_ROWS=16` (verify windows of up to 16 rows) |
| Long context / KV | 0050, 0060, 0065, 0220, 0290 | `GLM53_TF_LATENT_KV=1`, `GLM53_TF_KV_DTYPE=bf16\|fp8`, `GLM53_TF_KV_POOL_TOKENS=1048576` |
| Prefill | 0003-0006, 0080-0085, 0170, 0190, 0320, 0335, 0360, 0390 | `GLM53_TF_FAST_PREFILL=1`, `GLM53_TF_LEAN_PREFILL=1`, `GLM53_TF_PREFILL_ROWS=auto`, `GLM53_TF_FAST_EXPERTS=fat`, `GLM53_TF_MOE_GLUE=5`, `GLM53_TF_ATTN_BM32=1`, `GLM53_TF_MTP_PREFILL_CACHE=1`, `GLM53_TF_PREFILL_PP=1`, `GLM53_TF_SOLO_PIECE=4096`, `GLM53_TF_B12X=4`, `GLM53_TF_MLA_EXPAND=v2` |
| Sessions / batching | 0110, 0120, 0180, 0200, 0250, 0310 | `GLM53_TF_SESSION_GIB`, `GLM53_TF_BATCH=4`, `GLM53_TF_BATCH_SESSIONS=1`, `GLM53_TF_SESSION_DISK=/sessions`, `GLM53_TF_PREFIX_SHARE=1` |
| Communication / decode host work | 0230, 0350, 0370 | `GLM53_TF_COMM_BACKEND=roce` (NCCL fallback + failure marker), `GLM53_TF_DECODE_OVERLAP=1` |
| Per request | 0090-0093 | `"tf_knobs": {...}` in the request body |
| Serving / ops | 0002, 0140, 0150, 0160, 0210, 0300 | prepared folders, `/health`, `/metrics`, `reasoning_effort`, `stop`, prompt-token cache, request log (`GLM53_TF_REQUEST_LOG`, no text) |
| Measured, not adopted (off) | 0240 (bits 1-2), 0260, 0270, 0280, 0330, 0340, 0400, 0410, 0370's `GLM53_TF_CPU_PIN` | see [Limits](#limits-and-negatives) |

## Limits and negatives

| | TensorFold + patches (production) | vLLM kit |
| --- | --- | --- |
| Single-stream decode | 1.20-1.96x faster on every measured cell | baseline |
| Prefill | ~1,607 tok/s at 24.5k and ~1,600 at 98k, against 1,448 measured for the kit at 28k (~1.11x; not the same prompt length). MiaAI-Lab publishes 1,492-1,587 on base weights (table b) | measured 1,340-1,448 here |
| 4 concurrent streams | ~78 tok/s aggregate (median; 70-80 across runs) | Reederey87 publishes 63-66 warm; **MiaAI-Lab publishes 146.5 aggregate on 4-stream structured output** (base weights, their client), higher than anything we measured at 4 streams (`docs/RESEARCH-NIGHT.md` §5) |
| Context | 4 requests share one 1,048,576-token pool: a request can grow to 1M, but not four at once (admission waits or spills idle sessions to the store) | 850k-1M in one context |
| KV precision | **FP8** latent KV: greedy replies diverge from bf16 KV within the first 0-78 tokens on 15 of 20 prompts (quality checks above held; long-session recall checked by needle at 314k-358k only) | FP8 KV too |
| Memory margin | the worker node binds (2 GiB less memory). The 4 x 250k stress bottoms at 9.36 GiB MemAvailable there, but a 314k needle right after the stress and MMLU dipped to **7.39 GiB** (under our 8 GiB target, no OOM); 8,192-row prefill chunks (+~4% prefill) were rejected for memory (stress minimum 7.23 GiB) | - |
| API | no `logprobs`, `n > 1` rejected, no images; in single-stream mode a `stop` match ends the reply but the engine keeps decoding silently to EOS / `max_tokens` before the next queued request starts | full OpenAI surface of vLLM |
| Maturity | **work in progress**: one pair of Sparks, one checkpoint, three days of measurements | production kits with many contributors |

Other negatives and trade-offs, measured:

- **FP8 prefill (0083) was rejected**: +7-11% prefill, but greedy replies diverged from bf16 prefill on 25 of 30
  prompts; off.
- **Decode-step kernels (0130) regressed** on the real model (tf code greedy 64.3 -> 62.4 / 56.4 tok/s); off.
- **`hc_fused` (0190)** is bit-exact but 7-8x slower on GB10 (shared-memory limits); off. `mtp_window` (0190) cost
  decode after long prompts; off. (`attn_bm32` is on in production since W1: +5% prefill, same bits.)
- **Patches measured and not adopted** (they stay in the tree, off; `docs/PATCHES.md` and `docs/RESULTS.md` have the
  numbers): 0240 b12x bits 1-2 (slower than today's kernels; only bit 4 is used, via 0360), 0260 `once` expert
  kernel (never beats `fat`), 0270 `FAST_EXPERTS=auto` (-2% end to end despite faster isolated kernels), 0280
  batch round buckets (-5% at 4 streams), 0330 warp-specialized `tc` expert kernels (cfg 1/2 cannot launch on GB10,
  cfg 3 -4%), 0340 per-slot drafter choice (simulated -0.4%), 0400 KDA recurrence v2 (+1.0-1.2%, under its bar),
  0410 sparse attention v2 (+0%), 0370's CPU pinning (+0.4%), and 8,192-row lone chunks (memory, above).
- **The single-stream config died of unified-memory OOM** once, with a 12 GiB session store filling under agent
  traffic at 524k context. Its example config uses 6 GiB; the watchdog (`scripts/systemd/`) restarts a dead pair.
- `q4mse` is a re-quantization of the BF16 non-expert weights: not bit-identical to BF16 (13 of 200 MMLU answers
  differ; accuracy 87.0% -> 88.0%); exactness (drafted == serial) holds within each mode.
- The RoCE all-gather (0230/0350) is limited to 256 KiB a message: one unexplained 2 MiB mismatch was seen once in a
  harness run (above that limit). A run-time RoCE failure writes a marker and the next start uses NCCL.
- Only this checkpoint and this two-node topology have been tested.

## Tests

The patch tests run in the image on one GPU with TensorFold's synthetic checkpoint (no real weights):

```bash
docker run --rm --gpus all -e PYTHONDONTWRITEBYTECODE=1 -v $PWD:/work --entrypoint bash glm53-tensorfold:dev \
    -c "bash /work/scripts/run_tests_in_image.sh /work/results/tests -- tests/cuda/test_patches.py tests/test_glm_tool_calls.py"
```

Host-only (no GPU, no Docker): `python -m pytest -q tests/test_serve_ops.py tests/test_gpuwatch.py` (launcher, canary,
Xid parser, GPU clock watch); the other `tests/test_*.py` run against a patched tree (`PYTHONPATH=<tree>/src`), several
of them in Triton's CPU interpreter. Against a
running server: `python3 bench/glmbench.py --base http://127.0.0.1:8000 --model GLM-5.3-Flash-EXL3 --suites exact`
checks drafted == serial on the real model. Known GPU-test failures are listed in `docs/RESULTS.md` (for example the
engine-level FP8 KV tests do not run on the synthetic model).

Before publishing a fork: `scripts/check-public.sh` scans the tree for private IPs, hostnames, keys and tokens.

## Layout

| Path | What |
| --- | --- |
| `vendor/TensorFold` | TensorFold, pinned submodule (`2f8e514`, 0.3.4), unmodified |
| `patches/` | engine patches, applied in order at image build |
| `docker/` | Dockerfile, entrypoint, compose file |
| `scripts/` | `serve.sh` (build / start / stop / status / logs / canary / watchdog / gpucheck), `prepare.sh`, `gpuwatch.py` (GB10 clock / slow-state watch), `traffic-report.py` (request-log summary), systemd units, `check-public.sh` |
| `config/` | `mia-512k.env.example` (this fork's serve), `prod.env.example` (imported Jayleaton default), earlier configs, `minimal.env.example` (32k debugging baseline) |
| `AGENTS.md` | step-by-step setup for AI coding agents: checks, commands, expected logs, failures and fixes |
| `bench/` | benchmark clients, MMLU-200 subset, tool-call harness, shared-prefix bench, draft-policy and lookup simulators |
| `tests/` | patch tests (GPU) and launcher tests (host) |
| `results/` | `mia-512k/` (this fork's 2026-09-29 benches) plus the imported windows' JSON and scripts (W1-W10); logs omitted |
| `docs/` | `MIA-512K.md` (this serve), then the imported results, patch notes, and design notes |

## Licensing

| Part | License |
| --- | --- |
| This project's code, patches, scripts, benchmarks and docs | **Apache License 2.0** ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). Redistributions, modified or not, must keep the copyright line and the NOTICE attributions and state their changes. |
| TensorFold (`vendor/TensorFold`) | MIT, Copyright (c) 2026 TensorFold contributors; unmodified submodule, the patches are applied at build time. The TensorFold code the patches modify stays under its MIT License. Its third-party notices: `vendor/TensorFold/THIRD_PARTY_NOTICES.md`. |
| RoCE all-gather in `patches/0230`, fast-prefill kernels in `patches/0240` | adapted from / re-implementing [b12x](https://github.com/local-inference-lab/b12x) (Apache-2.0, Luke Alonso and the b12x contributors); details in [`NOTICE`](NOTICE). |
| Fat-expert MoE kernel structure in `patches/0170` | adapted from the Apache-2.0 [Reederey87 kit](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) (code MiaAI-Lab contributed under MIT before 2026-09-07); its NOTICE is reproduced in [`NOTICE`](NOTICE). The BF16 KDA copy in the same patch re-implements an idea from MiaAI-Lab PR #233 without its code. |
| Docker base image | NVIDIA Deep Learning Container License (`nvcr.io/nvidia/pytorch:26.07-py3`) |
| Model weights, this serve (not included) | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` @ `25a44fdb`, a byte-identical mirror of `brandonmusic/GLM-5.3-Flash-tr3-4bpw` @ `5ab363a8`. ShapleyMcg License 1.0 (Brandon M. Music): source-available, attribution required, no rights granted to the person known as "0xSero." The required notice is at the top of this README. Base model `zai-org/GLM-5.3-Flash` is MIT. These shards are not pre-abliterated; the transplant is a runtime copy and is not shipped here. Do not relicense the shards as MIT. |
| Model weights, imported measurements (not included) | `neko-legends/GLM-5.3-Flash-Uncensored-EXL3`: ShapleyMCG License 1.0 per its model card. Sources `orcarouter/GLM-5.3-Flash-Uncensored-FP8` and `zai-org/GLM-5.3-Flash` are MIT per their cards. Those weights are already abliterated. The tables below [Imported stack](#imported-stack-jayleaton-other-weights) use them. You are responsible for how you use either checkpoint. |
| DFlash2 drafter (not included) | `incoai/GLM-5.3-Flash-DFlash2`: **CC BY-NC-ND 4.0, non-commercial only**. This serve pins `dc77ff1c`; the imported configs pin `7d74cdd`. Never bundled; download it yourself, or run without it (MTP drafts only). |

## Credits

- [Brandon M. Music](https://github.com/brandonmmusic-max/shapleymcg): the TR3 / ShapleyMcg checkpoint this serve
  loads (`brandonmusic/GLM-5.3-Flash-tr3-4bpw`, mirrored by Mia-AiLab).
- [MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks): the 2× Spark recipe, the TR3 mirror,
  and the Dealign `o_proj` transplant this serve applies at load (layers 15–44 here; their vLLM recipe text says
  15–45 including MTP).
- [dealignai](https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4): the donor checkpoint the
  transplanted `o_proj` tensors come from. The donor file is not in this repo.
- [Ash Hart / TensorFold](https://github.com/ashhart/TensorFold): the engine, kernels, drafting and server this
  project patches.
- [Jay Leaton](https://github.com/jayleaton/glm53-tensorfold-spark): the imported engine patches, launcher, and
  the neko-legends measurements kept below.
- [MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) and its contributors, for the
  imported engine work: the fat-expert MoE design and the ops ideas listed in `docs/MIA-AUDIT.md`.
- [Reederey87](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark): the production vLLM kit we measured
  against and the Apache-2.0 kernel code `patches/0170` adapts.
- [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) (Luke Alonso and contributors): the RoCE
  one-shot all-gather `patches/0230` ports and the kernel designs `patches/0240` / `0360` re-implement.
- [0xSero](https://huggingface.co/0xSero): imported credit for other GLM-5.3-Flash EXL3 builds and Spark recipes.
  That credit is not a right to the ShapleyMcg checkpoint this serve loads. The ShapleyMcg License grants none.
- [neko-legends](https://huggingface.co/neko-legends) (abliterated EXL3 weights, under Local Inference Lab's
  ShapleyMCG license) and [orcarouter](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-FP8) (the
  uncensored FP8 source).
- [turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3): the EXL3 format.
- [incoai](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2): the DFlash2 drafter.
- [Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash): GLM-5.3-Flash.
- [Vontra](https://huggingface.co/Vontra): the MLX checkpoints TensorFold's GLM recipe uses.
- NVIDIA: the PyTorch container (`nvcr.io/nvidia/pytorch`).
- The [vLLM project](https://github.com/vllm-project/vllm).
