# Mia TR3 EXL3 on TensorFold, 524,288 context

This is the serve this fork is actually running. The rest of `docs/RESULTS.md` and the lower half of the README
are the imported Jayleaton stack (`neko-legends` weights, `q4mse`, a 4-request 1M pool, RoCE). Do not mix the
numbers.

Measured 2026-09-29 on two GB10 Sparks, tensor parallel 2. The engine reported ready 970 s after start. The
greedy bench finished 19:14 UTC and the sampled bench 19:20 UTC. Tables and receipts:
[the README](../README.md#speed-2026-09-29) and [`results/mia-512k/`](../results/mia-512k/README.md).

## Weights

[`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw) at
`25a44fdbf16862a46b7cc9921142c6c81350af2f`.

Mia's model card says this is not an original quantization. It is a byte-identical redistribution of
[`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) at
`5ab363a8dcf6405955fd5f99671e01a1c9fb124b`, published so
[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
keeps a fetch target. Brandon M. Music created the EXL3/TR3 checkpoint (ShapleyMcg). Z.AI created the base
model (`zai-org/GLM-5.3-Flash`, MIT). Mia re-hosts the snapshot.

> This work includes or was produced using ShapleyMcg, created by Brandon M. Music
> (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0,
> an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg
> without this attribution is unlicensed.

The checkpoint is about 164 GiB, 120 safetensor shards, quant `exl3` / 4-bit mcg. `num_hidden_layers` is 45,
`kv_lora_rank` is 512, `num_nextn_predict_layers` is 1. Routed experts are the 4-bit trellis. The other
matrices (attention projections, shared expert, dense layers, head) stay BF16. This serve does not run
`q4mse` on them.

These weights are the stock TR3 body. They are not the `neko-legends` abliterated checkpoint, and they are
not TensorFold's MLX recipe (`Vontra/GLM-5.3-Flash-MLX-4bit-MTP`). The engine prints that MLX line on every
EXL3 launch. The argv of this boot was the Mia snapshot path.

The drafter is [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) at
`dc77ff1c99eeb2df044ee3d4f0094eb033fee410`. That revision is not the `7d74cdd` pin in `config/prod.env.example`.
DFlash2 is CC BY-NC-ND 4.0. It is not in this repo. Download it yourself, or leave `DRAFTER` empty and the
server uses MTP drafts only.

Inside the container the paths are the Hugging Face snapshot directories under
`/root/.cache/huggingface/hub/`. Each node's cache is mounted there. The weights are not committed.

## Abliteration

The edit is a load-time transplant, `patches/0420-glm-ablit-transplant.patch`. The submodule
`vendor/TensorFold` stays at `2f8e514` (0.3.4). The patch is the diff. The image applies patches with
`git apply`, and falls back to `patch -p1` (the image copy has no submodule git dir). 0420 is a new file
plus hooks in the fast-boot path and `weights.py`, so a rebuild picks it up without moving the submodule SHA.

`GLM53_TF_ABLIT=1` copies BF16 `self_attn.o_proj` from the donor into the rank's in-memory checkpoint, on
CPU, before `.to(device)`, before non-expert quant, and before graph capture. The prepared-folder cache key
includes the ablit extra and a digest of the loader code. A folder built with the transplant off does not
satisfy a boot with it on. `serve.sh` sets `GLM53_TF_PREPARED=/prepared` and write-back on.

The donor is the full tensor, not a rank shard. `o_proj` is a column split of the last axis. Rank `r` of
world 2 takes columns `[r * half : (r + 1) * half]`.

What is copied:

| | |
| --- | --- |
| Source | [`dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4`](https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4), method `dealign-oproj-transplant`, as published with [drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock](https://huggingface.co/drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock) and fetched by the Mia vLLM kit's `ablit/fetch_transplant.py` |
| Tensors | 31, named `model.language_model.layers.{15–45}.self_attn.o_proj.weight`. No key contains `mtp`. BF16. |
| Shapes | `[4096, 16384]` for layers in `{15,19,23,27,31,35,39,43,45}`; `[4096, 8192]` otherwise |
| Applied | layers **15–44** (`GLM53_TF_ABLIT_LAYERS=15-44`) |
| Left stock | layers **0–14** (hashed; the load fails if they change) and **layer 45** |

Layer 45 stays stock on purpose. `num_hidden_layers` is 45, so the main stack is layers 0–44 and
`layers.45.self_attn.o_proj` is the MTP block (the engine loads it as the next-n layer, not as a 46th
transformer block). The donor contains that tensor. The log says `donor_layer_45=present_not_applied` and
`mtp=False`. Extending the range to 45 is refused, because 45 is not a main-stack layer. Do not treat a
missing MTP edit as a failed transplant.

Projection orthogonalization is not this method. The Mia kit measured the shipped refusal direction as noise
against stock `o_proj` (the projection variants did not reproduce the published edit; the byte copy did).
`GLM53_TF_NONEXPERT` must stay `bf16`. `q4` and `q4mse` would quantize the copy, and the loader refuses them.

This boot, both ranks, after the copy: `mean rel_l2=0.0000`, `mtp=False`, edited 15 through 44, guarded 0
through 14. Pre-copy distance was 0.1280 on rank 0 and 0.1269 on rank 1. The post-copy mean is the check
that the rank's half of the donor landed. The donor file is a host path, the same path on both nodes,
mounted read-only at `/ablit/donor.safetensors`. It is not in git.

## TensorFold config

`scripts/serve.sh` sources `config/prod.env` unless `CONFIG=` is set. This profile is not that file.
`config/prod.env.example` is Jayleaton's 1,048,576-token, batch-4, q4mse, RoCE serve on neko-legends weights.
`config/prod-single.env.example` is 524,288 with **bf16** latent KV, still q4mse, still those weights.

```bash
cp config/mia-512k.env.example config/mia-512k.env
# fill WORKER_SSH, HEAD_IP, HEAD_HF, WORKER_HF, ABLIT_DONOR_HOST
CONFIG=config/mia-512k.env scripts/serve.sh build
CONFIG=config/mia-512k.env scripts/serve.sh preflight
CONFIG=config/mia-512k.env scripts/serve.sh start
```

`config/*.env` is gitignored. Do not commit the filled file.

Knobs that define this boot:

| Setting | Value |
| --- | --- |
| `IMAGE` | `glm53-tensorfold:dev` (`sha256:cdcc670ac30458d27aa2bd7f2e0e1a67d8785646071a0289bc5778288dda2bc7`, 2026-09-29T18:19:59Z, from `nvcr.io/nvidia/pytorch:26.07-py3`) |
| `SERVED_NAME` / `PORT` / `HOST` | `GLM-5.3-Flash-EXL3` / `8888` / `0.0.0.0` (no authentication; put a proxy in front before exposing it) |
| `CONTEXT` / `MAX_TOKENS` | `524288` / `32768` |
| `GLM53_TF_LATENT_KV` | `1` |
| `GLM53_TF_KV_DTYPE` | `fp8` |
| `GLM53_TF_NONEXPERT` | `bf16` |
| `GLM53_TF_BATCH` | `2` |
| `GLM53_TF_COMM_BACKEND` | `nccl` |
| `GLM53_TF_AUTO_FDRAFTS` | `7` |
| `GLM53_TF_DEPTH` | `threshold` (upstream thresholds; not patch 0071's `cost` policy) |
| `GLM53_TF_LONGCTX_GRAPHS` | `1` |
| `NO_DRAFTS` | `0` |
| `GLM53_TF_ABLIT` / `GLM53_TF_ABLIT_LAYERS` | `1` / `15-44` |
| `GLM53_TF_ABLIT_DONOR` | `/ablit/donor.safetensors` |
| `PREFLIGHT` | `strict` |
| `MEM_GATE_GIB` | `8` (MemFree, checked before launch) |
| `CANARY` | `off` |
| Head NCCL | `enp1s0f0np0` / `rocep1s0f0` |
| Worker NCCL | `enp1s0f1np1` / `rocep1s0f1` |

`serve.sh` passes a different interface to each rank when `HEAD_NCCL_*` and `WORKER_NCCL_*` are set. On a
crossed cable the port that faces the other Spark is not the same name on both machines. Same-named HCAs are
not on one subnet. Passing every HCA makes NCCL sit in init. `HEAD_IP` is the head's address on that CX7
link, not its management address. The worker's Hugging Face cache has to be a directory the worker user can
read. Export it from the head over the CX7 address (NFS). The management network crawls, and a Docker volume
the worker user cannot read fails the weight check.

Left off, on purpose, relative to the imported production file and to the vLLM recipe this profile was
aiming at the shape of:

| Left off | Why it is not in this profile |
| --- | --- |
| `q4mse` / `q4` | Re-quantizes the BF16 non-experts, including the tensors the transplant just wrote. The loader refuses the combination. The imported speed tables use `q4mse`; they do not describe this boot. |
| RoCE (`GLM53_TF_COMM_BACKEND=roce`) | This boot is NCCL on the crossed cable. |
| Fat MoE, KDA BF16 large-M | Not set. The vLLM kit's fused-MoE and KDA flags have no equivalent turned on here. |
| KV pool, session GiB, NVMe session tier, prefix share | Not set. `serve.sh` still mounts a sessions directory; the quota knob is off, so it is not a session cache. |
| Decode overlap, CPU pin | Off (patch 0370's adopted W10 settings are the other stack). |
| `GLM53_TF_EFFORT_FIELD` | Unset (default 0). Top-level `reasoning_effort` is ignored. Pass `chat_template_kwargs`. |
| `FORCE_CONTEXT=1` | Must stay unset. It is the override that lets a per-head cache try a context that does not fit. |
| TensorFold 0.3.7 | Not rebased. This image is 0.3.4 plus the patches in this tree. |
| A third Spark | The launcher is `--tp 2` only. Per-head KV at 512k is about 195 GiB a rank and still does not fit if the cache were split three ways. Latent FP8 is the reason 512k loads. |

Context check in `serve.sh`: with latent KV off, a context past 131,072 is refused. With `GLM53_TF_LATENT_KV=1`,
524,288 is allowed. Per-head KV is about 390 KB a token a rank. Latent bf16 is about 13.6 KB. Latent FP8 is
7,664 bytes a token a rank (patch 0220: 512 e4m3 values plus one fp32 scale, 528-byte rows). DSA layers use
that cache. Linear-attention state is separate and is not the 390 KB figure.

## What the boot logged

Rank 0, copied from the container log:

```text
[tensorfold] latent KV cache (fp8 rows): 7.4 KB a token a rank (524296 slots, 3.72 GB)
[tensorfold] GLM53_TF_BATCH=2: only 1 sequence(s) fit with GLM53_TF_BATCH_RESERVE_GB=4 GB kept free (3.96 GB a sequence at 524296 cache slots); serving that many
[tensorfold] serving GLM-5.3-Flash-EXL3 at http://0.0.0.0:8888/v1 on CUDA, rank 0 of 2 (sampling: temperature 1.0, top_p 0.95; drafts: on; loaded in 969.7s)
[boot] r0 +969.9s engine ready (session store, batcher) (0.1s) | MemFree 6.9 GiB, MemAvailable 19.5 GiB
```

The transplant line on both ranks was `method=transplant layers=15-44 mtp=False mean rel_l2=0.0000
donor_layer_45=present_not_applied`. Requested batch 2 did not fit: the default reserve keeps 4 GB free and
one 512k FP8 sequence is 3.96 GB, so the server is **one sequence**. That is the measured concurrency. Do
not read the README's imported "4 requests" as this process.

`/v1/models` does not advertise `max_model_len`. The engine log is the context proof (524,288 requested,
524,296 slots). The sampling line is the checkpoint's generation config before the server merges its default
`top_k` of 20. A request that omits `top_k` still gets 20. `temperature` 0 is greedy.

First boot logged `weights: no usable prepared folder (no folder): building from the checkpoint; will write it`. Rank 0 then wrote
`/prepared/Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdbf168/4780c85a172687a9/rank0` (88.6 GB in 117.0 s)
and the drafter folder (0.7 GB). Those directories are on the hosts, not in git. A later start reuses them
only when the cache key matches, including the ablit extra.

The memory gate (`MEM_GATE_GIB=8`) reads MemFree and runs before the load. This boot got past it. The 6.9 GiB
MemFree in the ready line is after the weights and the cache were resident. MemAvailable at that line was
19.5 GiB on rank 0.

## API used for the benches

`POST /v1/chat/completions`, model `GLM-5.3-Flash-EXL3`, `max_tokens` 256 on the measured rows (24 on the
discarded warmup). Thinking and effort:

```json
{ "chat_template_kwargs": { "enable_thinking": true, "reasoning_effort": "low" } }
```

`enable_thinking` false, and no effort, is the Off column. The checkpoint template sets the effort to `max`
unless `reasoning_effort` is `low` or `high`. Off is not that path: the thinking-off template strips the
`Reasoning Effort` system line and closes an empty think block.

Low and High still often produced no `reasoning_content` on these prompts. Max on Explain and Code spent the
256-token cap inside the think block and returned an empty answer. That is what the tables record. It is not
a claim about refusal, and it is not a 512k needle: the long prompt is 3,556 tokens and names the token in
the first sentence.

## Rebuild notes that are easy to get wrong

- Do not commit `config/mia-512k.env`, the donor, the weights, the drafter, or a dirty `vendor/TensorFold`.
- Do not point `MODEL_PATH` at `neko-legends` or at the MLX repo and then describe the result as this boot.
- Do not set `GLM53_TF_NONEXPERT=q4mse` with the transplant on. Preflight and the loader both refuse it.
- Do not set `GLM53_TF_ABLIT_LAYERS` to `15-45`. Layer 45 is the MTP block on this config.
- Do not set `FORCE_CONTEXT=1` to "try per-head 512k". It will not fit.
- `scripts/check-public.sh` scans tracked and untracked files for private addresses, home directories, keys
  and tokens before a push. `vendor/TensorFold` is excluded (upstream docs use a documentation address).
