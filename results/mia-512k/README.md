# 2026-09-29 speed receipts (Mia TR3, transplant, 524288 latent FP8)

One sequence on the serve in [`docs/MIA-512K.md`](../../docs/MIA-512K.md). Not the W10 / RigMark numbers
elsewhere in `results/` (those are `neko-legends` weights, `q4mse`, RoCE, a 4-request pool).

| File | What |
| --- | --- |
| `greedy-20260929.jsonl` | `temperature` 0. File finished 2026-09-29 19:14 UTC. |
| `sampled-20260929.jsonl` | `temperature` 1.0, `top_p` 0.95, `top_k` 20. File finished 2026-09-29 19:20 UTC. |

Both files include four warmup rows (`"warmup": true`, `max_tokens` 24, thinking off). The README tables
drop those. Every measured row is HTTP 200. `cached_tokens` is 0 on every row.

## Prompts

User message only. Off sets `chat_template_kwargs.enable_thinking` false and sends no effort. Low, High and
Max set `enable_thinking` true and `reasoning_effort` to `low`, `high` or `max`. The server's
`GLM53_TF_EFFORT_FIELD` is unset, so a top-level `reasoning_effort` would have been ignored.

1. **Short question.** `What is the capital of France? Reply with the city name only.`
2. **Explain.** `Explain how binary search works. Use a few short paragraphs and a tiny numeric example.`
3. **Code.** `Write a Python function that merges two sorted lists into one sorted list. Include a short docstring and nothing else.`
4. **Long passage.** The first sentence is `The archive token for this passage is 184729.` Then the sentence
   `The river kept the same gray color through the afternoon and the road stayed empty.` repeated 220 times,
   then a newline and `Question: What archive token was named in the first sentence? Reply with the number only.`
   Measured prompt tokens: 3,556 with thinking off, 3,562 otherwise.

Measured `max_tokens` is 256. Order: warm each prompt once, then each prompt at Off, Low, High, Max.

## Rates

Taken from the response, not from a separate timer, except `wall_s` (client round trip):

- `prefill_tps` = `(prompt_tokens - cached_tokens) / tensorfold.prefill_s`
- `decode_tps` = `completion_tokens / tensorfold.decode_s`
- `tok_per_round` = `(completion_tokens - 1) / tensorfold.rounds`

`content_head` is the first 80 characters of `message.content`, newlines replaced with spaces. It is not the
full reply. Empty `content_head` with a `length` finish and a large `reasoning_chars` means the cap was spent
inside the think block.

Rates on fewer than 16 completion tokens, and prefill rates on the 20–35 token prompts, are single-step
latency, not throughput. Explain and Code are the decode rows. The long passage is the prefill row.
There is one shot per cell, not a median.

## What came back

`Paris` on every short-question row. `184729` on every long-passage row, greedy and sampled. Explain Max and
Code Max returned no answer text in either file. Low and High wrote no reasoning text except Code High
(70 characters greedy, 64 sampled).
