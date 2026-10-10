# OPD and MOPD on Tinker: results (2026-09-27)

All runs used the Tinker API (SA RL Infra Research account) with thinking on and a 32k context. The full
records are kept outside Git, in `opd-runs/`:

| Directory | Contents |
|---|---|
| `tinker-sa-sft-teachers-20260927T115503Z/` | The two SFT teachers and the base-model benchmark (`sft.py`) |
| `tinker-sa-opd-caesar-20260927T123742Z/` | OPD on caesar_cipher (`opd.py`), 853 MB of records |
| `tinker-sa-mopd-20260927T123742Z/` | MOPD on caesar_cipher + simple_geometry (`mopd.py`), 538 MB of records |
| `tinker-sa-opd-mopd-report-20260927/` | `REPORT.md`, `summary.json`, and the figures (PNG and SVG) from `report.py` |

Weights stay on Tinker; the records hold only their paths. Each run used one seed, so these are single
observations, not a controlled ranking.

## Pipeline

1. Teachers by SFT (`sft.py`): one rank-64 LoRA per domain, one epoch over the verified traces of the frozen
   GRPO teachers
   ([dataset](https://huggingface.co/datasets/semianalysisai/Qwen3.6-35B-A3B-teacher-sft-caesar-geometry)).
2. OPD (`opd.py`): a fresh rank-64 student, 20 updates × 128 caesar_cipher prompts, from the caesar teacher.
3. MOPD (`mopd.py`): a fresh rank-64 student, 20 updates × (64 caesar_cipher + 64 simple_geometry) prompts,
   each response scored by its own domain's teacher.

Both students use the shared recipe: LR 1e-5, one response per prompt, up to 30,720 response tokens, and
sampled-token reverse KL as the only signal. These runs were synchronous (policy lag 0). The recipe default,
and now `common.py`'s default, is asynchronous with policy lag 1 (`--policy-lag 1`).

## Quality

Same benchmark for every model: the first 100 dev problems of each domain (never trained on), 3 samples each
at temperature 1, avg@3.

| Model | caesar_cipher | simple_geometry |
|---|---|---|
| Base Qwen3.6-35B-A3B | 27.3% | 51.3% |
| SFT teacher (own domain) | 79.3% | 99.7% |
| OPD, update 10 | 65.3% | 64.0% (not trained) |
| OPD, update 20 | 74.7% | 70.7% (not trained) |
| MOPD, update 10 | 59.0% | 98.0% |
| MOPD, update 20 | 75.3% | 99.3% |

- MOPD matches OPD on caesar_cipher (75.3% vs 74.7%) with half as many caesar prompts (1,280 vs 2,560), and
  also reaches the geometry teacher (99.3% vs 99.7%). The second teacher cost nothing measurable on the
  first domain.
- Both students close most of the gap between the base model and the caesar teacher (27.3% to 79.3%) in 20
  updates: OPD 91%, MOPD 92%. Geometry is easier; MOPD is at 98% by update 10.
- OPD improves geometry without training on it (53% to 71%), so distilling the caesar teacher transfers
  partially. MOPD's geometry teacher closes the rest of the gap.
- Noise: the base model was benchmarked three times (caesar 27.3 / 28.0 / 27.0%, geometry 51.3 / 53.3 /
  56.0%). Differences of 2–3 points are within sampling noise at 300 samples.

## Throughput and time

Client wall time on a shared service, not GPU utilization. OPD and MOPD ran at the same time on one account,
and the SFT teachers also trained concurrently, so neither run had the service to itself.

| | OPD | MOPD |
|---|---|---|
| Wall time (20 updates + 3 evaluations) | 67.4 min | 59.7 min |
| Median update | 173 s | 152 s |
| … rollout (sample + teacher score) | 142 s (83%) | 127 s (83%) |
| … train step (`forward_backward` + `optim_step`) | 16 s (9%) | 14 s (9%) |
| … weight sync (save + new sampler) | 4 s (3%) | 5 s (4%) |
| … client-side (writing rollouts, building the batch) | 6% | 4% |
| Median / p90 / last response finishes | 65 / 115 / 140 s | 33 / 88 / 125 s |
| Teacher scoring left after the last response | 2 s | 2 s |
| Sampled tokens/s (aggregate, median update) | 11,790 | 8,355 |
| Trained tokens/s (median update) | 103,907 | 74,876 |
| Evaluation (600 samples) | ~2.5 min each | ~2.5 min each |

- Long thinking responses set the pace. A single response decodes at a median of 183 tokens/s (p10 152,
  p90 232), so a response that hits the 30,720-token cap takes about 170 s. Every update had some truncated
  responses (122 OPD and 73 MOPD training responses in total), so a synchronous update cannot go much below
  2.5 minutes regardless of parallelism. The median response finished in 65 s (OPD) or 33 s (MOPD); the rest
  of the rollout is waiting for stragglers.
- Teacher scoring costs almost no wall time. Scoring a response is one prefill (`compute_logprobs`), about
  2 s, and it starts as soon as the response finishes. It overlaps the remaining generation, leaving only 2 s
  after the last response.
- Training is fast relative to sampling. One update's 1.4M (OPD) or 0.9M (MOPD) tokens train in about 15 s
  (75–104k tokens/s). SFT, which only trains, ran at 196k tokens/s on caesar (1.5M-token batches, 7.7 s per
  update, pipelined) and 66k tokens/s on geometry (0.4M-token batches, 6.1 s per update).
- Ways to cut time: asynchronous (off-policy) sampling, so update n+1 samples while n trains; a lower token
  cap; or dropping and resampling stragglers. Each changes the protocol and would be reported as a separate
  variant.
- Trainer and sampler agree closely: the mean absolute logprob gap on sampled tokens is 0.015 nats.

## Cost

Estimates from token counts at Tinker's published prices (sampling $1.335/M, prefill $0.54/M, training
$1.177/M), not invoices. Prefill caching is ignored.

| Stage | Tokens | Estimated cost |
|---|---|---|
| SFT teachers (both) | 292M trained | ~$345 |
| Base and teacher benchmarks | ~5M sampled | ~$7 |
| OPD student | 48M sampled, 34M prefill, 34M trained | ~$122 |
| MOPD student | 35M sampled, 22M prefill, 21M trained | ~$83 |

The SFT teachers cost more than both students combined: one epoch covers 292M tokens of thinking traces,
while each student needs only 20 × 128 responses.

## Observations

- The API surface is small. The loop uses only `sample`, `compute_logprobs`, `forward_backward`,
  `optim_step`, `save_weights_for_sampler` and `save_state` (`backend.py`). The loss is the built-in
  `importance_sampling` with per-token advantages. Multi-teacher routing is a dict of sampling clients. No
  GPUs, serving or weight transfer are managed client-side.
- Weights never leave the service. Teachers and students are addressed by `tinker://` paths. Every update's
  student weights were saved as sampler checkpoints, so any update can be re-evaluated later.
- The SFT loss barely moved (caesar 0.316 to 0.308 nats/token, geometry 0.342 to 0.331) while accuracy
  jumped (27% to 79%, 51% to 99.7%). The traces come from a GRPO-trained copy of the same base, so they are
  nearly on-distribution and the useful signal is a small per-token shift. OPD shows the same pattern: the
  reverse KL starts at only 0.003–0.005 nats/token and still drives large accuracy gains.
- For large `forward_backward` requests that the service chunks internally, its `loss:sum` metric came back
  2–3× the true sum. The per-token logprobs it returns were exact (checked against `forward` on the same
  batches), so the NLL is computed from those.
- The first SFT step spikes (caesar NLL 0.305 to 0.350, geometry 0.339 to 0.421 after update 1) and recovers
  by update 3. This is Adam's first step at the cookbook's recommended LR of about 5e-4.
- Thinking at 32k dominates response length (11–13k tokens median on caesar), step time (the straggler
  floor), and cost (sampling is the largest cost for both students).
- No stage logged an error, warning or retry (the SDK may retry internally without logging). The runs used
  about 1% CPU and a few MB/min of network on the laptop that drove them.

## Reproduce

```bash
python hosted/tinker/launch.py --root /abs/sft-teachers --teachers sft --teachers-only
python hosted/opd/opd.py --domain caesar_cipher --teacher <caesar sampler path> --output /abs/opd
python hosted/opd/mopd.py --teachers /abs/sft-teachers/teachers.json --output /abs/mopd
python hosted/opd/report.py /abs/opd /abs/mopd --output /abs/report --teachers-campaign /abs/sft-teachers
```
