# OPD and MOPD on Tinker: results and notes (2026-09-27)

Everything here ran on the Tinker API (SA RL Infra Research account), thinking on, with a 32k context.
The full records are outside Git, in `opd-runs/`:

| Directory | Contents |
|---|---|
| `tinker-sa-sft-teachers-20260927T115503Z/` | The two SFT teachers and the base-model benchmark (`hosted/tinker/sft.py`) |
| `tinker-sa-opd-caesar-20260927T123742Z/` | OPD on caesar_cipher (`opd.py`), 853 MB of records |
| `tinker-sa-mopd-20260927T123742Z/` | MOPD on caesar_cipher + simple_geometry (`mopd.py`), 538 MB of records |
| `tinker-sa-opd-mopd-report-20260927/` | `REPORT.md`, `summary.json`, and the figures (PNG and SVG) from `report.py` |

All weights stay on Tinker; the records hold their paths only. One seed per run: these are single
observations, not a controlled ranking.

## Pipeline

1. **Teachers by SFT** (`hosted/tinker/sft.py`): one rank-64 LoRA per domain, one epoch over the
   verified traces of the frozen GRPO teachers
   ([dataset](https://huggingface.co/datasets/semianalysisai/Qwen3.6-35B-A3B-teacher-sft-caesar-geometry)).
2. **OPD** (`opd.py`): a fresh rank-64 student, 20 updates × 128 caesar_cipher prompts, from the caesar teacher.
3. **MOPD** (`mopd.py`): a fresh rank-64 student, 20 updates × (64 caesar_cipher + 64 simple_geometry)
   prompts, each response scored by its own domain's teacher.

Both students use the shared recipe: LR 1e-5, one response per prompt, up to 30,720 response tokens,
sampled-token reverse KL as the only signal, and synchronous updates (policy lag 0). The recipe's default is
asynchronous with policy lag 1 (`--policy-lag 1`, now common.py's default); these runs are the synchronous variant.

## Quality

The same benchmark everywhere: the first 100 dev problems of each domain (never trained on), 3 samples each at
temperature 1, avg@3.

| Model | caesar_cipher | simple_geometry |
|---|---|---|
| Base Qwen3.6-35B-A3B | 27.3% | 51.3% |
| SFT teacher (own domain) | 79.3% | 99.7% |
| OPD, update 10 | 65.3% | 64.0% (not trained) |
| **OPD, update 20** | **74.7%** | 70.7% (not trained) |
| MOPD, update 10 | 59.0% | 98.0% |
| **MOPD, update 20** | **75.3%** | **99.3%** |

- **MOPD matches OPD on caesar_cipher** (75.3% vs 74.7%) while seeing half as many caesar prompts
  (1,280 vs 2,560), and it also recovers the geometry teacher (99.3% vs 99.7%). Adding the second
  teacher cost nothing measurable on the first domain.
- **Both students close most of the gap to the caesar teacher:** OPD 91% and MOPD 92% of the
  27.3% → 79.3% gap, in 20 updates. Geometry is easier: MOPD is at 98% by update 10.
- **OPD improves geometry without training on it** (53% → 71%). Distilling the caesar teacher's
  reasoning transfers partially; MOPD's geometry teacher gets it the rest of the way.
- **Noise:** the base model was benchmarked three times (27.3 / 28.0 / 27.0% caesar, 51.3 / 53.3 / 56.0%
  geometry). Differences of 2–3 points are within sampling noise at 300 samples.

## Throughput and time

Client wall time on a shared service; not GPU utilization. OPD and MOPD ran **at the same time** on one
account (the SFT teachers also trained concurrently), so neither had the service to itself.

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

- **Long thinking responses set the pace.** A single response decodes at a median of **183 tokens/s**
  (p10 152, p90 232), so a response that reaches the 30,720-token cap needs about 170 s. Every update had
  some (122 OPD and 73 MOPD training responses were truncated in total), so a synchronous update cannot go
  much below 2.5 minutes, however many requests run in parallel. The median response finished in 65 s
  (OPD) or 33 s (MOPD); the rest of the rollout waits for the stragglers.
- **The teachers were almost free in wall time.** Scoring a response is one prefill (`compute_logprobs`),
  about 2 s per response, and it starts as soon as that response finishes. It overlaps the remaining
  generation, and only 2 s is left after the last response.
- **Training is fast next to sampling.** One update's 1.4M (OPD) or 0.9M (MOPD) tokens train in about
  15 s (75–104k tokens/s). SFT, which only trains, ran at 196k tokens/s on caesar (1.5M-token
  batches, 7.7 s per update, pipelined) and 66k tokens/s on geometry (0.4M-token batches, 6.1 s per update).
- **Where the time could go down:** asynchronous (off-policy) sampling, so update n+1 samples while n trains;
  a lower token cap; or dropping and resampling stragglers. Each changes the protocol and would be
  reported as a separate variant.
- The trainer and sampler agree closely: the mean absolute logprob gap on sampled tokens is 0.015 nats.

## Cost

These are estimates from token counts at Tinker's published prices (sampling $1.335/M, prefill $0.54/M,
training $1.177/M), not invoices. Prefill caching is ignored.

| Stage | Tokens | Estimated cost |
|---|---|---|
| SFT teachers (both) | 292M trained | ~$345 |
| Base and teacher benchmarks | ~5M sampled | ~$7 |
| OPD student | 48M sampled, 34M prefill, 34M trained | ~$122 |
| MOPD student | 35M sampled, 22M prefill, 21M trained | ~$83 |

The SFT teachers cost more than both students together, because one epoch covers 292M tokens of thinking
traces. The students only need 20 × 128 responses.

## Notes for the article

- **The API surface is small.** The whole loop uses the `sample`, `compute_logprobs`, `forward_backward`,
  `optim_step`, `save_weights_for_sampler` and `save_state` calls (`backend.py`). The loss is
  the built-in `importance_sampling` with per-token advantages. Multi-teacher routing is simply a dict of
  sampling clients, and no GPUs, serving, or weight transfer are managed by us.
- **Weights never leave the service.** Teachers and students are addressed by `tinker://` paths.
  Every update's student weights were saved as sampler checkpoints, so any update can be re-evaluated later.
- **The SFT loss barely moved** (caesar 0.316 → 0.308 nats/token; geometry 0.342 → 0.331) while accuracy
  jumped (27% → 79%, 51% → 99.7%). The traces come from a GRPO copy of the same base, so they are nearly
  on-distribution: the useful signal is a small per-token shift. The same shows in OPD, where the
  reverse KL starts at only 0.003–0.005 nats/token and still drives large accuracy gains.
- **One surprise in the metrics:** for large, internally chunked `forward_backward` requests, the service's
  `loss:sum` metric came back 2–3× the true sum. The per-token logprobs it returns were exact (checked
  against `forward` on the same batches), so the NLL is computed from those.
- **The first SFT step spikes** (caesar NLL 0.305 → 0.350, geometry 0.339 → 0.421 after update 1) and recovers
  by update 3: Adam's first step at the cookbook's recommended LR of about 5e-4.
- **Thinking at 32k dominates everything:** response lengths (11–13k tokens median on caesar), step time
  (the straggler floor), and cost (sampling is the largest line in both students).
- **Nothing failed:** no stage logged an error, warning, or retry (the SDK may retry internally without logging). The runs used about 1% CPU and a few MB/min of
  network on the laptop that drove them.

## Reproduce

```bash
python hosted/tinker/launch.py --root /abs/sft-teachers --teachers sft --teachers-only
python hosted/opd/opd.py --domain caesar_cipher --teacher <caesar sampler path> --output /abs/opd
python hosted/opd/mopd.py --teachers /abs/sft-teachers/teachers.json --output /abs/mopd
python hosted/opd/report.py /abs/opd /abs/mopd --output /abs/report --teachers-campaign /abs/sft-teachers
```
