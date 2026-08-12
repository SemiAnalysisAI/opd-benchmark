# MOPD framework benchmark

This benchmark compares multi-teacher on-policy distillation (MOPD) in
Prime-RL, verl, and Miles on the same two-GPU host. It follows the structure of
Miles' `run-qwen3-8B-opd-multi-teacher.sh` recipe while reducing model sizes and
sequence lengths to fit 2x H100 80 GB.

## Common recipe

- Student: `Qwen/Qwen2.5-0.5B-Instruct`
- Math teacher: `Qwen/Qwen2.5-Math-1.5B-Instruct`
- Code teacher: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Data: an equal mixture of GSM8K math prompts and HumanEval code prompts
- Routing: every math row goes only to the math teacher and every code row goes
  only to the code teacher
- Training steps: 15
- Prompts per step: 16
- Student rollouts per prompt: 4
- Total trajectories per step: 64
- Maximum prompt/response lengths: 1024/256 tokens
- Sampling: temperature 1.0, top-p 1.0
- Learning rate: 1e-6, constant schedule
- Optimizer: AdamW, betas (0.9, 0.98), weight decay 0.1, gradient clip 1.0
- OPD signal: sampled-token reverse KL (`k1`), coefficient 1.0
- Task reward in the training loss: disabled
- Seed: 42 where exposed

The three checkpoints have byte-identical `tokenizer.json` files. This is a
hard requirement for comparing teacher log probabilities at student-sampled
token IDs.

## Deliberate differences from the full Miles example

Miles' published example uses an 8B student, 32B math teacher, 30B-A3B code
teacher, eight GPUs, 16-way top-k teacher logits, and 16K-token responses. That
configuration cannot fit this two-GPU machine. The benchmark retains the
multi-teacher routing and optimizer structure but uses smaller same-tokenizer
models, shorter generations, and sampled-token (`top_k=0`) log probabilities.

`top_k=0` is the only reverse-KL objective implemented with equivalent semantics
by all three frameworks. Miles supports reverse KL over a teacher/student top-k
union, while verl's top-k mode is a different forward-KL objective and Prime-RL
does not expose a top-k OPD mode.

## Framework routing

- Miles reads `metadata.opd_teacher` and selects a named entry from
  `--opd-teacher-urls`.
- verl reads `data_source` through `distillation.teacher_key`.
- Prime-RL uses two train sources, each with its own OPD algorithm block and
  teacher endpoint.

verl normally reserves one complete Ray GPU slot for every teacher in addition
to the actor GPU, making stock two-teacher MOPD require at least three GPUs. The
included `verl_external_teachers.patch` adds an opt-in external HTTP teacher
manager so its actor can use GPU 0 while two small SGLang teachers share GPU 1.
The loss, per-row routing, and trainer remain verl-native.

## Usage

Copy this directory to `/root/opd-bench/artifacts/mopd-frameworks` on the GPU
host. `setup_remote.sh` pins the three repositories and models, starts the two
container environments, creates every framework's copy of the shared dataset,
validates routing/tokenizer identity, and converts the Miles student checkpoint.

Run the full suite serially on the host:

```bash
BENCH_ROOT=/root/opd-bench \
  bash /root/opd-bench/artifacts/mopd-frameworks/setup_remote.sh

STEPS=15 RUN_ID=full BENCH_ROOT=/root/opd-bench \
  bash /root/opd-bench/artifacts/mopd-frameworks/run_all.sh
```

Run the matched one-physical-teacher OPD ablation with the same recipe shape:

```bash
TEACHER_MODE=single STEPS=15 RUN_ID=opd-matched BENCH_ROOT=/root/opd-bench \
  bash /root/opd-bench/artifacts/mopd-frameworks/run_all.sh
```

In this mode, both logical domain routes use the math-policy endpoint. It is a
systems-footprint ablation, not a quality-equivalent control. `prime_opd.toml`
contains Prime-RL's matching configuration; the Miles and verl launchers switch
their endpoint maps through `TEACHER_MODE`.

`STEPS=1` runs an optimizer-step smoke test. The three launchers can also be run
independently. Prime-RL and verl launch from the host; Miles launches inside the
`opd-miles` container, as `run_all.sh` demonstrates.

Download the compact logs and reduce them locally:

```bash
RUN_ID=full REMOTE=root@86.38.238.166 \
  bash benchmarks/mopd-frameworks/collect_remote_results.sh

python3 benchmarks/mopd-frameworks/analyze_results.py \
  --output benchmarks/mopd-frameworks/results/summary.json
```

For the one-teacher run, use `TEACHER_MODE=single RUN_ID=opd-matched` and set
`DESTINATION` to `benchmarks/mopd-frameworks/results-opd` when collecting.

See `REPORT.md` for the measured result. `mopd-benchmark-bundle.tar.gz` contains
the report, raw compact logs, data generator, setup script, launchers, configs,
verl adapter, analyzer, and the publication-ready SVG/PNG chart. Run
`generate_blog_chart.py` after updating `results/summary.json` to regenerate the
chart from measured data.

`OPD_VS_MOPD.md` contains the teacher-count ablation and fixed-rollout routing
probe. Run the latter inside the Miles container:

```bash
RUN_ID=routing-full BENCH_ROOT=/root/opd-bench \
  bash /root/opd-bench/artifacts/mopd-frameworks/run_routing_probe.sh
```

The probe generates the same 8 math + 8 code prompts × 4 rollouts as one
training batch and cross-scores all 64 trajectories with both policies.
`routing_probe.py` is the downloadable implementation. Regenerate the ablation
JSON and both publication charts with:

```bash
python3 benchmarks/mopd-frameworks/compare_opd_mopd.py
python3 benchmarks/mopd-frameworks/generate_ablation_charts.py
```
