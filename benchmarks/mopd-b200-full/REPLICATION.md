# Full-size MOPD benchmark replication guide

This guide recreates the 15-step Prime-RL, Miles, and verl benchmark using the
models and workload from Miles' Qwen3-8B multi-teacher OPD example. It was
validated on one 8× NVIDIA B200 node. The rented machine was described as B300,
but `nvidia-smi` identified every GPU as B200; report the detected hardware.

The benchmark keeps logs, metrics, configs, environment manifests, GPU samples,
and all failed-attempt traces. It deliberately does not retain or download
generated checkpoints, broadcast weights, model caches, or base-model weights.

## 1. Exact experiment

| Item | Value |
|---|---|
| Student | `Qwen/Qwen3-8B` |
| Math teacher | `Qwen/Qwen3-32B` |
| Code teacher | `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| Train pool | 128 GSM8K + 128 HumanEval prompts |
| Eval pool | 32 GSM8K + 32 HumanEval prompts |
| Per step | 16 prompts × 4 rollouts = 64 trajectories |
| Run | 15 optimizer steps, 960 real trajectories |
| Sequence limits | 1,024 prompt + 16,384 response tokens |
| Sampling | temperature 1.0, top-p 1.0, no top-k truncation |
| Optimizer | AdamW, LR 1e-6, betas (0.9, 0.98), WD 0.1, clip 1.0 |
| Objective | sampled-token reverse KL (`k1`), task reward disabled |

Physical GPU assignment:

| GPUs | Prime-RL | Miles | verl |
|---|---|---|---|
| 0–1 | trainer | Megatron trainer TP=2 | FSDP actor + colocated vLLM |
| 2–5 | vLLM rollout DP=4 | SGLang rollout DP=4 | FSDP actor + colocated vLLM |
| 6 | math teacher | math teacher | external SGLang math teacher |
| 7 | code teacher | code teacher | external SGLang code teacher |

verl's actor and rollout are a six-GPU hybrid pool, so its training data-parallel
size is six. A 64-row batch is not divisible by six. `ppo_mini_batch_size=6`
causes verl's native balancing path to append eight documented, two-token,
zero-loss rows after generating the same 64 real trajectories. The included
padding patch clears the MOPD teacher fields on those synthetic rows. Analysis
also removes their 16 tokens per step from trained-token throughput.

## 2. Deliberate differences from the Miles example

The source recipe is
`examples/on_policy_distillation/run-qwen3-8B-opd-multi-teacher.sh` at Miles
commit `862ac1ea1fba864171b006d45e0d5e92ff008c6a`.

- The original recipe requests teacher top-k 16. The comparison uses `top_k=0`
  because sampled-token reverse KL is the only equivalent objective exposed by
  all three frameworks. Prime-RL has no top-k OPD mode, and verl's top-k mode is
  a different forward-KL objective.
- Miles' example leaves the code source as a placeholder. This benchmark uses
  a real 50/50 GSM8K/HumanEval prompt pool and routes each row to one specialist.
- Miles' stock `flash` trainer attention backend hung during the first B200
  backward pass. The measured run uses Miles' `fused` backend. Set
  `TRAIN_ATTENTION_BACKEND=flash` only to reproduce the preserved failure.
- Prime-RL is asynchronous and uses filesystem weight broadcast because the
  pinned repository warns against its NCCL broadcast path. Miles and verl are
  synchronous. Scheduling differences are reported, not disguised.
- verl keeps its native data path, trainer, routing decision, and distillation
  loss, but an opt-in adapter sends teacher scoring to the same two dedicated
  SGLang endpoints used by Miles. This makes the physical role split explicit.

## 3. Capacity and host prerequisites

Use a dedicated Linux host with:

- exactly eight approximately 192 GB NVIDIA GPUs;
- a working NVIDIA driver and Docker GPU runtime;
- at least 180 GB free in an executable filesystem for three pinned HF models,
  Miles' converted student, datasets, and source trees;
- enough separate Docker storage for the two pinned images;
- outbound HTTPS for GitHub, Hugging Face, package indexes, and Prime's first-run
  FlashInfer cubin download.

On the measured host, `/` had only about 193 GB total capacity. `/dev/shm` was a
670 GB executable tmpfs, so all benchmark state lived under
`/dev/shm/opd-bench`. This is fast and avoids Prime's filesystem-broadcast
penalty, but it is volatile: copy evidence off the machine before shutdown.

## 4. Copy and set up a clean node

From the local repository:

```bash
ssh root@31.22.104.163 'mkdir -p /dev/shm/opd-bench/artifacts/mopd-b200-full'
rsync -a benchmarks/mopd-b200-full/ \
  root@31.22.104.163:/dev/shm/opd-bench/artifacts/mopd-b200-full/
```

On the GPU host:

```bash
export BENCH_ROOT=/dev/shm/opd-bench
export ARTIFACT_ROOT=$BENCH_ROOT/artifacts/mopd-b200-full
bash "$ARTIFACT_ROOT/setup_remote_full.sh"
```

The setup script performs and logs all of the following:

1. verifies the GPU count, HBM capacity, and filesystem space;
2. installs `python3.12-dev` and build tools;
3. installs `uv`;
4. checks out Prime-RL, Miles, and verl at their recorded commits;
5. rewrites Prime's SSH submodule fetches to HTTPS for the checkout command;
6. builds Prime's complete environment and installs the local prompt taskset;
7. downloads all three models at immutable Hugging Face revisions in parallel;
8. generates each framework's losslessly equivalent dataset representation;
9. proves semantic token-to-ID equivalence across all three tokenizers;
10. pulls the two digest-pinned containers;
11. pins Accelerate 1.14.0 in the verl container;
12. converts the student checkpoint to Miles' Megatron torch-dist format; and
13. syntax-checks all launchers.

Repository pins:

```text
Prime-RL ec92686fbceb9375d2155cd05c6e87652bf68441
Miles    862ac1ea1fba864171b006d45e0d5e92ff008c6a
verl     2b0fe5158be73b30e749f5c63c1c3b6d5db5d614
```

Model pins:

```text
Qwen/Qwen3-8B                         b968826d9c46dd6066d109eabc6255188de91218
Qwen/Qwen3-32B                        9216db5781bf21249d130ec9da846c4624c16137
Qwen/Qwen3-Coder-30B-A3B-Instruct     b2cff646eb4bb1d68355c01b18ae02e7cf42d120
```

The student and math `tokenizer.json` files are byte-identical; the code file is
not. Do not reject this setup based on file bytes. The validator checks complete
vocab maps, added vocab, special IDs, vocab sizes, and representative encodings;
all token IDs are equivalent, which is the contract required for scoring the
student's sampled IDs under each teacher.

## 5. Smoke test real training

A process starting successfully is insufficient. A smoke test must generate 64
rollouts, contact both teachers, report a non-zero distillation loss and gradient
norm, run backward, and exit after optimizer step 1.

Run all three serially:

```bash
STEPS=1 RUN_ID=smoke BENCH_ROOT="$BENCH_ROOT" ARTIFACT_ROOT="$ARTIFACT_ROOT" \
  bash "$ARTIFACT_ROOT/run_all_full.sh"
```

Or run one framework:

```bash
STEPS=1 RUN_ID=smoke-prime bash "$ARTIFACT_ROOT/run_prime_full.sh"

docker exec -i \
  -e STEPS=1 -e RUN_ID=smoke-miles \
  -e BENCH_ROOT="$BENCH_ROOT" -e ARTIFACT_ROOT="$ARTIFACT_ROOT" \
  opd-miles-full bash "$ARTIFACT_ROOT/run_miles_full.sh"

STEPS=1 RUN_ID=smoke-verl bash "$ARTIFACT_ROOT/run_verl_full.sh"
```

## 6. Run the measured 15-step suite

The three runs are intentionally serial so they never contend for GPUs. A
detached driver survives an SSH disconnect:

```bash
nohup env \
  STEPS=15 RUN_ID=b200-full \
  BENCH_ROOT="$BENCH_ROOT" ARTIFACT_ROOT="$ARTIFACT_ROOT" \
  bash "$ARTIFACT_ROOT/run_all_full.sh" \
  > "$BENCH_ROOT/full-driver.log" 2>&1 < /dev/null &
echo $! > "$BENCH_ROOT/full-driver.pid"
```

Monitor without modifying the run:

```bash
ps -p "$(cat "$BENCH_ROOT/full-driver.pid")" -o pid,etime,stat,args
tail -f "$BENCH_ROOT/full-driver.log"
nvidia-smi dmon -s pucm
```

Framework logs live at:

```text
$BENCH_ROOT/runs-full/prime-b200-full
$BENCH_ROOT/runs-full/miles-b200-full
$BENCH_ROOT/runs-full/verl-b200-full
```

Every launcher records one-second GPU samples, separate teacher-start and
framework timestamps, native framework logs, both endpoint logs, resolved
configs, and metrics. Failed smoke runs use different run IDs and must not be
deleted; they are part of the software-experience record.

## 7. Validate completion

After the driver exits, require all four checks to print 15. Prime writes five
metric records per optimizer step, so count distinct step IDs rather than raw
lines:

```bash
jq -s 'map(.step) | unique | length' \
  "$BENCH_ROOT/runs-full/prime-b200-full/output/metrics.jsonl"

grep -ac '"progress/tokens"' \
  "$BENCH_ROOT/runs-full/prime-b200-full/output/run_default/metrics.jsonl"

grep -ac 'log_utils.py:544 - step ' \
  "$BENCH_ROOT/runs-full/miles-b200-full/run.log"

grep -ac 'step:.*global_seqlen/min:' \
  "$BENCH_ROOT/runs-full/verl-b200-full/run.log"
```

Also require exit metadata (`wall_time.env`), non-zero loss and grad norm on all
steps, requests in both teacher logs, and zero fatal traceback after the final
metric. Prime's console says `Trainable 0/64`: that counter only represents
non-zero RL advantages. Pure OPD intentionally has `advantages=None`; the
trainer still consumes `ref_logprobs`, reports reverse KL, and updates weights.

## 8. Capture environment, then remove weights

Generate the complete software/hardware manifest while containers and source
trees still exist:

```bash
BENCH_ROOT="$BENCH_ROOT" ARTIFACT_ROOT="$ARTIFACT_ROOT" \
  bash "$ARTIFACT_ROOT/collect_environment.sh"
```

Prime's filesystem broadcast writes approximately 16 GB per retained step.
After completion and only after confirming metrics exist, delete the exact
broadcast directory. It is a generated copy of model weights, not benchmark
evidence:

```bash
target="$BENCH_ROOT/runs-full/prime-b200-full/output/run_default/broadcasts"
test -d "$target"
find "$target" -type f -size +100M -print
rm -rf -- "$target"
```

Do not delete `metrics.jsonl`, `final_summary.json`, `configs/`, `logs/`, teacher
logs, `gpu.csv`, or `wall_time.env`.

## 9. Download only compact evidence

Back on the local machine:

```bash
REMOTE=root@31.22.104.163 \
BENCH_ROOT=/dev/shm/opd-bench \
RUN_ID=b200-full \
  bash benchmarks/mopd-b200-full/collect_remote_results.sh
```

The collector allowlists log/config/data formats, rejects checkpoint extensions,
and fails if any downloaded file exceeds 100 MB. It includes every smoke failure
and setup log under `results/all-runs` and `results/setup-logs`.

## 10. Recompute the report and chart

```bash
python3 benchmarks/mopd-b200-full/analyze_full_results.py \
  --output benchmarks/mopd-b200-full/results/summary.json

python3 benchmarks/mopd-b200-full/generate_blog_chart.py
```

The analyzer fails unless each framework has exactly 15 native step records. It
derives route counts from trained batches or observed endpoint requests,
integrates GPU energy from one-second samples, and reports active pipeline time,
end-to-end framework wall time, cold startup, generation, trainer compute,
teacher scoring, weight synchronization, MFU, HBM, and policy staleness where
the framework exposes them.

## 11. What is and is not comparable

This is a systems benchmark, not a model-quality comparison. The prompt pool,
models, token-ID contract, batch shape, sequence limits, optimizer, and loss
direction are matched. Stochastic samples are not identical. Framework-native
runtimes and update schedules remain intact: Prime is asynchronous, Miles has
separate trainer/rollout roles, and verl uses a hybrid actor/rollout pool with
zero-loss DP padding. Compare operational throughput and software behavior;
do not infer that a lower KL estimate or different response length means one
framework trained a better policy.
