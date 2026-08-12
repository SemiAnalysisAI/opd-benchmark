# Full-size MOPD software experience

This is the chronological bring-up record for Prime-RL, Miles, and verl on the
same 8×B200 MOPD recipe. Failed attempts are benchmark evidence: they show what
a user actually encounters before the first valid optimizer step. The compact
result bundle retains their logs under `results/all-runs/` and setup logs under
`results/setup-logs/`.

## Host discovery

The node was sold or described as “8×B300.” It was not. `nvidia-smi` reported
eight NVIDIA B200 GPUs with 183,359 MiB each, driver 580.126.09, and NV18 links
between every GPU pair. The system exposed 240 CPUs and roughly 1.3 TiB RAM.

The root filesystem was about 193 GB—too small for the 8B student, 32B math
teacher, 30B-A3B code teacher, converted Miles checkpoint, containers, and
temporary Prime broadcasts. `/dev/shm` was a 670 GB executable tmpfs, so the
benchmark moved to `/dev/shm/opd-bench`. This made Prime's filesystem weight
broadcast fast but made evidence volatile.

The first post-run environment collector also exited at its mount check because
it called `findmnt` as though `$BENCH_ROOT` were itself a mount point. It is a
directory inside `/dev/shm`, so `findmnt -T` is required. The failed trace is
retained as `environment-collection-retry.log`; the corrected collector records
the containing tmpfs mount.

## Setup failures before any framework ran

### Prime submodules assumed GitHub SSH access

`git submodule update --init --recursive` failed with host-key verification on
SSH-form GitHub URLs. The benchmark node had outbound HTTPS but no configured
GitHub SSH trust or key. A command-scoped SSH-to-HTTPS URL rewrite fixed the
checkout without weakening SSH verification globally.

An interrupted retry then left `deps/pydantic-config` empty and staged as
deleted. `uv sync` reported that it was not a Python project, which looked like
a dependency problem but was checkout damage. Restoring the pinned worktree and
rerunning the recursive submodule update fixed it. The replication script does
that restore before submodule setup.

Preserved setup traces:

```text
prime-setup.log
prime-setup-retry.log
prime-setup-retry2.log
prime-setup-retry3.log
```

### Model tokenizer files were not byte-identical

The student and math teacher had the same `tokenizer.json` hash; the code
teacher's bytes differed. Rejecting it on hash alone would have been a false
negative. Transformers reported the same tokenizer length (151,669), base vocab
size (151,643), complete token-to-ID map, added vocab, EOS/PAD IDs, and sample
encodings for all three models. MOPD scores student-sampled token IDs, so semantic
ID equivalence—not JSON serialization identity—is the relevant contract.

## Miles

### Attempt 1: the submission client disconnected while the job kept running

The first launcher piped `ray job submit` through logging. Its WebSocket closed
while Ray still marked the job `RUNNING`; shell control continued as though the
command were complete, and cleanup killed a healthy job. This is a dangerous UX
failure because a launcher's exit did not represent the training process.

Evidence: `miles-smoke1/`.

### Attempt 2: explicit polling hit a broken dashboard address

The next launcher polled Ray job status. Ray's dashboard agent advertised the
node's public-IP ephemeral endpoint, then status/log requests returned HTTP 500
and `ConnectionRefusedError`. The job remained stale as `RUNNING` and still did
not provide a reliable blocking lifecycle.

Evidence: `miles-smoke2/`.

### Resolution: keep Ray internals, bypass only dashboard job submission

The working launcher still starts the Ray head and uses Miles' Ray actors, but
invokes `train.py` directly as the blocking foreground process. Exit status,
signals, cleanup, and `tee` now correspond to the real training job.

### Attempt 3: stock `flash` attention hung on B200 backward

With Miles' example-default `--attention-backend flash`, the system generated
and teacher-scored all 64 trajectories, then stalled in actor training for more
than twenty minutes. Trainer GPUs showed 100% SM activity, near-zero memory
throughput, low power relative to normal training, and one spinning CPU thread
per rank. There was no Python exception. Restarting the dedicated container was
required to terminate the wedged CUDA work.

Evidence: `miles-smoke3/`.

The same recipe with Miles' `fused` backend completed. The valid smoke reported
reverse-KL loss 0.5847, grad norm 5.88, 47.4 s rollout, 9.6 s actor train, and a
real optimizer update. The full launcher therefore defaults to `fused` while
retaining `TRAIN_ATTENTION_BACKEND=flash` as a failure-reproduction switch.

Evidence: `miles-smoke4-fused/`.

### Warning quality

Miles emitted a high volume of non-fatal warnings: a ModelOpt/Transformers
version warning, torchao notices, a Qwen3-ASR `cache_position` message labeled
`ERROR` even though no ASR model was used, legacy-tokenizer notices, and a TP
configuration warning. None invalidated the successful step. The false-severity
ASR line is especially costly during incident triage.

## Prime-RL

### Attempt 1: first B200 kernel compilation needed Python headers

Prime passed environment setup but failed before model execution when Triton
compiled a helper and could not find `/usr/include/python3.12/Python.h`.
Installing `python3.12-dev` resolved it. The Python package lock alone was not a
complete build environment.

Evidence: `prime-smoke1/`.

### First-run network dependency and cold start

Prime downloaded B200 FlashInfer cubins from NVIDIA at runtime. A supposedly
preinstalled Python environment therefore still needed outbound network during
first model startup. This belongs in capacity and air-gapped deployment docs,
and setup time must be separated from step time.

### Misleading `Trainable 0/64` on successful OPD

Both successful Prime smokes printed `Trainable 0/64 (0.0%)`. That counter is
tied to non-zero RL advantages. Pure OPD deliberately has zero task reward and
`advantages=None`, but its rows contain teacher `ref_logprobs` and are consumed
by the OPD trainer. The proof is in the real trainer metrics: non-zero reverse
KL, loss, gradient norm, forward/backward time, weight broadcast, and changed
step. This is an observability bug, not a failed training run.

Evidence: `prime-smoke2/` and `prime-smoke3-no-zero-filter/`.

The benchmark explicitly configures all pre/post filters in monitor-only mode
to make pure-OPD intent visible. The default zero-advantage filter did not in
fact drop these rows because their advantage is absent, but relying on that
implementation detail is confusing.

### Filesystem broadcast creates large disposable artifacts

Prime's pinned revision warns against its NCCL weight-broadcast path, so the run
uses filesystem broadcast. Each retained 8B BF16 broadcast is roughly 16 GB.
The two successful smoke runs each left one copy. Those exact `broadcasts/`
directories were removed after logs and metrics were verified; no model weight
entered the downloaded evidence. The full guide repeats this cleanup after the
15-step run.

## verl

### Attempt 1: external teachers still had to satisfy config accounting

The adapter manages two teacher endpoints outside verl's Ray pool, but the
distillation dataclass still validates logical replicas. Setting
`distillation.n_gpus_per_node=8` produced:

```text
ValueError: Sum of teacher replicas (2) must match distillation resource pool (8)
```

Setting the logical distillation pool to two satisfied metadata validation;
the trainer process still sees only physical GPUs 0–5, while the external
servers own 6–7.

Evidence: `verl-smoke1/`.

### Attempt 2: unpinned Accelerate was incompatible with Transformers 5

The digest-pinned verl image contained PyTorch 2.11.0+cu130, Transformers 5.5.3,
Accelerate 1.12.0, and vLLM 0.24.0. Actor construction failed with:

```text
TypeError: Parameter.__new__() got unexpected keyword '_is_hf_initialized'
```

The verl project allowed Accelerate to float. Pinning Accelerate 1.14.0 fixed
the handoff because that release removes and restores the Transformers internal
field instead of passing it into `torch.nn.Parameter`.

Evidence: `verl-smoke2/` and `verl-accelerate-upgrade.log`.

### Attempt 3: 64 is not divisible by six data-parallel ranks

verl reached generation, both teachers, and actor update, then asserted:

```text
AssertionError: Got mini_batch_size=64 and self.engine.get_data_parallel_size()=6
```

The six-GPU hybrid actor/rollout layout is the natural counterpart to Miles'
two trainer plus four rollout GPUs, but its FSDP data parallelism imposes a
batch-divisibility constraint. Setting `ppo_mini_batch_size=6` lets verl's
native balancer append eight no-op rows to the 64 real trajectories.

Evidence: `verl-smoke3-accelerate114/`.

### Attempt 4: the stock padding helper did not know about MOPD fields

verl generated 64 real trajectories and correctly reported padding from 64 to
72. Its native padding helper zeroed PPO fields but cloned the real sample's
ragged `teacher_logprobs` and `teacher_ids`. Distillation then failed in
`no_padding_2_padding` because the log-prob value length no longer matched the
synthetic two-token sequence.

Evidence: `verl-smoke4-dp-padding/`.

The included 11-line patch zeros both teacher fields with the correct sequence
length and preserves dtype/device and any top-k trailing dimension. The next
smoke completed generation, routing, reverse KL, backward, and optimizer update:
loss 0.538, grad norm 5.82, 64.6 s generation, 6.0 s actor update, and 91.7 s
total step time. Each specialist served 32 scored rows plus one startup request.

Evidence: `verl-smoke5-padding-fields/`.

### Warning quality and startup

verl's first step starts six vLLM HTTP servers, loads six FSDP shards, captures
CUDA graphs, and runs FlashInfer autotuning. Logs include repeated warnings from
every Ray worker, deprecation messages, and “blocking ray.get inside async
actor.” The run is healthy, but useful lifecycle events are buried. The separate
teacher-start, framework-wall, and native step timers prevent this cold start
from contaminating steady-state throughput.

### Successful full run still printed a shutdown traceback

After the progress bar reached 15/15, verl printed a traceback from Python's
weak-reference cleanup because Ray killed a DataLoader worker during teardown:

```text
RuntimeError: DataLoader worker (...) is killed by signal: Killed.
```

This occurred between the completed progress bar and the final step-15 metric
line. The step-15 reverse-KL loss and gradient norm, final validation message,
wall-time metadata, and successful suite exit were all present afterward. It
did not invalidate the run, but presenting a teardown traceback on a successful
job makes automation and human incident triage unnecessarily ambiguous.

## Comparative takeaways

### Local chart export needed a native renderer fallback

The first PNG export installed CairoSVG in an isolated Python environment but
failed because the macOS host did not expose a loadable native Cairo library.
The SVG had already been written correctly. The chart script now catches both
missing Python packages and missing native Cairo, then uses macOS `sips` when it
is available; systems without either renderer still receive the complete SVG.

- A pinned Git commit or container digest is necessary but not sufficient.
  System headers, runtime-downloaded kernels, and unconstrained transitive
  packages materially changed whether a run worked.
- A process reaching “startup complete” is weak evidence. Every framework had
  at least one failure after substantial initialization; only a non-zero loss,
  backward pass, and optimizer step count as a smoke-test pass.
- Framework console counters are not objective-neutral. Prime's “trainable”
  means reward-advantage trainable, not OPD trainable.
- Equal model and prompt settings do not guarantee equal systems semantics.
  Prime is asynchronous, Miles separates trainer and generator roles, and verl
  uses hybrid workers plus zero-loss DP padding. These differences must remain
  visible in the report.
- Prime's teacher logs contained 1,036 successful scoring calls for 960 consumed
  trajectories. The 76-call excess is consistent with queued or in-flight
  asynchronous producer work at shutdown and remains a real systems cost.
- Preserve raw failures. Without the four verl attempts and three Miles launch
  attempts, a clean final command would badly understate the integration work.
