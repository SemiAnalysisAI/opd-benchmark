# On-Policy Distillation Framework Benchmark

## Bottom line

All three open-source stacks completed the same 15-update policy-distillation
job on two H100s. All three transferred the reverse-text policy: their mean
observed rollout score over the last five batches landed in the same narrow
band, 0.762–0.773, while sampled reverse KL fell substantially.

verl was the fastest complete run at 229 seconds and 8.38 trajectories/s.
Prime-RL finished in 276 seconds, but its framework window was slightly faster
than verl at 8.93 trajectories/s after excluding its separately launched
teacher's 61-second boot. Miles finished in 335 seconds. Its Megatron actor
reported high training throughput, but the complete system spent 85.6% of its
steady-state step in the framework's wait phase, including the colocated
rollout/train handoff.

This is policy transfer, not model compression or knowledge-distillation
benchmarking. The student and teacher are both Qwen3-0.6B: the teacher is the
RL-trained checkpoint and the student starts from the SFT checkpoint. The
experiment follows the systems-first shape of SemiAnalysis' [RL systems
benchmark](https://newsletter.semianalysis.com/p/rl-systems-mind-the-gap-matching):
hold the training job constant, then measure the whole system rather than rank
isolated kernels.

## Comparable results

| Framework | End-to-end | Trajectories/s | Generated response tokens/s | First update | Native steady step | GPU energy |
|---|---:|---:|---:|---:|---:|---:|
| Prime-RL | 276 s | 6.96 | 535.7 | 162 s | 6.79 s | 18.7 Wh |
| verl | **229 s** | **8.38** | **553.9** | ~158 s | **5.05 s** | **14.9 Wh** |
| Miles | 335 s | 5.73 | 392.6 | 161 s | 11.49 s | 22.4 Wh |

End-to-end time starts before teacher/framework initialization and ends after
the fifteenth optimizer update. Each system processed 1,920 trajectories. The
generated-token rate uses the observed response lengths, so it remains useful
when policy quality causes one framework to stop earlier than another. GPU
energy is a trapezoidal estimate from one-second `nvidia-smi` samples for both
GPUs; it excludes CPU and host energy.

The native steady-step counters are not perfectly interchangeable. Prime-RL's
counter is the decoupled trainer update, verl reports the synchronous
generation/scoring/update step, and Miles reports its train/wait cycle. The
end-to-end columns are the primary systems comparison.

## Did the policy transfer?

| Framework | Initial score | Last-5 mean | Final batch | Best batch | Final truncation | Final sampled reverse KL |
|---|---:|---:|---:|---:|---:|---:|
| Prime-RL | 0.133 | 0.762 | **0.832** | **0.832** | **0.8%** | 0.128 |
| verl | 0.110 | 0.766 | 0.802 | 0.802 | 7.0% | **0.118** |
| Miles | 0.086 | **0.773** | 0.751 | 0.805 | 10.9% | 0.164 |

Yes, within the limits of this short run. Final-batch ranking is mostly noise:
Miles reached 0.805 on batch 14 and fell to 0.751 on batch 15, while all three
last-five means were effectively tied. The more useful shared pattern is that
long, truncated reasoning collapsed into shorter valid answers as the student
moved toward the RL teacher. Prime-RL's mean response fell from 117.3 to 42.9
tokens; verl fell from 116.5 to 55.1; Miles fell from 120.2 to 57.0.

These are scores on the on-policy rollout batches consumed by training, not a
held-out post-training evaluation. Task score was never included in the
training objective. Prime-RL and verl logged the canonical reverse-text score
during rollout. Miles' [external-teacher OPD
path](https://github.com/radixark/miles/blob/main/docs/advanced/on-policy-distillation.md)
deliberately emits a zero task-reward placeholder, so a read-only hook scored
all 128 completed responses after each generation batch without changing
rewards, advantages, or training.

Reverse-KL values should be read as convergence curves, not an exact
cross-framework scalar ranking. Prime-RL exposes the negative of its
`ref_kl/mean`, verl exposes its k1 distillation loss, and Miles exposes
`opd_reverse_kl`; masking and clamping details differ.

## The job we held constant

- Student: `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT`
- Teacher: `PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL`
- Task source: `PrimeIntellect/Reverse-Text-RL`
- 15 optimizer updates
- 8 prompts per update, 16 student samples per prompt, 128 trajectories per update
- 128-token maximum response
- Temperature 1.0, top-p 1.0
- Sampled-token reverse KL with coefficient 1.0
- No task reward in the objective
- AdamW, learning rate 3e-6, constant schedule, betas 0.9/0.999,
  weight decay 0.01, gradient clipping at 1.0
- Two GPUs total and seed 42 where exposed

The host had two NVIDIA H100 80 GB HBM3 GPUs with NVLink, driver 580.126.09,
CUDA 13.0, 80 AMD EPYC 9654 vCPUs, and 363 GiB RAM. The software revisions and
exact launchers are pinned in the adjacent [README](README.md).

The runtimes' internal architecture was intentionally left native. Prime-RL
separated its trainer from inference and shared the inference GPU between
student rollout and the external teacher. verl colocated actor training and
rollout on one GPU and placed the teacher on the second. Miles colocated its
Megatron trainer and SGLang rollout engine through explicit sleep/wake phases,
with an external SGLang teacher on GPU 1. Forcing one topology on all three
would test a custom harness rather than the frameworks users actually run.

## Where the time went

Time to the first completed update was almost identical—roughly 158–162
seconds—but startup ownership differed:

- Prime-RL spent 61 seconds booting its external teacher, then about 101
  seconds initializing the main runtime and completing the cold update. Once
  warm, trainer updates averaged 6.79 seconds. Excluding teacher startup, its
  215-second framework window delivered 8.93 trajectories/s.
- verl owned teacher initialization inside one 229-second process. Its inferred
  initialization was about 136 seconds, its first step took 22.5 seconds, and
  later steps held close to 5.05 seconds. The actor reported about 9.9% steady
  MFU—still low because a 0.6B model is tiny for an H100 and the OPD phases are
  sequential.
- Miles spent 29 seconds on the external teacher and 6 seconds on Ray before
  the 300-second framework window. Warm steps averaged 11.49 seconds. Its
  actor-training counter was 12.1k tokens/s, but this did not translate into
  system throughput: the native wait ratio averaged 85.6% after the first
  step. Model sleep, rollout, wake, scoring, and weight transfer dominate this
  small colocated job.

That Miles result is the clearest systems lesson here. A fast train kernel can
coexist with the slowest end-to-end loop. Conversely, Prime-RL's separate
teacher startup hurts a 15-step benchmark but would amortize over a longer run;
the measured framework-window lead over verl was only 6.5%, so that is a
hypothesis to test at longer duration, not a declared winner.

End-to-end average GPU utilization was low for every stack because cold start
dominated and teacher scoring arrived in short bursts. Prime-RL averaged
12.5%/6.3% utilization on GPUs 0/1, verl 9.4%/0.8%, and Miles 10.4%/0.2%.
Peak sampled memory was 64.7/13.1 GiB for Prime-RL, 41.5/33.3 GiB for verl,
and 44.4/44.7 GiB for Miles. Prime-RL's high GPU-0 peak reflects colocated
student inference plus teacher service, not actor training alone; its native
trainer reported an 11.8 GiB peak.

## Setup and operational experience

### Prime-RL

[Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl) ran natively in its
uv environment. Three Git submodules used SSH GitHub URLs, so an unauthenticated
machine needed a one-shot HTTPS rewrite and an explicit submodule checkout.
The vLLM runtime shim also needed the Python 3.12 development headers. After
that, the native OPD task and external OpenAI-compatible teacher path worked.

Two metrics need interpretation. `Trainable 0/128` describes the absence of
scalar task advantages in pure OPD; trainer gradients and `ref_kl` confirm that
updates occurred. `mismatch_kl` is rollout/trainer policy staleness, not the
student/teacher distillation KL.

### verl

verl's official vLLM container was the shortest path to a complete run. Its
[OPD configuration](https://verl.readthedocs.io/en/latest/algo/opd.html)
accepted an external teacher and pure distillation objective directly. The
main integration work was supplying the task scorer because rollout still
computes task metrics even when task reward is excluded from OPD.

The sharp configuration edge was batch semantics: with v1 workers,
`ppo_mini_batch_size` counts prompts and is expanded by `rollout.n`. Setting it
to the already-expanded trajectory count pads the synthetic batch and breaks
teacher-logprob alignment. The correct value here was 8, which becomes 128
trajectories after 16 samples per prompt.

After reporting 15/15 updates and 100% progress, Ray killed a DataLoader worker
during teardown and printed a traceback. The launcher still exited zero and
all measured step records were complete. It did not affect the benchmark
window, but it is worth flagging as an operational cleanup issue.

### Miles

Miles' official container brought Megatron and SGLang together, but its actor
needed the supplied Hugging Face-to-TorchDist conversion before training. The
converted student checkpoint occupied about 1.2 GB. In this build, the direct
bridge path dropped the top-level bf16 setting before actor construction and
failed, while the documented converted-checkpoint path completed cleanly.

The external SGLang teacher initially captured dozens of prefill graph sizes.
Disabling teacher CUDA graphs reduced teacher startup to 29 seconds and kept
the teacher mode comparable to the other external service. The task-score hook
added only CPU string comparison after generation and did not enter Miles'
reward path.

## What this benchmark does not establish

- It is one seed and one 15-step run per framework, not a variance study.
- The score is measured on training rollouts. No final checkpoint was saved or
  evaluated on the reserved 128-prompt common evaluation set.
- Prime-RL used its native 1,000-row task source. verl and Miles used the first
  872 rows after reserving 128 for evaluation. The source dataset, prompt
  format, and task were shared, but exact prompt order across frameworks was
  not guaranteed.
- The 0.6B reverse-text task is deliberately cheap and exposes orchestration
  overhead. It does not predict throughput for a 32B model, long contexts,
  multi-node training, or multi-teacher OPD.
- Framework-native token-throughput and MFU counters use different
  denominators. The report does not rank them as if they were interchangeable.
- The one-second GPU telemetry can miss short teacher bursts, and the energy
  estimate excludes CPUs, networking, and storage.
- Runtime versions differed because each project was run in its supported
  environment: Prime-RL used vLLM 0.26, verl vLLM 0.24, and Miles SGLang
  0.5.17dev.

The next serious benchmark should repeat three seeds, save every final policy,
run the same held-out evaluator, extend to enough updates that startup is
amortized, and then add a larger model where GPU saturation matters. This run
is the small preliminary systems baseline before analyzing the main policy-
distillation and multi-teacher designs described in [opd.md](../../opd.md).

## Artifacts

- Machine-readable aggregate: [results/summary.json](results/summary.json)
- Per-step comparison: [results/step_metrics.csv](results/step_metrics.csv)
- Reproducibility guide and revisions: [README.md](README.md)
- Launchers: [run_prime.sh](run_prime.sh), [run_verl.sh](run_verl.sh), and
  [run_miles.sh](run_miles.sh)
- Data adapter and scoring: [prepare_data.py](prepare_data.py),
  [reverse_text_reward.py](reverse_text_reward.py), and
  [miles_task_score.py](miles_task_score.py)
- Raw native logs and one-second GPU telemetry: [results](results)

For context on why OPD uses student-generated trajectories and dense teacher
feedback, see Thinking Machines' [On-policy distillation
post](https://thinkingmachines.ai/blog/on-policy-distillation/). This benchmark
tests the open-source implementations rather than re-explaining the method.
