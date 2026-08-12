# OPD framework benchmark

This benchmark compares the native on-policy distillation paths in Prime-RL,
verl, and Miles on one two-GPU host.

## Common recipe

- Student: `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT`
- Teacher: `PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL`
- Task: `PrimeIntellect/Reverse-Text-RL`
- Training steps: 15
- Prompts per step: 8
- Student rollouts per prompt: 16
- Total trajectories per step: 128
- Maximum response length: 128 tokens
- Sampling: temperature 1.0, top-p 1.0
- Learning rate: 3e-6, constant schedule
- Optimizer: AdamW, betas (0.9, 0.999), weight decay 0.01, gradient clip 1.0
- OPD signal: sampled-token reverse KL (`k1`), coefficient 1.0
- Task reward in training loss: disabled
- Hardware: two GPUs total; placement follows each native runtime (Prime-RL
  separates training from inference, while verl and Miles colocate student
  rollout and training)
- Seed: 42 where the framework exposes it

The teacher and student use the same architecture and parameter count. The
experiment measures transfer of an RL-trained policy into an SFT checkpoint,
not compression from a larger model.

## Pinned environment

- Host: Ubuntu 24.04.4, 80 vCPU AMD EPYC 9654, 363 GiB RAM
- GPUs: 2x NVIDIA H100 80 GB HBM3 with NVLink
- Driver/CUDA: 580.126.09 / 13.0
- Prime-RL: `ec92686fbceb9375d2155cd05c6e87652bf68441`
  (native uv environment, Prime-RL 0.7.0, PyTorch 2.11/cu128, vLLM 0.26)
- verl: `2b0fe5158be73b30e749f5c63c1c3b6d5db5d614`
  (`verlai/verl:vllm024.dev2`, PyTorch 2.11/cu130, vLLM 0.24)
- Miles: `862ac1ea1fba864171b006d45e0d5e92ff008c6a`
  (`radixark/miles:latest`, PyTorch 2.11/cu130, SGLang 0.5.17dev)

## Measurements

Each launcher records:

- end-to-end startup and run time;
- time to first optimizer step;
- steady-state step time;
- trajectories and generated tokens per second;
- sampled-token reverse-KL trajectory;
- per-GPU utilization, memory, and power at one-second resolution;
- setup failures, workarounds, and framework-native defaults that could not be
  made identical.

Miles' native external-teacher OPD path uses a zero task-reward placeholder.
`miles_task_score.py` observes each completed rollout and logs the same
reverse-text score used by the other two runs without modifying Miles'
training rewards or advantages.

Run `prepare_data.py` once, then execute the three launchers independently. The
launchers expect the remote workspace at `/root/opd-bench`; override
`BENCH_ROOT` if needed.

After copying each run directory into `results/<framework>`, run
`python3 analyze_results.py` to regenerate `results/summary.json` and
`results/step_metrics.csv`.
