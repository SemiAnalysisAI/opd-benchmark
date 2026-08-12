## 2026-08-05 15:01 UTC

**Step**: initialization / 20
**Health**: Healthy

**Progress**: The GRPO and OPD training environments and reverse-text evaluation environment are ready. Policy inference is completing startup.
**Stability**: No training metrics are available before the first batch.
**Performance**: Startup only; no step timing is available yet.

**Notes**: Online W&B run: https://wandb.ai/joey00072/algorithms-debug/runs/af1d4bdd32dd4e87b082747e38422708. The frozen OPD teacher is serving on GPU 1; policy inference and training use GPUs 0 and 2. NCCL P2P and shared-memory transports are explicitly enabled on this no-NVLink host.

## 2026-08-05 15:04 UTC

**Step**: 20 / 20
**Health**: Healthy

**Progress**: Final mixed train reward 0.7763; GRPO reward 0.7639; OPD reward 0.8133. Final eval reward 0.8329. Train errors 0.0%, train truncation 0.8%, eval errors 0.0%, and eval truncation 0.0%.
**Stability**: Final entropy 0.7315, mismatch KL 0.0402, and grad norm 248.9664. Gradient norm peaked at 339.2741 on step 8. No NaNs or fatal training errors occurred.
**Performance**: Final trainer step took 2.1s at 6,539 tokens/s and 9.0% reported MFU. Peak trainer memory was 12.3 GiB. The orchestrator loop completed in 1m 29s.

**Notes**: W&B synced five files, seven media files, and fourteen artifact files. A blog-ready bundle with charts, CSVs, representative traces, configs, environment metadata, reproduction commands, and a SHA-256 manifest is stored under `blog_bundle/`.
