# Primary robustness recovery on the 3090 host

The committed recovery plan contains the 72 unsuccessful model/cell/round
slots from the grid snapshot completed on 5 August 2026. It preserves each
cell's original temperature, top-p, sampling mode, output suffix, and generation
seed. Successful slots are not included.

After pulling the `dna/experiment-script` branch on the 3090 host, activate the
same Python environment used for LLM-DNA and run a command preview:

```bash
DRY_RUN=1 bash scripts/recover_primary_robustness_failures_3090.sh
```

Launch the recovery on GPU 0:

```bash
mkdir -p logs/primary_robustness_recovery_20260810
nohup env GPUS=0 MAX_CONCURRENT_GPUS=1 \
  bash scripts/recover_primary_robustness_failures_3090.sh \
  > logs/primary_robustness_recovery_20260810/nohup.log 2>&1 &
```

Set `HF_TOKEN` in the environment first if private or gated Hugging Face models
require it. To use multiple GPUs, set both `GPUS` and `MAX_CONCURRENT_GPUS`, for
example `GPUS=0,1 MAX_CONCURRENT_GPUS=2`.

The launcher is restart-safe. On a later invocation it continues missing work,
does not rerun recorded successes, and retries recorded failures. Outputs are
kept separate from the original grid under:

```text
out/primary_robustness_recovery_20260810/
```

Copy that complete directory back into the main repository's `out/` directory
after the run. The grid audit treats a successful recovery record as satisfying
the corresponding historical failure.
