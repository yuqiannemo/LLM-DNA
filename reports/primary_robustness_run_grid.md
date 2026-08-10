# Primary robustness experiment — run grid

Updated: 2026-08-05T18:55:39+08:00

Revised plan:

- Fixed cohort: **the latest validated100 model list** from `top_p_control_t05_t05_p06_r2/manifest.json`.
- Historical artifacts are reused only when their model ID and decoding cell match this cohort/grid.
- Two repeats per condition. The shared `temperature=0.5` row uses **seed 42 for both rounds** so its `top_p=0.8/0.9/1.0` cells can also serve the validated top-p control. Other proposed grid backfills remain **r1 / 42001** and **r2 / 42002**.
- The historical sweep did not explicitly set or record a response-generation seed. Its recorded `random_seed=42` controlled probe selection and DNA reduction, not the generation sampler.
- Stochastic grid: temperature `{0.2, 0.3, 0.5, 0.7}` × top-p `{0.8, 0.9, 1.0}`.
- Deterministic control: `temperature=0`, `top_p=1.0` (top-p inactive).
- Total: **13 settings × 2 repeats × 100 models = 2,600 successful model artifacts required**.

Important cohort note: the two validated100 variants overlap by 97 models, but the older temperature/top-p sweep used here overlaps the latest validated100 cohort by **87 models**. Therefore those grid cells require 13 cohort backfills plus retries for any failed/pending overlapping models.

Legend: `✅` both repeats have 100 successful models; `♻️` historical results are reusable but the cell is incomplete; `⬜` no matching run was found; `—` is not a planned condition.

| Temperature \\ Top-p | 0.8 | 0.9 | 1.0 |
|---:|---:|---:|---:|
| 0 | — | — | ✅ 200/200 |
| 0.2 | ♻️ 199/200；还需 1 | ✅ 200/200 | ♻️ 198/200；还需 2 |
| 0.3 | ♻️ 199/200；还需 1 | ♻️ 189/200；还需 11 | ♻️ 185/200；还需 15 |
| 0.5 | ♻️ 198/200；还需 2 | ♻️ 196/200；还需 4 | ♻️ 176/200；还需 24 |
| 0.7 | ♻️ 196/200；还需 4 | ♻️ 197/200；还需 3 | ♻️ 195/200；还需 5 |

## Overall progress

- Successful artifacts: **2528/2600**.
- Settings started: **13/13**.
- Settings fully complete: **2/13**.
- Remaining successful artifacts required: **72**.

## Repeat-level detail

| Temperature | Top-p | Round | Historical generation seed | Proposed backfill seed (not run) | Reusable success | Failed | Pending/no status | Still needed | State |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 0 | 1 | r1 | not explicitly set or recorded | 42001 | 100 | 0 | 0 | 0 | ✅ complete |
| 0 | 1 | r2 | 42002 | 42002 | 100 | 0 | 0 | 0 | ✅ complete |
| 0.2 | 0.8 | r1 | not explicitly set or recorded | 42001 | 100 | 0 | 0 | 0 | ✅ complete |
| 0.2 | 0.8 | r2 | not explicitly set or recorded | 42002 | 99 | 1 | 0 | 1 | ♻️ reusable/incomplete |
| 0.2 | 0.9 | r1 | not explicitly set or recorded | 42001 | 100 | 0 | 0 | 0 | ✅ complete |
| 0.2 | 0.9 | r2 | not explicitly set or recorded | 42002 | 100 | 0 | 0 | 0 | ✅ complete |
| 0.2 | 1 | r1 | not explicitly set or recorded | 42001 | 99 | 1 | 0 | 1 | ♻️ reusable/incomplete |
| 0.2 | 1 | r2 | not explicitly set or recorded | 42002 | 99 | 1 | 0 | 1 | ♻️ reusable/incomplete |
| 0.3 | 0.8 | r1 | not explicitly set or recorded | 42001 | 100 | 0 | 0 | 0 | ✅ complete |
| 0.3 | 0.8 | r2 | not explicitly set or recorded | 42002 | 99 | 1 | 0 | 1 | ♻️ reusable/incomplete |
| 0.3 | 0.9 | r1 | not explicitly set or recorded | 42001 | 96 | 4 | 0 | 4 | ♻️ reusable/incomplete |
| 0.3 | 0.9 | r2 | not explicitly set or recorded | 42002 | 93 | 7 | 0 | 7 | ♻️ reusable/incomplete |
| 0.3 | 1 | r1 | not explicitly set or recorded | 42001 | 94 | 6 | 0 | 6 | ♻️ reusable/incomplete |
| 0.3 | 1 | r2 | not explicitly set or recorded | 42002 | 91 | 9 | 0 | 9 | ♻️ reusable/incomplete |
| 0.5 | 0.8 | r1 | not explicitly set or recorded | 42 | 100 | 0 | 0 | 0 | ✅ complete |
| 0.5 | 0.8 | r2 | 42 | 42 | 98 | 2 | 0 | 2 | ♻️ reusable/incomplete |
| 0.5 | 0.9 | r1 | not explicitly set or recorded | 42 | 98 | 2 | 0 | 2 | ♻️ reusable/incomplete |
| 0.5 | 0.9 | r2 | not explicitly set or recorded | 42 | 98 | 2 | 0 | 2 | ♻️ reusable/incomplete |
| 0.5 | 1 | r1 | not explicitly set or recorded | 42 | 98 | 2 | 0 | 2 | ♻️ reusable/incomplete |
| 0.5 | 1 | r2 | 42 | 42 | 78 | 22 | 0 | 22 | ♻️ reusable/incomplete |
| 0.7 | 0.8 | r1 | not explicitly set or recorded | 42001 | 97 | 3 | 0 | 3 | ♻️ reusable/incomplete |
| 0.7 | 0.8 | r2 | 42002 | 42002 | 99 | 1 | 0 | 1 | ♻️ reusable/incomplete |
| 0.7 | 0.9 | r1 | not explicitly set or recorded | 42001 | 99 | 1 | 0 | 1 | ♻️ reusable/incomplete |
| 0.7 | 0.9 | r2 | 42002 | 42002 | 98 | 2 | 0 | 2 | ♻️ reusable/incomplete |
| 0.7 | 1 | r1 | not explicitly set or recorded | 42001 | 96 | 4 | 0 | 4 | ♻️ reusable/incomplete |
| 0.7 | 1 | r2 | 42002 | 42002 | 99 | 1 | 0 | 1 | ♻️ reusable/incomplete |

Ignored 43 manifest(s) outside the revised primary grid; see the CSV/source tree for audit.
