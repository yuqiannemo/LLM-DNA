# RBF-MMD Pilot

## Setup
- Data dir: `out/rand_chinese`
- Reference setting: `0.2:0.8`
- Comparison settings: `0.2:0.8`, `0.3:0.8`, `0.3:0.9`
- Selected models: `8`
- Selected prompts: `20`
- Calibration prompts: `4`
- Evaluation prompts: `16`
- Embedding features: `4096`

## Notes
- `single_cosine` uses one response sample per prompt from the selected run split.
- `mean_cosine` averages repeated samples per prompt before concatenation.
- `exact_rbf_mmd` computes prompt-wise squared RBF-MMD and averages across prompts.
- `same_*` compares two repeat splits from the same decoding setting.
- `cross_*` compares the reference setting against another setting.
- NA in intermediate figures means the model or prompt was not present in the required split, or the run lacked enough repeats.

## Results
- `same_t02_p08` / `single_cosine`: top1=1.000, top3=1.000, top5=1.000, mrr=1.000
- `same_t02_p08` / `mean_cosine`: top1=1.000, top3=1.000, top5=1.000, mrr=1.000
- `same_t02_p08` / `exact_rbf_mmd`: top1=1.000, top3=1.000, top5=1.000, mrr=1.000
- `cross_t02_p08_to_t03_p08` / `single_cosine`: top1=1.000, top3=1.000, top5=1.000, mrr=1.000
- `cross_t02_p08_to_t03_p08` / `mean_cosine`: top1=1.000, top3=1.000, top5=1.000, mrr=1.000
- `cross_t02_p08_to_t03_p08` / `exact_rbf_mmd`: top1=1.000, top3=1.000, top5=1.000, mrr=1.000
- `cross_t02_p08_to_t03_p09` / `single_cosine`: top1=1.000, top3=1.000, top5=1.000, mrr=1.000
- `cross_t02_p08_to_t03_p09` / `mean_cosine`: top1=1.000, top3=1.000, top5=1.000, mrr=1.000
- `cross_t02_p08_to_t03_p09` / `exact_rbf_mmd`: top1=1.000, top3=1.000, top5=1.000, mrr=1.000

## Current Result

The first small pilot run on the cached `out/rand_chinese` subset selected 8 models and 20 prompts. On that tiny subset, all three methods reached perfect top1/top3/top5/MRR, so the current signal is mainly in the distance magnitudes rather than ranking changes. The exact RBF-MMD self-distance was lower than single-sample cosine on the same split, which is consistent with the distributional formulation but is not yet a decisive retrieval gain.
