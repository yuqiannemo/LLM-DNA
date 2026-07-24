# Revised LLM-DNA robustness experiment

## Primary claim and task

The primary task is closed-set **exact-model identification**. Every compared
method ranks the same model labels for the same query DNA and gallery. Pairwise
same-or-related verification is retained only as a secondary diagnostic; it is
not compared directly with 300-way Top-1 accuracy.

The main method matrix is:

| Representation | Classifier | Role |
|---|---|---|
| Original 128-D DNA | cosine centroid | retrieval baseline |
| Original 128-D DNA | multiclass linear SVM | classifier-only ablation |
| Original 128-D DNA | multiclass RBF SVM | nonlinear classifier ablation |
| Multi-sample response distribution | mean cosine | budget-matched distributional baseline |
| Multi-sample response distribution | exact RBF-MMD | proposed representation |

## Data protocol

- Use the predeclared, performance-independent 300-model cohort in
  `configs/rand_chinese_stratified_300.jsonl`. Its audit JSON records the
  selection rule, eligible pool, seed, metadata, and SHA-256 hash.
- Generate one deterministic `T=0` cell and the complete factorial grid
  `T in {0.2, 0.3, 0.5, 0.7}` by `top_p in {0.8, 0.9, 1.0}`.
- Generate four independent response repeats. Repeats 1-2 form training and
  galleries; repeats 3-4 are held-out queries.
- The generation seed changes by repeat (`42001` through `42004`), while the
  prompt/projection seed remains 42. Both are stored in artifacts.
- A result is blocked when fewer than 250 common models remain. Failed models
  should be backfilled rather than silently shrinking the final cohort.
- All methods use the same prompts, model cohort, train/test repeats, and query
  budget. Distributional results must include the matched-budget mean-centroid
  baseline, not only the single-sample baseline.

Launch generation with:

```bash
python3 scripts/build_experiment_cohort.py --target 300
nohup bash scripts/run_primary_robustness_experiment.sh \
  > logs/primary_robustness_master.log 2>&1 &
```

The launcher dynamically uses all visible GPUs up to ten workers. Override
`GPUS` to pin cards or `MAX_CONCURRENT_GPUS` to change the cap.

After all four repeats are complete, run:

```bash
bash scripts/run_primary_analysis.sh
```

## Required analyses

1. Per-setting exact Top-1, Top-3, Top-5, MRR, and model-bootstrap 95% CIs.
2. Temperature-agnostic deployment at each bound
   `tau in {0.2, 0.3, 0.5, 0.7}`. The classifier sees pooled eligible training
   settings but not temperature/top-p. Report equal-cell mixed accuracy and
   worst-setting accuracy.
3. Paired method comparisons on identical held-out model queries. The key
   comparison is the degradation slope/retained accuracy, not unrelated task
   accuracies.
4. Tree-aware error severity as a secondary analysis: exact, direct one-hop,
   two-hop, same-component, and disconnected predictions. Export direct HF
   edges with `export_hf_model_tree.py`; connected-component closure alone is
   not treated as a direct relationship.
5. Legacy pairwise verification uses every test pair at natural prevalence.
   Training may downsample negatives. Report PR-AUC, MCC, specificity, and
   false positives; separate same-model and lineage-only labels before making
   lineage claims.

## Interpretation guardrails

- The unsuffixed historical gallery has unknown decoding provenance and is a
  legacy sensitivity analysis only. The primary experiment uses explicit,
  independently generated repeat galleries.
- A scalar linear SVM on cosine distance cannot change closed-set ranking; the
  primary SVM therefore consumes the full 128-D DNA in a multiclass task.
- `T=0` appears once because top-p is inactive without sampling.
- Do not claim a top-p effect, RBF advantage, or lineage robustness unless the
  paired confidence interval supports it.
- Report failure rates/reasons by cell and compare excluded versus retained
  model size and architecture distributions.

## Final outputs

- Figure 1: exact accuracy versus temperature, faceted by top-p, with one curve
  per representation/classifier and 95% CIs.
- Figure 2: accuracy loss relative to `T=0` plus the method-by-temperature
  interaction/degradation slope.
- Figure 3: mixed and worst-setting accuracy versus the unknown-temperature
  upper bound `tau`.
- Figure 4: stacked tree-aware error severity versus temperature.
- Tables: protocol/cohort, complete per-setting results, bounded-deployment
  results, and matched-budget method/efficiency ablations.
