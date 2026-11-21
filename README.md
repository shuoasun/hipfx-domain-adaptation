# Domain adaptation for cross-cohort hip fracture prediction (SOF/MrOS → UK Biobank)

Code for the manuscript *Domain Adaptation and Predictor Harmonization for Cross-Cohort Generalization of Hip Fracture
Risk Prediction: Evaluation in SOF, MrOS, and the UK Biobank*.

Neural networks are trained on a source cohort (SOF for females, MrOS for males) and transferred to sex-matched UK Biobank
target cohorts. Eight variants are compared: a source-only baseline, MMD, CORAL and gradient reversal (GRL), and their
combinations. Hyperparameters are chosen by random search, and the selected model is evaluated on held-out target test sets
with a participant-level bootstrap.

## Repository layout

```
src/
  da_bmd_sensitivity_std.py  data preparation, models (MLP feature extractor, classifier, domain discriminator),
                             MMD / CORAL / GRL losses, random search space, label-free proxy score
  run_chunk.py               analysis definitions (JOBS) and resumable runs with label-free proxy selection
  run_tval.py                main analysis: model selection on a labelled target validation set (saves every trial)
  run_ft.py                  target fine-tuning of the selected models (supplementary analysis)
  collect_tval.py            one CSV per selection rule (proxy / target validation / test upper bound)
  ci_participant.py          participant-level bootstrap confidence intervals for AUC and AUC differences
  make_fu2.py                UKB target cohorts restricted to >=2 years of potential follow-up
  make_subsets.py            UKB females aged >=70 years
scripts/                     shell launchers (4 parallel workers, resumable; run until "ALL DONE")
data/README.md               expected input files and columns (no data included)
```

## Requirements

Python 3.10+, `pip install -r requirements.txt`. All runs use the CPU (`torch.set_num_threads(1)` per worker).

## Reproducing the analyses

```bash
export DATA_DIR=$PWD/data RESULTS_DIR=$PWD/results
python src/make_fu2.py                       # UKB cohorts with >=2 years of potential follow-up
python src/make_subsets.py                   # females aged >=70 (supplementary)

# main analysis: target-validation selection, 4 settings (females/males x original/standardized BMD)
bash scripts/run_tval_long.sh
# confidence intervals, e.g. males with original-scale BMD
python src/collect_tval.py --job male_primary --out_dir results/ci
python src/ci_participant.py --res results/ci/male_primary_fu2_tval_tval.csv \
       --uk data/UKmale_for_DA_aligned_fu2.csv --out results/ci/male_primary --B 500

# supplementary analyses
bash scripts/run_female_explore.sh           # females: without grip strength, core predictors, aged >=70
bash scripts/run_tval_hip_female.sh          # females: femur BMD subset (needs UK*_fu2_hipBMD.csv, see make_subsets.py)
bash scripts/run_tval30_female.sh            # females: 30 trials per variant
bash scripts/run_ft_long.sh                  # target fine-tuning (after run_tval_long.sh)
bash scripts/run_fu2.sh <job> ...            # label-free proxy selection (one ~3-minute chunk per call)
```

Analysis names (`JOBS` in `src/run_chunk.py`): `female_primary`, `male_primary` (original-scale BMD),
`female_C1_std_all`, `male_C1_std_all` (BMD standardized within each cohort), `female_C1_noGrip`, `female_C1_core`, and
the BMD sensitivity analyses A–C. Each analysis has 6 repetitions × 3 seeds × 8 variants = 144 units; runs can be
interrupted and resumed.

## Design notes

- Test sets: a random 20% of the target cohort per repetition (seed `12000 + rep`), identical for all models.
- Target validation set: a stratified 25% of the remaining target data (seed `55000 + rep`), disjoint from the test set;
  its labels are used only to choose among trials. Test labels are never used for training or selection.
- Unlabelled target features for the alignment losses come from the whole target training pool.
- The "oracle" rule in `collect_tval.py` (selection by test AUC) is an upper bound for diagnostics only.

## Citation

Please cite the manuscript when using this code.

## License

MIT (see `LICENSE`).
