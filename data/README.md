# Data

No participant-level data are included in this repository. The cohorts are available to approved researchers:

- **SOF** (phs000510.v1.p1) and **MrOS** (phs000373.v1.p1): dbGaP, https://www.ncbi.nlm.nih.gov/gap/
- **UK Biobank**: https://www.ukbiobank.ac.uk/ (this study: application 61915)

Place the harmonized analysis files here (or set `DATA_DIR`):

| File | Cohort |
|---|---|
| `SOF_for_DA_enriched_smokefix.csv` | SOF (female source) |
| `MrOS_for_DA_rawBMD.csv` | MrOS (male source) |
| `UKfemale_for_DA_aligned.csv`, `UKmale_for_DA_aligned.csv` | UKB targets, full cohorts |
| `UK_female_incident10y_HAready_FINAL.csv`, `UK_male_incident10y_HAready_FINAL.csv` | UKB rows with `index_date` (Instance-2 visit), row-aligned with the files above; used only by `make_fu2.py` |

Each analysis file has one row per participant and these columns:

`HA_HIPFX` (incident hip fracture, 0/1), `HA_AGE`, `HA_BMI`, `HA_HEIGHT`, `HA_WEIGHT`, `HA_GRIPAVG`,
`HA_THD` (total hip BMD; whole-body BMD, field 23236, in UKB), `HA_LSD` (lumbar spine BMD), `HA_FND` (femoral neck BMD),
`HA_WRSTFX`, `HA_SLDFX` (prior wrist / spine fracture, 0/1), `SMOKE_ORD` (0 never, 1 former, 2 current), `WALK_ORD` (walking pace, ordinal).
