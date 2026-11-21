#!/bin/bash
# Rerun on follow-up-restricted UKB target cohorts (>=2 y before end of records). Results go to <job>_fu2/.
# usage: bash run_fu2.sh <job> [<job>...]   (repeat until --status shows all units done; each call ~170 s)
export UKF_NAME=UKfemale_for_DA_aligned_fu2.csv UKM_NAME=UKmale_for_DA_aligned_fu2.csv JOB_SUFFIX=_fu2
bash "$(dirname "$0")/go.sh" "$@"
