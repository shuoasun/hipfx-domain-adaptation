#!/bin/bash
# Exploratory female analyses, all with target-validation selection (run_tval.py). Run after run_tval_hip_female.sh finishes.
#  1) feature sets on the >=2y cohort: no grip strength; core set (age, BMI, 3 BMD sites, prior fractures), BMD standardized
#  2) UKB females aged >=70 (closer to the SOF age distribution), original-scale and standardized BMD
cd "$(dirname "$0")/../src"; mkdir -p ../results
UKF_NAME=UKfemale_for_DA_aligned_fu2.csv JOB_SUFFIX=_fu2 bash -c 'sed -e "s/^export .*/export UKF_NAME=$UKF_NAME UKM_NAME=UKmale_for_DA_aligned_fu2.csv JOB_SUFFIX=_fu2 BUDGET=\${BUDGET:-1500}/" -e s/long_tval.log/long_explore1.log/g -e s/log_tval_w/log_ex1_w/g ../scripts/run_tval_long.sh > ../scripts/_ex1.sh'
sed -e "s/^export .*/export UKF_NAME=UKfemale_for_DA_aligned_fu2_age70.csv UKM_NAME=UKmale_for_DA_aligned_fu2.csv JOB_SUFFIX=_fu2age70 BUDGET=\${BUDGET:-1500}/" -e s/long_tval.log/long_explore2.log/g -e s/log_tval_w/log_ex2_w/g ../scripts/run_tval_long.sh > ../scripts/_ex2.sh
bash ../scripts/_ex1.sh female_C1_noGrip female_C1_core
bash ../scripts/_ex2.sh female_primary female_C1_std_all
echo "EXPLORE ALL DONE $(date)"
