#!/bin/bash
# Long continuous run of the target-validation analysis. Run once in Terminal:  bash run_tval_long.sh
cd "$(dirname "$0")/../src"; mkdir -p ../results
export UKF_NAME=UKfemale_for_DA_aligned_fu2.csv UKM_NAME=UKmale_for_DA_aligned_fu2.csv JOB_SUFFIX=_fu2 BUDGET=${BUDGET:-1500}
JOBS=${@:-female_primary male_primary female_C1_std_all male_C1_std_all}
caffeinate -i -w $$ 2>/dev/null &
while true; do
  pids=""; for w in 0 1 2 3; do python3 run_tval.py --worker $w --nworkers 4 --budget $BUDGET --jobs $JOBS >> ../results/log_tval_w$w.txt 2>&1 & pids="$pids $!"; done; wait $pids
  st=$(python3 run_tval.py --status --jobs $JOBS); echo "$(date '+%F %T') $st" | tr '\n' ' ' | tee -a ../results/long_tval.log; echo | tee -a ../results/long_tval.log
  echo "$st" | awk '{split($NF,a,"/"); if(a[1]!=a[2]) bad=1} END{exit bad}' && break
done
python3 run_tval.py --merge --jobs $JOBS | tee -a ../results/long_tval.log; echo "ALL DONE $(date)" | tee -a ../results/long_tval.log
