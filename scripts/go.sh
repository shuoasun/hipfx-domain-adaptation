#!/bin/bash
# one ~170 s chunk with 4 parallel workers; usage: bash go.sh <job> [<job> ...]
cd "$(dirname "$0")/../src"; mkdir -p ../results; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for k in 0 1 2 3; do python3 run_chunk.py --worker $k --nworkers 4 --budget ${BUDGET:-163} --jobs "$@" > ../results/log_w$k.txt 2>&1 & done; wait
cat ../results/log_w*.txt | grep -c AUC; grep -h -E "Error|Traceback" ../results/log_w*.txt | head -3; python3 run_chunk.py --status --jobs "$@"
