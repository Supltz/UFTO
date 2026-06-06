#!/usr/bin/env bash
set -euo pipefail

run_one() {
  local dataset="$1"
  nohup python train_stage_1.py --dataset "${dataset}" > "nohup_stage1_${dataset}.log" 2>&1
}

# run_one "rafdb"
run_one "affectnet"
run_one "emotionet"
run_one "rafau"
