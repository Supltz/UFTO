#!/usr/bin/env bash
set -euo pipefail

run_one() {
  local dataset="$1"
  nohup python test.py --dataset "${dataset}" > "nohup_test_${dataset}.log" 2>&1 &
}

run_one "rafdb"
run_one "affectnet"
run_one "emotionet"
run_one "rafau"
