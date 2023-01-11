#! /bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MASS_ROOT=${SCRIPT_DIR}/../
cd ${MASS_ROOT}
SAMPLES=${SAMPLES:-100}
NUM_FEATURES=${NUM_FEATURES:-2}
BENCHES="dist uni" 
for BENCH in $BENCHES; do
  mkdir -p ${BENCH}
  rm -rf ${BENCH}/*
  echo "Training ${BENCH}"
  Rscript graph/${BENCH}fit.R ${SAMPLES} data ${NUM_FEATURES}
  PYTHONPATH=${PYTHONPATH}:./scripts python3 scripts/normalize.py ${BENCH}
done
PYTHONPATH=${PYTHONPATH}:./scripts python3 scripts/normalize.py timegan
