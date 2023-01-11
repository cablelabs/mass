#! /bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MASS_ROOT=${SCRIPT_DIR}/../
BATCH_SIZE=${SAMPLES:-100}
SEQ_LEN=${SEQ_LEN:-12}
NUM_FEATURES=${NUM_FEATURES:-2}
DENORM=${DENORM:-0}
DIST=${DIST:-0}
cd ${MASS_ROOT}
mkdir -p gan${CONTEXT}
rm -f gan${CONTEXT}/*
PYTHONPATH=${PYTHONPATH}:${MASS_ROOT}/scripts python3 ${MASS_ROOT}/scripts/generate_mass.py --num_features=${NUM_FEATURES} --batch_size=${BATCH_SIZE} --seq_len=${SEQ_LEN} -n 1 --context=${CONTEXT}
#./bin/splitmass.sh ${SEQ_LEN}
for i in `ls gan${CONTEXT}/*.mass`; do cp $i ${i}n; done

if [ ${DENORM} -eq 1 ]; then
  for i in `seq 0 $(expr ${BATCH_SIZE} - 1)`; do
    mv gan/${i}.mass gan/${i}.predist
    Rscript graph/denormalize.R $i ${BATCH_SIZE} gan ${NUM_FEATURES}
  done
fi
if [ ${DIST} -eq 1 ]; then
  for i in `seq 0 $(expr ${BATCH_SIZE} - 1)`; do
    mv gan/${i}.mass gan/${i}.predist
    Rscript graph/todist.R $i ${BATCH_SIZE} gan ${NUM_FEATURES}
  done
fi
cd -
