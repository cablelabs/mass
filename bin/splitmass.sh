#! /bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MASS_ROOT=${SCRIPT_DIR}/../
NUM_FEATURES=${NUM_FEATURES:-2}
SEQ_LEN=$1
cd ${MASS_ROOT}
i=0
rm -f gan/*
split -l ${SEQ_LEN} sample.mass split_
for f in `ls split_*`; do
  mv $f gan/${i}.mass
  i=`expr $i + 1`
done
