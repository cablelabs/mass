#! /bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MASS_ROOT=${SCRIPT_DIR}/../
SAMPLES=100
cd ${MASS_ROOT}
mkdir -p results
rm -rf results/*
SOURCES="${SOURCES:-data test}"
BENCHES="${BENCHES:-dist gan timegan uni test}"
NUM_FEATURES=${NUM_FEATURES:-2}
echo "source -> bench KLDIVERG ACFUNCTION CENTMOMENT USERNOVELTY HURSTCOEF CROSSCORR" | tr ' ' '\t' | tee -a results/summary.dat
for SOURCE in $SOURCES; do
for BENCH in $BENCHES; do
if [ "$SOURCE" != "${BENCH}" ]; then
Rscript ${MASS_ROOT}/graph/score.R ${SAMPLES} $SOURCE ${SAMPLES} ${BENCH} ${NUM_FEATURES} 2>/dev/null > results/${SOURCE}_${BENCH}.log
RESULT=`cat results/${SOURCE}_${BENCH}.log | grep RESULT`
echo "$SOURCE -> $BENCH $RESULT" | sed 's/RESULT //' | tr ' ' '\t' | tee -a results/summary.dat
mv train.png results/train_${SOURCE}.png
mv test.png results/test_${SOURCE}_${BENCH}.png
fi
done
done

