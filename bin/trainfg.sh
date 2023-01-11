#! /bin/bash
NUM_FEATURES=${1:-4}
BATCH_SIZE=${2:-100}
CONTEXT=${3:-""}
LOAD=${4:-""}
echo -e "Started $(date)"
START_TIME=`date +%s`
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MASS_ROOT=${SCRIPT_DIR}/../
cd ${MASS_ROOT}
#./bin/tomass.sh
#./bin/trainbench.sh
if [ "${LOAD}" == "load" ]; then
  FLAGS="--load_g"
  EPOCHS=${LOAD_EPOCHS:-2000}
else
  FLAGS=""
  EPOCHS=${EPOCHS:-5000}
fi
CONTEXT=${CONTEXT} PYTHONPATH=${PYTHONPATH}:${MASS_ROOT}/scripts python3 -u ${MASS_ROOT}/scripts/train_mass.py --g_lrn_rate=0.002 --d_lrn_rate=0.0004 --g_pretraining_epochs=10 --d_pretraining_epochs=10 --label_smoothing --seq_len=12 --num_features=${NUM_FEATURES} --num_epochs=${EPOCHS} --corr_matching --corr_alpha=1.0 --batch_size=${BATCH_SIZE} --gauss_histogram --normalized --no_save_g --conditional_freezing 
END_TIME=`date +%s`
DUR_SEC=`expr ${END_TIME} - ${START_TIME}`
DUR_HOUR_REAL=`echo -e "scale=5;${DUR_SEC}/3600" | bc`
DUR_HOUR=`echo -e "scale=0;${DUR_SEC}/3600" | bc`
DUR_HOUR_REM=`echo -e "scale=5;(${DUR_HOUR_REAL}-${DUR_HOUR})/1.0" | bc`
DUR_MIN_REAL=`echo -e "scale=5;(${DUR_HOUR_REM}*60)/1.0" | bc`
DUR_MIN=`echo -e "scale=0;(${DUR_HOUR_REM}*60)/1.0" | bc`
DUR_MIN_REM=`echo -e "scale=5;(${DUR_MIN_REAL}-${DUR_MIN})/1.0" | bc`
DUR_SEC=`echo -e "scale=0;(${DUR_MIN_REM}*60)/1.0" | bc`
echo -e "Training Duration: ${DUR_HOUR} hours, ${DUR_MIN} minutes, and ${DUR_SEC} seconds" 
echo -e "Training Ended $(date)"
#SAMPLES=`ls data/*.mass | wc -l | awk '{print $1}'`
#SAMPLES=${SAMPLES:-100}
#CONTEXT=${CONTEXT} SAMPLES=${SAMPLES} ./bin/gengan.sh
#echo -e "Generation Ended $(date)"
#Rscript graph/corrcheck.R ${SAMPLES} gan 1 2
#./bin/score.sh
cd -
