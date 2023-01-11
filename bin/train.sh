#! /bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
NUM_FEATURES=${NUM_FEATURES:-2}
CONTEXT=${CONTEXT:-""}
BATCH_SIZE=${BATCH_SIZE:-100}
rm -f screenlog.0
screen -d -m -L ${SCRIPT_DIR}/trainfg.sh ${NUM_FEATURES} ${BATCH_SIZE} ${CONTEXT}
