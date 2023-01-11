#! /bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MASS_ROOT=${SCRIPT_DIR}/../
cd ${MASS_ROOT}
for i in `ls data/*.csv`; do
python3 scripts/extract.py $i
done
cd -

