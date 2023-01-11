#! /bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MASS_ROOT=${SCRIPT_DIR}/../
cd ${MASS_ROOT}
for i in `ls data/*.csv`; do
#cat $i |  awk 'BEGIN { FS=","}; {print $26"-"$27}' | sed 's/^-/0-/' | sed 's/-$/-0/'| sed 's/-/ /' > $i.rxtx
cat $i | grep "Data" | awk 'BEGIN { FS=","}; {print $26" "$27}' > $i.rxtx
done
cd -

