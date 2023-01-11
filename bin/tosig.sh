#! /bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MASS_ROOT=${SCRIPT_DIR}/../
cd ${MASS_ROOT}
for i in `ls data/*.csv`; do
#cat $i |  awk 'BEGIN { FS=","}; {print $26"-"$27}' | sed 's/^-/0-/' | sed 's/-$/-0/'| sed 's/-/ /' > $i.rxtx
cat $i | grep -e "Cell" | awk 'BEGIN { FS=","}; {print $24}' > $i.sigcell
cat $i | grep -e "Wifi" | awk 'BEGIN { FS=","}; {print $79}' > $i.sigwifi
paste $i.sigwifi $i.sigcell | awk '{print $1" "$2}' > $i.sigtmp
cat $i.sigtmp | grep ". ." >$i.sig
rm $i.sigcell $i.sigwifi
done
cd -

