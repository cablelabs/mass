#! /bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MASS_ROOT=${SCRIPT_DIR}/../
mkdir -p ${MASS_ROOT}/test
SMOOTH=${1:-6}
TEST_PORTION=${TEST_PORTION:-0}
DATA=${DATA:-metrics}
idx=0
test_idx=0
cd ${MASS_ROOT}
MAX_USERS=${SAMPLES:-100}
rm -rf data/*.mass
rm -rf data/*.massn
rm -rf data/*.raw
rm -rf test/*.mass
for i in `ls data/*.${DATA}`; do 
  python3 scripts/tomass.py $i $idx $SMOOTH
  #if test -f "data/${idx}.raw"; then
  #  #Rscript graph/smooth.R ${idx}
  #  mv data/${idx}.raw data/${idx}.mass
  #else
  #  continue
  #fi
  if test -f "data/${idx}.mass"; then
    echo "Smooth ok ${idx} ${test_idx}"
  else
    echo "Smooth error"
    continue
  fi

  R=`awk -v seed=$RANDOM 'BEGIN{srand(seed);print rand()}'`
  echo -e "scale=2;$R < ${TEST_PORTION}/100"
  IS_TEST=`echo -e "scale=2;$R < ${TEST_PORTION}/100" | bc`
  if [ $idx -ge ${MAX_USERS} ]; then
    IS_TEST=1
  fi
  if [ $test_idx -ge ${MAX_USERS} ]; then
    IS_TEST=0
  fi
  if [ ${IS_TEST} -eq 1 ]; then
    TEST_PATH=`echo -e "$i" | sed 's/^data/test/'`
    mv data/${idx}.mass test/${test_idx}.mass
    mv data/${idx}.massn test/${test_idx}.massn
    if [ ${test_idx} -ge ${MAX_USERS} ]; then
      rm test/${test_idx}.mass
      rm test/${test_idx}.massn
    fi
    test_idx=`expr ${test_idx} + 1`
    continue
  fi
  if [ $idx -ge ${MAX_USERS} ]; then
    rm data/${idx}.mass
    rm data/${idx}.massn
  fi
  if [ $idx -ge ${MAX_USERS} ] && [ ${test_idx} -ge ${MAX_USERS} ]; then
     break
  fi
  idx=`expr $idx + 1`
done
cd -
