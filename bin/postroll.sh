#! /bin/bash
BENCH=$1
SAMPLES=$2
DISTFIT=${3:-fit}
i=0
while [ $i -lt $SAMPLES ]; do
  echo "Rolling ${BENCH} sample ${i}..."
  if [ "${BENCH}" != "arima" ]; then
    Rscript graph/roll.R ${i} ${BENCH}
  fi
  if test -f ${BENCH}/${i}.mass; then
    if [ "$DISTFIT" == "fit" ]; then
      mv ${BENCH}/${i}.mass ${BENCH}/${i}.prebeta
      Rscript graph/transform.R $i $SAMPLES ${BENCH}
    fi
    i=`expr $i + 1`
  else
    rm ${BENCH}/${i}.raw
    Rscript graph/${BENCH}fit.R ${SAMPLES} data 2>/dev/null
  fi
done
