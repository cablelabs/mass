#! /bin/bash
mkdir -p results
rm -f results/train_curve.dat
for EPOCH in `seq 100 100 4000`; do
  D=0
  C=0
  M=0
  N=0
  for i in `seq 1 5`; do
    START_TIME=`date +%s`
    EPOCHS=$EPOCH ./bin/trainfg.sh 2 100 ""
    END_TIME=`date +%s`
    DURATION=`expr ${END_TIME} - ${START_TIME}`
    ./bin/gengan.sh
    RESULT=`Rscript graph/score.R 100 data 100 gan 2 | grep "RESULT"`
    CORR=`echo $RESULT | awk '{print $7}'`
    MOM=`echo $RESULT | awk '{print $4}'`
    NOV=`echo $RESULT | awk '{print $5}'`
    N=`echo "scale=5;$NOV + $N" | bc`
    C=`echo "scale=5;$CORR + $C" | bc`
    M=`echo "scale=5;$MOM + $M" | bc`
    D=`echo "scale=5;$DURATION + $D" | bc`
  done
  N=`echo "scale=5;$N / 5.0" | bc`
  C=`echo "scale=5;$C / 5.0" | bc`
  M=`echo "scale=5;$M / 5.0" | bc`
  D=`echo "scale=5;$D / 5.0" | bc`
  echo "${EPOCH} ${C} ${M} ${N} ${D}" >> results/train_curve.dat
done
