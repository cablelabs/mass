#! /bin/bash
CONTEXTS=`cat data/contexts`

for CONTEXT in ${CONTEXTS}; do
  CONTEXT=${CONTEXT} python3 scripts/normalize.py data >/dev/null 2>&1
  USERS=`ls data/*.massn.${CONTEXT} | wc -l | awk {'print $1'}`
  mkdir -p gan${CONTEXT}
  CONTEXT=${CONTEXT} SAMPLES=${USERS} ./bin/gengan.sh >/dev/null 2>&1
  RES=`CONTEXT=${CONTEXT} TROWS=12 Rscript graph/score.R ${USERS} data ${USERS} gan${CONTEXT} 2 | grep RESULT`
  COR=`echo "$RES" | awk '{print $7}'`
  MOM=`echo "$RES" | awk '{print $4}'`
  NOV=`echo "$RES" | awk '{print $5}'`
  RESGLOB=`CONTEXT=${CONTEXT} TROWS=12 Rscript graph/score.R ${USERS} data ${USERS} ganDEFAULT 2 | grep RESULT`
  CORGLOB=`echo "$RESGLOB" | awk '{print $7}'`
  MOMGLOB=`echo "$RESGLOB" | awk '{print $4}'`
  NOVGLOB=`echo "$RESGLOB" | awk '{print $5}'`
  COR=`echo "scale=2;(${CORGLOB}-${COR})/${CORGLOB}" | bc`
  MOM=`echo "scale=2;(${MOMGLOB}-${MOM})/${MOMGLOB}" | bc`
  NOV=`echo "scale=2;(${NOVGLOB}-${NOV})/${NOVGLOB}" | bc`
  echo "${CONTEXT} ${COR} ${MOM} ${NOV}"
done

