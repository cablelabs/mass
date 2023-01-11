#! /bin/bash
CONTEXTS=`python3 scripts/determine_context.py data 2>/dev/null`
echo "${CONTEXTS}" > data/contexts
for CONTEXT in ${CONTEXTS}; do
  python3 scripts/extractcontext.py data ${CONTEXT}
  CONTEXT=${CONTEXT} python3 scripts/normalize.py data
done
BATCH_SIZE=`ls data/*.massn | wc -l | awk '{print $1}'`
echo "context DEFAULT batch ${BATCH_SIZE}"
./bin/trainfg.sh 2 ${BATCH_SIZE} ""
cp models/c_rnn_gan_g.pth models/c_rnn_gan_g.pth.DEFAULT   
for CONTEXT in ${CONTEXTS}; do
  BATCH_SIZE=`ls data/*.massn.${CONTEXT} | wc -l | awk '{print $1}'`
  echo "context ${CONTEXT} batch ${BATCH_SIZE}"
  ./bin/trainfg.sh 2 ${BATCH_SIZE} ${CONTEXT} load
  cp models/c_rnn_gan_g.pth models/c_rnn_gan_g.pth.${CONTEXT}   
done
