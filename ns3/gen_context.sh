#! /bin/bash
MASS_HOST=${MASS_HOST:-localhost}
SEQ_LEN=${SEQ_LEN:-100}
CONTEXTS="STREAM STREAM_HIGH STREAM_LOW INTERACT INTERACT_LOW INTERACT_HIGH LOW HIGH"
curl -s -d '{"seq_len":'${SEQ_LEN}'}' http://${MASS_HOST}:7777/generate?format=txt > data/sample.trace.DEFAULT
for CONTEXT in ${CONTEXTS}; do
  curl -s -d '{"context":"'${CONTEXT}'","seq_len":'${SEQ_LEN}'}' http://${MASS_HOST}:7777/generate?format=txt > data/sample.trace.${CONTEXT}
done
