#! /bin/bash
if [ ${PROTO} == "udp" ]; then
  IPROTO="-u"
  PERF_PORT=$(expr ${PORT} + 1)
else
  PERF_PORT=${PORT}
fi
if [ ${DIRECTION} == "UP" ]; then
  DIRFLAG="-R"
else
  DIRFLAG=""
fi
echo "iperf3 -c ${PERF_HOST} -f m -p ${PERF_PORT} ${DIRFLAG} -t ${EPOCH_TIME} -b ${BPS} ${IPROTO} -l ${BUFFER}"
iperf3 -c ${PERF_HOST} -f m -p ${PERF_PORT} ${DIRFLAG} -t ${EPOCH_TIME} -b ${BPS} ${IPROTO} -l ${BUFFER}
