#! /bin/bash
PORT=$1
OUTPUT=$2
CLIENT=${3:-1}
iperf3 -s -p ${PORT} -f m -i 0 | grep --line-buffer -e "Mbits/sec.*sender\|Mbits/sec.*receiver" | stdbuf -o0 sed -e 's/.*KBytes[[:space:]]*\(.*\)Mbits.*/0 0 0 \1/' >> logs/${OUTPUT}_${CLIENT}.log
