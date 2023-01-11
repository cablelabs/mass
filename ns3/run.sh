#! /bin/bash
EXAMPLE=${1:-basic}
PROTO=${PROTO:-udp}
GDB=${GDB:-no}
mkdir -p data
SEQ_LEN=${SEQ_LEN} ./gen_context.sh
docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -e EXAMPLE=${EXAMPLE} -e PROTO=${PROTO} -e GDB=${GDB} -v `pwd`/data:/sim/data --name ns3mass --rm ns3mass
