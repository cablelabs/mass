#! /bin/bash
EXAMPLE=${EXAMPLE:-basic}
PROTO=${PROTO:-udp}
GDB=${GDB:-no}
touch sim.log
cd /usr/ns3/ns-3.29
GDP_ARG=""
if [ "${GDB}" == "yes" ]; then
  GDB_ARG="gdb --args "
fi
./waf --command-template="${GDB_ARG}%s --ns3::ConfigStore::Filename=input-defaults.txt --ns3::ConfigStore::Mode=Load --ns3::ConfigStore::FileFormat=RawText" --run mass_${EXAMPLE} 2>&1
cd /sim
cat sim.log 
