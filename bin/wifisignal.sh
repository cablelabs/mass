#! /bin/bash
WDEV=${WDEV:-wlan0}
ISMAC=1
which airport >/dev/null && ISMAC=1
if [ ${ISMAC} -eq 1 ]; then
  SIGNAL=`airport ${WDEV} -I | grep agrCtlRSSI | awk '{print $2}'`
else
  SIGNAL=`iw dev ${WDEV} link | grep "signal:" | awk '{print $2}'`
fi
echo -n "${SIGNAL}"
