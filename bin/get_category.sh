#! /bin/bash
PACKAGE=$1
CATEGORY=`curl -s https://play.google.com/store/apps/details?id=${PACKAGE} | grep "hrTbp R8zArc" | sed "s/.*\/store\/apps\/category\///" | sed 's/".*//'`
echo ${CATEGORY}
