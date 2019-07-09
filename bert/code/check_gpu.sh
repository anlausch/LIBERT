#!/usr/bin/env bash
while gpusage | grep -q 'dws07    0'; do
   echo "gpu still occupied"
   sleep 30
done
echo "starting process now";