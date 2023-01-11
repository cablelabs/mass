#! /usr/bin/env python3

import sys

category = {}
with open('metadata/categories.dat') as f:
  for line in f:
    cols = line.strip('\n').split(" ")
    if len(cols) == 2:
       category[cols[0]] = cols[1] 
    else:
       category[cols[0]] = "UNKNOWN"

def get_category(app):
  app = app.lstrip("+").lstrip("=")
  if app in category:
    cat = category[app]
  else:
    cat = "UNKNOWN"
  if cat.strip() == "":
    cat = "UNKNOWN"
  return cat

infile = sys.argv[1]
infile = sys.argv[1]
outfile = infile + ".metrics"

stamps = {}
last_app = None
with open(infile) as f:
  for line in f:
    line = line.strip('\n')
    cols = line.split(",")
    if cols[3] == "ts_raw":
      continue
    stamp = round(int(cols[3])/(1000*60))
    if stamp not in stamps:
      stamps[stamp] = {}

    if cols[4] == "Data":
      stamps[stamp]["rx"] = cols[25]
      stamps[stamp]["tx"] = cols[26]
    if cols[4] == "App":
      app = cols[10]
      current_app = get_category(app)
      if current_app == "UNKNOWN" and last_app is not None:
         current_app = last_app
      stamps[stamp]["app"] = current_app
      if current_app != "UNKNOWN":
        last_app = current_app
    if cols[4] == "CellTower":
      stamps[stamp]["cellsig"] = cols[23]
    if cols[4] == "Wifi":
      stamps[stamp]["wifisig"] = cols[78]

last_app = None
with open(outfile,'w') as f:
  for stamp in sorted(stamps.keys()):
    s = stamps[stamp]
    if "rx" in s and s["rx"] != "":
      rx = s["rx"]
    else:
      rx = "0"
    if "tx" in s and s["tx"] != "":
      tx = s["tx"]
    else:
      tx = "0"
    if "cellsig" not in s or s["cellsig"] == "":
      continue
    cellsig = s["cellsig"]
    if "wifisig" not in s or s["wifisig"] == "":
      continue
    wifisig = s["wifisig"]
    if "app" in s:
      app = s["app"]
    else:
      app = "UNKNOWN"
    if app == "UNKNOWN" and last_app is not None:
      app = last_app
    f.write(f"{rx} {tx} {cellsig} {wifisig} {app}\n")
    if app != "UNKNOWN":
      last_app = app
