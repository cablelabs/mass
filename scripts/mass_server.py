#! /usr/bin/env python3

from mass import Mass
from flask import Flask, jsonify, request
from os import path
import io
import numpy as np

def get_log(log_path):
  if not path.exists(log_path):
    return []
  log_data = []
  with open(log_path) as f:
    for line in f:
      # TODO: make use of all data points
      #log_data.append(map(lambda x: float(x),line.strip('\n').split(" ")))
      log_data.append(float(line.strip('\n').split(" ")[3]))
  return log_data

def get_monitor_data(client):
  udp_up = get_log(f"logs/udpUP_{client}.log")
  udp_down = get_log(f"logs/udpDOWN_{client}.log")
  tcp_up = get_log(f"logs/tcpUP_{client}.log")
  tcp_down = get_log(f"logs/tcpDOWN_{client}.log")
  return {"udpup": udp_up,
          "udpdown": udp_down,
          "tcpup": tcp_up,
          "tcpdown": tcp_down}

app = Flask(__name__)

@app.route("/data/<client>",methods=["POST"])
def get_data(client):
  return jsonify(get_monitor_data(client))

@app.route("/",methods=["GET"])
def index():
  with open("html/index.html") as f:
    html = f.read() 
  return html

def to_txt(trace, users):
  output = ""
  if users == 1:
    s = io.BytesIO()
    np.savetxt(s, trace, delimiter=" ")
    output =  s.getvalue().decode()
  else:
    sep = ""
    for i in range(0, users):
      s = io.BytesIO()
      np.savetxt(s, trace[i,:,:], delimiter=" ")
      output +=  sep + s.getvalue().decode()
      sep = "\n"
  return output

@app.route("/generate",methods=["POST"])
def gen():
    data = request.get_json(force=True)
    if 'context' in data:
      ctx = data['context'] 
    else:
      ctx = "DEFAULT"
    if 'users' in data:
      users = data['users']
    else:
      users = 1
    if 'seq_len' in data:
      seq_len = data['seq_len']
    else:
      seq_len = 100
    if 'normalize' in data:
      normalize = data['normalize']
    else:
      normalize = "pos"
    if 'shuffle' in data:
      do_shuffle = data['shuffle']
    else:
      do_shuffle = False

    mass = Mass(users, seq_len, do_shuffle, normalize)
    trace = mass.generate(context=ctx)

    output_format = request.args.get('format', default = "json", type=str)
    if output_format == "txt":
      return to_txt(trace, users)
    return jsonify({"trace":trace.tolist()})

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=7777)
