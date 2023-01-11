#! /usr/bin/env python3
from sh import validate as val

def validate():
  out = "%s" % val()
  out = out.split(" ")
  return (float(out[1]),float(out[2]))
