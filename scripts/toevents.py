#! /usr/bin/env python3
import midi
import sys
infile = sys.argv[1]
pattern = midi.read_midifile(infile)
track = pattern.pop()

for e in track:
  print(e)
