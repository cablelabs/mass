#! /usr/bin/env python3
import midi
import sys
infile = sys.argv[1]
pattern = midi.read_midifile(infile)
track = pattern.pop()
minmidi = 21
maxmidi = 108
midirange = maxmidi - minmidi

tone = 0
length = 0
pause = 0
volume = 0
tempo = 0
print("Rx RxStd Tx TxStd TxRx")
for e in track:
  if e.name == "Set Tempo":
    tempo = e.get_bpm()
  if e.name == "Note On":
    tone = e.data[0]
    pause = e.tick
    if tone < 21:
      tone = 21
    volume = e.data[1]
  if e.name == "Note Off":
    length = e.tick
    print(f"{tone} {length} {volume} {pause} {tempo}")
