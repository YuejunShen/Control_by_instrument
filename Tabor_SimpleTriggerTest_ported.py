"""Port of Tabor-Electronics/Python-Examples
'Notebooks/Using Triggers/SimpleTriggerTest.ipynb' cells to a plain script,
run against our actual P9484M, to see whether their reference example shows
the same below-zero shift we saw with our own triggered scripts.

Only changes from the notebook: the VISA address, the SourceFiles import
path, and holding the output on with a timed sleep instead of leaving the
notebook kernel running indefinitely.
"""

import os
import sys
import time

import numpy as np

srcpath = os.path.join(os.path.dirname(__file__), "Tabor electronics", "SourceFiles")
sys.path.append(srcpath)
from tevisainst import TEVisaInst


# Cell 3
inst_addr = "TCPIP::169.254.3.195::5025::SOCKET"
inst = TEVisaInst(inst_addr)
resp = inst.send_scpi_query("*IDN?")
print("connected to: " + resp)

# Cell 4
inst.default_paranoia_level = 2
inst.send_scpi_cmd("*CLS; *RST")
resp = inst.send_scpi_query(":SYST:ERR?")
print(resp)

# Cell 5
model_name = inst.send_scpi_query("SYST:INF:MODel?")
print("Model: {0} ".format(model_name))
resp = inst.send_scpi_query(":INST:CHAN? MAX")
print("Number of channels: " + resp)
num_channels = int(resp)

if model_name.startswith("P948"):
    bpp = 2
    max_dac = 65535
    wpt_type = np.uint16
    channels_per_dac = 2
elif model_name.startswith("P908"):
    bpp = 1
    max_dac = 255
    wpt_type = np.uint8
    channels_per_dac = 1
else:
    bpp = 2
    max_dac = 65535
    wpt_type = np.uint16
    channels_per_dac = 2

half_dac = max_dac / 2.0

resp = inst.send_scpi_query(":TRACe:SELect:SEGMent? MAX")
print("Max segment number: " + resp)
max_seg_number = int(resp)

resp = inst.send_scpi_query(":TRACe:FREE?")
arbmem_capacity = (int(resp) // 64) * 64
print("Available memory per DDR: {0:,} wave-bytes".format(arbmem_capacity))

max_seglen = arbmem_capacity // bpp
print("Max segment length: {0:,}".format(max_seglen))

# Cell 6
seglen = 4096
cyclelen = seglen
ncycles = seglen / cyclelen

x = np.linspace(start=0, stop=2 * np.pi * ncycles, num=seglen, endpoint=False)
y = (np.sin(x) + 1.0) * half_dac
y = np.round(y)
y = np.clip(y, 0, max_dac)
wav = y.astype(wpt_type)
print("waveform min/max codes:", wav.min(), wav.max(), "expected mid:", half_dac)

# Cell 7
ch = 1
cmd = ":INST:CHAN {0}".format(ch)
inst.send_scpi_cmd(cmd)

segnum = 1
print("Downloading segment {0} of channel {1}".format(segnum, ch))
cmd = ":TRAC:DEF {0}, {1}".format(segnum, seglen)
inst.send_scpi_cmd(cmd)

cmd = ":TRAC:SEL {0}".format(segnum)
inst.send_scpi_cmd(cmd)

inst.write_binary_data(":TRAC:DATA", wav)

resp = inst.send_scpi_query(":SYST:ERR?")
print(resp)

# Cell 8
cmd = ":INST:CHAN {0}".format(segnum)
inst.send_scpi_cmd(cmd)

cmd = ":SOUR:FUNC:MODE:SEGM {0}".format(1)
inst.send_scpi_cmd(cmd)

cmd = ":INIT:CONT OFF"
inst.send_scpi_cmd(cmd)

inst.send_scpi_cmd(":OUTP ON")

resp = inst.send_scpi_query(":SYST:ERR?")
print(resp)

# Cell 9
inst.send_scpi_cmd(":TRIG:SOURce:ENAB {}".format("TRG1"))
inst.send_scpi_cmd(":TRIG:SELECT {}".format("TRG1"))
inst.send_scpi_cmd(":TRIG:LEV 1.0")
inst.send_scpi_cmd(":TRIG:DEL 0E-9")
inst.send_scpi_cmd(":TRIG:SLOP POS")
inst.send_scpi_cmd(":TRIG:COUP ON")
inst.send_scpi_cmd(":TRIGger:STAT ON")

resp = inst.send_scpi_query(":SYST:ERR?")
print(resp)

print()
print("=== Readback after full notebook sequence ===")
for scpi in (
    ":SOUR:VOLT?", ":SOUR:VOLT:OFFS?", ":INIT:CONT?", ":TRIG:STAT?",
    ":TRIG:IDLE?", ":TRIG:IDLE:LEV?", ":OUTP?",
):
    print(f"{scpi:20s} -> {inst.send_scpi_query(scpi).strip()}")

print()
print("Armed for 90 seconds exactly per the notebook's own sequence. "
      "Trigger it now and check the scope.")
time.sleep(90)
print("Time is up, turning off.")

# Cell 10
inst.send_scpi_cmd(":OUTP OFF")
inst.send_scpi_cmd(":TRIG:STAT OFF")
inst.close_instrument()
