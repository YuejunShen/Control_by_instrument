"""Upload a sine burst to the Tabor P9484M and play it once per external
trigger edge.

Unlike Tabor_P9484M_PUND.py (which free-runs continuously), this script arms
the AWG so that it stays idle and outputs the programmed segment exactly once
each time a trigger event is received on the selected trigger input.

The sine burst starts at phase 0 (rising) at segment sample 0, so with
trigger_delay_s = 0 it starts right as the trigger edge is received. Edit
sine_frequency_hz / sine_cycle_count / sine_amplitude to change the shape.

NOTE: the trigger SCPI commands below (:TRIG:SEL, :TRIG:SOUR:ENAB, :TRIG:LEV,
:TRIG:SLOP, :TRIG:DEL, :TRIG:STAT) follow the standard Tabor Proteus trigger
subsystem. Double check them against the P9484M SCPI programming manual for
your firmware revision before relying on this in an experiment.

Run the script and press Enter when you want to turn the output off.
"""

import os
import sys

import numpy as np


# Use the Tabor drivers supplied with this project.
TABOR_SOURCE_PATH = os.path.join(os.path.dirname(__file__), "Tabor electronics", "SourceFiles")
if TABOR_SOURCE_PATH not in sys.path:
    sys.path.insert(0, TABOR_SOURCE_PATH)

from tevisainst import TEVisaInst


# -----------------------------
# User-editable settings
# -----------------------------
proteus_address = "TCPIP::169.254.3.195::5025::SOCKET"
awg_channel = 1
segment_number = 1

sample_rate_hz = 2.4e9
segment_granularity = 64
minimum_segment_points = 128
maximum_segment_points = 1_000_000

# Sine shape. The sine starts at phase 0 (sin(0) = 0, rising) at t=0
# (segment sample 0), so with trigger_delay_s = 0 it starts as soon as the
# trigger edge is received.
#
# sine_frequency_hz is deliberately chosen as sample_rate_hz/64 (64 samples
# per cycle) with sine_cycle_count=2, so the sine fills the *entire* segment
# (128 samples = the minimum segment length, already a multiple of the 64
# sample granularity) with no zero-padding needed. This matters: this P9484M
# channel is AC-coupled (per the Proteus manual), which cannot faithfully
# hold a static/unchanging DC level for any real duration -- any padded flat
# region droops. Tabor's own reference example (SimpleTriggerTest.ipynb)
# fills its whole segment with continuous sine the same way, with zero
# padding, and it measures correctly centered on this same instrument.
# Earlier versions of this script used a short burst + long zero baseline
# (padded to the segment length), which is exactly the static-level pattern
# an AC-coupled output can't reproduce -- that's what caused the below-zero
# shift, not a wrong SCPI setting.
sine_amplitude = 1.0                       # normalized, -1..1
sine_frequency_hz = sample_rate_hz / 128  # 64 samples per cycle
sine_cycle_count = 2                       # -> 128 samples, exactly one segment
baseline_after_s = 0.0                      # kept at 0: any padding here would
                                            # reintroduce a static AC-coupling droop

# AWG output settings.
awg_vpp = 0.5
# NOTE: on scope, a 0.2 Vpp sine with offset=0 measured between 0 and -0.2 V
# instead of the expected +-0.1 V (i.e. shifted down by awg_vpp/2). Setting
# awg_offset_v = awg_vpp/2 to compensate was REJECTED by the instrument
# (":SOUR:VOLT:OFFS 0.1" -> SCPI error 204, data out of range), so that is
# not a valid fix. Leaving this at 0 until the actual cause/valid offset
# range for this channel is confirmed against the P9484M SCPI manual.
awg_offset_v = 0.0

# Trigger settings: which input arms/starts playback, and how it is
# interpreted. trigger_source is one of "TRG1", "TRG2", "CPU", "INT".
# Keep trigger_delay_s at 0 so the sine burst starts immediately on the
# trigger edge, with no added latency beyond the instrument's fixed hardware
# delay.
trigger_source = "TRG1"
trigger_level_v = 1.0
trigger_slope = "POS"  # "POS" or "NEG"
trigger_delay_s = 0.0


def build_arb_waveform():
    """Build the normalized (-1..1) sine-burst waveform."""
    if sine_frequency_hz <= 0:
        raise ValueError("sine_frequency_hz must be positive")
    if sine_cycle_count <= 0:
        raise ValueError("sine_cycle_count must be positive")
    if abs(sine_amplitude) > 1.0:
        raise ValueError("sine_amplitude must stay within [-1, 1]")

    burst_duration_s = sine_cycle_count / sine_frequency_hz
    point_count = max(2, int(round(burst_duration_s * sample_rate_hz)))
    sample_times = np.arange(point_count) / sample_rate_hz
    waveform = sine_amplitude * np.sin(2.0 * np.pi * sine_frequency_hz * sample_times)

    if baseline_after_s > 0:
        baseline_count = max(1, int(round(baseline_after_s * sample_rate_hz)))
        waveform = np.concatenate((waveform, np.zeros(baseline_count)))

    required_length = max(len(waveform), minimum_segment_points)
    padded_length = (
        (required_length + segment_granularity - 1)
        // segment_granularity
        * segment_granularity
    )
    if padded_length > maximum_segment_points:
        raise ValueError(
            f"Waveform requires {padded_length} points; limit is {maximum_segment_points}"
        )
    if padded_length > len(waveform):
        # Padding here means part of the segment is a static, unchanging DC
        # level -- exactly what an AC-coupled output can't reproduce (see the
        # module docstring). sine_frequency_hz/sine_cycle_count are chosen so
        # this should never trigger; warn loudly if a future edit breaks that.
        print(
            f"WARNING: waveform padded from {len(waveform)} to {padded_length} "
            "samples -- this reintroduces a static DC region on an AC-coupled "
            "output. Pick sine_frequency_hz/sine_cycle_count so the sine "
            "itself already lands on a segment_granularity-aligned length."
        )
        waveform = np.pad(waveform, (0, padded_length - len(waveform)), mode="constant")

    return waveform


def encode_for_dac(normalized_waveform):
    if np.any(normalized_waveform < -1.0) or np.any(normalized_waveform > 1.0):
        raise ValueError("Normalized waveform exceeds the DAC range")
    return np.rint((normalized_waveform + 1.0) * 32767.5).astype(np.uint16)


def check_instrument_error(instrument, operation):
    response = instrument.send_scpi_query(":SYST:ERR?").strip()
    if not response.startswith("0"):
        raise RuntimeError(f"P9484M error after {operation}: {response}")


def upload_and_arm(instrument, dac_waveform):
    def send(command):
        instrument.send_scpi_cmd(command)
        check_instrument_error(instrument, command)

    send("*CLS;*RST")

    send(f":INST:CHAN {awg_channel}")
    send(":SOUR:MODE DIR")
    send(f":FREQ:RAST {sample_rate_hz}")
    send(":TRAC:FORM U16")
    send(":TRAC:DEL:ALL")
    # Set :INIT:CONT here, in the same position Tabor_P9484M_PUND.py sets it
    # (right after :TRAC:DEL:ALL, before :TRAC:DEF/:SOUR:VOLT), rather than
    # at the end. PUND's raw AWG output is confirmed centered at 0 V, and
    # this was the one structural ordering difference between the two
    # scripts, so match it in case this instrument's DAC centering is
    # established relative to when :INIT:CONT is set.
    send(":INIT:CONT OFF")
    send(f":TRAC:DEF {segment_number},{len(dac_waveform)}")
    send(f":TRAC:SEL {segment_number}")

    transfer_status = instrument.write_binary_data(":TRAC:DATA", dac_waveform)
    if transfer_status != 0:
        raise RuntimeError(f"Waveform transfer failed with code {transfer_status}")

    check_instrument_error(instrument, "waveform transfer")
    send(f":FUNC:MODE:SEGM {segment_number}")
    send(f":SOUR:VOLT {awg_vpp}")
    send(f":SOUR:VOLT:OFFS {awg_offset_v}")

    # Unlike Tabor_P9484M_PUND.py (:INIT:CONT ON, so the segment is always
    # actively playing and never idle), this script spends most of its time
    # between triggers in the AWG's separate "idle state" (Proteus
    # Programming Manual 4.18/4.19: :TRIG:IDLE / :TRIG:IDLE:LEV), which is a
    # DC level independent of the uploaded segment data. Set it explicitly
    # to mid-scale (0 V equivalent) instead of relying on it already being
    # at its documented default.
    send(":TRIG:IDLE DC")
    send(":TRIG:IDLE:LEV 32768")

    # Match the order used in Tabor's own reference example
    # (Using Triggers/SimpleTriggerTest.ipynb): turn the output on first,
    # then configure and arm the trigger in a separate step.
    send(":OUTP ON")

    send(f":TRIG:SOUR:ENAB {trigger_source}")
    send(f":TRIG:SEL {trigger_source}")
    if trigger_source in ("TRG1", "TRG2"):
        send(f":TRIG:LEV {trigger_level_v}")
        send(f":TRIG:SLOP {trigger_slope}")
    send(f":TRIG:DEL {trigger_delay_s}")
    send(":TRIG:COUP ON")
    send(":TRIG:STAT ON")


def main():
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    normalized_waveform = build_arb_waveform()
    dac_waveform = encode_for_dac(normalized_waveform)
    waveform_duration_s = len(dac_waveform) / sample_rate_hz

    instrument = TEVisaInst(proteus_address)
    try:
        instrument.default_paranoia_level = 1
        identity = instrument.send_scpi_query("*IDN?").strip()
        if "P9484" not in identity.upper():
            raise RuntimeError(f"Connected instrument is not a P9484: {identity}")
        print("Connected to:", identity)

        upload_and_arm(instrument, dac_waveform)
        print(f"Waveform points: {len(dac_waveform)}")
        print(f"Waveform duration: {waveform_duration_s:g} s")
        print(f"P9484M setting: {awg_vpp:g} Vpp, {awg_offset_v:g} V offset into 50 ohms")
        print(
            f"Armed on trigger source {trigger_source} "
            f"({trigger_slope} slope, {trigger_delay_s:g} s delay). "
            "Waveform plays once per trigger edge."
        )
        input("Output is armed. Press Enter to turn it OFF... ")
    finally:
        try:
            instrument.send_scpi_cmd(f":INST:CHAN {awg_channel}")
            instrument.send_scpi_cmd(":TRIG:STAT OFF")
            instrument.send_scpi_cmd(":OUTP OFF")
            print("P9484M output turned OFF.")
        finally:
            instrument.close_instrument()


if __name__ == "__main__":
    main()
