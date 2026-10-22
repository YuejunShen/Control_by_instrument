# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 09:39:30 2026

@author: leeau

v2 (2026-10-22): fixes the amplitude/offset problem on the Keysight 33511B.
  The 33511B outputs  V = offset + data * Vpp/2  for arb data in [-1, +1]; it does
  not rescale the data. v1 normalized by max|V| and used a hand-set 0.2 V offset,
  so the base sat at +0.2 V and every pulse was shifted by +0.2 V (VOLT:OFFS was
  also sent without the f-string, so setup failed before OUTP ON).
  v2 computes offset and Vpp from the lowest/highest levels (output_scaling()),
  so the base is exactly 0 V and the pulses land exactly on the set amplitudes,
  using the full DAC range. Also: 100 MSa/s sample grid, no zero-padding to the
  trigger period, STEP arb filter, load-dependent voltage limit, output off on error.
"""

import numpy as np
import pyvisa
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURATION FLAGS
# -----------------------------
SEND_TO_INSTRUMENT = False  # Set to False to only plot, True to send to Agilent
SHOW_PLOT = True            # Set to True to display plot in Spyder

# -----------------------------
# User-editable PUND parameters
# -----------------------------
# Reset pulse (negative)
reset_amplitude = -2.0         # V
reset_width = 1000e-9          # s
reset_rise_time = 20e-9        # s

# Positive pulses
pos_amplitude = 1              # V
pos_width = 500e-9             # s
pos_rise_time = 20e-9          # s

# Negative pulses
neg_amplitude = -1.5           # V
neg_width = 500e-9             # s
neg_rise_time = 20e-9          # s

# Gaps
gap_after_reset = 500e-9       # s
gap_between_pos = 500e-9       # s
gap_pos_to_neg = 500e-9        # s
gap_between_neg = 500e-9       # s

# Holds
init_hold = 50e-9              # s
final_hold = 50e-9             # s

# Repetition control
num_pund_repeats = 5           # Number of times PUND repeats per trigger
target_period = 1/60           # External trigger period (60 Hz = 16.67 ms)
trigger_margin = 1e-3          # NEW: 1 ms of slack for latency, re-arm, and jitter

# FIX: zero-padding to the trigger period is not needed in triggered burst mode
# (the AWG plays the arb once, then holds the first sample until the next trigger).
# Padding filled ~97% of arb memory and forced a low sample rate.
PAD_TO_PERIOD = False

# Gap between PUND repetitions (if multiple PUNDs per trigger)
gap_between_punds = 100e-6     # s (only used if num_pund_repeats > 1)

# NEW: the DC offset is no longer set by hand. output_scaling() computes the offset and
# amplitude so the base (0 V in the waveform) comes out at exactly 0 V and the pulses at
# exactly the amplitudes above.

# System parameters
arb_sample_rate_hz = 100e6 # FIX: 10 ns grid, so every duration above is a whole number of samples
                           # (at 62 MSa/s a 20 ns edge became 2 samples = 32 ns)
maximum_arb_points = 1000000
max_allowed_amplitude_v = 5.0  # ±5 V peak (10 Vpp) is the 33511B limit into 50 Ω (±10 V with load INF)
arb_filter = "STEP"            # NEW: "STEP" = least overshoot/ringing on pulse edges ("NORM" is the default)

# NEW: Output load setting.
#   "INF" -> high-impedance load (scope at 1 MΩ, sample/DUT input, etc.)
#   "50"  -> true 50 Ω termination
# If this is wrong, your voltages will be off by a factor of 2.
awg_output_load = "50"

# NEW: Trigger settings
trigger_slope = "POS"          # "POS" = rising edge, "NEG" = falling edge
trigger_delay_s = 0.0          # Delay from trigger edge to output start

#awg_address = "USB0::0x0957::0x2707::MY57301577::INSTR" # Keysight
awg_address = "USB0::2391::9991::MY52300442::0::INSTR"   # Agilent
awg_timeout_ms = 30000


def segment(value, duration_s, sample_rate):
    """Create a flat segment at a given value."""
    if duration_s < 0:
        raise ValueError("Durations cannot be negative")
    if duration_s == 0:
        return np.empty(0, dtype=float)
    point_count = max(1, int(round(duration_s * sample_rate)))
    return np.full(point_count, value, dtype=float)


def ramp(start_value, stop_value, duration_s, sample_rate):
    """Create a linear ramp between two values."""
    if duration_s < 0:
        raise ValueError("Ramp time cannot be negative")
    if duration_s == 0:
        return np.empty(0, dtype=float)
    point_count = max(2, int(round(duration_s * sample_rate)))
    return np.linspace(start_value, stop_value, point_count, endpoint=False)


def pulse(amplitude, width_s, rise_time_s, sample_rate):
    """Create a pulse with rise, flat-top, and fall."""
    return np.concatenate((
        ramp(0.0, amplitude, rise_time_s, sample_rate),
        segment(amplitude, width_s, sample_rate),
        ramp(amplitude, 0.0, rise_time_s, sample_rate),
    ))


def output_scaling():
    """
    NEW: choose AWG offset and half-span so the output window is exactly [v_low, v_high].

    The AWG outputs  V = offset + data * (Vpp / 2)  with data in [-1, +1].
    The window always includes 0 V, so the base is reachable and lands at exactly 0 V:
        data(V) = (V - offset) / half_span
    """
    v_high = max(0.0, reset_amplitude, pos_amplitude, neg_amplitude)
    v_low = min(0.0, reset_amplitude, pos_amplitude, neg_amplitude)
    if v_high == v_low:
        raise ValueError("All pulse amplitudes are zero")
    voltage_offset_v = (v_high + v_low) / 2
    half_span = (v_high - v_low) / 2      # Vpp / 2
    return voltage_offset_v, half_span


def build_single_pund():
    """Build a single PUND sequence in volts (base = 0 V, without zero padding)."""
    reset_norm = reset_amplitude   # CHANGED: built in volts, normalized in build_pund_burst_waveform
    pos_norm = pos_amplitude
    neg_norm = neg_amplitude

    single_pund = np.concatenate((
        segment(0.0, init_hold, arb_sample_rate_hz),                          # Initial hold

        pulse(reset_norm, reset_width, reset_rise_time, arb_sample_rate_hz),  # Reset
        segment(0.0, gap_after_reset, arb_sample_rate_hz),

        pulse(pos_norm, pos_width, pos_rise_time, arb_sample_rate_hz),        # P
        segment(0.0, gap_between_pos, arb_sample_rate_hz),

        pulse(pos_norm, pos_width, pos_rise_time, arb_sample_rate_hz),        # U
        segment(0.0, gap_pos_to_neg, arb_sample_rate_hz),

        pulse(neg_norm, neg_width, neg_rise_time, arb_sample_rate_hz),        # N
        segment(0.0, gap_between_neg, arb_sample_rate_hz),

        pulse(neg_norm, neg_width, neg_rise_time, arb_sample_rate_hz),        # D

        segment(0.0, final_hold, arb_sample_rate_hz),                         # Final hold
    ))

    # FIX: duration from the samples actually generated (rounding to the sample grid
    # can make it differ from the sum of the nominal parameters)
    single_pund_duration = len(single_pund) / arb_sample_rate_hz

    return single_pund, single_pund_duration


def build_pund_burst_waveform():
    """
    Build PUND burst waveform:
    Repeats PUND num_pund_repeats times, then pads with 0 V so the waveform
    finishes trigger_margin BEFORE the next trigger edge arrives.
    """
    single_pund, single_pund_duration = build_single_pund()

    if num_pund_repeats == 1:
        pund_sequences = single_pund
        total_pund_time = len(pund_sequences) / arb_sample_rate_hz
    else:
        pund_list = []
        for i in range(num_pund_repeats):
            pund_list.append(single_pund)
            if i < num_pund_repeats - 1:
                pund_list.append(segment(0.0, gap_between_punds, arb_sample_rate_hz))
        pund_sequences = np.concatenate(pund_list)
        total_pund_time = len(pund_sequences) / arb_sample_rate_hz

    # CHANGED: leave trigger_margin of slack before the next trigger
    usable_period = target_period - trigger_margin
    if usable_period <= 0:
        raise ValueError(
            f"trigger_margin ({trigger_margin*1e3:.3f} ms) is larger than the "
            f"trigger period ({target_period*1e3:.3f} ms)!"
        )

    zero_duration = usable_period - total_pund_time
    if not PAD_TO_PERIOD:
        zero_duration = 0.0

    if zero_duration < 0:
        raise ValueError(
            f"PUND sequences ({total_pund_time*1e3:.3f} ms) exceed the usable period "
            f"({usable_period*1e3:.3f} ms = trigger period - {trigger_margin*1e3:.1f} ms margin)!\n"
            f"Reduce num_pund_repeats, pulse widths, gaps, or trigger_margin."
        )

    waveform = np.concatenate((
        pund_sequences,
        segment(0.0, zero_duration, arb_sample_rate_hz),
    ))

    if len(waveform) > maximum_arb_points:
        raise ValueError(
            f"Waveform has {len(waveform)} points; limit is {maximum_arb_points}\n"
            f"Reduce sample rate, number of repetitions, or zero padding."
        )

    total_duration_s = len(waveform) / arb_sample_rate_hz

    # NEW: volts -> AWG data in [-1, +1]; 0 V maps to data = -offset / half_span
    voltage_offset_v, max_amplitude = output_scaling()
    waveform = (waveform - voltage_offset_v) / max_amplitude

    # NEW: sanity check, waveform must finish before the next trigger
    if total_duration_s >= target_period:
        raise ValueError(
            f"Waveform length ({total_duration_s*1e3:.3f} ms) is not shorter than the "
            f"trigger period ({target_period*1e3:.3f} ms); triggers would be skipped."
        )

    return waveform, total_duration_s, total_pund_time, single_pund_duration, zero_duration, max_amplitude, voltage_offset_v


def plot_pund_waveform(waveform, total_duration_s, total_pund_time, single_pund_duration, zero_duration, max_amplitude, voltage_offset_v):
    """Plot the PUND waveform with two views: full cycle and zoomed on first PUND."""
    time_ms = np.arange(len(waveform)) / arb_sample_rate_hz * 1e3
    voltage = waveform * max_amplitude + voltage_offset_v  # Scale to actual voltage

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Full waveform view
    ax1.plot(time_ms, voltage, linewidth=1.5, color='blue')
    ax1.axvline(x=total_pund_time*1e3, color='red', linestyle='--',
                linewidth=2, label=f'PUND end / Zero section ({zero_duration*1e3:.2f} ms)')

    ax1.axhline(y=0.0, color='purple', linestyle='--', linewidth=1,
                label=f'Base / idle level (0 V, AWG offset {voltage_offset_v:+.2f} V)')

    # show the trigger margin region and the next trigger edge
    ax1.axvspan(total_duration_s*1e3, target_period*1e3, alpha=0.15, color='gray',
                label=f'Trigger margin ({trigger_margin*1e3:.1f} ms)')
    ax1.axvline(x=target_period*1e3, color='black', linestyle='-.', linewidth=2,
                label=f'Next trigger ({target_period*1e3:.2f} ms)')

    # Mark each PUND repetition
    for i in range(num_pund_repeats):
        pund_start = i * (single_pund_duration + int(round(gap_between_punds * arb_sample_rate_hz)) / arb_sample_rate_hz)
        pund_end = pund_start + single_pund_duration
        ax1.axvspan(pund_start*1e3, pund_end*1e3, alpha=0.1, color='green')
        ax1.text((pund_start + single_pund_duration/2)*1e3, max_amplitude*0.5,
                 f'PUND {i+1}', ha='center', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Voltage (V)', fontsize=12)
    ax1.set_title(f'Full Trigger Period: {len(waveform)} points @ {arb_sample_rate_hz/1e6:.1f} MS/s | '
                  f'{num_pund_repeats} PUND(s) per trigger',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.legend(loc='lower right')
    ax1.set_xlim([0, target_period * 1e3 * 1.02])  # CHANGED: show up to the next trigger

    # Zoomed view of first PUND
    first_pund_points = int(round(single_pund_duration * arb_sample_rate_hz))
    first_pund_points = min(first_pund_points + 20, len(waveform))

    time_us = np.arange(first_pund_points) / arb_sample_rate_hz * 1e6

    ax2.plot(time_us, voltage[:first_pund_points], linewidth=2, color='green',
             marker='o', markersize=3, markevery=max(1, first_pund_points//50))
    ax2.set_xlabel('Time (µs)', fontsize=12)
    ax2.set_ylabel('Voltage (V)', fontsize=12)
    ax2.set_title(f'Zoomed: First PUND Sequence ({single_pund_duration*1e9:.1f} ns, {first_pund_points} points)',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.0, color='purple', linestyle='--', linewidth=1,
                label='Base / idle level (0 V)')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    # Pulse center timing
    print("\n=== Pulse Timing Calculation ===")
    t = 0.0
    t += init_hold

    t += reset_rise_time
    reset_center = t + reset_width / 2
    t += reset_width + reset_rise_time
    t += gap_after_reset

    t += pos_rise_time
    pos1_center = t + pos_width / 2
    t += pos_width + pos_rise_time
    t += gap_between_pos

    t += pos_rise_time
    pos2_center = t + pos_width / 2
    t += pos_width + pos_rise_time
    t += gap_pos_to_neg

    t += neg_rise_time
    neg1_center = t + neg_width / 2
    t += neg_width + neg_rise_time
    t += gap_between_neg

    t += neg_rise_time
    neg2_center = t + neg_width / 2
    t += neg_width + neg_rise_time

    t += final_hold

    print(f"Total time calculated: {t*1e9:.1f} ns ({t*1e6:.3f} µs)")
    print(f"Expected PUND duration: {single_pund_duration*1e9:.1f} ns ({single_pund_duration*1e6:.3f} µs)")
    if abs(single_pund_duration - t) < 1e-12:
        print("✓ Times match!")
    else:
        # durations were rounded to the sample grid; centres below are nominal
        print(f"⚠ Times don't match (difference {abs(single_pund_duration - t)*1e9:.1f} ns)"
              f" - some durations are not multiples of {1e9/arb_sample_rate_hz:.2f} ns")

    print(f"\nPulse Center Times (µs):")
    print(f"  Reset:  {reset_center*1e6:.3f}")
    print(f"  Pos1:   {pos1_center*1e6:.3f}")
    print(f"  Pos2:   {pos2_center*1e6:.3f}")
    print(f"  Neg1:   {neg1_center*1e6:.3f}")
    print(f"  Neg2:   {neg2_center*1e6:.3f}")
    print("="*50)

    label_offset = max(abs(reset_amplitude), abs(pos_amplitude), abs(neg_amplitude)) * 0.25

    annotations = [
        (reset_center, reset_amplitude - label_offset, 'Reset', 'red', 'lightcoral'),
        (pos1_center,  pos_amplitude  + label_offset, 'P', 'green', 'lightgreen'),
        (pos2_center,  pos_amplitude  + label_offset, 'U', 'green', 'lightgreen'),
        (neg1_center,  neg_amplitude  - label_offset, 'N', 'orange', 'lightyellow'),
        (neg2_center,  neg_amplitude  - label_offset, 'D', 'orange', 'lightyellow'),
    ]
    for center, y, label, color, face in annotations:
        ax2.axvline(x=center*1e6, color=color, linestyle=':', linewidth=1.5, alpha=0.7)
        ax2.text(center*1e6, y, label, fontsize=9, ha='center', color=color, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor=face, alpha=0.7))

    ax2.set_xlim([-0.025, time_us[-1]])  # CHANGED: -25 ns expressed in µs
    ax2.legend(loc='lower right')        # FIX: legend was never drawn

    plt.tight_layout()
    plt.show()


def check_errors(awg, context):
    """CHANGED: Check for AWG errors and raise if any are found."""
    errors = []
    while True:
        response = awg.query("SYST:ERR?").strip()
        code_str = response.split(",", 1)[0]
        if int(code_str) == 0:
            break
        errors.append(response)
    if errors:
        raise RuntimeError(f"AWG error(s) after '{context}':\n  " + "\n  ".join(errors))


def setup_awg(waveform, max_amplitude, voltage_offset_v):
    """Upload waveform to AWG and configure triggered burst output."""
    limit_v = 2 * max_allowed_amplitude_v if awg_output_load.upper().startswith("INF") else max_allowed_amplitude_v
    if max_amplitude <= 0 or max_amplitude > limit_v:
        raise ValueError(f"Amplitude {max_amplitude}V is outside the allowed range (0 to {limit_v}V)")

    # Check offset + peak must stay within the output range
    if abs(voltage_offset_v) + max_amplitude > limit_v:
        raise ValueError(
            f"|Offset| ({abs(voltage_offset_v):.3f} V) + peak amplitude ({max_amplitude:.3f} V) "
            f"= {abs(voltage_offset_v) + max_amplitude:.3f} V exceeds the {limit_v} V limit"
        )
    
    arb_name = "PUND_BURST"   # Must be 12 characters or fewer
    if len(arb_name) > 12:
        raise ValueError(f"Arb name '{arb_name}' is longer than 12 characters")

    rm = pyvisa.ResourceManager()
    awg = rm.open_resource(awg_address)
    awg.timeout = awg_timeout_ms
    awg.write_termination = "\n"
    awg.read_termination = "\n"

    try:
        print("\nConfiguring AWG...")
        awg.write("*RST")
        awg.write("*CLS")
        awg.query("*OPC?")  # NEW: wait for the reset to finish

        awg.write("SOURCE:DATA:VOLATILE:CLEAR")
        check_errors(awg, "SOURCE:DATA:VOLATILE:CLEAR")

        # NEW: explicit byte order to match is_big_endian=True
        awg.write("FORM:BORD NORM")
        check_errors(awg, "FORM:BORD NORM")

        print(f"Uploading waveform ({len(waveform)} points) via binary transfer...")
        awg.write_binary_values(
            f"SOURCE:DATA:ARBITRARY {arb_name},",
            waveform,
            datatype="f",
            is_big_endian=True,
        )
        awg.query("*OPC?")  # CHANGED: block until upload is processed
        check_errors(awg, f"SOURCE:DATA:ARBITRARY {arb_name} (upload)")

        awg.write(f"FUNC:ARB {arb_name}")
        check_errors(awg, f"FUNC:ARB {arb_name}")

        awg.write("FUNC ARB")
        check_errors(awg, "FUNC ARB")

        awg.write(f"FUNC:ARB:SRATE {arb_sample_rate_hz}")
        check_errors(awg, f"FUNC:ARB:SRATE {arb_sample_rate_hz}")

        awg.write(f"FUNC:ARB:FILT {arb_filter}")
        check_errors(awg, f"FUNC:ARB:FILT {arb_filter}")

        # NEW: set output load BEFORE amplitude
        awg.write(f"OUTP:LOAD {awg_output_load}")
        check_errors(awg, f"OUTP:LOAD {awg_output_load}")

        awg.write(f"VOLT {2.0 * max_amplitude}")
        check_errors(awg, f"VOLT {2.0 * max_amplitude}")

        awg.write(f"VOLT:OFFS {voltage_offset_v}")          # FIX: was missing the f-prefix
        check_errors(awg, f"VOLT:OFFS {voltage_offset_v}")

        # Burst / external trigger configuration ---
        awg.write("BURS:MODE TRIG")          # triggered (N-cycle), not gated
        check_errors(awg, "BURS:MODE TRIG")

        awg.write("BURS:NCYC 1")             # one pass through the arb per trigger
        check_errors(awg, "BURS:NCYC 1")

        awg.write("TRIG:SOUR EXT")           # rear-panel Ext Trig input
        check_errors(awg, "TRIG:SOUR EXT")

        awg.write(f"TRIG:SLOP {trigger_slope}")
        check_errors(awg, f"TRIG:SLOP {trigger_slope}")

        awg.write(f"TRIG:DEL {trigger_delay_s}")
        check_errors(awg, f"TRIG:DEL {trigger_delay_s}")

        awg.write("BURS:STAT ON")
        check_errors(awg, "BURS:STAT ON")

        # Enable output last
        awg.write("OUTP ON")
        check_errors(awg, "OUTP ON")

        print("✓ Waveform uploaded, triggered burst mode enabled, output ON")
        print(f"  Waiting for external trigger ({trigger_slope} edge) on rear Ext Trig input")
    except Exception:
        awg.write("OUTP OFF")  # NEW: never leave a half-configured output on
        raise
    finally:
        awg.close()
        rm.close()


def turn_off_awg():
    """Turn off AWG output."""
    rm = pyvisa.ResourceManager()
    awg = rm.open_resource(awg_address)
    awg.timeout = awg_timeout_ms
    try:
        awg.write("OUTP OFF")
        print("AWG output turned OFF.")
    finally:
        awg.close()
        rm.close()


def main():
    """Main function to build, plot, and/or upload PUND burst waveform."""
    print("="*60)
    print("PUND Burst Waveform Generator (Binary Upload, Ext Triggered)")
    print("="*60)

    try:
        waveform, total_duration_s, total_pund_time, single_pund_duration, zero_duration, max_amplitude, voltage_offset_v = build_pund_burst_waveform()
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        return

    print(f"\nWaveform Configuration:")
    print(f"  AWG offset: {voltage_offset_v:+.3f} V (auto, keeps the base at 0 V)")
    print(f"  Output range: {voltage_offset_v - max_amplitude:.3f} V to {voltage_offset_v + max_amplitude:.3f} V")
    print(f"  Base (0 V) data value: {-voltage_offset_v / max_amplitude:+.4f}")
    print(f"  Reset pulse: {reset_amplitude:.2f} V, {reset_width*1e6:.3f} µs width, {reset_rise_time*1e9:.1f} ns rise/fall")
    print(f"  Positive pulses: {pos_amplitude:.2f} V, {pos_width*1e6:.3f} µs width, {pos_rise_time*1e9:.1f} ns rise/fall")
    print(f"  Negative pulses: {neg_amplitude:.2f} V, {neg_width*1e6:.3f} µs width, {neg_rise_time*1e9:.1f} ns rise/fall")
    print(f"\nGaps:")
    print(f"  After reset: {gap_after_reset*1e6:.3f} µs")
    print(f"  Between positive pulses: {gap_between_pos*1e6:.3f} µs")
    print(f"  Positive to negative: {gap_pos_to_neg*1e6:.3f} µs")
    print(f"  Between negative pulses: {gap_between_neg*1e6:.3f} µs")
    if num_pund_repeats > 1:
        print(f"  Between PUND sequences: {gap_between_punds*1e6:.1f} µs")
    print(f"\nHolds:")
    print(f"  Initial hold: {init_hold*1e9:.1f} ns")
    print(f"  Final hold: {final_hold*1e9:.1f} ns")
    print(f"\nRepetition / Triggering:")
    print(f"  PUNDs per trigger: {num_pund_repeats}")
    print(f"  Single PUND duration: {single_pund_duration*1e6:.3f} µs")
    print(f"  Total PUND time: {total_pund_time*1e3:.4f} ms")
    print(f"  Zero section (auto-calculated): {zero_duration*1e3:.3f} ms")
    print(f"  Trigger period: {target_period*1e3:.3f} ms ({1/target_period:.1f} Hz, external)")
    print(f"  Trigger margin: {trigger_margin*1e3:.3f} ms")
    print(f"\nWaveform Details:")
    print(f"  Waveform length: {total_duration_s*1e3:.3f} ms")   # CHANGED label
    print(f"  Slack before next trigger: {(target_period - total_duration_s)*1e3:.3f} ms")
    print(f"  Total points: {len(waveform)} / {maximum_arb_points}")
    print(f"  Memory used: {len(waveform)/maximum_arb_points*100:.1f}%")
    print(f"  Sample rate: {arb_sample_rate_hz/1e6:.1f} MS/s")
    print(f"  Repetition rate: set by external trigger ({1/target_period:.1f} Hz)")
    print(f"  Half-span (Vpp/2): {max_amplitude:.3f} V")
    print(f"  Vpp on AWG: {2.0*max_amplitude:.3f} V (load setting: {awg_output_load})")

    if SEND_TO_INSTRUMENT:
        try:
            setup_awg(waveform, max_amplitude, voltage_offset_v)
        except (RuntimeError, ValueError) as e:
            print(f"\n❌ AWG SETUP FAILED: {e}")
            print("   Check the output state on the front panel before connecting your sample.")
    else:
        print("\n(SEND_TO_INSTRUMENT = False, not uploading to AWG)")

    if SHOW_PLOT:
        print("\nGenerating plot...")
        plot_pund_waveform(waveform, total_duration_s, total_pund_time, single_pund_duration, zero_duration, max_amplitude, voltage_offset_v)

    print("="*60)


if __name__ == "__main__":
    main()