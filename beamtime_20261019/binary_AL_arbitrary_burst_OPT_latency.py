# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 09:39:30 2026

@author: leeau

Modified 2026-10-19 for beamtime_20261019 (based on
https://github.com/autlee17/AWG/blob/main/binary_AL_arbitrary_burst_PUND_latency.py):
the standard trapezoidal PUND is replaced by the Bayesian-optimized waveform
from BO_pund_rise_BTO_F9_1737_rise_points8_rep1 (evaluation 223 = BO step 193,
the lowest-Work protocol with delta_P >= P_target). Every modified block is
tagged "CHANGED (beamtime_20261019)"; everything else (triggered burst,
zero padding, trigger margin, binary upload) is unchanged.
"""

import numpy as np
import pyvisa
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURATION FLAGS
# -----------------------------
SEND_TO_INSTRUMENT = False  # Set to False to only plot, True to send to Agilent
SHOW_PLOT = True            # Set to True to display plot in Spyder

# ---------------------------------------------------------------------------
# CHANGED (beamtime_20261019): PUND parameters replaced by the BO-optimal
# waveform. Source run:
#   D:\Ferroelectric_BTO_optimization\20260921\rise_points_scan\
#   BO_pund_rise_BTO_F9_1737_rise_points8_rep1\bo_history_20260921_173722_480265.csv
#   evaluation 223 (BO step 193), device BTO_F9
#   Work = 1.6485e-11 J (Work_NORM = 0.834, i.e. 16.6% below the linear rise)
#   delta_P = 0.169057 C/m^2 >= P_target = 0.169042 C/m^2 (feasible)
#
# The sequence reproduces the BO measurement waveform (waveform.py in
# Ferroelectric_PUND_trigger_sync_BO_useful) in physical volts:
#   pre-baseline -> reset (linear ramps) -> gap1 -> P (optimized rise ->
#   flat top -> linear fall) -> gap2 -> U (identical optimized pulse) ->
#   post-baseline
# There are no N/D pulses. The optimized part is only the 1 us rise of the
# positive pulse(s): 8 knot voltages, linearly interpolated onto this
# script's sample grid.
# ---------------------------------------------------------------------------
# BO-optimal rise knots (commanded volts, V0 ... V7), evaluation 223
opt_rise_knots_v = np.array([
    0.3932343312209508,
    0.147990189262185,
    0.1849243462729037,
    0.2275742208138708,
    0.2288907221788988,
    0.2831974366257075,
    0.3525412100597109,
    0.5243521159906094,
])
opt_rise_time = 1e-6           # s, duration of the optimized rise
bo_sample_rate_hz = 50e6       # sample rate used during BO; sets the knot time grid

# Positive (optimized) pulse: flat top and linear fall, as during BO
pos_amplitude = 0.5            # V, flat-top level (independent of the last knot)
pos_width = 2e-6               # s, flat-top duration
pos_fall_time = 1e-6           # s

# Reset pulse (negative): ramps of reset_rise_time, flat for 2 * pos_width
reset_amplitude = -0.5         # V
reset_width = 2 * pos_width    # s (4 us, as in BO)
reset_rise_time = 1e-6         # s

# Gaps / baselines, as during BO
pre_baseline = 2e-6            # s
gap_after_reset = 1e-6         # s (BO gap1)
gap_between_pos = 1e-6         # s (BO gap2)
post_baseline = 2e-6           # s

# True: P + U (identical optimized pulses, exactly as measured during BO).
# False: reset + a single optimized switching pulse only.
include_second_pulse = True

# Repetition control
num_pund_repeats = 5           # Number of times the optimized sequence repeats per trigger
target_period = 1/60           # External trigger period (60 Hz = 16.67 ms)
trigger_margin = 1e-3          # NEW: 1 ms of slack for latency, re-arm, and jitter

# Gap between sequence repetitions (if multiple sequences per trigger)
gap_between_punds = 100e-6     # s (only used if num_pund_repeats > 1)

# CHANGED (beamtime_20261019): was 0.2 V. The BO waveform idled at 0 V, and a
# nonzero offset would shift every voltage seen by the device (reset, rise
# knots and flat top), so the optimized protocol would no longer apply.
voltage_offset_v = 0.0

# System parameters
arb_sample_rate_hz = 6.2e7 # sample rate for short pulses above
maximum_arb_points = 1000000
max_allowed_amplitude_v = 5.0  # ±5 V peak (10 Vpp) is the 33511B limit into 50 Ω

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


# CHANGED (beamtime_20261019): new function. Interpolates the BO knots onto
# this script's sample grid.
def optimized_rise(sample_rate):
    """Linearly interpolate the BO rise knots onto the AWG sample grid.

    Knot times match waveform.build_shaped_rise in the BO code: the knots are
    evenly spaced from the first to the last *BO* sample of the rise, i.e.
    0 ... opt_rise_time - 1/bo_sample_rate_hz. Samples after the last knot
    hold its value, so the continuous shape is the same as during BO,
    independent of arb_sample_rate_hz.
    """
    point_count = int(round(opt_rise_time * sample_rate))
    sample_t = np.arange(point_count) / sample_rate
    knot_t = np.linspace(0.0, opt_rise_time - 1.0 / bo_sample_rate_hz, opt_rise_knots_v.size)
    return np.interp(sample_t, knot_t, opt_rise_knots_v)


# CHANGED (beamtime_20261019): new function (replaces the symmetric pulse()
# for the positive pulses).
def optimized_pulse(sample_rate):
    """Optimized rise -> flat top at pos_amplitude -> linear fall to 0 V.

    As in the BO code, the rise is concatenated directly into the flat top
    with no forced continuity (last knot 0.524 V, flat top 0.5 V).
    """
    return np.concatenate((
        optimized_rise(sample_rate),
        segment(pos_amplitude, pos_width, sample_rate),
        ramp(pos_amplitude, 0.0, pos_fall_time, sample_rate),
    ))


# CHANGED (beamtime_20261019): body rewritten. Builds the BO-optimal sequence
# instead of P-U-N-D (name kept so the rest of the script is unchanged).
def build_single_pund():
    """Build a single optimized sequence (without zero padding)."""
    positive_pulse = optimized_pulse(arb_sample_rate_hz)
    parts = [
        segment(0.0, pre_baseline, arb_sample_rate_hz),                       # Pre-baseline
        pulse(reset_amplitude, reset_width, reset_rise_time, arb_sample_rate_hz),  # Reset
        segment(0.0, gap_after_reset, arb_sample_rate_hz),
        positive_pulse,                                                       # P (optimized)
    ]
    if include_second_pulse:
        parts += [
            segment(0.0, gap_between_pos, arb_sample_rate_hz),
            positive_pulse,                                                   # U (optimized)
        ]
    parts.append(segment(0.0, post_baseline, arb_sample_rate_hz))             # Post-baseline
    single_pund_v = np.concatenate(parts)

    # Normalize to the -1 to +1 range. The rise overshoots the flat top
    # (0.524 V > 0.5 V), so use the waveform's actual peak, not the
    # pulse amplitudes.
    max_amplitude = float(np.max(np.abs(single_pund_v)))
    single_pund = single_pund_v / max_amplitude

    positive_pulse_duration = opt_rise_time + pos_width + pos_fall_time
    single_pund_duration = (pre_baseline +
                            2*reset_rise_time + reset_width + gap_after_reset +
                            positive_pulse_duration +
                            (gap_between_pos + positive_pulse_duration if include_second_pulse else 0.0) +
                            post_baseline)

    return single_pund, single_pund_duration, max_amplitude


def build_pund_burst_waveform():
    """
    Build PUND burst waveform:
    Repeats PUND num_pund_repeats times, then pads with 0 V so the waveform
    finishes trigger_margin BEFORE the next trigger edge arrives.
    """
    single_pund, single_pund_duration, max_amplitude = build_single_pund()

    if num_pund_repeats == 1:
        pund_sequences = single_pund
        total_pund_time = single_pund_duration
    else:
        pund_list = []
        for i in range(num_pund_repeats):
            pund_list.append(single_pund)
            if i < num_pund_repeats - 1:
                pund_list.append(segment(0.0, gap_between_punds, arb_sample_rate_hz))
        pund_sequences = np.concatenate(pund_list)
        total_pund_time = num_pund_repeats * single_pund_duration + (num_pund_repeats - 1) * gap_between_punds

    # CHANGED: leave trigger_margin of slack before the next trigger
    usable_period = target_period - trigger_margin
    if usable_period <= 0:
        raise ValueError(
            f"trigger_margin ({trigger_margin*1e3:.3f} ms) is larger than the "
            f"trigger period ({target_period*1e3:.3f} ms)!"
        )

    zero_duration = usable_period - total_pund_time

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

    # NEW: sanity check, waveform must finish before the next trigger
    if total_duration_s >= target_period:
        raise ValueError(
            f"Waveform length ({total_duration_s*1e3:.3f} ms) is not shorter than the "
            f"trigger period ({target_period*1e3:.3f} ms); triggers would be skipped."
        )

    return waveform, total_duration_s, total_pund_time, single_pund_duration, zero_duration, max_amplitude


def plot_pund_waveform(waveform, total_duration_s, total_pund_time, single_pund_duration, zero_duration, max_amplitude):
    """Plot the PUND waveform with two views: full cycle and zoomed on first PUND."""
    time_ms = np.arange(len(waveform)) / arb_sample_rate_hz * 1e3
    voltage = waveform * max_amplitude + voltage_offset_v  # Scale to actual voltage

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Full waveform view
    ax1.plot(time_ms, voltage, linewidth=1.5, color='blue')
    ax1.axvline(x=total_pund_time*1e3, color='red', linestyle='--',
                linewidth=2, label=f'PUND end / Zero section ({zero_duration*1e3:.2f} ms)')

    ax1.axhline(y=voltage_offset_v, color='purple', linestyle='--', linewidth=1,
                label=f'Offset / idle level ({voltage_offset_v:.2f} V)')

    # show the trigger margin region and the next trigger edge
    ax1.axvspan(total_duration_s*1e3, target_period*1e3, alpha=0.15, color='gray',
                label=f'Trigger margin ({trigger_margin*1e3:.1f} ms)')
    ax1.axvline(x=target_period*1e3, color='black', linestyle='-.', linewidth=2,
                label=f'Next trigger ({target_period*1e3:.2f} ms)')

    # Mark each PUND repetition
    for i in range(num_pund_repeats):
        pund_start = i * (single_pund_duration + gap_between_punds)
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
    ax2.set_title(f'Zoomed: First Optimized Sequence ({single_pund_duration*1e9:.1f} ns, {first_pund_points} points)',  # CHANGED (beamtime_20261019)
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=voltage_offset_v, color='purple', linestyle='--', linewidth=1,
                label=f'Offset / idle level ({voltage_offset_v:.2f} V)')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    # CHANGED (beamtime_20261019): timing follows the optimized sequence
    # (reset, P, optional U; no N/D). Also records each optimized rise onset.
    print("\n=== Pulse Timing Calculation ===")
    t = 0.0
    t += pre_baseline

    t += reset_rise_time
    reset_center = t + reset_width / 2
    t += reset_width + reset_rise_time
    t += gap_after_reset

    pos1_rise_start = t
    t += opt_rise_time
    pos1_center = t + pos_width / 2
    t += pos_width + pos_fall_time

    pos2_rise_start = pos2_center = None
    if include_second_pulse:
        t += gap_between_pos
        pos2_rise_start = t
        t += opt_rise_time
        pos2_center = t + pos_width / 2
        t += pos_width + pos_fall_time

    t += post_baseline

    print(f"Total time calculated: {t*1e9:.1f} ns ({t*1e6:.3f} µs)")
    print(f"Expected PUND duration: {single_pund_duration*1e9:.1f} ns ({single_pund_duration*1e6:.3f} µs)")
    if abs(single_pund_duration - t) < 1e-12:
        print("✓ Times match!")
    else:
        print(f"⚠ Times don't match (difference {abs(single_pund_duration - t)*1e9:.1f} ns)")

    # CHANGED (beamtime_20261019): reset / P / U only, plus optimized rise onsets
    print(f"\nPulse Center Times (µs):")
    print(f"  Reset:  {reset_center*1e6:.3f}")
    print(f"  P:      {pos1_center*1e6:.3f}  (optimized rise starts at {pos1_rise_start*1e6:.3f})")
    if include_second_pulse:
        print(f"  U:      {pos2_center*1e6:.3f}  (optimized rise starts at {pos2_rise_start*1e6:.3f})")
    print("="*50)

    label_offset = max_amplitude * 0.1

    annotations = [
        (reset_center, reset_amplitude + voltage_offset_v - label_offset, 'Reset', 'red', 'lightcoral'),
        (pos1_center,  pos_amplitude  + voltage_offset_v + label_offset, 'P (opt)', 'green', 'lightgreen'),
    ]
    if include_second_pulse:
        annotations.append(
            (pos2_center, pos_amplitude + voltage_offset_v + label_offset, 'U (opt)', 'green', 'lightgreen'))
    for rise_start in (pos1_rise_start, pos2_rise_start):
        if rise_start is not None:
            ax2.axvspan(rise_start*1e6, (rise_start + opt_rise_time)*1e6, alpha=0.15, color='green')
    for center, y, label, color, face in annotations:
        ax2.axvline(x=center*1e6, color=color, linestyle=':', linewidth=1.5, alpha=0.7)
        ax2.text(center*1e6, y, label, fontsize=9, ha='center', color=color, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor=face, alpha=0.7))

    ax2.set_xlim([-0.025, time_us[-1]])  # CHANGED: -25 ns expressed in µs

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


def setup_awg(waveform, max_amplitude):
    """Upload waveform to AWG and configure triggered burst output."""
    if max_amplitude <= 0 or max_amplitude > max_allowed_amplitude_v:
        raise ValueError(f"Amplitude {max_amplitude}V is outside the allowed range (0 to {max_allowed_amplitude_v}V)")

    # Check offset + peak must stay within the output range
    if abs(voltage_offset_v) + max_amplitude > max_allowed_amplitude_v:
        raise ValueError(
            f"|Offset| ({abs(voltage_offset_v):.3f} V) + peak amplitude ({max_amplitude:.3f} V) "
            f"= {abs(voltage_offset_v) + max_amplitude:.3f} V exceeds the {max_allowed_amplitude_v} V limit"
        )
    
    arb_name = "OPT_BURST"    # CHANGED (beamtime_20261019): was "PUND_BURST". Must be 12 characters or fewer
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

        # NEW: set output load BEFORE amplitude
        awg.write(f"OUTP:LOAD {awg_output_load}")
        check_errors(awg, f"OUTP:LOAD {awg_output_load}")

        awg.write(f"VOLT {2.0 * max_amplitude}")
        check_errors(awg, f"VOLT {2.0 * max_amplitude}")

        # CHANGED (beamtime_20261019): bug fix, the f-string prefix was missing,
        # so the literal text "{voltage_offset_v}" was sent to the AWG.
        awg.write(f"VOLT:OFFS {voltage_offset_v}")
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
    # CHANGED (beamtime_20261019): summary prints describe the optimized sequence
    print("BO-Optimized Burst Waveform Generator (Binary Upload, Ext Triggered)")
    print("="*60)

    try:
        waveform, total_duration_s, total_pund_time, single_pund_duration, zero_duration, max_amplitude = build_pund_burst_waveform()
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        return

    print(f"\nWaveform Configuration:")
    print(f"  DC offset: {voltage_offset_v:.3f} V (idle level between bursts)")
    print(f"  Output range: {voltage_offset_v - max_amplitude:.3f} V to {voltage_offset_v + max_amplitude:.3f} V")
    print(f"  Reset pulse: {reset_amplitude:.2f} V, {reset_width*1e6:.3f} µs width, {reset_rise_time*1e9:.1f} ns rise/fall")
    print(f"  Optimized rise: {opt_rise_knots_v.size} knots over {opt_rise_time*1e9:.1f} ns, "
          f"{int(round(opt_rise_time * arb_sample_rate_hz))} samples")
    print(f"    knots (V): " + ", ".join(f"{v:.4f}" for v in opt_rise_knots_v))
    print(f"  Positive pulse{'s (P, U)' if include_second_pulse else ' (P only)'}: flat top {pos_amplitude:.2f} V, "
          f"{pos_width*1e6:.3f} µs width, {pos_fall_time*1e9:.1f} ns fall")
    print(f"\nGaps / baselines:")
    print(f"  Pre-baseline: {pre_baseline*1e6:.3f} µs")
    print(f"  After reset: {gap_after_reset*1e6:.3f} µs")
    if include_second_pulse:
        print(f"  Between positive pulses: {gap_between_pos*1e6:.3f} µs")
    print(f"  Post-baseline: {post_baseline*1e6:.3f} µs")
    if num_pund_repeats > 1:
        print(f"  Between sequences: {gap_between_punds*1e6:.1f} µs")
    print(f"\nRepetition / Triggering:")
    print(f"  Sequences per trigger: {num_pund_repeats}")
    print(f"  Single sequence duration: {single_pund_duration*1e6:.3f} µs")
    print(f"  Total sequence time: {total_pund_time*1e3:.4f} ms")
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
    print(f"  Max amplitude: {max_amplitude:.3f} V")
    print(f"  Vpp on AWG: {2.0*max_amplitude:.3f} V (load setting: {awg_output_load})")

    if SHOW_PLOT:
        print("\nGenerating plot...")
        plot_pund_waveform(waveform, total_duration_s, total_pund_time, single_pund_duration, zero_duration, max_amplitude)

    if SEND_TO_INSTRUMENT:
        try:
            setup_awg(waveform, max_amplitude)
        except (RuntimeError, ValueError) as e:
            print(f"\n❌ AWG SETUP FAILED: {e}")
            print("   Check the output state on the front panel before connecting your sample.")
    else:
        print("\n(SEND_TO_INSTRUMENT = False, not uploading to AWG)")

    print("="*60)


if __name__ == "__main__":
    main()