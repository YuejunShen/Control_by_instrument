# -*- coding: utf-8 -*-
"""
Reset pulse + triangle burst for the Keysight 33511B (external trigger, binary upload).

Based on binary_AL_burst_Triangle_latency.py (leeau, 2026-09-24).

Sequence per trigger:
    0 V hold -> reset pulse -> 0 V delay -> triangle burst -> 0 V hold

Amplitude/offset (same convention as binary_AL_arbitrary_burst_PUND_latency_v2.py):
  The 33511B outputs  V = offset + data * Vpp/2  for arb data in [-1, +1]; it does
  not rescale the data. The waveform is therefore built in VOLTS with the base at
  0 V, and output_scaling() picks offset and Vpp from the lowest/highest levels.
  The base comes out at exactly 0 V, the reset and triangle peaks land exactly on
  the amplitudes below, and the full DAC range is used. No hand-set DC offset.
  (The original script added a 0.2 V offset to the whole waveform and sent
  VOLT:OFFS without the f-prefix.)
"""

import numpy as np
import pyvisa
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURATION FLAGS
# -----------------------------
SEND_TO_INSTRUMENT = False  # Set to False to only plot, True to send to the 33511B
SHOW_PLOT = True            # Set to True to display plot in Spyder

# -----------------------------
# User-editable parameters
# -----------------------------
# Reset pulse (sign sets the reset direction)
reset_amplitude = -2.0         # V
reset_width = 1000e-9          # s, flat top
reset_rise_time = 20e-9        # s, rise = fall

# Delay from the END of the reset pulse (fall edge back at 0 V) to the START of the triangle
reset_to_triangle_delay = 500e-9   # s

# Triangle burst: starts at 0 V going up, symmetric about 0 V, ends at 0 V
triangle_amplitude = 1.0       # V, peak (output swings -A ... +A)
triangle_freq = 100000         # Hz
triangle_duration = 800e-6     # s, must be a whole number of triangle periods

# Holds
init_hold = 50e-9              # s, 0 V before the reset (first sample = idle level)
final_hold = 50e-9             # s, 0 V after the triangle

# Cycle timing
target_period = 1/60           # External trigger period (60 Hz = 16.67 ms)
trigger_margin = 1e-3          # 1 ms of slack for latency, re-arm, and jitter

# System parameters
arb_sample_rate_hz = 100e6     # 10 ns grid: 20 ns edges = 2 samples, 1000 points per 100 kHz cycle
maximum_arb_points = 1000000
max_allowed_amplitude_v = 5.0  # ±5 V peak (10 Vpp) is the 33511B limit into 50 Ω (±10 V with load INF)
arb_filter = "STEP"            # "STEP" = least overshoot/ringing on the reset edges ("NORM" is the default)

# Output load setting.
#   "INF" -> high-impedance load (scope at 1 MΩ, sample/DUT input, etc.)
#   "50"  -> true 50 Ω termination
# If this is wrong, your voltages will be off by a factor of 2.
awg_output_load = "50"

# Trigger settings
trigger_slope = "POS"          # "POS" = rising edge, "NEG" = falling edge
trigger_delay_s = 0.0          # Delay from trigger edge to output start

#awg_address = "USB0::0x0957::0x2707::MY57301577::INSTR"  # Keysight 33511B (other unit)
awg_address = "USB0::2391::9991::MY52300442::0::INSTR"    # 33511B used by the other beamtime scripts
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


def triangle(amplitude, freq_hz, duration_s, sample_rate):
    """Triangle starting at 0 V going up, peak ±amplitude, whole number of cycles."""
    n_cycles = duration_s * freq_hz
    if abs(n_cycles - round(n_cycles)) > 1e-6:
        raise ValueError(
            f"triangle_duration holds {n_cycles:.4f} cycles; use a whole number of "
            f"periods ({1/freq_hz*1e6:.3f} µs each) so the triangle ends at 0 V."
        )
    points_per_cycle = sample_rate / freq_hz
    if abs(points_per_cycle - round(points_per_cycle)) > 1e-6 or round(points_per_cycle) % 4:
        raise ValueError(
            f"{points_per_cycle:.3f} samples per triangle cycle; choose arb_sample_rate_hz so "
            f"this is a whole multiple of 4 (peaks and zero crossings land on samples)."
        )
    n_points = int(round(duration_s * sample_rate))
    t = np.arange(n_points) / sample_rate
    # (2/pi) arcsin(sin x) is an exact triangle: 0 at x=0, +1 at pi/2, -1 at 3pi/2
    return amplitude * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * freq_hz * t))


def output_scaling():
    """
    Choose AWG offset and half-span so the output window is exactly [v_low, v_high].

    The AWG outputs  V = offset + data * (Vpp / 2)  with data in [-1, +1].
    The window always includes 0 V, so the base is reachable and lands at exactly 0 V:
        data(V) = (V - offset) / half_span
    """
    v_high = max(0.0, reset_amplitude, abs(triangle_amplitude))
    v_low = min(0.0, reset_amplitude, -abs(triangle_amplitude))
    if v_high == v_low:
        raise ValueError("Reset and triangle amplitudes are both zero")
    voltage_offset_v = (v_high + v_low) / 2
    half_span = (v_high - v_low) / 2      # Vpp / 2
    return voltage_offset_v, half_span


def build_reset_triangle_waveform():
    """Build the waveform in volts (base 0 V), then convert it to AWG data in [-1, +1]."""
    parts = {
        "init_hold": segment(0.0, init_hold, arb_sample_rate_hz),
        "reset": pulse(reset_amplitude, reset_width, reset_rise_time, arb_sample_rate_hz),
        "delay": segment(0.0, reset_to_triangle_delay, arb_sample_rate_hz),
        "triangle": triangle(triangle_amplitude, triangle_freq, triangle_duration, arb_sample_rate_hz),
        "final_hold": segment(0.0, final_hold, arb_sample_rate_hz),
    }

    # Start/end time of each part from the samples actually generated
    timing, n = {}, 0
    for name, samples in parts.items():
        timing[name] = (n / arb_sample_rate_hz, (n + len(samples)) / arb_sample_rate_hz)
        n += len(samples)

    voltage = np.concatenate(list(parts.values()))

    if len(voltage) > maximum_arb_points:
        raise ValueError(
            f"Waveform has {len(voltage)} points; limit is {maximum_arb_points}\n"
            f"Reduce sample rate or triangle duration."
        )

    total_duration_s = len(voltage) / arb_sample_rate_hz
    usable_period = target_period - trigger_margin
    if total_duration_s > usable_period:
        raise ValueError(
            f"Waveform ({total_duration_s*1e3:.3f} ms) exceeds the usable period "
            f"({usable_period*1e3:.3f} ms = trigger period - {trigger_margin*1e3:.1f} ms margin)!\n"
            f"Reduce triangle_duration or the delay."
        )

    # volts -> AWG data in [-1, +1]; 0 V maps to data = -offset / half_span
    voltage_offset_v, half_span = output_scaling()
    waveform = (voltage - voltage_offset_v) / half_span
    if np.max(np.abs(waveform)) > 1 + 1e-9:
        raise ValueError("Internal error: scaled waveform exceeds [-1, +1]")
    waveform = np.clip(waveform, -1.0, 1.0)

    return waveform, voltage, timing, total_duration_s, half_span, voltage_offset_v


def plot_reset_triangle_waveform(voltage, timing, total_duration_s, half_span, voltage_offset_v):
    """Plot the full trigger period and a zoom on the reset -> triangle transition."""
    time_s = np.arange(len(voltage)) / arb_sample_rate_hz

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Full view up to the next trigger edge
    ax1.plot(time_s * 1e3, voltage, linewidth=1.0, color='blue')
    ax1.axhline(y=0.0, color='purple', linestyle='--', linewidth=1,
                label=f'Base / idle level (0 V, AWG offset {voltage_offset_v:+.2f} V)')
    tri_start, tri_end = timing["triangle"]
    ax1.axvspan(tri_start * 1e3, tri_end * 1e3, alpha=0.1, color='green',
                label=f'Triangle {triangle_freq/1e3:g} kHz, ±{triangle_amplitude:g} V')
    ax1.axvspan(total_duration_s * 1e3, target_period * 1e3, alpha=0.15, color='gray',
                label='Idle until next trigger')
    ax1.axvline(x=target_period * 1e3, color='black', linestyle='-.', linewidth=2,
                label=f'Next trigger ({target_period*1e3:.2f} ms)')
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Voltage (V)', fontsize=12)
    ax1.set_title(f'Full Trigger Period: {len(voltage)} points @ {arb_sample_rate_hz/1e6:.1f} MS/s',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    ax1.set_xlim([0, target_period * 1e3 * 1.02])

    # Zoom: reset pulse, delay, and the first few triangle cycles
    zoom_end = min(tri_start + 3 / triangle_freq, total_duration_s)
    m = time_s <= zoom_end
    ax2.plot(time_s[m] * 1e6, voltage[m], linewidth=2, color='green',
             marker='o', markersize=2, markevery=max(1, np.count_nonzero(m) // 200))
    ax2.axhline(y=0.0, color='purple', linestyle='--', linewidth=1, label='Base / idle level (0 V)')
    reset_start, reset_end = timing["reset"]
    ax2.axvspan(reset_start * 1e6, reset_end * 1e6, alpha=0.15, color='red',
                label=f'Reset {reset_amplitude:g} V, {reset_width*1e9:g} ns')
    ax2.annotate('', xy=(tri_start * 1e6, 0.5 * reset_amplitude),
                 xytext=(reset_end * 1e6, 0.5 * reset_amplitude),
                 arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax2.text((reset_end + tri_start) / 2 * 1e6, 0.6 * reset_amplitude,
             f'delay {reset_to_triangle_delay*1e9:g} ns', ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax2.set_xlabel('Time (µs)', fontsize=12)
    ax2.set_ylabel('Voltage (V)', fontsize=12)
    ax2.set_title('Zoomed: Reset -> Delay -> First Triangle Cycles', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')
    ax2.set_xlim([0, zoom_end * 1e6])

    plt.tight_layout()
    plt.show()


def check_errors(awg, context):
    """Check for AWG errors and raise if any are found."""
    errors = []
    while True:
        response = awg.query("SYST:ERR?").strip()
        code_str = response.split(",", 1)[0]
        if int(code_str) == 0:
            break
        errors.append(response)
    if errors:
        raise RuntimeError(f"AWG error(s) after '{context}':\n  " + "\n  ".join(errors))


def setup_awg(waveform, half_span, voltage_offset_v):
    """Upload waveform to AWG and configure triggered burst output."""
    limit_v = 2 * max_allowed_amplitude_v if awg_output_load.upper().startswith("INF") else max_allowed_amplitude_v
    if half_span <= 0 or half_span > limit_v:
        raise ValueError(f"Half-span {half_span}V is outside the allowed range (0 to {limit_v}V)")

    # offset + half-span must stay within the output range
    if abs(voltage_offset_v) + half_span > limit_v:
        raise ValueError(
            f"|Offset| ({abs(voltage_offset_v):.3f} V) + half-span ({half_span:.3f} V) "
            f"= {abs(voltage_offset_v) + half_span:.3f} V exceeds the {limit_v} V limit"
        )

    arb_name = "RST_TRI"   # Must be 12 characters or fewer
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
        awg.query("*OPC?")  # wait for the reset to finish

        awg.write("SOURCE:DATA:VOLATILE:CLEAR")
        check_errors(awg, "SOURCE:DATA:VOLATILE:CLEAR")

        # explicit byte order to match is_big_endian=True
        awg.write("FORM:BORD NORM")
        check_errors(awg, "FORM:BORD NORM")

        print(f"Uploading waveform ({len(waveform)} points) via binary transfer...")
        awg.write_binary_values(
            f"SOURCE:DATA:ARBITRARY {arb_name},",
            waveform,
            datatype="f",
            is_big_endian=True,
        )
        awg.query("*OPC?")  # block until upload is processed
        check_errors(awg, f"SOURCE:DATA:ARBITRARY {arb_name} (upload)")

        awg.write(f"FUNC:ARB {arb_name}")
        check_errors(awg, f"FUNC:ARB {arb_name}")

        awg.write("FUNC ARB")
        check_errors(awg, "FUNC ARB")

        awg.write(f"FUNC:ARB:SRATE {arb_sample_rate_hz}")
        check_errors(awg, f"FUNC:ARB:SRATE {arb_sample_rate_hz}")

        awg.write(f"FUNC:ARB:FILT {arb_filter}")
        check_errors(awg, f"FUNC:ARB:FILT {arb_filter}")

        # set output load BEFORE amplitude
        awg.write(f"OUTP:LOAD {awg_output_load}")
        check_errors(awg, f"OUTP:LOAD {awg_output_load}")

        awg.write(f"VOLT {2.0 * half_span}")              # Vpp
        check_errors(awg, f"VOLT {2.0 * half_span}")

        awg.write(f"VOLT:OFFS {voltage_offset_v}")        # auto offset, keeps the base at 0 V
        check_errors(awg, f"VOLT:OFFS {voltage_offset_v}")

        # Burst / external trigger configuration
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
        awg.write("OUTP OFF")  # never leave a half-configured output on
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
    """Build, plot, and/or upload the reset + triangle burst waveform."""
    print("="*60)
    print("Reset + Triangle Burst Generator (Keysight 33511B, Ext Triggered)")
    print("="*60)

    try:
        waveform, voltage, timing, total_duration_s, half_span, voltage_offset_v = build_reset_triangle_waveform()
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        return

    reset_start, reset_end = timing["reset"]
    tri_start, tri_end = timing["triangle"]
    print(f"\nWaveform Configuration:")
    print(f"  AWG offset: {voltage_offset_v:+.3f} V (auto, keeps the base at 0 V)")
    print(f"  Vpp on AWG: {2.0*half_span:.3f} V (load setting: {awg_output_load})")
    print(f"  Output range: {voltage_offset_v - half_span:.3f} V to {voltage_offset_v + half_span:.3f} V")
    print(f"  Base (0 V) data value: {-voltage_offset_v / half_span:+.4f}")
    print(f"  Reset pulse: {reset_amplitude:+.2f} V, {reset_width*1e9:.1f} ns width, "
          f"{reset_rise_time*1e9:.1f} ns rise/fall ({reset_start*1e6:.3f}-{reset_end*1e6:.3f} µs)")
    print(f"  Reset end -> triangle start: {(tri_start - reset_end)*1e9:.1f} ns")
    print(f"  Triangle: ±{triangle_amplitude:.2f} V, {triangle_freq/1e3:g} kHz, "
          f"{triangle_freq*triangle_duration:.0f} cycles ({tri_start*1e6:.3f}-{tri_end*1e6:.3f} µs)")
    print(f"\nWaveform Details:")
    print(f"  Waveform length: {total_duration_s*1e3:.3f} ms")
    print(f"  Slack before next trigger: {(target_period - total_duration_s)*1e3:.3f} ms "
          f"(margin required: {trigger_margin*1e3:.1f} ms)")
    print(f"  Total points: {len(waveform)} / {maximum_arb_points} "
          f"({len(waveform)/maximum_arb_points*100:.1f}% of memory)")
    print(f"  Sample rate: {arb_sample_rate_hz/1e6:.1f} MS/s, "
          f"{arb_sample_rate_hz/triangle_freq:.0f} points per triangle cycle")
    print(f"  Repetition rate: set by external trigger ({1/target_period:.1f} Hz)")

    if SEND_TO_INSTRUMENT:
        try:
            setup_awg(waveform, half_span, voltage_offset_v)
        except (RuntimeError, ValueError) as e:
            print(f"\n❌ AWG SETUP FAILED: {e}")
            print("   Check the output state on the front panel before connecting your sample.")
    else:
        print("\n(SEND_TO_INSTRUMENT = False, not uploading to AWG)")

    if SHOW_PLOT:
        print("\nGenerating plot...")
        plot_reset_triangle_waveform(voltage, timing, total_duration_s, half_span, voltage_offset_v)

    print("="*60)


if __name__ == "__main__":
    main()
