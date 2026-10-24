# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:23:26 2026

@author: leeau
"""

import numpy as np
import pyvisa
import matplotlib.pyplot as plt
from scipy import signal

# -----------------------------
# CONFIGURATION FLAGS
# -----------------------------
SEND_TO_INSTRUMENT = False  # Set to False to only plot, True to send to Agilent
SHOW_PLOT = True           # Set to True to display plot in Spyder

# -----------------------------
# User-editable Triangle Burst parameters
# -----------------------------
triangle_duration = 800e-6       # Duration of triangle wave section (s)
triangle_freq = 100000           # Triangle wave frequency (Hz)
triangle_amplitude = 1.0       # Peak amplitude (V)

# Cycle timing
target_period = 1/60          # 60 Hz = 16.67 ms
trigger_margin = 1e-3   # 1 ms of slack for latency, re-arm, and jitter

voltage_offset_v = 0.2 # 200 mV, DC offset (V) applied to the ENTIRE waveform, including the 0 V hold

# System parameters
arb_sample_rate_hz = 1e7  #If the peaks get shaved off, increase the sample rate; e.g. 1 MHz needs sample rate 6e7
maximum_arb_points = 1000000
max_allowed_amplitude_v = 5.0  # ±5 V peak (10 Vpp) is the 33511B limit into 50 Ω

# Output load setting.
#   "50"  -> true 50 Ω termination
#   "INF" -> high-impedance load (scope at 1 MΩ, etc.)
# If this is wrong, your voltages will be off by a factor of 2.
awg_output_load = "50"

#_address = "USB0::0x0957::0x2707::MY57301577::INSTR" # Keysight
awg_address = "USB0::2391::9991::MY52300442::0::INSTR" # Agilent
awg_timeout_ms = 30000


def segment(value, duration_s, sample_rate):
    """Create a flat segment at a given value."""
    """Creates the 0 V hold after the triangle waves."""
    if duration_s < 0:
        raise ValueError("Durations cannot be negative")
    if duration_s == 0:
        return np.empty(0, dtype=float)
    point_count = max(1, int(round(duration_s * sample_rate)))
    return np.full(point_count, value, dtype=float) #creates an array filled entirely with int in value variable


def build_triangle_burst_waveform():
    """
    Build triangle burst waveform:
    - Triangle wave for triangle_duration
    - Zero voltage for remainder of target_period (auto-calculated)
    Total repeats at 60 Hz.
    """
    
    # Calculate number of points for triangle section
    n_triangle = int(round(triangle_duration * arb_sample_rate_hz))
    
    # Generate time array for triangle section
    t_triangle = np.linspace(0, triangle_duration, n_triangle, endpoint=False)
    
    # Generate triangular wave using scipy
    # sawtooth with width=0.5 creates symmetric triangle wave
    triangle_wave = signal.sawtooth(2 * np.pi * triangle_freq * t_triangle + np.pi/2, width=0.5)
    #2 * np.pi * triangle_freq * t_triangle builds the phase argument (converts time into an angle in radians)
    # one full triangle cycle = 2 pi radians, so at time t the phase is 2*pi*triangle freq * t
    # sawtooth makes a ramp, the width parameter sets the symmetry point, 0 would be a reverse sawtooth.
    
    # Calculate zero duration to reach target period
    zero_duration = (target_period - trigger_margin) - triangle_duration
    
    if zero_duration < 0:
        raise ValueError(
            f"Triangle duration ({triangle_duration*1e3:.3f} ms) exceeds target period "
            f"({target_period*1e3:.3f} ms)!\n"
            f"Reduce triangle_duration."
        )
    
    # Create zero section
    zero_section = segment(0.0, zero_duration, arb_sample_rate_hz)
    
    # Build complete waveform
    waveform = np.concatenate([triangle_wave, zero_section])
    
    if len(waveform) > maximum_arb_points:
        raise ValueError(
            f"Waveform has {len(waveform)} points; limit is {maximum_arb_points}\n"
            f"Reduce sample rate or triangle duration."
        )
    
    total_duration_s = len(waveform) / arb_sample_rate_hz
    
    return waveform, total_duration_s, triangle_duration, zero_duration, triangle_amplitude


def plot_triangle_waveform(waveform, total_duration_s, triangle_duration, zero_duration, max_amplitude):
    """Plot the triangle burst waveform with two views: full cycle and zoomed."""
    time_ms = np.arange(len(waveform)) / arb_sample_rate_hz * 1e3 #converting to ms
    voltage = waveform * max_amplitude + voltage_offset_v  # Scale to actual voltage
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Full waveform view (entire 60 Hz cycle)
    ax1.plot(time_ms, voltage, linewidth=1.5, color='blue')
    ax1.axvline(x=triangle_duration*1e3, color='red', linestyle='--', 
               linewidth=2, label=f'Triangle end / Zero section ({zero_duration*1e3:.2f} ms)')
    ax1.axhline(y=voltage_offset_v, color='purple', linestyle='--', linewidth=1,
                label=f'Offset / idle level ({voltage_offset_v:.2f} V)')
    ax1.axvspan(0, triangle_duration*1e3, alpha=0.1, color='green') #colors the triangle section light green
    ax1.text(triangle_duration*1e3/2, max_amplitude*0.5, 
            f'Triangle Wave\n{triangle_freq} Hz', ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Voltage (V)', fontsize=12)
    ax1.set_title(f'Full 60 Hz Cycle: {len(waveform)} points @ {arb_sample_rate_hz/1e6:.1f} MS/s', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.legend()
    ax1.set_xlim([0, total_duration_s * 1e3])
    
    # Zoomed view of triangle section
    # Show first few cycles for detail
    zoom_duration = min(5 / triangle_freq, triangle_duration)  # Show ~5 cycles
    zoom_points = int(round(zoom_duration * arb_sample_rate_hz))
    zoom_points = min(zoom_points, len(waveform)) # min function returns whichever value is smaller of the two
    #this ensures that the zoom window never exceeds the actual triangle section length
    
    time_ms_zoom = np.arange(zoom_points) / arb_sample_rate_hz * 1e3
    
    ax2.plot(time_ms_zoom, voltage[:zoom_points], linewidth=2, color='green', 
             marker='o', markersize=2, markevery=max(1, zoom_points//100))
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Voltage (V)', fontsize=12)
    
    num_cycles_shown = triangle_freq * zoom_duration
    ax2.set_title(f'Zoomed: First {num_cycles_shown:.1f} Cycles of {triangle_freq} Hz Triangle Wave', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.set_xlim([0, zoom_duration * 1e3])
    
    # Add annotation showing one period
    if zoom_points > arb_sample_rate_hz / triangle_freq:
        period_ms = (1 / triangle_freq) * 1e3
        ax2.annotate('', xy=(period_ms, max_amplitude*0.8), xytext=(0, max_amplitude*0.8),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax2.text(period_ms/2, max_amplitude*0.9, f'Period = {1/triangle_freq*1e6:.1f} µs', 
                ha='center', fontsize=9, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n=== Triangle Wave Details ===")
    print(f"Triangle frequency: {triangle_freq} Hz")
    print(f"Triangle period: {1/triangle_freq*1e6:.3f} µs")
    print(f"Number of complete cycles: {triangle_freq * triangle_duration:.1f}")
    print(f"Points per cycle: {arb_sample_rate_hz / triangle_freq:.1f}")
    print(f"Triangle duration: {triangle_duration*1e3:.3f} ms")
    print(f"Zero duration: {zero_duration*1e3:.3f} ms")
    print(f"="*50)


def check_errors(awg, context):
    """Check for errors on the AWG."""
    while True:
        response = awg.query("SYST:ERR?").strip()
        code_str = response.split(",", 1)[0]
        if int(code_str) == 0:
            break
        print(f"AWG error after '{context}': {response}")


def setup_awg(waveform, max_amplitude):
    """Upload waveform to AWG and configure output."""
    if max_amplitude <= 0 or max_amplitude > max_allowed_amplitude_v:
        raise ValueError(f"Amplitude {max_amplitude}V is outside the allowed range (0 to {max_allowed_amplitude_v}V)")

    # offset + peak must stay within the output range
    if abs(voltage_offset_v) + max_amplitude > max_allowed_amplitude_v:
        raise ValueError(
            f"|Offset| ({abs(voltage_offset_v):.3f} V) + peak amplitude ({max_amplitude:.3f} V) "
            f"= {abs(voltage_offset_v) + max_amplitude:.3f} V exceeds the {max_allowed_amplitude_v} V limit"
        )
        
    arb_name = "TRI_BURST"
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

        print(f"Uploading waveform ({len(waveform)} points) via binary transfer...")
        awg.write_binary_values(
            f"SOURCE:DATA:ARBITRARY {arb_name},",
            waveform,
            datatype="f",
            is_big_endian=True,
        )
        awg.write("*WAI")
        check_errors(awg, f"SOURCE:DATA:ARBITRARY {arb_name} (upload)")

        awg.write(f"FUNC:ARB {arb_name}")
        check_errors(awg, f"FUNC:ARB {arb_name}")

        awg.write("FUNC ARB")
        check_errors(awg, "FUNC ARB")

        awg.write(f"FUNC:ARB:SRATE {arb_sample_rate_hz}")
        check_errors(awg, f"FUNC:ARB:SRATE {arb_sample_rate_hz}")
        
        awg.write(f"FUNC:ARB:SRATE {arb_sample_rate_hz}")
        check_errors(awg, f"FUNC:ARB:SRATE {arb_sample_rate_hz}")

        awg.write(f"OUTP:LOAD {awg_output_load}")      # NEW: set before amplitude
        check_errors(awg, f"OUTP:LOAD {awg_output_load}")

        awg.write(f"VOLT {2.0 * max_amplitude}")
        check_errors(awg, f"VOLT {2.0 * max_amplitude}")

        awg.write("VOLT:OFFS {voltage_offset_v}")
        check_errors(awg, "VOLT:OFFS {voltage_offset_v}")

        
        # --- Burst / external trigger configuration ---
        awg.write("BURS:MODE TRIG")      # triggered (N-cycle), not gated
        check_errors(awg, "BURS:MODE TRIG")

        awg.write("BURS:NCYC 1")         # one pass through the arb per trigger
        check_errors(awg, "BURS:NCYC 1")

        awg.write("TRIG:SOUR EXT")       # rear-panel Ext Trig input
        check_errors(awg, "TRIG:SOUR EXT")

        awg.write("TRIG:SLOP POS")       # rising edge (use NEG for falling)
        check_errors(awg, "TRIG:SLOP POS")

        # Optional: delay the output relative to the trigger edge
        # awg.write("TRIG:DEL 0")

        awg.write("BURS:STAT ON")
        check_errors(awg, "BURS:STAT ON")

        awg.write("OUTP ON")
        check_errors(awg, "OUTP ON")
        
        print("✓ Waveform uploaded and AWG output enabled")
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
    """Main function to build, plot, and/or upload triangle burst waveform."""
    print("="*60)
    print("Triangle Burst Waveform Generator (Binary Upload)")
    print("="*60)
    
    # Build waveform
    try:
        waveform, total_duration_s, triangle_duration, zero_duration, max_amplitude = build_triangle_burst_waveform()
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        return
    
    # Print configuration
    print(f"\nWaveform Configuration:")
    print(f"  Triangle frequency: {triangle_freq} Hz")
    print(f"  Triangle amplitude: ±{triangle_amplitude:.2f} V")
    print(f"  Vpp on AWG: {2.0*max_amplitude:.3f} V (load setting: {awg_output_load} Ω)")
    print(f"  DC offset: {voltage_offset_v:.3f} V (idle level between bursts)")
    print(f"  Output range: {voltage_offset_v - max_amplitude:.3f} V to {voltage_offset_v + max_amplitude:.3f} V")
    print(f"  Triangle duration: {triangle_duration*1e3:.3f} ms")
    print(f"  Number of cycles: {triangle_freq * triangle_duration:.1f}")
    print(f"  Zero section (auto-calculated): {zero_duration*1e3:.3f} ms")
    print(f"  Target period (60 Hz): {target_period*1e3:.3f} ms")
    print(f"\nWaveform Details:")
    print(f"  Total duration: {total_duration_s*1e3:.3f} ms")
    print(f"  Total points: {len(waveform)} / {maximum_arb_points}")
    print(f"  Memory used: {len(waveform)/maximum_arb_points*100:.1f}%")
    print(f"  Sample rate: {arb_sample_rate_hz/1e6:.1f} MS/s")
    print(f"  Points per triangle cycle: {arb_sample_rate_hz/triangle_freq:.1f}")
    print(f"  Actual repetition rate: {1/total_duration_s:.3f} Hz")
    print(f"  Vpp on AWG: {2.0*max_amplitude:.3f} V")
    
    # Plot if requested
    if SHOW_PLOT:
        print("\nGenerating plot...")
        plot_triangle_waveform(waveform, total_duration_s, triangle_duration, zero_duration, max_amplitude)
    
    # Send to instrument if requested
    if SEND_TO_INSTRUMENT:
        setup_awg(waveform, max_amplitude)
    else:
        print("\n(SEND_TO_INSTRUMENT = False, not uploading to AWG)")
    
    print("="*60)


if __name__ == "__main__":
    main()