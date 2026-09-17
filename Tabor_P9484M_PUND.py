"""Upload and continuously output a PUND waveform from the Tabor P9484M.

Connection:
    P9484M AWG Channel 1 (DC output) -> X-20 high-impedance input

The requested amplitude is the peak voltage wanted at the sample. Because the
X-20 is inverting, the waveform uploaded to the P9484M is inverted so that the
sample receives a negative reset pulse followed by positive P and U pulses.

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
sync_marker_number = 1
sync_width_s = 1e-9

# Desired waveform at the sample:
sample_peak_voltage_v = 0.55
pulse_width_s = 2e-9
rise_time_s = 0.1e-9
gap_reset_to_positive_s = 1e-9
gap_positive_to_positive_s = 1e-9
baseline_before_s = 1e-9
baseline_after_s = 1e-9

sample_rate_hz = 9e9
# Apply no programmed marker delay. Marker byte 0 corresponds directly to
# waveform sample 0.
marker_coarse_delay_samples = 0
marker_fine_delay_s = 0.0
segment_granularity = 64
# Segment 1 is a Proteus fast segment. The supplied P9484M example uses a
# 128-point segment; the final length is still padded to the 64-point boundary.
minimum_segment_points = 128
maximum_segment_points = 1_000_000

# Voltage-chain parameters:
x20_gain = 1
generator_high_z_factor = 2.0
max_allowed_sample_peak_v = 0.6

# P9484M DC-output range, specified as Vpp into 50 ohms.
minimum_awg_vpp = 0.050
maximum_awg_vpp = 0.55


def constant_segment(value, duration_s):
    if duration_s < 0:
        raise ValueError("Waveform durations cannot be negative")
    if duration_s == 0:
        return np.empty(0, dtype=np.float64)
    count = max(1, int(round(duration_s * sample_rate_hz)))
    return np.full(count, value, dtype=np.float64)


def ramp(start_value, stop_value, duration_s):
    if duration_s < 0:
        raise ValueError("rise_time_s cannot be negative")
    if duration_s == 0:
        return np.empty(0, dtype=np.float64)
    count = max(2, int(round(duration_s * sample_rate_hz)))
    return np.linspace(start_value, stop_value, count, endpoint=False)


def pulse(polarity, flat_top_s):
    return np.concatenate(
        (
            ramp(0.0, polarity, rise_time_s),
            constant_segment(polarity, flat_top_s),
            ramp(polarity, 0.0, rise_time_s),
        )
    )


def build_sample_waveform():
    """Construct normalized voltage wanted at the sample."""
    if pulse_width_s <= 0:
        raise ValueError("pulse_width_s must be greater than zero")

    waveform = np.concatenate(
        (
            constant_segment(0.0, baseline_before_s),
            pulse(-1.0, 2.0 * pulse_width_s),  # Negative reset pulse
            constant_segment(0.0, gap_reset_to_positive_s),
            pulse(+1.0, pulse_width_s),        # P pulse
            constant_segment(0.0, gap_positive_to_positive_s),
            pulse(+1.0, pulse_width_s),        # U pulse
            constant_segment(0.0, baseline_after_s),
        )
    )

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
        waveform = np.pad(waveform, (0, padded_length - len(waveform)))

    return waveform


def choose_awg_amplitude():
    """Return the P9484M Vpp setting and required digital waveform scale."""
    if sample_peak_voltage_v <= 0:
        raise ValueError("sample_peak_voltage_v must be greater than zero")
    if sample_peak_voltage_v > max_allowed_sample_peak_v:
        raise ValueError(
            f"Requested sample peak {sample_peak_voltage_v:g} V exceeds the "
            f"{max_allowed_sample_peak_v:g} V safety limit"
        )

    # sample peak = (AWG Vpp / 2) * high-Z doubling * X-20 gain
    required_awg_vpp = (
        2.0 * sample_peak_voltage_v / (generator_high_z_factor * x20_gain)
    )

    # If the required setting is below the AWG analog minimum, use the minimum
    # analog range and reduce the uploaded waveform digitally.
    programmed_awg_vpp = max(required_awg_vpp, minimum_awg_vpp)
    if programmed_awg_vpp > maximum_awg_vpp:
        raise ValueError(
            f"Required P9484M output {programmed_awg_vpp:g} Vpp exceeds "
            f"its {maximum_awg_vpp:g} Vpp limit"
        )
        

    digital_scale = required_awg_vpp / programmed_awg_vpp
    return programmed_awg_vpp, digital_scale


def encode_for_dac(normalized_waveform):
    if np.any(normalized_waveform < -1.0) or np.any(normalized_waveform > 1.0):
        raise ValueError("Normalized waveform exceeds the DAC range")
    return np.rint((normalized_waveform + 1.0) * 32767.5).astype(np.uint16)


def build_sync_marker(waveform_point_count):
    """Build marker data with a rising edge at sample 0 of every segment.

    Above 2.5 GSa/s, each P9484M marker byte controls eight waveform samples.
    At lower rates in 16-bit mode, each state controls two samples and two
    consecutive 4-bit states are packed into a byte. Marker 1 occupies bit 0.
    """
    if sync_width_s <= 0:
        raise ValueError("sync_width_s must be greater than zero")
    if sample_rate_hz > 2.5e9:
        samples_per_state = 8
        if waveform_point_count % samples_per_state != 0:
            raise ValueError("High-rate waveform length must be divisible by eight")
        marker_state_count = waveform_point_count // samples_per_state
        high_state_count = max(
            1, int(round(sync_width_s * sample_rate_hz / samples_per_state))
        )
        high_state_count = min(high_state_count, marker_state_count)
        marker_bytes = np.zeros(marker_state_count, dtype=np.uint8)
        marker_bytes[:high_state_count] = 1
    else:
        samples_per_state = 2
        if waveform_point_count % 4 != 0:
            raise ValueError("Waveform length must be divisible by four for marker packing")
        marker_state_count = waveform_point_count // samples_per_state
        high_state_count = max(
            1, int(round(sync_width_s * sample_rate_hz / samples_per_state))
        )
        high_state_count = min(high_state_count, marker_state_count)
        marker_states = np.zeros(marker_state_count, dtype=np.uint8)
        marker_states[:high_state_count] = 1
        marker_bytes = marker_states[0::2] + (marker_states[1::2] << 4)

    marker_bytes = marker_bytes.astype(np.uint8)
    if not (marker_bytes[0] & 1):
        raise RuntimeError("Marker 1 is not high at waveform sample 0")
    if marker_bytes[-1] != 0:
        raise RuntimeError("Marker must be low before the next segment boundary")

    actual_width_s = high_state_count * samples_per_state / sample_rate_hz
    return marker_bytes, actual_width_s


def check_instrument_error(instrument, operation):
    response = instrument.send_scpi_query(":SYST:ERR?").strip()
    if not response.startswith("0"):
        raise RuntimeError(f"P9484M error after {operation}: {response}")


def upload_and_start(
    instrument,
    dac_waveform,
    marker_data,
    awg_vpp,
    marker_coarse_delay_samples,
    marker_fine_delay_s,
):
    def send(command):
        instrument.send_scpi_cmd(command)
        check_instrument_error(instrument, command)

    send("*CLS;*RST")

    # CH1: continuously repeating PUND waveform.
    send(f":INST:CHAN {awg_channel}")
    send(":SOUR:MODE DIR")
    send(f":FREQ:RAST {sample_rate_hz}")
    send(":TRAC:FORM U16")
    send(":TRAC:DEL:ALL")
    send(":INIT:CONT ON")
    send(f":TRAC:DEF {segment_number},{len(dac_waveform)}")
    send(f":TRAC:SEL {segment_number}")

    transfer_status = instrument.write_binary_data(":TRAC:DATA", dac_waveform)
    if transfer_status != 0:
        raise RuntimeError(f"Waveform transfer failed with code {transfer_status}")

    marker_status = instrument.write_binary_data(":MARK:DATA 0,", marker_data)
    if marker_status != 0:
        raise RuntimeError(f"Marker transfer failed with code {marker_status}")

    check_instrument_error(instrument, "waveform transfer")
    send(f":FUNC:MODE:SEGM {segment_number}")
    send(f":SOUR:VOLT {awg_vpp}")
    send(":SOUR:VOLT:OFFS 0")

    send(f":MARK:SEL {sync_marker_number}")
    send(f":MARK:DEL:COAR {marker_coarse_delay_samples}")
    send(f":MARK:DEL:FINE {marker_fine_delay_s}")
    # This installed marker module uses a fixed logic-output voltage; its
    # programmable PTOP/OFFS commands return SCPI 210 (not implemented).
    send(":MARK ON")
    send(":OUTP ON")


def main():
    if sample_rate_hz <= 0 or x20_gain <= 0 or generator_high_z_factor <= 0:
        raise ValueError("Sample rate and voltage conversion factors must be positive")

    sample_waveform = build_sample_waveform()
    awg_vpp, digital_scale = choose_awg_amplitude()

    # The X-20 is inverting, so invert the AWG waveform. After amplification,
    # the sample receives the polarity defined by sample_waveform.
    awg_waveform = -sample_waveform * digital_scale
    dac_waveform = encode_for_dac(awg_waveform)
    marker_data, actual_sync_width_s = build_sync_marker(len(dac_waveform))
    waveform_period_s = len(dac_waveform) / sample_rate_hz

    instrument = TEVisaInst(proteus_address)
    try:
        instrument.default_paranoia_level = 1
        identity = instrument.send_scpi_query("*IDN?").strip()
        if "P9484" not in identity.upper():
            raise RuntimeError(f"Connected instrument is not a P9484: {identity}")
        print("Connected to:", identity)

        upload_and_start(
            instrument,
            dac_waveform,
            marker_data,
            awg_vpp,
            marker_coarse_delay_samples,
            marker_fine_delay_s,
        )
        print(f"Waveform points: {len(dac_waveform)}")
        print(f"Waveform period: {waveform_period_s:g} s")
        print(f"Repetition rate: {1.0 / waveform_period_s:g} Hz")
        print(f"P9484M setting: {awg_vpp:g} Vpp into 50 ohms")
        print(f"Digital waveform scale: {digital_scale:g}")
        print(f"Requested sample voltage: {sample_peak_voltage_v:g} V peak")
        print(
            f"Marker {sync_marker_number} trigger: rising edge at segment sample 0 "
            f"with {marker_coarse_delay_samples / sample_rate_hz + marker_fine_delay_s:g} s "
            f"physical delay compensation, width {actual_sync_width_s:g} s, "
            "fixed hardware logic level"
        )
        input("Output is ON. Press Enter to turn it OFF... ")
    finally:
        try:
            instrument.send_scpi_cmd(f":INST:CHAN {awg_channel}")
            instrument.send_scpi_cmd(":MARK OFF")
            instrument.send_scpi_cmd(":OUTP OFF")
            print("P9484M output turned OFF.")
        finally:
            instrument.close_instrument()


if __name__ == "__main__":
    main()
