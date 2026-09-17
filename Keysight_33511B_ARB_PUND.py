import numpy as np
import pyvisa


# -----------------------------
# User-editable PUND parameters
# -----------------------------
zero_before_s = 0e-3            # Flat baseline at 0 V before the reset pulse
ramp_time_s = 10e-9             # Rise/fall time, shared by all pulses
pulse_length_s = 1e-3           # Flat-top duration of each positive (P/U) pulse
gap_reset_to_positive_s = 5e-3  # Gap between the reset pulse and the first positive pulse
gap_positive_to_positive_s = 3e-3  # Gap between the two positive pulses
zero_after_s = 5e-3             # Flat baseline at 0 V after the last pulse
amplitude_v = 0.8                # Peak pulse amplitude (V)

arb_sample_rate_hz = 20e6
maximum_arb_points = 1_000_000
max_allowed_amplitude_v = 1.5

awg_address = "USB0::0x0957::0x2707::MY57301577::INSTR"
awg_timeout_ms = 30000


def segment(value, duration_s):
    if duration_s < 0:
        raise ValueError("Durations cannot be negative")
    if duration_s == 0:
        return np.empty(0, dtype=float)
    point_count = max(1, int(round(duration_s * arb_sample_rate_hz)))
    return np.full(point_count, value, dtype=float)


def ramp(start_value, stop_value, duration_s):
    if duration_s < 0:
        raise ValueError("Ramp time cannot be negative")
    if duration_s == 0:
        return np.empty(0, dtype=float)
    point_count = max(2, int(round(duration_s * arb_sample_rate_hz)))
    return np.linspace(start_value, stop_value, point_count, endpoint=False)


def pulse(polarity, width_s):
    return np.concatenate((
        ramp(0.0, polarity, ramp_time_s),
        segment(polarity, width_s),
        ramp(polarity, 0.0, ramp_time_s),
    ))


def build_pund_waveform():
    waveform = np.concatenate((
        segment(0.0, zero_before_s),
        pulse(-1.0, 2 * pulse_length_s),
        segment(0.0, gap_reset_to_positive_s),
        pulse(1.0, pulse_length_s),
        segment(0.0, gap_positive_to_positive_s),
        pulse(1.0, pulse_length_s),
        segment(0.0, zero_after_s),
    ))
    if len(waveform) > maximum_arb_points:
        raise ValueError(f"Waveform has {len(waveform)} points; limit is {maximum_arb_points}")
    total_duration_s = len(waveform) / arb_sample_rate_hz
    return waveform, total_duration_s


def check_errors(awg, context):
    while True:
        response = awg.query("SYST:ERR?").strip()
        code_str = response.split(",", 1)[0]
        if int(code_str) == 0:
            break
        print(f"AWG error after '{context}': {response}")


def setup_awg(waveform, amplitude_v):
    amplitude_v = float(amplitude_v)
    if amplitude_v <= 0 or amplitude_v > max_allowed_amplitude_v:
        raise ValueError("Pulse amplitude is outside the allowed range")
    arb_name = "PUND1"
    rm = pyvisa.ResourceManager()
    awg = rm.open_resource(awg_address)
    awg.timeout = awg_timeout_ms
    awg.write_termination = "\n"
    awg.read_termination = "\n"
    try:
        awg.write("*RST")
        awg.write("*CLS")

        awg.write("SOURCE:DATA:VOLATILE:CLEAR")
        check_errors(awg, "SOURCE:DATA:VOLATILE:CLEAR")

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

        awg.write(f"VOLT {2.0 * amplitude_v}")
        check_errors(awg, f"VOLT {2.0 * amplitude_v}")

        awg.write("VOLT:OFFS 0")
        check_errors(awg, "VOLT:OFFS 0")

        awg.write("OUTP ON")
        check_errors(awg, "OUTP ON")
    finally:
        awg.close()
        rm.close()


def turn_off_awg():
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
    waveform, total_duration_s = build_pund_waveform()
    print(
        f"Generating PUND: zero_before={zero_before_s:g}s, ramp={ramp_time_s:g}s, "
        f"pulse_length={pulse_length_s:g}s, "
        f"gap_reset_to_positive={gap_reset_to_positive_s:g}s, "
        f"gap_positive_to_positive={gap_positive_to_positive_s:g}s, "
        f"zero_after={zero_after_s:g}s, amplitude={amplitude_v:g}V, period={total_duration_s:g}s"
    )
    setup_awg(waveform, amplitude_v)


if __name__ == "__main__":
    main()
