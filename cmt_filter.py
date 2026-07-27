#!/usr/bin/env python3
"""MSX and MSX-like Cassette Magnetic Tape Audio Input and Output Shaping
======================================================================

1. Circuit Overview
-------------------
The Cassette Magnetic Tape (CMT) interface circuits in MSX and
MSX-like computers handle signal conversion between analog audio
signals on magnetic tape and digital logic signals inside the MSX
or MSX-like system.

2. Input Circuit Analysis: CMT-IN -> IOA7 (Fig. 5-5-9)
------------------------------------------------------
This circuit takes the raw, noisy analog audio signal coming from a cassette player
(CMT-IN / CN4-5) and converts it into a clean digital 0V / 5V square wave fed into
IOA7 (Bit 7 of I/O Port A on the AY-3-8910 / YM2149 PSG sound chip).

Component & Circuit Effects:
- High-Pass Filter (C31 = 0.1 uF, R21 = 2.7 kOhm):
  Cutoff frequency fc_hp = 1 / (2 * pi * R21 * C31) ~= 589.5 Hz.
  Blocks DC offsets and eliminates low-frequency power supply hum (50/60 Hz) and
  cassette motor rumble.

- Low-Pass Filter (R20 = 12 kOhm, C29 = 0.0015 uF / 1.5 nF):
  Cutoff frequency fc_lp = 1 / (2 * pi * R20 * C29) ~= 8841.9 Hz.
  Suppresses high-frequency tape hiss, noise spikes, and tape bias tones.

- Combined Bandpass Transfer Function H_in(s):
  H_in(s) = (C31 * R21 * s) / [ (C29 * C31 * R20 * R21) * s^2 + (C29*R20 + C29*R21 + C31*R21) * s + 1 ]
  Peak center frequency is tuned to ~2263 Hz, providing maximum passband gain
  right between the cassette carrier FSK tones (1200 Hz space / 2400 Hz mark).

- Diode Clamping (Diodes D1, D2):
  Anti-parallel diodes connected to ground clamp signal amplitude to ~ +/-0.6V
  to protect the comparator and normalize playback levels.

- Comparator with Hysteresis (R33 = 4.7 kOhm, JRC-311B):
  Acts as a zero-crossing Schmitt trigger comparator, converting the filtered
  AC waveform into a sharp 0V / 5V square wave for digital reading by IOA7.

3. Output Circuit Analysis: PC5 -> CMT OUT (Fig. 5-4-10)
--------------------------------------------------------
This circuit takes digital 0V / 5V pulse streams from PPI (8255) pin PC5, smooths
the sharp transitions into a band-limited wave, attenuates the voltage to
microphone/line level, and sends it to CN4-4 CMT OUT. It also includes a relay motor
control circuit driven by PC4 via driver 75451.

Component & Circuit Effects:
- Logic Inverter (74LS14 4B, R37 = 10 kOhm pull-up):
  Inverts logic state from PPI PC5 and provides clean driving capability.

- DC Blocking / High-Pass Filter (C31 = 3.3 uF):
  fc_hp ~= 8.0 Hz (with ~6 kOhm load impedance). Removes +2.5V DC offset,
  centering the output waveform around 0V AC.

- Low-Pass Waveform Shaping (R41 = 1.2 kOhm, C32 = 0.022 uF / 22 nF):
  fc_lp = 1 / (2 * pi * R41 * C32) ~= 6028.6 Hz.
  Rounds off sharp square-wave edges into a pseudo-sine wave, eliminating
  high harmonics that would saturate magnetic tape.

- Attenuator Voltage Divider (R40 = 4.7 kOhm, R39 = 100 Ohm):
  Attenuation factor = 100 / (4700 + 100) ~= 0.0208 (-33.6 dB).
  Reduces 5V peak-to-peak TTL output down to ~100 mV peak-to-peak MIC/LINE level.

- Combined Exact Transfer Function H_out(s):
  H_out(s) = (C31 * R39 * s) / [ (C31 * C32 * R41 * (R39 + R40)) * s^2 + (C31*(R39+R40+R41) + C32*(R39+R40)) * s + 1 ]

"""

import sys
import os
import wave
import argparse
import unittest
import numpy as np
import scipy.signal as signal


class CMTAudioOutputFilterExact:
    """
    Streaming filter modeling cassette output circuit (Fig. 5-4-10).
    Exact transfer function derived from circuit components:
    H(s) = (C31*R39*s) / [ (C31*C32*R41*(R39+R40))*s^2 + (C31*(R39+R40+R41) + C32*(R39+R40))*s + 1 ]

    Voltage-to-WAV-scale mapping:
    The raw H(s) above reflects the true physical attenuation of this circuit
    (it steps a 0/5V logic swing down to ~17-52 mV mic level, i.e. its DC-referenced
    attenuator ratio is R39/(R39+R40) ~= -33.6 dB, and its actual frequency-response
    peak magnitude is smaller still). If that raw response were written straight to a
    16-bit WAV, "full scale" logic input would only ever reach a few percent of the
    file's dynamic range.

    To make the WAV's full scale (+/-1.0, i.e. +/-32767 in int16) represent the true
    electrical peak of the CMT OUT line, this class finds the peak magnitude of
    |H(jw)| across all frequencies (the frequency at which the circuit outputs its
    largest possible voltage for a given input amplitude) and applies a fixed makeup
    gain so that a full-scale input driven at that frequency produces a full-scale
    (+/-1.0) output. Every other frequency's output is scaled by the same fixed
    factor, so relative amplitudes -- and thus the circuit's real bandpass shape --
    are preserved; only the reference point changes from "raw volts" to "this
    circuit's own electrical peak = full scale".
    """

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

        # Circuit component values
        C31, R41, C32, R40, R39 = 3.3e-6, 1200.0, 0.022e-6, 4700.0, 100.0

        b_num = [C31 * R39, 0.0]
        a_den = [
            C31 * C32 * R41 * (R39 + R40),
            C31 * (R39 + R40 + R41) + C32 * (R39 + R40),
            1.0,
        ]

        # Find the true peak of |H(jw)| across all frequencies (continuous-time,
        # before discretization) -- this is the circuit's actual electrical peak
        # response, not just the DC-referenced attenuator ratio.
        freqs_hz = np.logspace(
            0, 6, 20000
        )  # 1 Hz .. 1 MHz, comfortably covers the passband
        _, h = signal.freqs(b_num, a_den, worN=2 * np.pi * freqs_hz)
        mag = np.abs(h)
        peak_idx = np.argmax(mag)
        self.peak_gain = mag[peak_idx]
        self.peak_freq_hz = freqs_hz[peak_idx]
        self.makeup_gain = 1.0 / self.peak_gain
        self.makeup_gain_db = 20 * np.log10(self.makeup_gain)

        # Discretize via bilinear transform, then fold the makeup gain into the
        # numerator so the streaming filter directly outputs the normalized scale.
        b_d, self.a_d = signal.bilinear(b_num, a_den, fs=sample_rate)
        self.b_d = b_d * self.makeup_gain
        self.zi = signal.lfilter_zi(self.b_d, self.a_d)

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Processes a streaming chunk of float audio samples (-1.0 to +1.0).
        """
        # 1. Logic Inverter (74LS14)
        inverted = -chunk

        # 2. Linear continuous circuit (AC coupling, LP smoothing, attenuation),
        #    normalized so the circuit's true electrical peak maps to +/-1.0.
        out, self.zi = signal.lfilter(self.b_d, self.a_d, inverted, zi=self.zi)
        return out


class CMTAudioInputFilterExact:
    """
    Streaming filter modeling cassette input circuit (Fig. 5-5-9).
    Exact transfer function derived from circuit components:
    H(s) = (C31*R21*s) / [ (C29*C31*R20*R21)*s^2 + (C29*R20 + C29*R21 + C31*R21)*s + 1 ]
    Followed by anti-parallel diode clamping and Schmitt trigger comparator.
    """

    def __init__(self, sample_rate=44100, v_clamp=0.6, hysteresis_v=0.05):
        self.sample_rate = sample_rate
        self.v_clamp = v_clamp
        self.hysteresis_v = hysteresis_v

        # Circuit component values
        C31, R21, R20, C29 = 0.1e-6, 2700.0, 12000.0, 0.0015e-6

        b_num = [C31 * R21, 0.0]
        a_den = [C29 * C31 * R20 * R21, C29 * R20 + C29 * R21 + C31 * R21, 1.0]

        # Discretize via bilinear transform
        self.b_d, self.a_d = signal.bilinear(b_num, a_den, fs=sample_rate)
        self.zi = signal.lfilter_zi(self.b_d, self.a_d)
        self.schmitt_state = -1.0  # -1.0 or +1.0

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Processes a streaming chunk of float audio samples.
        """
        # 1. Bandpass filter
        filtered, self.zi = signal.lfilter(self.b_d, self.a_d, chunk, zi=self.zi)

        # 2. Diode clamping (+/- 0.6V)
        clamped = np.clip(filtered, -self.v_clamp, self.v_clamp)

        # 3. Comparator with hysteresis (Schmitt Trigger)
        output = np.zeros_like(clamped)
        state = self.schmitt_state
        v_upper = self.hysteresis_v
        v_lower = -self.hysteresis_v

        for i in range(len(clamped)):
            val = clamped[i]
            if state <= 0 and val > v_upper:
                state = 1.0
            elif state >= 0 and val < v_lower:
                state = -1.0
            output[i] = state

        self.schmitt_state = state
        return output


class TapeChannelGain:
    """
    OPTIONAL stage modeling real-world tape recording/playback level loss
    (e.g. weak recording level, worn tape, misadjusted deck) between the
    OUTPUT and INPUT circuits.

    This is no longer needed to make a basic output->input chain work: since
    CMTAudioOutputFilterExact now normalizes its own true electrical peak
    to +/-1.0 (see its docstring), a chained output->input signal already
    lands well inside CMTAudioInputFilterExact's clamp/hysteresis range.
    Use --tape-gain-db to deliberately attenuate (negative dB) or boost
    (positive dB) the signal between stages to test how the input comparator
    behaves with a degraded or over-driven tape signal. gain_db=0 (default)
    means "no extra loss" -- i.e. this stage is only inserted at all if the
    user explicitly asks for it.
    """

    def __init__(self, gain_db=0.0):
        self.gain = 10 ** (gain_db / 20.0)

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        return chunk * self.gain


def _build_filter_chain(mode_spec, sample_rate, tape_gain_db=None):
    """
    Builds a list of (name, filter_obj) stages from a mode spec like
    'output', 'input', 'output+input', or 'output,input'.
    If tape_gain_db is given, a TapeChannelGain stage is inserted right after
    each 'output' stage to model tape recording/playback level loss or gain
    (see TapeChannelGain docstring). By default (tape_gain_db=None) no such
    stage is added, since the output filter's own peak-normalization already
    makes output->input chaining work correctly.
    """
    modes = [
        m.strip().lower() for m in mode_spec.replace("+", ",").split(",") if m.strip()
    ]
    for m in modes:
        if m not in ("input", "output"):
            raise ValueError(f"Unknown mode '{m}' (expected 'input' or 'output')")

    stages = []
    for m in modes:
        if m == "input":
            stages.append(
                ("input", CMTAudioInputFilterExact(sample_rate=sample_rate))
            )
        else:
            stages.append(
                ("output", CMTAudioOutputFilterExact(sample_rate=sample_rate))
            )
            if tape_gain_db is not None:
                stages.append(
                    ("tape-channel-gain", TapeChannelGain(gain_db=tape_gain_db))
                )
    return stages


class ChainedFilter:
    """Runs a chunk through a sequence of filter stages in order."""

    def __init__(self, stages):
        self.stages = stages  # list of (name, filter_obj)

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        for _, stage in self.stages:
            chunk = stage.process_chunk(chunk)
        return chunk


def process_wav_file(
    input_wav_path: str, output_wav_path: str, filter_obj, chunk_size: int = 1024
):
    """
    Streams a WAV file chunk-by-chunk through the given filter object and saves to output_wav_path.
    """
    with wave.open(input_wav_path, "rb") as infile:
        nchannels = infile.getnchannels()
        sampwidth = infile.getsampwidth()
        framerate = infile.getframerate()

        with wave.open(output_wav_path, "wb") as outfile:
            outfile.setnchannels(1)  # mono output
            outfile.setsampwidth(2)  # 16-bit PCM
            outfile.setframerate(framerate)

            while True:
                frames = infile.readframes(chunk_size)
                if not frames:
                    break

                # Convert PCM bytes to float (-1.0 to 1.0)
                if sampwidth == 2:
                    data = (
                        np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                elif sampwidth == 1:
                    data = (
                        np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0
                    ) / 128.0
                else:
                    raise ValueError(f"Unsupported sample width: {sampwidth}")

                if nchannels > 1:
                    data = data[::nchannels]

                processed = filter_obj.process_chunk(data)

                # Convert back to int16 PCM
                out_int16 = np.clip(processed * 32767.0, -32768, 32767).astype(np.int16)
                outfile.writeframes(out_int16.tobytes())


class TestCMTAudioFiltersExact(unittest.TestCase):
    def test_output_filter_peak_gain(self):
        """
        The output filter is normalized so the circuit's own frequency-response
        peak (its true electrical peak, found at ~246 Hz for these component
        values) maps to exactly +/-1.0 WAV full scale. Other frequencies land
        at or below that, scaled by the same fixed factor.
        """
        fs = 44100
        out_filter = CMTAudioOutputFilterExact(sample_rate=fs)

        # Drive it at its own resonant/peak-response frequency: full-scale in
        # should map to (almost exactly) full-scale out once settled.
        t = np.linspace(0, 1.0, int(fs * 1.0), endpoint=False)
        sine_in = np.sin(2 * np.pi * out_filter.peak_freq_hz * t).astype(np.float32)

        chunk1 = out_filter.process_chunk(sine_in[: fs // 2])
        chunk2 = out_filter.process_chunk(sine_in[fs // 2 :])
        out = np.concatenate([chunk1, chunk2])

        settled = out[-int(fs * 0.1) :]  # well past the startup transient
        self.assertAlmostEqual(np.max(np.abs(settled)), 1.0, delta=0.01)

        # A generic 1 kHz tone (off-peak) should land just under full scale,
        # never over it, since makeup gain is calibrated to the true peak.
        sine_1k = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        out_filter2 = CMTAudioOutputFilterExact(sample_rate=fs)
        out_1k = out_filter2.process_chunk(sine_1k)
        settled_1k = out_1k[-int(fs * 0.1) :]
        self.assertLess(np.max(np.abs(settled_1k)), 1.0)
        self.assertGreater(np.max(np.abs(settled_1k)), 0.9)

    def test_input_filter_reconstruction(self):
        fs = 44100
        in_filter = CMTAudioInputFilterExact(sample_rate=fs)

        # Test signal with 50 Hz hum + DC offset + 2400 Hz data tone
        t = np.linspace(0, 0.1, int(fs * 0.1), endpoint=False)
        sine_in = (
            0.5 * np.sin(2 * np.pi * 2400 * t) + 0.2 * np.sin(2 * np.pi * 50 * t) + 0.1
        )

        chunk1 = in_filter.process_chunk(sine_in[:2000])
        chunk2 = in_filter.process_chunk(sine_in[2000:])
        out = np.concatenate([chunk1, chunk2])

        settled_out = out[500:]
        unique_vals = np.unique(settled_out)
        self.assertTrue(all(v in [-1.0, 1.0] for v in unique_vals))

    def test_output_to_input_chain_toggles(self):
        """
        Regression test: chaining output->input must NOT get stuck latched at
        one comparator state. Now that CMTAudioOutputFilterExact normalizes
        its own peak response to +/-1.0, its output for realistic FSK tones
        (1200/2400 Hz) lands well above the input filter's default 50 mV
        Schmitt hysteresis, with no extra gain stage required.
        """
        fs = 44100
        t = np.arange(int(fs * 1.0)) / fs
        bits = np.sign(np.sin(2 * np.pi * 1200 * t)).astype(np.float32)

        stages = _build_filter_chain("output+input", fs)  # no tape_gain_db needed
        self.assertEqual([name for name, _ in stages], ["output", "input"])
        chained = ChainedFilter(stages)
        out = chained.process_chunk(bits)

        # After the initial filter-settling transient, the comparator must
        # actually be toggling, not stuck at a single latched value.
        settled_tail = out[-2000:]
        self.assertEqual(set(np.unique(settled_tail)), {-1.0, 1.0})
        self.assertGreater(np.sum(np.diff(out) != 0), 100)

    def test_wav_processing(self):
        fs = 44100
        test_in = "temp_test_in.wav"
        test_out = "temp_test_out.wav"

        t = np.linspace(0, 0.1, int(fs * 0.1), endpoint=False)
        sig = np.sin(2 * np.pi * 1200 * t).astype(np.float32)

        with wave.open(test_in, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes((sig * 32767).astype(np.int16).tobytes())

        out_filt = CMTAudioOutputFilterExact(sample_rate=fs)
        process_wav_file(test_in, test_out, out_filt, chunk_size=256)

        self.assertTrue(os.path.exists(test_out))
        if os.path.exists(test_in):
            os.remove(test_in)
        if os.path.exists(test_out):
            os.remove(test_out)


def run_tests():
    print("Running CMT Audio Filter Test Suite...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCMTAudioFiltersExact)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="CMT Audio Shaping Circuits Streaming WAV Filter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument(
        "-m",
        "--mode",
        help="Filter mode: 'input' (CMT-IN -> IOA7), 'output' (PC5 -> CMT OUT), "
        "or a chain such as 'output+input' / 'output,input' to simulate a full "
        "record->playback round trip through a cassette deck.",
    )
    parser.add_argument("-i", "--input-file", help="Path to input WAV file")
    parser.add_argument("-o", "--output-file", help="Path to output WAV file")
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=1024,
        help="Chunk size in frames for streaming processing (default: 1024)",
    )
    parser.add_argument(
        "--tape-gain-db",
        type=float,
        default=None,
        help="OPTIONAL: inserts a gain stage right after each 'output' stage "
        "to simulate real-world tape recording/playback level loss or gain "
        "(e.g. a weak recording, worn tape, or misadjusted deck). Not needed "
        "for normal use -- the output filter already normalizes its own "
        "electrical peak to full WAV scale. Omit to leave levels as-is "
        "(default: no extra stage added).",
    )

    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    if not args.mode or not args.input_file or not args.output_file:
        parser.print_help()
        print(
            "\nError: --mode, --input-file, and --output-file are required when not running in --test mode."
        )
        sys.exit(1)

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)

    # Read input sample rate
    with wave.open(args.input_file, "rb") as wf:
        sample_rate = wf.getframerate()

    try:
        stages = _build_filter_chain(
            args.mode, sample_rate, tape_gain_db=args.tape_gain_db
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    stage_desc = " -> ".join(name for name, _ in stages)
    print(
        f"Applying CMT Audio filter chain [{stage_desc}] on '{args.input_file}'..."
    )
    filter_obj = ChainedFilter(stages)

    process_wav_file(
        args.input_file, args.output_file, filter_obj, chunk_size=args.chunk_size
    )
    print(f"Processing complete. Output written to '{args.output_file}'.")


if __name__ == "__main__":
    main()
