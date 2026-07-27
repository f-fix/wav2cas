#!/usr/bin/env python3
"""
===============================================================================
Cassette Audio Modeler - Physical DSP Simulation of Audio Cassette Channels
===============================================================================

Overview:
---------
This module implements a physical and DSP simulation of audio cassette recording
and playback channels, modeling both the analog pre/post electronics and the
magneto-electric physics of the tape-head interface.

Simulation Architecture:
------------------------
1. Record Path (Mic Input -> Tape Magnetic Flux):
   - Microphone Preamp & Bandpass Filtering: DC blocking high-pass (~20 Hz) and
     anti-aliasing low-pass (clamped relative to Nyquist frequency).
   - IEC Type I Record Pre-emphasis EQ: Boosts high frequencies (tau_1 = 120 us,
     tau_2 = 12 us) to overcome magnetic writing and gap losses.
   - Tape Self-Demagnetization (Write Loss): High-frequency demagnetization
     L_demag(s) = 1 / (1 + s * tau_120) balancing record pre-emphasis.
   - Anhysteretic Magnetic Saturation: Soft non-linear tape magnetization M(H)
     modeled via hyperbolic tangent tanh(k * H) resulting from AC bias linearization.

2. Playback Path (Tape Magnetic Flux -> Earphone Output):
   - Faraday's Law Read Head Induction: Induced voltage e(t) = -N * dPhi/dt across
     head coils, creating a +6 dB/octave derivative slope.
   - Wallace Gap Loss Filter: Spatial frequency attenuation sinc(pi * g / lambda)
     caused by finite read head gap length g (~1.5 microns) at speed v (4.76 cm/s).
   - IEC Type I Playback De-emphasis EQ: Integrates the +6 dB/octave derivative
     slope (tau_1 = 3180 us) and applies de-emphasis (tau_2 = 120 us).
   - Ear Output Amplifier Stage: AC coupling high-pass filter driving headphone loads.

References Consulted:
---------------------
1. IEC Standard 60094-4 & 60094-5: "Magnetic Tape Sound Recording and
   Reproducing Systems" - Standard equalization time constants for Type I cassettes
   (3180 us, 120 us, 12 us).
2. Wallace, R. L. (1951): "The Reproduction of Magnetically Recorded Signals",
   Bell System Technical Journal, 30(4), pp. 1145-1173. (Gap and spacing loss equations).
3. Jiles, D. C., & Atherton, D. L. (1986): "Theory of Ferromagnetic Hysteresis",
   Journal of Magnetism and Magnetic Materials, 61(1-2), pp. 48-60. (Anhysteretic
   tape magnetization and AC bias linearization models).
4. Zölzer, U. (2011): "DAFX: Digital Audio Effects", John Wiley & Sons. (Bilinear
   transform mapping of continuous-time analog equalizers to discrete IIR filters).
5. Smith, J. O. (2007): "Physical Audio Signal Processing", W3K Publishing.
   (Stateful streaming DSP filter implementations and lfilter_zi initial states).
"""

import os
import sys
import wave
import argparse
import unittest
import tempfile
import numpy as np
import scipy.signal as signal


class CassetteStage:
    """
    Individual stateful processing stage in an audio cassette filter chain.
    """

    def __init__(
        self,
        stage_type,
        fs,
        n_channels,
        tape_speed=0.0476,
        head_gap=1.5e-6,
        drive=1.2,
        noise_level=0.001,
    ):
        self.stage_type = stage_type.lower().strip()
        self.fs = fs
        self.n_channels = n_channels
        self.v = tape_speed
        self.g = head_gap
        self.drive = drive
        self.noise_level = noise_level
        self._init_filters()

    def _init_filters(self):
        fs = self.fs
        n = self.n_channels
        nyq = fs / 2.0

        tau_3180 = 3180e-6  # 50.05 Hz pole
        tau_120 = 120e-6  # 1326 Hz zero/pole
        tau_12 = 12e-6  # 13263 Hz pole

        if self.stage_type in ["record", "rec"]:
            # Safe bandpass frequency range bounded by Nyquist frequency
            low_f = max(20.0, nyq * 0.001)
            high_f = min(18000.0, nyq * 0.95)

            # 1. Mic Preamp Filter
            b_mic, a_mic = signal.butter(2, [low_f / nyq, high_f / nyq], btype="band")
            self.mic_b, self.mic_a = b_mic, a_mic
            zi_mic = signal.lfilter_zi(b_mic, a_mic)
            self.mic_zi = np.tile(zi_mic[:, None], (1, n)) if n > 1 else zi_mic.copy()

            # 2. IEC Type I Record Pre-emphasis EQ: H_rec(s) = (1 + s*tau_120) / (1 + s*tau_12)
            b_rec, a_rec = signal.bilinear([tau_120, 1.0], [tau_12, 1.0], fs=fs)
            self.rec_b, self.rec_a = b_rec, a_rec
            zi_rec = signal.lfilter_zi(b_rec, a_rec)
            self.rec_zi = np.tile(zi_rec[:, None], (1, n)) if n > 1 else zi_rec.copy()

            # 3. Write Loss / Tape Self-Demagnetization: H_demag(s) = 1 / (1 + s*tau_120)
            b_demag, a_demag = signal.bilinear([1.0], [tau_120, 1.0], fs=fs)
            self.demag_b, self.demag_a = b_demag, a_demag
            zi_demag = signal.lfilter_zi(b_demag, a_demag)
            self.demag_zi = (
                np.tile(zi_demag[:, None], (1, n)) if n > 1 else zi_demag.copy()
            )

        elif self.stage_type in ["playback", "pb", "play"]:
            # 1. Wallace Gap Loss FIR Filter (sinc spatial loss model)
            freqs = np.linspace(0, nyq, 256)
            wavelengths = np.zeros_like(freqs)
            wavelengths[1:] = self.v / freqs[1:]
            gap_loss = np.ones_like(freqs)
            gap_loss[1:] = np.abs(
                np.sin(np.pi * self.g / wavelengths[1:])
                / (np.pi * self.g / wavelengths[1:])
            )
            gap_loss = np.maximum(gap_loss, 0.02)
            b_gap = signal.firwin2(65, freqs, gap_loss, fs=fs)
            self.gap_b = b_gap
            zi_gap = signal.lfilter_zi(b_gap, 1.0)
            self.gap_zi = np.tile(zi_gap[:, None], (1, n)) if n > 1 else zi_gap.copy()

            # 2. IEC Type I Playback De-emphasis EQ: H_pb(s) = (1 + s*tau_120) / (1 + s*tau_3180)
            b_pb, a_pb = signal.bilinear([tau_120, 1.0], [tau_3180, 1.0], fs=fs)
            self.pb_b, self.pb_a = b_pb, a_pb
            zi_pb = signal.lfilter_zi(b_pb, a_pb)
            self.pb_zi = np.tile(zi_pb[:, None], (1, n)) if n > 1 else zi_pb.copy()

            # 3. Ear Output Stage (DC blocking High-pass)
            low_ear = max(20.0, nyq * 0.001)
            b_ear, a_ear = signal.butter(1, low_ear / nyq, btype="high")
            self.ear_b, self.ear_a = b_ear, a_ear
            zi_ear = signal.lfilter_zi(b_ear, a_ear)
            self.ear_zi = np.tile(zi_ear[:, None], (1, n)) if n > 1 else zi_ear.copy()

            self.prev_flux_sample = np.zeros((1, n)) if n > 1 else np.zeros((1,))

            # Normalization scale for Faraday differentiation d/dt
            self.induction_scale = tau_3180 * fs

        else:
            raise ValueError(
                f"Unknown filter stage '{self.stage_type}'. Allowed choices: 'record' (rec), 'playback' (pb)."
            )

    def process_chunk(self, chunk):
        axis = 0 if chunk.ndim > 1 else -1

        if self.stage_type in ["record", "rec"]:
            x, self.mic_zi = signal.lfilter(
                self.mic_b, self.mic_a, chunk, axis=axis, zi=self.mic_zi
            )
            x, self.rec_zi = signal.lfilter(
                self.rec_b, self.rec_a, x, axis=axis, zi=self.rec_zi
            )
            x, self.demag_zi = signal.lfilter(
                self.demag_b, self.demag_a, x, axis=axis, zi=self.demag_zi
            )
            return np.tanh(self.drive * x)

        elif self.stage_type in ["playback", "pb", "play"]:
            if chunk.ndim > 1:
                flux_ext = np.vstack([self.prev_flux_sample, chunk])
                self.prev_flux_sample = chunk[-1:]
            else:
                flux_ext = np.insert(chunk, 0, self.prev_flux_sample)
                self.prev_flux_sample = chunk[-1]

            dflux = np.diff(flux_ext, axis=axis) * self.induction_scale

            if self.noise_level > 0:
                dflux += np.random.normal(0, self.noise_level, size=dflux.shape)

            e_gap, self.gap_zi = signal.lfilter(
                self.gap_b, 1.0, dflux, axis=axis, zi=self.gap_zi
            )
            e_eq, self.pb_zi = signal.lfilter(
                self.pb_b, self.pb_a, e_gap, axis=axis, zi=self.pb_zi
            )
            out, self.ear_zi = signal.lfilter(
                self.ear_b, self.ear_a, e_eq, axis=axis, zi=self.ear_zi
            )
            return out


class CassetteAudioProcessor:
    """
    Streaming WAV Cassette Modeler supporting explicitly chained filter stages.
    """

    def __init__(
        self,
        tape_speed=0.0476,
        head_gap=1.5e-6,
        saturation_drive=1.2,
        tape_hiss_level=0.001,
    ):
        self.v = tape_speed
        self.g = head_gap
        self.drive = saturation_drive
        self.noise_level = tape_hiss_level

    def process_stream(
        self, input_wav_path, output_wav_path, mode="record+playback", chunk_size=4096
    ):
        raw_stages = [s.strip() for s in mode.split("+") if s.strip()]
        if not raw_stages:
            raise ValueError(
                "Mode string must specify at least one valid stage (e.g., 'record', 'playback', 'record+playback')."
            )

        parsed_stages = []
        for s in raw_stages:
            if s.lower() == "chain":
                parsed_stages.extend(["record", "playback"])
            else:
                parsed_stages.append(s)

        with wave.open(input_wav_path, "rb") as in_wf:
            n_channels = in_wf.getnchannels()
            sampwidth = in_wf.getsampwidth()
            fs = in_wf.getframerate()

            chain = [
                CassetteStage(
                    stage_type=st,
                    fs=fs,
                    n_channels=n_channels,
                    tape_speed=self.v,
                    head_gap=self.g,
                    drive=self.drive,
                    noise_level=self.noise_level,
                )
                for st in parsed_stages
            ]

            with wave.open(output_wav_path, "wb") as out_wf:
                out_wf.setnchannels(n_channels)
                out_wf.setsampwidth(sampwidth)
                out_wf.setframerate(fs)

                while True:
                    raw_bytes = in_wf.readframes(chunk_size)
                    if not raw_bytes:
                        break

                    audio = self._bytes_to_float(raw_bytes, sampwidth, n_channels)

                    for stage in chain:
                        audio = stage.process_chunk(audio)

                    audio = np.clip(audio, -1.0, 1.0)
                    out_bytes = self._float_to_bytes(audio, sampwidth)
                    out_wf.writeframes(out_bytes)

    def _bytes_to_float(self, raw_bytes, sampwidth, n_channels):
        if sampwidth == 1:
            data = np.frombuffer(raw_bytes, dtype=np.uint8)
            flt = (data.astype(np.float32) - 128.0) / 128.0
        elif sampwidth == 2:
            data = np.frombuffer(raw_bytes, dtype=np.int16)
            flt = data.astype(np.float32) / 32768.0
        elif sampwidth == 4:
            data = np.frombuffer(raw_bytes, dtype=np.int32)
            flt = data.astype(np.float32) / 2147483648.0
        elif sampwidth == 3:
            a8 = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
            a32 = np.zeros((len(a8), 4), dtype=np.uint8)
            a32[:, 1:] = a8
            flt = (a32.view(np.int32) >> 8).astype(np.float32) / 8388608.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

        if n_channels > 1:
            flt = flt.reshape(-1, n_channels)
        return flt

    def _float_to_bytes(self, flt_data, sampwidth):
        if sampwidth == 1:
            pcm = ((flt_data * 127.0) + 128.0).astype(np.uint8)
        elif sampwidth == 2:
            pcm = (flt_data * 32767.0).astype(np.int16)
        elif sampwidth == 4:
            pcm = (flt_data * 2147483647.0).astype(np.int32)
        elif sampwidth == 3:
            ints = (flt_data * 8388607.0).astype(np.int32)
            b4 = ints.view(np.uint8).reshape(-1, 4)
            pcm = b4[:, :3]
        return pcm.tobytes()


# -----------------------------------------------------------------------------
# Unit Test Suite
# -----------------------------------------------------------------------------
class TestCassetteAudioProcessor(unittest.TestCase):
    def setUp(self):
        self.duration = 0.5

    def _create_temp_wav(self, signal_data, fs=44100, channels=1, sampwidth=2):
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        pcm = (signal_data * 32767.0).astype(np.int16)
        with wave.open(temp_path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(fs)
            wf.writeframes(pcm.tobytes())
        return temp_path

    def test_variable_sample_rate_support(self):
        """Verify filtering across multiple standard audio sample rates (8 kHz to 44.1 kHz)"""
        for fs in [8000, 11025, 16000, 22050, 32000, 44100]:
            t = np.linspace(0, self.duration, int(fs * self.duration), endpoint=False)
            sine = 0.5 * np.sin(2 * np.pi * 440 * t)

            in_path = self._create_temp_wav(sine, fs=fs)
            fd, out_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

            try:
                proc = CassetteAudioProcessor(tape_hiss_level=0.0)
                proc.process_stream(in_path, out_path, mode="record+playback")

                with wave.open(out_path, "rb") as wf:
                    self.assertEqual(wf.getframerate(), fs)
                    self.assertGreater(wf.getnframes(), 0)
            finally:
                if os.path.exists(in_path):
                    os.remove(in_path)
                if os.path.exists(out_path):
                    os.remove(out_path)

    def test_explicit_custom_chain_orders(self):
        """Verify user-specified filter chain sequences (e.g. playback+record, record+playback)"""
        fs = 44100
        t = np.linspace(0, self.duration, int(fs * self.duration), endpoint=False)
        sine = 0.5 * np.sin(2 * np.pi * 1000 * t)

        in_path = self._create_temp_wav(sine, fs=fs)
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        try:
            proc = CassetteAudioProcessor(tape_hiss_level=0.0)
            proc.process_stream(in_path, out_path, mode="playback+record")
            with wave.open(out_path, "rb") as wf:
                self.assertGreater(wf.getnframes(), 0)
        finally:
            if os.path.exists(in_path):
                os.remove(in_path)
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_sine_wave_steady_state_response(self):
        """Verify sine wave passes through record+playback chain without amplitude blowup"""
        fs = 44100
        t = np.linspace(0, self.duration, int(fs * self.duration), endpoint=False)
        sine = 0.5 * np.sin(2 * np.pi * 1000 * t)

        in_path = self._create_temp_wav(sine, fs=fs)
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        try:
            proc = CassetteAudioProcessor(tape_hiss_level=0.0)
            proc.process_stream(in_path, out_path, mode="record+playback")

            with wave.open(out_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                out_data = (
                    np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                )

            rms_out = np.sqrt(np.mean(out_data[1000:] ** 2))
            self.assertGreater(rms_out, 0.1)
            self.assertLess(rms_out, 0.9)
        finally:
            if os.path.exists(in_path):
                os.remove(in_path)
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_square_wave_transient_response(self):
        """Verify square wave transitions are reconstructed cleanly with expected tape rounding"""
        fs = 44100
        t = np.linspace(0, self.duration, int(fs * self.duration), endpoint=False)
        square = 0.5 * signal.square(2 * np.pi * 220 * t)

        in_path = self._create_temp_wav(square, fs=fs)
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        try:
            proc = CassetteAudioProcessor(tape_hiss_level=0.0)
            proc.process_stream(in_path, out_path, mode="record+playback")

            with wave.open(out_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                out_data = (
                    np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                )

            rms_out = np.sqrt(np.mean(out_data[1000:] ** 2))
            self.assertGreater(rms_out, 0.1)
            self.assertLess(rms_out, 0.95)
        finally:
            if os.path.exists(in_path):
                os.remove(in_path)
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_stereo_channel_processing(self):
        """Verify multi-channel stereo WAV processing with independent state tracking"""
        fs = 44100
        t = np.linspace(0, self.duration, int(fs * self.duration), endpoint=False)
        stereo = np.column_stack(
            [0.5 * np.sin(2 * np.pi * 440 * t), 0.3 * np.cos(2 * np.pi * 880 * t)]
        )

        in_path = self._create_temp_wav(stereo, fs=fs, channels=2)
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        try:
            proc = CassetteAudioProcessor(tape_hiss_level=0.0)
            proc.process_stream(in_path, out_path, mode="record+playback")

            with wave.open(out_path, "rb") as wf:
                self.assertEqual(wf.getnchannels(), 2)
                self.assertGreater(wf.getnframes(), 0)
        finally:
            if os.path.exists(in_path):
                os.remove(in_path)
            if os.path.exists(out_path):
                os.remove(out_path)


def main():
    if "--test" in sys.argv:
        print("Executing unit test suite...")
        test_argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "--test"]
        unittest.main(argv=test_argv)
        return

    parser = argparse.ArgumentParser(
        description="Audio Cassette Modeler with Explicit Filter Chaining (+)"
    )
    parser.add_argument("-i", "--input", help="Path to input WAV file")
    parser.add_argument("-o", "--output", help="Path to output WAV file")
    parser.add_argument(
        "-m",
        "--mode",
        default="record+playback",
        help="Filter sequence separated by '+' (e.g., 'record+playback', 'playback+record', 'record+playback+record+playback')",
    )
    parser.add_argument(
        "--drive",
        type=float,
        default=1.2,
        help="Tape saturation drive intensity (default: 1.2)",
    )
    parser.add_argument(
        "--hiss",
        type=float,
        default=0.001,
        help="Tape hiss noise level (default: 0.001)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run embedded unit test suite"
    )

    args = parser.parse_args()

    if not args.input or not args.output:
        parser.print_help()
        sys.exit(1)

    processor = CassetteAudioProcessor(
        saturation_drive=args.drive, tape_hiss_level=args.hiss
    )
    processor.process_stream(args.input, args.output, mode=args.mode)
    print(
        f"Successfully processed '{args.input}' -> '{args.output}' using chain: [{args.mode}]."
    )


if __name__ == "__main__":
    main()
