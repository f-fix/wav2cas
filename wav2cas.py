#!/usr/bin/env python3
"""wav2cas.py - Decode MSX cassette audio (WAV) into a .CAS file.

Pure Python, standard library only (wave, struct, argparse, sys).

FORMAT ASSUMPTIONS (standard MSX BIOS cassette encoding, FSK / Kansas-City style):

  * A "0" data bit is encoded as ONE cycle at the base frequency for the
    current baud rate.
  * A "1" data bit is encoded as TWO cycles at double the base frequency.
  * Supported baud rates (auto-detected from the pilot tone frequency),
    covering the range real MSX hardware and turbo-loaders use:

        baud    bit-0 freq   bit-1 freq
         600     600 Hz       1200 Hz
        1200    1200 Hz       2400 Hz
        2400    2400 Hz       4800 Hz
        3600    3600 Hz       7200 Hz
        4800    4800 Hz       9600 Hz
        7200    7200 Hz      14400 Hz
        9600    9600 Hz      19200 Hz

  * Each block (header or data) is preceded on tape by a pilot tone, which
    is simply a long run of "1" bits. The decoder locks onto whichever
    baud rate the pilot tone matches, then starts interpreting bits as
    UART frames: 1 start bit (0), 8 data bits (LSB first), then a stop
    bit / mark period.

  * STOP BITS: by default the decoder is strict, exactly like older tape
    tools - after the 8 data bits it requires --stop-bits (default 2)
    genuine "1" bits before accepting the byte, and drops the byte (and
    resyncs on the pilot search) if that doesn't hold. This is what keeps
    stray noise or pilot/data transition artifacts from being misread as
    spurious extra bytes right before or after a real block. Pass
    --lenient-stop-bits to fall back to the older, more permissive
    behavior that just looks for the next start bit with no fixed count
    (useful for tapes whose stop-bit timing is irregular).

  * Bit classification uses a single adaptive midpoint threshold between
    the nominal "short" (bit-1) and "long" (bit-0) cycle lengths, rather
    than requiring each pulse to closely match an exact target frequency.
    This avoids a "dead zone" between the two tones and tolerates tape
    wow/flutter, jitter, and general noise far better than tight
    frequency-matching would. The threshold slowly tracks the actually
    measured pulse lengths as decoding proceeds, adapting to gradual speed
    drift over the length of a tape.

  * AMPLITUDE / AGC: instead of one fixed trigger threshold for the whole
    file, an envelope follower tracks the local signal amplitude (fast
    attack, slow release, like an audio compressor's envelope detector) and
    the Schmitt-trigger threshold is a fraction of that *local* envelope.
    This lets the decoder ride out gradual volume changes over a
    recording - fade in/out, azimuth wobble, etc. Pass --no-agc to fall
    back to one fixed threshold based on the whole file's peak amplitude.

  * CONFIDENCE: every classified pulse gets a confidence score in [0, 1]
    based on how far its length is from the long/short decision threshold,
    relative to the gap between the two reference lengths (a pulse right
    at a reference length scores 1.0; one sitting exactly on the threshold
    scores 0.0). Per-bit confidence is the pulse confidence for a "0" bit,
    or the average of its two pulses for a "1" bit. Per-byte confidence is
    the average over the start bit, the 8 data bits, and (in strict mode)
    the stop bit(s). Per-block confidence is the average of its bytes'
    confidences. Block confidence is printed as a diagnostic, and blocks
    scoring below --min-confidence (default 0.8) are
    left out of the output file - useful for automatically discarding
    blocks decoded from a garbled/noisy stretch of tape.

  * Every time a new pilot tone run is detected after at least one byte has
    already been decoded, the previously accumulated bytes are flushed out
    as one completed block, and decoding starts fresh for the next block.
    The baud rate is re-checked at every such pilot, since nothing stops a
    tape from switching baud rates between blocks.

  * Each decoded block is written to the .cas output preceded by the
    standard 8-byte CAS block marker:

        1F A6 DE BA CC 13 7D 74

    (this is the same convention used by real MSX emulators' .cas files -
    it lets a loader locate block boundaries in the file since a raw .cas
    does not otherwise store timing/pilot information).

  * With --pad, 0-7 zero bytes are inserted before each CAS block header so
    it starts at a file offset that's a multiple of 8, which some MSX tools
    expect. Off by default.

LIMITATIONS:
  * Only handles integer PCM WAV data (8/16/24/32-bit), not floating point.
"""

import argparse
import io
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import wave

CAS_HEADER = bytes([0x1F, 0xA6, 0xDE, 0xBA, 0xCC, 0x13, 0x7D, 0x74])

# baud rate -> (bit-0 frequency, bit-1 frequency)
BAUD_TABLE = {
    600: (600, 1200),
    1200: (1200, 2400),
    2400: (2400, 4800),
    3600: (3600, 7200),
    4800: (4800, 9600),
    7200: (7200, 14400),
    9600: (9600, 19200),
}


# --------------------------------------------------------------------------
# Optional (`--filter`) RC Bandpass Simulation (MSX CMTIN stage)
# --------------------------------------------------------------------------


def apply_msx_hardware_filter(samples, framerate):
    """Simulates the MSX cassette hardware input circuit response.

    Applies a DC-blocking high-pass filter (~300 Hz) and a gentle smoothing
    low-pass filter (~6 kHz) to clean up tape rumble and high-frequency hiss
    without phase distortion or breaking clean captures.
    """
    if not samples:
        return samples

    hp_rc = 1.0 / (2.0 * math.pi * 300.0)
    dt = 1.0 / framerate
    hp_alpha = hp_rc / (hp_rc + dt)

    lp_rc = 1.0 / (2.0 * math.pi * 6000.0)
    lp_alpha = dt / (lp_rc + dt)

    filtered = [0.0] * len(samples)

    prev_in = samples[0]
    prev_out = 0.0
    hp_out = [0.0] * len(samples)
    for i in range(len(samples)):
        cur_in = float(samples[i])
        prev_out = hp_alpha * (prev_out + cur_in - prev_in)
        hp_out[i] = prev_out
        prev_in = cur_in

    prev_val = hp_out[0]
    for i in range(len(hp_out)):
        prev_val = prev_val + lp_alpha * (hp_out[i] - prev_val)
        filtered[i] = prev_val

    return filtered


# --------------------------------------------------------------------------
# WAV, FLAC, etc. reading
# --------------------------------------------------------------------------


class PurePythonFlacDecoder:
    """A low-performance pure Python FLAC decoder using memory buffering."""

    def __init__(self, file_path):
        with open(file_path, "rb") as f:
            self.data = f.read()

        self.data_len = len(self.data)
        self.offset = 0

        self.sample_rate = 44100
        self.channels = 2
        self.bits_per_sample = 16
        self.total_samples = 0

        # Bit stream buffer state
        self.bit_buffer = 0
        self.bit_count = 0

        self.pcm_data = bytearray()
        self._parse_flac()

    def _read_bytes(self, n):
        if self.offset + n > self.data_len:
            return None
        res = self.data[self.offset : self.offset + n]
        self.offset += n
        return res

    def _parse_flac(self):
        header = self._read_bytes(4)
        if header != b"fLaC":
            raise ValueError("Invalid FLAC file: Missing 'fLaC' signature marker.")

        # Read metadata blocks
        is_last = False
        while not is_last:
            block_header = self._read_bytes(4)
            if not block_header or len(block_header) < 4:
                break

            is_last = (block_header[0] & 0x80) != 0
            block_type = block_header[0] & 0x7F
            block_length = struct.unpack(">I", b"\x00" + block_header[1:4])[0]
            block_data = self._read_bytes(block_length)

            if block_type == 0:  # STREAMINFO
                self._parse_streaminfo(block_data)

        # Read audio frames with compact carriage-return progress feedback
        while self.offset < self.data_len:
            percent = min(100.0, (self.offset / self.data_len) * 100.0)
            sys.stdout.write(f"\rDecoding FLAC: {percent:5.1f}%")
            sys.stdout.flush()

            sync_marker = self._read_bytes(2)
            if not sync_marker:
                break

            # Check for sync code (14 bits set to 1 -> 0xFFF)
            if len(sync_marker) == 2 and (
                sync_marker[0] == 0xFF
                and (sync_marker[1 & 0xFE] == 0xF8 or (sync_marker[1] & 0xFC) == 0xF0)
            ):
                self.offset -= 2  # Rewind sync marker
                self._parse_frame()
            else:
                self.offset -= 1  # Resync scan step

        sys.stdout.write("\rDecoding FLAC: 100.0% - Complete!\n")
        sys.stdout.flush()

    def _parse_streaminfo(self, data):
        if len(data) < 34:
            return
        sr_ch_bps_and_samples = data[10:26]
        combined_val = struct.unpack(">I", sr_ch_bps_and_samples[:4])[0]

        self.sample_rate = combined_val >> 12
        self.channels = ((combined_val >> 9) & 0x07) + 1
        self.bits_per_sample = ((combined_val >> 4) & 0x1F) + 1
        self.total_samples = (
            int.from_bytes(sr_ch_bps_and_samples[4:12], "big") & 0xFFFFFFFFF
        )

    def _parse_frame(self):
        self.bit_count = 0
        self.bit_buffer = 0

        header_start = self._read_bytes(2)
        if not header_start or len(header_start) < 2:
            return

        if not self._read_bytes(1):  # block strategy / numbering
            return

        bs_sr = self._read_bytes(1)
        if not bs_sr:
            return
        block_size_type = (bs_sr[0] >> 4) & 0x0F

        ch_bps = self._read_bytes(1)
        if not ch_bps:
            return
        channel_assignment = (ch_bps[0] >> 4) & 0x0F

        block_size = self._get_block_size(block_size_type)
        if block_size is None:
            return

        self._read_bytes(1)  # Skip CRC-8 header byte

        subframe_samples = []
        for ch in range(self.channels):
            sub_samples = self._parse_subframe(block_size)
            subframe_samples.append(sub_samples)

        # Apply Channel Decorrelation if stereo assignment matches
        if self.channels == 2 and len(subframe_samples) == 2:
            left = subframe_samples[0]
            right = subframe_samples[1]
            if channel_assignment == 8:  # Left/Side -> R = L - S
                right = [l - s for l, s in zip(left, right)]
            elif channel_assignment == 9:  # Side/Right -> L = R + S
                left = [r + s for r, s in zip(right, left)]
            elif channel_assignment == 10:  # Mid/Side -> L = M + S/2, R = M - S/2
                new_left = []
                new_right = []
                for m, s in zip(left, right):
                    res = s >> 1
                    new_left.append(m + res - (s & 1 & (s < 0)))
                    new_right.append(m - res)
                left, right = new_left, new_right
            subframe_samples = [left, right]

        self.bit_count = 0
        self.bit_buffer = 0

        for i in range(block_size):
            for ch in range(self.channels):
                if i < len(subframe_samples[ch]):
                    sample = subframe_samples[ch][i]

                    if self.bits_per_sample > 16:
                        sample >>= self.bits_per_sample - 16
                    elif self.bits_per_sample < 16:
                        sample <<= 16 - self.bits_per_sample

                    sample = max(-32768, min(32767, sample))
                    self.pcm_data.extend(struct.pack("<h", sample))

    def _get_block_size(self, bs_type):
        if bs_type == 1:
            return 192
        elif 2 <= bs_type <= 5:
            return 576 << (bs_type - 2)
        elif bs_type == 6:
            b = self._read_bytes(1)
            return (b[0] + 1) if b else None
        elif bs_type == 7:
            b = self._read_bytes(2)
            return (struct.unpack(">H", b)[0] + 1) if b else None
        elif 8 <= bs_type <= 15:
            return 256 << (bs_type - 8)
        return None

    def _parse_subframe(self, block_size):
        sub_header = self._read_bytes(1)
        if not sub_header:
            return [0] * block_size

        byte_val = sub_header[0]
        sub_type = (byte_val >> 1) & 0x3F

        wasted_bits = 0
        if (byte_val & 0x01) != 0:
            wasted_bits = self._read_unary() + 1

        samples = [0] * block_size

        if sub_type == 0:
            sample = self._read_signed_bits(self.bits_per_sample)
            samples = [sample] * block_size
        elif sub_type == 1:
            for i in range(block_size):
                samples[i] = self._read_signed_bits(self.bits_per_sample)
        elif 8 <= sub_type <= 12:
            order = sub_type - 8
            for i in range(order):
                samples[i] = self._read_signed_bits(self.bits_per_sample)
            samples = self._parse_residual(block_size, order, samples)
        elif 32 <= sub_type <= 63:
            order = sub_type - 31
            for i in range(order):
                samples[i] = self._read_signed_bits(self.bits_per_sample)
            self._read_bytes(4)
            samples = self._parse_residual(block_size, order, samples)

        if wasted_bits > 0:
            samples = [s << wasted_bits for s in samples]

        return samples

    def _parse_residual(self, block_size, order, samples):
        rice_header = self._read_bits(2)
        if rice_header is None:
            return samples

        param_len = 4 if rice_header == 0 else 5
        partition_order = self._read_bits(4)
        num_partitions = 1 << partition_order
        part_samples = block_size >> partition_order

        for p in range(num_partitions):
            param = self._read_bits(param_len)
            if param is None:
                break

            escape = (1 << param_len) - 1
            start_idx = order if p == 0 else 0
            end_idx = part_samples

            for i in range(start_idx, end_idx):
                idx = p * part_samples + i
                if idx >= block_size:
                    break
                val = self._read_rice_signed(param if param < escape else 0)
                prev_val = samples[idx - 1] if idx > 0 else 0
                samples[idx] = val + prev_val

        return samples

    def _read_bits(self, n):
        if n == 0:
            return 0
        while self.bit_count < n:
            if self.offset >= self.data_len:
                return None
            self.bit_buffer = (self.bit_buffer << 8) | self.data[self.offset]
            self.offset += 1
            self.bit_count += 8

        self.bit_count -= n
        return (self.bit_buffer >> self.bit_count) & ((1 << n) - 1)

    def _read_signed_bits(self, n):
        unsigned_val = self._read_bits(n)
        if unsigned_val is None:
            return 0
        if unsigned_val & (1 << (n - 1)):
            unsigned_val -= 1 << n
        return unsigned_val

    def _read_unary(self):
        count = 0
        while True:
            bit = self._read_bits(1)
            if bit is None or bit == 1:
                break
            count += 1
        return count

    def _read_rice_signed(self, param):
        val = self._read_unary()
        if param > 0:
            low = self._read_bits(param)
            if low is not None:
                val = (val << param) | low
        if val & 1:
            return -((val >> 1) + 1)
        else:
            return val >> 1


def open_flac_as_wav(file_path):
    """Standalone helper function that accepts a FLAC file path and returns a standard wave.Wave_read object."""
    decoder = PurePythonFlacDecoder(file_path)
    pcm_bytes = bytes(decoder.pcm_data)
    data_length = len(pcm_bytes)

    wav_io = io.BytesIO()
    riff_chunk_size = 36 + data_length
    audio_format = 1  # PCM
    channels = decoder.channels
    sample_rate = decoder.sample_rate
    bits_per_sample = 16  # Forced standard 16-bit downsampled layout
    block_align = channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align

    wav_io.write(b"RIFF")
    wav_io.write(struct.pack("<I", riff_chunk_size))
    wav_io.write(b"WAVE")

    # fmt sub-chunk
    wav_io.write(b"fmt ")
    wav_io.write(struct.pack("<I", 16))
    wav_io.write(struct.pack("<H", audio_format))
    wav_io.write(struct.pack("<H", channels))
    wav_io.write(struct.pack("<I", sample_rate))
    wav_io.write(struct.pack("<I", byte_rate))
    wav_io.write(struct.pack("<H", block_align))
    wav_io.write(struct.pack("<H", bits_per_sample))

    # data sub-chunk
    wav_io.write(b"data")
    wav_io.write(struct.pack("<I", data_length))
    wav_io.write(pcm_bytes)

    wav_io.seek(0)
    return wave.open(wav_io, "rb")


class DirectWavReader:
    """A lightweight wrapper that mimics wave.Wave_read with context manager support and minimal memory bloat."""

    def __init__(self, file_path, wave_obj):
        self._file_path = file_path
        self._wave_obj = wave_obj

    def getnchannels(self):
        return self._wave_obj.getnchannels()

    def getsampwidth(self):
        return self._wave_obj.getsampwidth()

    def getframerate(self):
        return self._wave_obj.getframerate()

    def getnframes(self):
        return self._wave_obj.getnframes()

    def readframes(self, n):
        return self._wave_obj.readframes(n)

    def close(self):
        try:
            self._wave_obj.close()
        finally:
            # Clean up the temporary file from disk when closed
            if os.path.exists(self._file_path):
                os.remove(self._file_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def open_via_ffmpeg_as_wav(file_path):
    """
    Decodes audio via FFmpeg straight to disk and returns a lightweight
    context-manager compatible reader object.
    """
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav.close()  # Close handle so FFmpeg can write to it cleanly

    cmd = [
        "ffmpeg",
        "-v",
        "quiet",
        "-i",
        file_path,
        "-acodec",
        "pcm_s16le",
        "-y",
        temp_wav.name,
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        if os.path.exists(temp_wav.name):
            os.remove(temp_wav.name)
        raise RuntimeError(
            f"FFmpeg decoding failed: {result.stderr.decode('utf-8', errors='ignore')}"
        )

    wav_obj = wave.open(temp_wav.name, "rb")
    return DirectWavReader(temp_wav.name, wav_obj)


def open_wave_or_flac_for_reading(path):
    try:
        # should be fast when it works at all, and supports more
        # formats
        return open_via_ffmpeg_as_wav(path)
    except:
        pass
    try:
        # very slow but lets us use FLAC where we otherwise could not
        return open_flac_as_wav(path)
    except:
        pass
    # normal WAV-only path, limited to old "traditional" Windows WAV
    # codec support (8bit unsigned/16bit signed PCM)
    return wave.open(path, "rb")


def read_wav_mono(path):
    """Read a WAV file and return (samples, framerate). FLAC also
    works, possibly much more slowly. other formats may also work if
    you have ffmpeg installed and available on your PATH.

    samples is a flat list of ints/floats, mono (channels averaged if the
    file is stereo/multi-channel), DC offset NOT removed (the edge detector
    below handles that implicitly via zero-crossing + hysteresis).

    """
    with open_wave_or_flac_for_reading(path) as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    samples = _unpack_samples(raw, sampwidth)

    if nchannels > 1:
        mono = []
        for i in range(0, len(samples) - nchannels + 1, nchannels):
            frame = samples[i : i + nchannels]
            mono.append(sum(frame) / nchannels)
        samples = mono

    return samples, framerate


def _unpack_samples(raw, sampwidth):
    if sampwidth == 1:
        # WAV 8-bit PCM is unsigned, centered on 128
        return [b - 128 for b in raw]
    elif sampwidth == 2:
        count = len(raw) // 2
        return list(struct.unpack("<%dh" % count, raw[: count * 2]))
    elif sampwidth == 3:
        count = len(raw) // 3
        out = [0] * count
        for i in range(count):
            b0, b1, b2 = raw[i * 3], raw[i * 3 + 1], raw[i * 3 + 2]
            v = b0 | (b1 << 8) | (b2 << 16)
            if v & 0x800000:
                v -= 0x1000000
            out[i] = v
        return out
    elif sampwidth == 4:
        count = len(raw) // 4
        return list(struct.unpack("<%di" % count, raw[: count * 4]))
    else:
        raise ValueError("Unsupported sample width: %d bytes" % sampwidth)


# --------------------------------------------------------------------------
# Edge detection -> pulse (cycle) list
# --------------------------------------------------------------------------


def find_rising_edges(
    samples,
    threshold_ratio,
    agc=True,
    agc_attack=0.3,
    agc_release=0.005,  # Restore original release speed to pass tests
    agc_floor_ratio=0.02,
):
    """Schmitt-trigger rising zero-crossing detector with DC bias tracking.

    Real MSX hardware uses an AC-coupled input. This function mimics that by
    tracking the local 'center' of the wave (bias) and subtracting it.
    This allows the Schmitt trigger to arm and fire even on recordings with
    significant DC offset (like File 4) while maintaining compatibility
    with existing tests.
    """
    peak_file = max((abs(s) for s in samples), default=0) or 1

    edges = []
    armed = True

    # DC Bias Tracker (slow average)
    bias = float(samples[0])
    bias_alpha = 0.002

    # AGC state (tracked relative to the AC signal)
    floor = peak_file * agc_floor_ratio
    envelope = floor

    prev_v_adj = samples[0] - bias

    for i in range(1, len(samples)):
        cur = float(samples[i])

        # 1. Update Bias Tracker (find the center of the wave)
        bias = bias * (1.0 - bias_alpha) + cur * bias_alpha

        # 2. Extract the AC component
        v_adj = cur - bias

        if agc:
            abs_v = abs(v_adj)
            if abs_v > envelope:
                envelope = envelope * (1.0 - agc_attack) + abs_v * agc_attack
            else:
                envelope = envelope * (1.0 - agc_release) + abs_v * agc_release

            if envelope < floor:
                envelope = floor
            local_threshold = envelope * threshold_ratio
        else:
            local_threshold = peak_file * threshold_ratio

        # 3. Schmitt Trigger logic on the biased-removed signal
        if not armed:
            if v_adj < -local_threshold:
                armed = True
        else:
            # Check for rising crossing of the center (v_adj = 0)
            if prev_v_adj < 0 <= v_adj:
                # Linear interpolation for sub-sample accuracy
                frac = (
                    (0 - prev_v_adj) / (v_adj - prev_v_adj)
                    if v_adj != prev_v_adj
                    else 0.0
                )
                edges.append((i - 1) + frac)
                armed = False

        prev_v_adj = v_adj

    return edges


def _interp_zero(i0, v0, v1):
    if v1 == v0:
        return float(i0)
    frac = (0 - v0) / (v1 - v0)
    frac = min(max(frac, 0.0), 1.0)
    return i0 + frac


def edges_to_periods(edges):
    """Convert consecutive rising-edge positions into a list of cycle
    lengths (in samples), one entry per full cycle. Working in the sample-
    length ("period") domain rather than frequency makes the long/short
    midpoint classification used by decode() a simple comparison.
    """
    periods = []
    for i in range(len(edges) - 1):
        length = edges[i + 1] - edges[i]
        if length <= 0:
            continue
        periods.append(length)
    return periods


# --------------------------------------------------------------------------
# Bit / byte / block decoding
# --------------------------------------------------------------------------


def _best_baud_match(freq, tolerance):
    """Return (baud, f0, f1) for the baud rate whose bit-1 frequency is the
    closest relative match to freq, or None if none are within tolerance.
    """
    best = None
    best_err = None
    for baud, (f0, f1) in BAUD_TABLE.items():
        err = abs(freq - f1) / f1
        if err <= tolerance and (best_err is None or err < best_err):
            best = (baud, f0, f1)
            best_err = err
    return best


def _trim_block_edges(data, confidences, threshold):
    """Drop low-confidence bytes from the start and/or end of a block,
    stopping as soon as a byte at or above threshold is encountered.
    """
    start = 0
    end = len(data)
    while start < end and confidences[start] < threshold:
        start += 1
    while end > start and confidences[end - 1] < threshold:
        end -= 1
    return bytes(data[start:end]), list(confidences[start:end])


def decode(
    periods,
    framerate,
    tolerance=0.30,
    min_pilot_pulses=40,
    adapt=True,
    adapt_rate=0.1,
    adapt_clamp=0.35,
    strict_stop=True,
    stop_bits=2,
    edge_trim=True,
    edge_trim_threshold=0.5,
    max_gap_multiple=5.0,
    verbose=False,
):
    """Decode a list of cycle-length ("period", in samples) values into a
    list of (baud, bytes, confidence) blocks.
    """
    blocks = []
    i = 0
    n = len(periods)

    state = "SEARCH"
    pilot_count = 0
    candidate = None
    baud = None

    long_nom = short_nom = None
    long_avg = short_avg = None
    threshold = None

    current = bytearray()
    current_confidences = []

    ones_run = 0
    flushed_this_run = False

    def log(msg):
        if verbose:
            print(msg, file=sys.stderr)

    def classify(p):
        if p < threshold:
            return "short"
        if p <= long_avg * max_gap_multiple:
            return "long"
        return "gap"

    def pulse_confidence(p):
        gap = long_avg - short_avg
        if gap <= 0:
            return 0.0
        conf = abs(p - threshold) / (gap / 2)
        return min(1.0, max(0.0, conf))

    def update_long(p):
        nonlocal long_avg, threshold
        if not adapt:
            return
        if abs(p - long_nom) <= long_nom * adapt_clamp:
            long_avg = long_avg * (1 - adapt_rate) + p * adapt_rate
            threshold = (long_avg + short_avg) / 2

    def update_short(p):
        nonlocal short_avg, threshold
        if not adapt:
            return
        if abs(p - short_nom) <= short_nom * adapt_clamp:
            short_avg = short_avg * (1 - adapt_rate) + p * adapt_rate
            threshold = (long_avg + short_avg) / 2

    def read_bit(idx):
        if idx >= n:
            return None, idx, 0.0
        p = periods[idx]
        c = classify(p)
        if c == "long":
            conf = pulse_confidence(p)
            update_long(p)
            return 0, idx + 1, conf
        elif c == "short":
            if idx + 1 < n and classify(periods[idx + 1]) == "short":
                p2 = periods[idx + 1]
                conf = (pulse_confidence(p) + pulse_confidence(p2)) / 2
                update_short(p)
                update_short(p2)
                return 1, idx + 2, conf
            return None, idx + 1, 0.0
        else:
            return None, idx + 1, 0.0

    def flush_block():
        if not current:
            return
        data = bytes(current)
        confs = list(current_confidences)
        if edge_trim:
            trimmed_data, trimmed_confs = _trim_block_edges(
                data, confs, edge_trim_threshold
            )
            dropped = len(data) - len(trimmed_data)
            if dropped:
                log(
                    "[trim] dropped %d low-confidence byte(s) from block edges "
                    "(kept %d of %d)" % (dropped, len(trimmed_data), len(data))
                )
            data, confs = trimmed_data, trimmed_confs
        if not data:
            log("[block] entire block trimmed away as low-confidence noise")
            return
        block_conf = sum(confs) / len(confs)
        blocks.append((baud, data, block_conf))
        log("[block] %d bytes decoded, confidence=%.3f" % (len(data), block_conf))

    while i < n:
        period = periods[i]

        if state == "SEARCH":
            freq = framerate / period if period > 0 else 0
            match = _best_baud_match(freq, tolerance)
            if match is not None:
                if pilot_count == 0 or match[0] != candidate[0]:
                    pilot_count = 1
                    candidate = match
                else:
                    pilot_count += 1
                i += 1
                if pilot_count >= min_pilot_pulses:
                    baud, f0, f1 = candidate
                    long_nom = framerate / f0
                    short_nom = framerate / f1
                    long_avg = long_nom
                    short_avg = short_nom
                    threshold = (long_avg + short_avg) / 2
                    log(
                        "[pilot] locked baud=%d (f0=%gHz f1=%gHz) near pulse %d"
                        % (baud, f0, f1, i)
                    )
                    state = "SYNCED"
                    ones_run = 0
                    flushed_this_run = False
            else:
                pilot_count = 0
                i += 1

        elif state == "SYNCED":
            c = classify(period)
            if c == "short":
                update_short(period)
                ones_run += 1
                i += 1
                if ones_run >= min_pilot_pulses and not flushed_this_run:
                    window = periods[max(0, i - min_pilot_pulses) : i]
                    avg_period = sum(window) / len(window)
                    freq = framerate / avg_period if avg_period > 0 else 0
                    match = _best_baud_match(freq, tolerance)
                    if match is not None and match[0] != baud:
                        baud, f0, f1 = match
                        long_nom = framerate / f0
                        short_nom = framerate / f1
                        long_avg = long_nom
                        short_avg = short_nom
                        threshold = (long_avg + short_avg) / 2
                        log(
                            "[pilot] re-locked baud=%d (f0=%gHz f1=%gHz) near pulse %d"
                            % (baud, f0, f1, i)
                        )
                    flush_block()
                    current = bytearray()
                    current_confidences = []
                    flushed_this_run = True
            elif c == "long":
                ones_run = 0
                flushed_this_run = False
                state = "BYTE"
            else:
                log("[gap] dropout/silence period (%.1f samples) - resyncing" % period)
                flush_block()
                current = bytearray()
                current_confidences = []
                ones_run = 0
                flushed_this_run = False
                state = "SEARCH"
                pilot_count = 0
                i += 1

        elif state == "BYTE":
            start_bit, j, sconf = read_bit(i)
            if start_bit != 0:
                state = "SEARCH"
                pilot_count = 0
                i = j if start_bit is not None else i + 1
                continue
            i = j

            value = 0
            ok = True
            bit_confs = [sconf]
            for bitpos in range(8):
                bit, j, c = read_bit(i)
                if bit is None:
                    ok = False
                    break
                value |= bit << bitpos
                bit_confs.append(c)
                i = j
            if not ok:
                state = "SEARCH"
                pilot_count = 0
                continue

            if strict_stop:
                for _ in range(stop_bits):
                    bit, j, c = read_bit(i)
                    if bit != 1:
                        ok = False
                        break
                    bit_confs.append(c)
                    i = j
                if not ok:
                    state = "SEARCH"
                    pilot_count = 0
                    continue

            current.append(value)
            current_confidences.append(sum(bit_confs) / len(bit_confs))
            state = "SYNCED"
            ones_run = 0
            flushed_this_run = False

    flush_block()

    return blocks


# --------------------------------------------------------------------------
# CAS output
# --------------------------------------------------------------------------


def write_cas(path, blocks, pad=False):
    """Write (baud, data, confidence) blocks to a .cas file."""
    with open(path, "wb") as f:
        for _baud, data, _conf in blocks:
            if pad:
                pad_len = (-f.tell()) % 8
                if pad_len:
                    f.write(b"\x00" * pad_len)
            f.write(CAS_HEADER)
            f.write(data)


# --------------------------------------------------------------------------
# Self-tests (run with --test)
# --------------------------------------------------------------------------

_TEST_FRAMERATE = 44100


def _test_add_cycle(
    samples, freq, amp, framerate=_TEST_FRAMERATE, noise_amp=0.0, jitter=0.0, rng=None
):
    rng = rng or random
    actual_freq = freq * (1 + rng.uniform(-jitter, jitter))
    n = max(2, int(round(framerate / actual_freq)))
    for k in range(n):
        v = amp * math.sin(2 * math.pi * k / n)
        v += rng.uniform(-noise_amp, noise_amp) * amp
        samples.append(v)


def _test_gen_block(
    samples,
    baud,
    payload,
    pilot_seconds,
    amp=20000,
    noise_amp=0.0,
    jitter=0.0,
    stop_bits=2,
    rng=None,
    framerate=_TEST_FRAMERATE,
):
    f0, f1 = baud, baud * 2
    n_pilot_bits = int(pilot_seconds * baud)
    for _ in range(n_pilot_bits):
        _test_add_cycle(samples, f1, amp, framerate, noise_amp, jitter, rng)
        _test_add_cycle(samples, f1, amp, framerate, noise_amp, jitter, rng)
    for byte in payload:
        _test_add_cycle(samples, f0, amp, framerate, noise_amp, jitter, rng)
        for b in range(8):
            bit = (byte >> b) & 1
            if bit:
                _test_add_cycle(samples, f1, amp, framerate, noise_amp, jitter, rng)
                _test_add_cycle(samples, f1, amp, framerate, noise_amp, jitter, rng)
            else:
                _test_add_cycle(samples, f0, amp, framerate, noise_amp, jitter, rng)
        for _ in range(stop_bits):
            _test_add_cycle(samples, f1, amp, framerate, noise_amp, jitter, rng)
            _test_add_cycle(samples, f1, amp, framerate, noise_amp, jitter, rng)


def _test_decode_samples(
    samples, framerate=_TEST_FRAMERATE, threshold_ratio=0.2, agc=True, **decode_kwargs
):
    edges = find_rising_edges(samples, threshold_ratio, agc=agc)
    periods = edges_to_periods(edges)
    return decode(periods, framerate, **decode_kwargs)


def _t_basic_multiblock():
    samples = []
    _test_gen_block(samples, 1200, b"HEADERID", 1.0)
    _test_gen_block(samples, 2400, bytes(range(256)), 0.3)
    _test_gen_block(samples, 2400, b"Second block at 2400 baud, testing 123.", 0.5)
    samples += [0] * int(_TEST_FRAMERATE * 0.3)

    blocks = _test_decode_samples(samples)
    if len(blocks) != 3:
        return False, "expected 3 blocks, got %d" % len(blocks)
    if blocks[0][1] != b"HEADERID":
        return False, "block 0 mismatch: %r" % (blocks[0][1],)
    if blocks[1][1] != bytes(range(256)):
        return False, "block 1 mismatch"
    if blocks[2][1] != b"Second block at 2400 baud, testing 123.":
        return False, "block 2 mismatch: %r" % (blocks[2][1],)
    return True, "ok"


def _t_strict_vs_lenient_stop_bits():
    rng = random.Random(123)
    payload = b"IRRSTOP1"
    samples = []
    f0, f1 = 1200, 2400
    for _ in range(int(1.0 * 1200)):
        _test_add_cycle(samples, f1, 20000, rng=rng)
        _test_add_cycle(samples, f1, 20000, rng=rng)
    for idx, byte in enumerate(payload):
        _test_add_cycle(samples, f0, 20000, rng=rng)
        for b in range(8):
            bit = (byte >> b) & 1
            if bit:
                _test_add_cycle(samples, f1, 20000, rng=rng)
                _test_add_cycle(samples, f1, 20000, rng=rng)
            else:
                _test_add_cycle(samples, f0, 20000, rng=rng)
        stop_n = 1 if idx % 2 == 0 else 3
        for _ in range(stop_n):
            _test_add_cycle(samples, f1, 20000, rng=rng)
            _test_add_cycle(samples, f1, 20000, rng=rng)
    samples += [0] * int(_TEST_FRAMERATE * 0.3)

    strict_blocks = _test_decode_samples(samples, strict_stop=True, stop_bits=2)
    lenient_blocks = _test_decode_samples(samples, strict_stop=False)

    problems = []
    if any(b[1] == payload for b in strict_blocks):
        problems.append(
            "strict mode unexpectedly decoded the full irregular-stop payload"
        )
    if not any(b[1] == payload for b in lenient_blocks):
        problems.append(
            "lenient mode failed to decode the irregular-stop payload: got %r"
            % ([b[1] for b in lenient_blocks],)
        )
    return (not problems), "; ".join(problems) or "ok"


def _t_noise_jitter_tolerance():
    rng = random.Random(42)
    samples = []
    _test_gen_block(
        samples, 1200, b"HEADERID", 1.0, noise_amp=0.15, jitter=0.08, rng=rng
    )
    _test_gen_block(
        samples, 1200, bytes(range(256)), 0.3, noise_amp=0.15, jitter=0.08, rng=rng
    )
    samples += [0] * int(_TEST_FRAMERATE * 0.3)

    blocks = _test_decode_samples(samples)
    if len(blocks) != 2:
        return False, "expected 2 blocks, got %d" % len(blocks)
    if blocks[0][1] != b"HEADERID":
        return False, "header block mismatch: %r" % (blocks[0][1],)
    if blocks[1][1] != bytes(range(256)):
        return False, "binary block mismatch"
    return True, "ok"


def _t_agc_recovers_quiet_block():
    payload_loud = b"LOUDMSG1"
    payload_quiet = b"QUIETMS2"
    samples = []
    _test_gen_block(samples, 1200, payload_loud, 0.5, amp=20000)
    samples += [0] * 200
    _test_gen_block(samples, 1200, payload_quiet, 0.5, amp=20000 * 0.08)
    samples += [0] * int(_TEST_FRAMERATE * 0.2)

    agc_blocks = _test_decode_samples(samples, agc=True)
    noagc_blocks = _test_decode_samples(samples, agc=False)
    agc_payloads = [b[1] for b in agc_blocks]
    noagc_payloads = [b[1] for b in noagc_blocks]

    if payload_loud not in agc_payloads or payload_quiet not in agc_payloads:
        return False, "AGC failed to recover both payloads: got %r" % (agc_payloads,)
    if payload_quiet in noagc_payloads:
        return False, (
            "no-agc unexpectedly recovered the quiet payload too - "
            "test isn't discriminating, needs a bigger amplitude gap"
        )
    return True, "ok"


def _t_gap_does_not_produce_spurious_byte():
    payload1 = b"FIRSTMSG"
    payload2 = b"SECONDMSG"
    samples = []
    _test_gen_block(samples, 1200, payload1, 0.5, amp=20000)
    samples += [0] * 2000
    _test_gen_block(samples, 1200, payload2, 0.5, amp=20000)
    samples += [0] * int(_TEST_FRAMERATE * 0.2)

    blocks = _test_decode_samples(samples)
    payloads = [b[1] for b in blocks]
    if payload1 not in payloads:
        return False, "block 1 wasn't decoded cleanly: got %r" % (payloads,)
    if payload2 not in payloads:
        return False, "block 2 wasn't decoded cleanly: got %r" % (payloads,)
    return True, "ok"


def _t_confidence_filtering():
    rng = random.Random(99)
    samples = []
    _test_gen_block(
        samples, 1200, b"CLEANMSG", 0.5, noise_amp=0.02, jitter=0.01, rng=rng
    )
    samples += [0] * 300
    _test_gen_block(
        samples, 1200, b"GARBLED!", 0.5, noise_amp=0.35, jitter=0.25, rng=rng
    )
    samples += [0] * int(_TEST_FRAMERATE * 0.2)

    blocks = _test_decode_samples(samples)
    if len(blocks) < 2:
        return False, "expected at least 2 blocks, got %d" % len(blocks)
    clean, garbled = blocks[0], blocks[1]
    if clean[1] != b"CLEANMSG":
        return False, "clean block content mismatch: %r" % (clean[1],)
    if clean[2] < 0.85:
        return False, "clean block confidence unexpectedly low: %.3f" % clean[2]
    if garbled[2] >= clean[2]:
        return False, "garbled block confidence not lower than clean block's"
    kept = [b for b in blocks if b[2] >= 0.85]
    if garbled in kept:
        return False, "garbled block incorrectly passed threshold"
    return True, "ok"


def _t_edge_trim_mixed_confidence():
    data = bytes([0xFF, 0xFE, ord("H"), ord("I"), 0xFD])
    confs = [0.1, 0.2, 0.95, 0.93, 0.15]
    trimmed_data, trimmed_confs = _trim_block_edges(data, confs, 0.5)
    if trimmed_data != b"HI":
        return False, "expected b'HI', got %r" % (trimmed_data,)
    if trimmed_confs != [0.95, 0.93]:
        return False, "confidence list wasn't trimmed correctly: %r" % (trimmed_confs,)
    return True, "ok"


def _t_edge_trim_all_low_confidence():
    data = bytes([1, 2, 3])
    confs = [0.1, 0.2, 0.05]
    trimmed_data, trimmed_confs = _trim_block_edges(data, confs, 0.5)
    if trimmed_data != b"" or trimmed_confs != []:
        return False, "expected an empty result, got %r" % (trimmed_data,)
    return True, "ok"


def _t_edge_trim_untouched_interior():
    data = bytes([ord("A"), 0x00, ord("B")])
    confs = [0.9, 0.1, 0.9]
    trimmed_data, trimmed_confs = _trim_block_edges(data, confs, 0.5)
    if trimmed_data != data or trimmed_confs != confs:
        return False, "interior byte was incorrectly trimmed: %r" % (trimmed_data,)
    return True, "ok"


def _t_pad_alignment():
    blocks = [(1200, b"ABC", 1.0), (1200, b"HELLO", 1.0)]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.cas")
        write_cas(path, blocks, pad=True)
        with open(path, "rb") as f:
            data = f.read()

    offsets = []
    idx = 0
    while True:
        pos = data.find(CAS_HEADER, idx)
        if pos == -1:
            break
        offsets.append(pos)
        idx = pos + 1

    if any(o % 8 != 0 for o in offsets):
        return False, "header offsets not 8-byte aligned: %r" % offsets
    parts = data.split(CAS_HEADER)[1:]
    if not parts[0].startswith(b"ABC"):
        return False, "block 0 content wrong: %r" % (parts[0],)
    if parts[1] != b"HELLO":
        return False, "block 1 content wrong: %r" % (parts[1],)
    return True, "ok"


_SELF_TESTS = [
    ("basic multi-block decode", _t_basic_multiblock),
    ("strict vs. lenient stop bits", _t_strict_vs_lenient_stop_bits),
    ("noise & jitter tolerance", _t_noise_jitter_tolerance),
    ("AGC recovers a quiet block", _t_agc_recovers_quiet_block),
    (
        "gap/dropout doesn't produce a spurious byte",
        _t_gap_does_not_produce_spurious_byte,
    ),
    ("confidence-based block filtering", _t_confidence_filtering),
    ("edge trim: mixed confidence", _t_edge_trim_mixed_confidence),
    ("edge trim: all low confidence", _t_edge_trim_all_low_confidence),
    ("edge trim: interior untouched", _t_edge_trim_untouched_interior),
    ("CAS --pad alignment", _t_pad_alignment),
]


def run_self_tests():
    passed = failed = 0
    for name, fn in _SELF_TESTS:
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, "exception: %r" % (e,)
        print(
            "[%s] %s%s"
            % (
                "PASS" if ok else "FAIL",
                name,
                "" if (not msg or msg == "ok") else (" - " + msg),
            )
        )
        if ok:
            passed += 1
        else:
            failed += 1
    print("%d passed, %d failed" % (passed, failed))
    return failed == 0


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Decode MSX cassette audio (WAV) into a .CAS file."
    )
    parser.add_argument("input", nargs="?", help="input .wav file")
    parser.add_argument("output", nargs="?", help="output .cas file")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.30,
        help="relative frequency tolerance used only for initial pilot-tone "
        "baud detection, as a fraction (default: 0.30)",
    )
    parser.add_argument(
        "--min-pilot-pulses",
        type=int,
        default=40,
        help="consecutive pilot pulses required to lock onto a baud rate (default: 40)",
    )
    parser.add_argument(
        "--threshold-ratio",
        type=float,
        default=0.2,
        help="Schmitt-trigger threshold as a fraction of the amplitude envelope (default: 0.2)",
    )
    parser.add_argument(
        "--no-agc",
        action="store_true",
        help="disable local amplitude tracking (AGC)",
    )
    parser.add_argument(
        "--agc-attack",
        type=float,
        default=0.3,
        help="AGC attack rate, 0-1 (default: 0.3)",
    )
    parser.add_argument(
        "--agc-release",
        type=float,
        default=0.005,
        help="AGC release rate, 0-1 (default: 0.005)",
    )
    parser.add_argument(
        "--agc-floor-ratio",
        type=float,
        default=0.02,
        help="floor for the AGC envelope (default: 0.02)",
    )
    parser.add_argument(
        "--no-adapt",
        action="store_true",
        help="disable adaptive threshold tracking",
    )
    parser.add_argument(
        "--adapt-rate",
        type=float,
        default=0.1,
        help="adaptive threshold tracking rate, 0-1 (default: 0.1)",
    )
    parser.add_argument(
        "--adapt-clamp",
        type=float,
        default=0.35,
        help="maximum fractional deviation for adaptation (default: 0.35)",
    )
    parser.add_argument(
        "--lenient-stop-bits",
        action="store_true",
        help="don't require a fixed number of stop bits after each byte",
    )
    parser.add_argument(
        "--stop-bits",
        type=int,
        default=2,
        help="number of stop bits required per byte in strict mode (default: 2)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.8,
        help="minimum block confidence threshold (default: 0.8)",
    )
    parser.add_argument(
        "--no-edge-trim",
        action="store_true",
        help="disable automatic trimming of low-confidence bytes",
    )
    parser.add_argument(
        "--edge-trim-threshold",
        type=float,
        default=0.5,
        help="edge trim confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--max-gap-multiple",
        type=float,
        default=5.0,
        help="maximum gap multiple for dropout detection (default: 5.0)",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Apply built-in MSX hardware filter model for robust compatibility across tapes",
    )
    parser.add_argument(
        "--pad",
        action="store_true",
        help="insert 0-7 zero-byte padding before each CAS block header",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="run internal self-tests and exit",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print decoding progress to stderr"
    )
    args = parser.parse_args()

    if args.test:
        sys.exit(0 if run_self_tests() else 1)

    if args.input is None or args.output is None:
        parser.error("input and output are required (unless --test is given)")

    print("Reading %s ..." % args.input, file=sys.stderr)
    samples, framerate = read_wav_mono(args.input)
    print("%d samples at %d Hz" % (len(samples), framerate), file=sys.stderr)

    if args.filter:
        samples = apply_msx_hardware_filter(samples, framerate)

    edges = find_rising_edges(
        samples,
        args.threshold_ratio,
        agc=not args.no_agc,
        agc_attack=args.agc_attack,
        agc_release=args.agc_release,
        agc_floor_ratio=args.agc_floor_ratio,
    )
    print("%d rising edges detected" % len(edges), file=sys.stderr)

    periods = edges_to_periods(edges)

    blocks = decode(
        periods,
        framerate,
        tolerance=args.tolerance,
        min_pilot_pulses=args.min_pilot_pulses,
        adapt=not args.no_adapt,
        adapt_rate=args.adapt_rate,
        adapt_clamp=args.adapt_clamp,
        strict_stop=not args.lenient_stop_bits,
        stop_bits=args.stop_bits,
        edge_trim=not args.no_edge_trim,
        edge_trim_threshold=args.edge_trim_threshold,
        max_gap_multiple=args.max_gap_multiple,
        verbose=args.verbose,
    )

    if not blocks:
        print(
            "No data blocks decoded - try adjusting parameters.",
            file=sys.stderr,
        )

    kept = [b for b in blocks if b[2] >= args.min_confidence]

    if not kept:
        print(
            "No blocks met the --min-confidence threshold (%.2f) - nothing written to %s"
            % (args.min_confidence, args.output),
            file=sys.stderr,
        )
        sys.exit(1)

    write_cas(args.output, kept, pad=args.pad)

    print(
        "Decoded %d block(s), wrote %d (min-confidence=%.2f):"
        % (len(blocks), len(kept), args.min_confidence),
        file=sys.stderr,
    )
    for idx, (baud, data, conf) in enumerate(blocks):
        status = "kept" if conf >= args.min_confidence else "DROPPED (below threshold)"
        print(
            "  block %d: baud=%d bytes=%d confidence=%.3f [%s]"
            % (idx, baud, len(data), conf, status),
            file=sys.stderr,
        )

    print("Wrote %s" % args.output, file=sys.stderr)


if __name__ == "__main__":
    main()
