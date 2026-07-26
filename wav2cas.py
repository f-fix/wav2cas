#!/usr/bin/env python3
"""wav2cas.py - Decode MSX and MSX-like cassette audio (WAV) into a .CAS file.

Pure Python, standard library only (wave, struct, argparse, sys).

FORMAT ASSUMPTIONS (standard MSX BIOS cassette encoding, FSK / Kansas-City style):

  * A "0" data bit is encoded as ONE cycle at the base frequency for the
    current baud rate.

  * A "1" data bit is encoded as TWO cycles at double the base frequency.

  * Supported baud rates (auto-detected from the pilot tone
    frequency), covering the range typical MSX tapes and FSK turbo-loaders
    use, and also some adjacent systems:

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

  * STOP BITS: by default the decoder is strict, exactly like older
    tape tools - after the 8 data bits it requires --stop-bits
    (default 2) genuine "1" bits before accepting the byte, and drops
    the byte (and resyncs on the pilot search) if that doesn't
    hold. This is what keeps stray noise or pilot/data transition
    artifacts from being misread as spurious extra bytes right before
    or after a real block. Pass --lenient-stop-bits to fall back to
    more permissive behavior that just looks for the next start bit
    with no fixed count (useful for tapes whose stop-bit timing is
    irregular).

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
    spurious blocks decoded from a garbled/noisy stretch of tape.

  * Every time a new pilot tone run is detected after at least one byte has
    already been decoded, the previously accumulated bytes are flushed out
    as one completed block, and decoding starts fresh for the next block.
    The baud rate is re-checked at every such pilot, since nothing stops a
    tape from switching baud rates between blocks.

  * Each decoded block is written to the .cas output preceded by the
    standard 8-byte CAS block marker:

        1F A6 DE BA CC 13 7D 74

    (this is the same convention used by real MSX emulators' .cas
    files - it lets a loader locate block boundaries in the file since
    a raw .cas does not otherwise store timing/pilot
    information. Essentially it represents "silence").

  * With --pad, 0-7 zero bytes are inserted before each CAS block header so
    it starts at a file offset that's a multiple of 8, which some MSX tools
    expect. Off by default.

LIMITATIONS:

  * Only handles integer PCM WAV data (8/16/24/32-bit), not floating
    point or more obscure codecs. However, there is also support (on
    by default) for converting all inputs to 16-bit signed linear PCM
    using external `ffmpeg`. The `--no-native-formats` flag turns it
    off. it will automatically fall back to a slow internal decoder
    for FLAC inputs if `ffmpeg` doesn't work, but
    `--no-native-formats` likewise turns this off and makes it
    WAV-only. The `--no-ffmpeg` flag skips ffmpeg but still allows
    slow pure Python FLAC decoding.

"""

import argparse
import array
import base64
import gzip
import io
import math
import os
import random
import struct
import subprocess  # comment out as necessary in sandboxed environments
import sys
import tempfile
import types
import wave

# In a sandboxed environment `subprocess` might not be available. If
# that's the case, comment out the `import subprocess` line above and
# let this mock replace it:

if "subprocess" not in sys.modules:
    sys.modules["subprocess"] = type(
        "Mock",
        (types.ModuleType,),
        {
            "__getattr__": lambda s, n: (_ for _ in ()).throw(
                OSError(f"subprocess.{n} disabled")
            ),
        },
    )("subprocess")

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


def apply_band_filter(samples, framerate):
    """Crudely simulates cassette hardware input circuit response.

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

    # Use a single output array and single pass to save memory and time
    filtered = array.array("f", [0.0]) * len(samples)

    prev_in = float(samples[0])
    prev_hp_out = 0.0
    prev_lp_val = 0.0

    for i in range(len(samples)):
        cur_in = float(samples[i])

        # High-pass step
        hp_out = hp_alpha * (prev_hp_out + cur_in - prev_in)
        prev_hp_out = hp_out
        prev_in = cur_in

        # Low-pass step
        lp_out = prev_lp_val + lp_alpha * (hp_out - prev_lp_val)
        prev_lp_val = lp_out

        filtered[i] = lp_out

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
                and (sync_marker[1] & 0xFE == 0xF8 or (sync_marker[1] & 0xFC) == 0xF0)
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

        header_start = self._read_bytes(
            2
        )  # sync(14) + reserved(1) + blocking strategy(1)
        if not header_start or len(header_start) < 2:
            return

        bs_sr = self._read_bytes(1)  # block_size_code(4) | sample_rate_code(4)
        if not bs_sr:
            return
        block_size_type = (bs_sr[0] >> 4) & 0x0F
        sample_rate_code = bs_sr[0] & 0x0F

        ch_bps = self._read_bytes(
            1
        )  # channel_assignment(4) | sample_size_code(3) | reserved(1)
        if not ch_bps:
            return
        channel_assignment = (ch_bps[0] >> 4) & 0x0F

        # variable-length UTF-8-style encoded frame/sample number: the count of
        # leading 1-bits in the first byte gives the number of continuation
        # bytes that follow (same scheme as UTF-8 codepoint encoding)
        first = self._read_bytes(1)
        if not first:
            return
        first_byte = first[0]
        if first_byte & 0x80 == 0x00:
            continuation_bytes = 0
        elif first_byte & 0xE0 == 0xC0:
            continuation_bytes = 1
        elif first_byte & 0xF0 == 0xE0:
            continuation_bytes = 2
        elif first_byte & 0xF8 == 0xF0:
            continuation_bytes = 3
        elif first_byte & 0xFC == 0xF8:
            continuation_bytes = 4
        elif first_byte & 0xFE == 0xFC:
            continuation_bytes = 5
        elif first_byte == 0xFE:
            continuation_bytes = 6
        else:
            continuation_bytes = 0
        if continuation_bytes and not self._read_bytes(continuation_bytes):
            return

        block_size = self._get_block_size(block_size_type)
        if block_size is None:
            return

        # extra explicit sample-rate bytes (not otherwise used, just consumed
        # so the CRC byte below lines up correctly)
        if sample_rate_code == 12:
            self._read_bytes(1)
        elif sample_rate_code in (13, 14):
            self._read_bytes(2)

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

            # Apply Fixed predictors
            if order == 1:
                for i in range(order, block_size):
                    samples[i] += samples[i - 1]
            elif order == 2:
                for i in range(order, block_size):
                    samples[i] += 2 * samples[i - 1] - samples[i - 2]
            elif order == 3:
                for i in range(order, block_size):
                    samples[i] += (
                        3 * samples[i - 1] - 3 * samples[i - 2] + samples[i - 3]
                    )
            elif order == 4:
                for i in range(order, block_size):
                    samples[i] += (
                        4 * samples[i - 1]
                        - 6 * samples[i - 2]
                        + 4 * samples[i - 3]
                        - samples[i - 4]
                    )

        elif 32 <= sub_type <= 63:
            order = sub_type - 31
            for i in range(order):
                samples[i] = self._read_signed_bits(self.bits_per_sample)

            # Properly read LPC header: precision (4 bits), shift (5 bits signed), and coefficients
            prec = self._read_bits(4) + 1
            shift = self._read_signed_bits(5)
            coeffs = [self._read_signed_bits(prec) for _ in range(order)]

            samples = self._parse_residual(block_size, order, samples)

            # Apply LPC predictors
            for i in range(order, block_size):
                pred = sum(coeffs[j] * samples[i - 1 - j] for j in range(order))
                if shift >= 0:
                    samples[i] += pred >> shift
                else:
                    samples[i] += pred << (-shift)

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

        idx = order
        for p in range(num_partitions):
            param = self._read_bits(param_len)
            if param is None:
                break

            escape = (1 << param_len) - 1

            # Handle unencoded binary residual payload
            if param == escape:
                bps = self._read_bits(5)
            else:
                bps = 0

            # Partition 0 is short by `order` samples
            samples_in_partition = part_samples if p > 0 else part_samples - order

            for _ in range(samples_in_partition):
                if idx >= block_size:
                    break

                if param == escape:
                    val = self._read_signed_bits(bps) if bps > 0 else 0
                else:
                    val = self._read_rice_signed(param)

                samples[idx] = val
                idx += 1

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
        if n == 0:
            return 0
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


def open_wave_or_flac_for_reading(path, allow_native_formats=True, allow_ffmpeg=True):
    if allow_native_formats:
        try:
            # should be fast when it works at all
            return wave.open(path, "rb")
        except Exception:
            pass
        if allow_ffmpeg:
            try:
                # should be relatively fast when it works at all, and
                # supports more formats
                return open_via_ffmpeg_as_wav(path)
            except Exception:
                pass
        try:
            # very slow but lets us use FLAC where we otherwise could not
            return open_flac_as_wav(path)
        except Exception:
            pass
    # normal WAV-only path, limited to old "traditional" Windows WAV
    # codec support (8bit unsigned/16bit signed PCM)
    return wave.open(path, "rb")


def read_wav_mono(
    path, allow_native_formats=True, allow_ffmpeg=True, stereo_to_mono="mix"
):
    """Read a WAV file and return (samples, framerate). FLAC also
    works, possibly much more slowly. other formats may also work if
    you have ffmpeg installed and available on your PATH.

    If allow_native_formats is False, the ffmpeg/FLAC fallbacks are
    skipped entirely and the file is read as plain WAV via the stdlib
    `wave` module only - no `subprocess` call is made. Use this if you'd
    rather convert non-WAV sources to WAV yourself ahead of time (e.g. via
    your own `ffmpeg` invocation) than have this tool shell out to ffmpeg
    on your behalf.

    If allow_native_formats is True and allow_ffmpeg is False, the
    slow pure Python FLAC decoder will be used for FLAC files.

    stereo_to_mono controls how a multi-channel file is collapsed to
    mono: "mix" (default) averages all channels together, "left" uses
    only the first channel, "right" uses only the second channel. Ignored
    entirely for mono input.

    """
    with open_wave_or_flac_for_reading(path, allow_native_formats, allow_ffmpeg) as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    samples = _unpack_samples(raw, sampwidth)

    if nchannels > 1:
        if stereo_to_mono == "left":
            samples = samples[0::nchannels]
        elif stereo_to_mono == "right":
            samples = samples[1::nchannels]
        else:
            mono = array.array("f")
            for i in range(0, len(samples) - nchannels + 1, nchannels):
                frame = samples[i : i + nchannels]
                mono.append(sum(frame) / nchannels)
            samples = mono

    return samples, framerate


# Reinterpreting an unsigned byte (0..255) as signed-after-subtracting-128 is
# the same as flipping its sign bit (XOR 0x80) and then reading it as signed
# two's-complement - this lets the whole 8-bit case go through bytes.translate
# (fast, in C) instead of a Python-level loop.
_UNSIGNED8_TO_SIGNED8 = bytes(b ^ 0x80 for b in range(256))


def _unpack_samples(raw, sampwidth):
    """Unpack raw little-endian PCM bytes into an array.array of samples.

    Returns a compact typed array (2-4 bytes/sample, no per-sample Python
    object) rather than a list of boxed ints/floats - for a several-
    hundred-megasample file that's the difference between tens of
    megabytes and multiple gigabytes of memory. Everything downstream
    (indexing, slicing, iteration, len()) works the same as it would on a
    plain list.
    """
    if sampwidth == 1:
        # WAV 8-bit PCM is unsigned, centered on 128
        arr = array.array("b")
        arr.frombytes(raw.translate(_UNSIGNED8_TO_SIGNED8))
        return arr
    elif sampwidth == 2:
        count = len(raw) // 2
        arr = array.array("h")
        arr.frombytes(raw[: count * 2])
        if sys.byteorder != "little":
            arr.byteswap()
        return arr
    elif sampwidth == 3:
        count = len(raw) // 3
        arr = array.array("i", bytes(4 * count))
        for i in range(count):
            b0, b1, b2 = raw[i * 3], raw[i * 3 + 1], raw[i * 3 + 2]
            v = b0 | (b1 << 8) | (b2 << 16)
            if v & 0x800000:
                v -= 0x1000000
            arr[i] = v
        return arr
    elif sampwidth == 4:
        count = len(raw) // 4
        arr = array.array("i")
        arr.frombytes(raw[: count * 4])
        if sys.byteorder != "little":
            arr.byteswap()
        return arr
    else:
        raise ValueError("Unsupported sample width: %d bytes" % sampwidth)


# --------------------------------------------------------------------------
# Edge detection -> pulse (cycle) list
# --------------------------------------------------------------------------


def find_all_edges(
    samples,
    threshold_ratio,
    agc=True,
    agc_attack=0.3,
    agc_release=0.005,
    agc_floor_ratio=0.02,
):
    """Schmitt-trigger zero-crossing detector, polarity-independent.

    Detects EVERY zero-crossing - both rising and falling - rather than
    only rising ones. This ensures polarity-agnostic decoding.

    Requires the signal to cross back past the opposite threshold before
    the next crossing is accepted, which gives noise immunity (a
    bidirectional Schmitt trigger). Returns a generator of (fractional,
    linearly-interpolated) sample indices of each accepted zero-crossing,
    in chronological order regardless of direction.

    This function mimics an AC-coupled input by tracking the local
    'center' of the wave (a slow-moving bias estimate) and subtracting
    it before thresholding, so recordings with significant DC offset
    are handled the same as clean ones.

    If agc is True, the threshold at each sample is threshold_ratio times
    a local amplitude envelope (fast attack / slow release, like an audio
    compressor's envelope follower) rather than a value derived from the
    whole file's peak amplitude, riding out gradual volume changes over a
    recording. A floor (agc_floor_ratio times the file's overall peak)
    keeps the envelope from collapsing during quiet/silent stretches.

    """
    peak_file = max((abs(s) for s in samples), default=0) or 1

    # DC bias tracker (slow moving average, defines the AC signal's center)
    bias = float(samples[0])
    bias_alpha = 0.002

    floor = peak_file * agc_floor_ratio
    envelope = floor

    prev_v_adj = float(samples[0]) - bias
    # state: which side of the hysteresis band we're currently "armed" to
    # cross away from. Starts unarmed in both directions until the signal
    # settles onto one side.
    armed_high = True
    armed_low = True

    # Performance optimization: Localize loop variables
    interp = _interp_zero

    if agc:
        # Hot loop version with AGC
        for i, sample in enumerate(samples):
            cur = float(sample)
            bias += (cur - bias) * bias_alpha
            v_adj = cur - bias

            abs_v = abs(v_adj)
            if abs_v > envelope:
                envelope += (abs_v - envelope) * agc_attack
            else:
                envelope += (abs_v - envelope) * agc_release

            if envelope < floor:
                envelope = floor

            local_threshold = envelope * threshold_ratio

            if armed_high and prev_v_adj < 0 <= v_adj:
                yield interp(i - 1, prev_v_adj, v_adj)
                armed_high = False
            elif armed_low and prev_v_adj > 0 >= v_adj:
                yield interp(i - 1, prev_v_adj, v_adj)
                armed_low = False

            if v_adj < -local_threshold:
                armed_high = True
            if v_adj > local_threshold:
                armed_low = True
            prev_v_adj = v_adj
    else:
        # Hot loop version without AGC
        local_threshold = peak_file * threshold_ratio
        for i, sample in enumerate(samples):
            cur = float(sample)
            bias += (cur - bias) * bias_alpha
            v_adj = cur - bias

            if armed_high and prev_v_adj < 0 <= v_adj:
                yield interp(i - 1, prev_v_adj, v_adj)
                armed_high = False
            elif armed_low and prev_v_adj > 0 >= v_adj:
                yield interp(i - 1, prev_v_adj, v_adj)
                armed_low = False

            if v_adj < -local_threshold:
                armed_high = True
            if v_adj > local_threshold:
                armed_low = True
            prev_v_adj = v_adj


def _interp_zero(i0, v0, v1):
    if v1 == v0:
        return float(i0)
    frac = (0 - v0) / (v1 - v0)
    frac = min(max(frac, 0.0), 1.0)
    return i0 + frac


def edges_to_half_periods(edges_gen):
    """Convert consecutive zero-crossing positions into a generator of
    half-cycle lengths (in samples). Working in the sample-length domain
    rather than frequency makes the long/short midpoint classification
    used by decode() a simple comparison. Since find_all_edges detects
    every zero-crossing (not just rising ones), each entry here is one
    half-cycle: two of them make a full cycle.
    """
    prev_edge = None
    for edge in edges_gen:
        if prev_edge is not None:
            length = edge - prev_edge
            if length > 0:
                yield length
        prev_edge = edge


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
    half_periods,
    framerate,
    tolerance=0.30,
    min_pilot_pulses=80,
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
    """Decode a list of half cycle-length ("half-period", in samples) values into a
    list of (baud, bytes, confidence) blocks.
    """
    blocks = []
    i = 0
    n = len(half_periods)

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
        """Read one encoded bit starting at half_periods[idx], where each
        entry is a half-cycle (see find_all_edges/edges_to_half_periods). A
        "0" bit is one full cycle of the low tone = 2 long half-cycles; a
        "1" bit is two full cycles of the high tone = 4 short
        half-cycles. Returns (bit, next_idx, confidence); bit is None on
        failure/EOF."""
        if idx >= n:
            return None, idx, 0.0
        c0 = classify(half_periods[idx])
        if c0 == "long":
            if idx + 1 < n and classify(half_periods[idx + 1]) == "long":
                p1, p2 = half_periods[idx], half_periods[idx + 1]
                conf = (pulse_confidence(p1) + pulse_confidence(p2)) / 2
                update_long(p1)
                update_long(p2)
                return 0, idx + 2, conf
            return None, idx + 1, 0.0
        elif c0 == "short":
            if idx + 3 < n and all(
                classify(half_periods[idx + k]) == "short" for k in (1, 2, 3)
            ):
                ps = half_periods[idx : idx + 4]
                conf = sum(pulse_confidence(p) for p in ps) / 4
                for p in ps:
                    update_short(p)
                return 1, idx + 4, conf
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
        if CAS_HEADER in data:
            print(
                "Warning: byte sequence matching CAS header found in data, this probably won't work",
                file=sys.stderr,
            )
        blocks.append((baud, data, block_conf))
        log("[block] %d bytes decoded, confidence=%.3f" % (len(data), block_conf))

    while i < n:
        period = half_periods[i]

        if state == "SEARCH":
            freq = framerate / (2 * period) if period > 0 else 0
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
                    long_nom = framerate / (2 * f0)
                    short_nom = framerate / (2 * f1)
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
                    window = half_periods[max(0, i - min_pilot_pulses) : i]
                    avg_period = sum(window) / len(window)
                    freq = framerate / (2 * avg_period) if avg_period > 0 else 0
                    match = _best_baud_match(freq, tolerance)
                    if match is not None and match[0] != baud:
                        baud, f0, f1 = match
                        long_nom = framerate / (2 * f0)
                        short_nom = framerate / (2 * f1)
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
    edges = find_all_edges(samples, threshold_ratio, agc=agc)
    half_periods = list(edges_to_half_periods(edges))
    return decode(half_periods, framerate, **decode_kwargs)


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
    rng = random.Random(42)
    payload_loud = b"LOUDMSG1"
    payload_quiet = b"QUIETMS2"
    samples = []

    # 1. Generate loud block (20,000 amplitude)
    _test_gen_block(samples, 1200, payload_loud, 0.5, amp=20000, rng=rng)

    # 2. Add silence gap (500 samples)
    # This allows the AGC envelope follower to decay toward the target volume
    samples += [0] * 500

    # 3. Generate quiet block (1,600 amplitude / 8% of previous)
    _test_gen_block(samples, 1200, payload_quiet, 0.5, amp=1600, rng=rng)

    # 4. Add trailing silence (1,000 samples)
    # Critical: ensures the final wave cycles complete so the edges are detected
    samples += [0] * 1000

    # Decode with AGC enabled
    agc_blocks = _test_decode_samples(samples, agc=True)
    agc_payloads = [b[1] for b in agc_blocks]

    if payload_loud not in agc_payloads:
        return False, f"Missing loud block. Decoded: {agc_payloads}"
    if payload_quiet not in agc_payloads:
        return False, f"Missing quiet block. Decoded: {agc_payloads}"
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
        samples, 1200, b"GARBLED!", 0.5, noise_amp=0.28, jitter=0.18, rng=rng
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


def _t_polarity_invariance():
    # A polarity-inverted capture must decode identically to the original
    rng = random.Random(7)
    samples = []
    _test_gen_block(
        samples, 1200, b"HEADERID", 0.5, noise_amp=0.05, jitter=0.03, rng=rng
    )
    _test_gen_block(
        samples, 1200, bytes(range(64)), 0.3, noise_amp=0.05, jitter=0.03, rng=rng
    )
    samples += [0] * int(_TEST_FRAMERATE * 0.2)

    inverted = [-s for s in samples]

    normal_blocks = _test_decode_samples(samples)
    inverted_blocks = _test_decode_samples(inverted)

    normal_payloads = [b[1] for b in normal_blocks]
    inverted_payloads = [b[1] for b in inverted_blocks]

    if normal_payloads != [b"HEADERID", bytes(range(64))]:
        return False, "normal-polarity decode wasn't clean: %r" % (normal_payloads,)
    if inverted_payloads != normal_payloads:
        return False, (
            "inverted-polarity decode didn't match normal: %r vs %r"
            % (inverted_payloads, normal_payloads)
        )
    return True, "ok"


def _t_stereo_to_mono():
    left_val, right_val = 100, -100
    n = 200
    interleaved = []
    for _ in range(n):
        interleaved += [left_val, right_val]

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "stereo.wav")
        with wave.open(path, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(struct.pack("<%dh" % len(interleaved), *interleaved))

        mix, _ = read_wav_mono(path, stereo_to_mono="mix")
        left, _ = read_wav_mono(path, stereo_to_mono="left")
        right, _ = read_wav_mono(path, stereo_to_mono="right")

        mono_path = os.path.join(d, "mono.wav")
        with wave.open(mono_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(struct.pack("<%dh" % n, *([42] * n)))
        mono_mix, _ = read_wav_mono(mono_path, stereo_to_mono="mix")
        mono_left, _ = read_wav_mono(mono_path, stereo_to_mono="left")
        mono_right, _ = read_wav_mono(mono_path, stereo_to_mono="right")

    if any(v != 0 for v in mix):
        return False, "mix should average +100/-100 to 0: %r" % (mix[:3],)
    if any(v != left_val for v in left):
        return False, "left should be all +100: %r" % (left[:3],)
    if any(v != right_val for v in right):
        return False, "right should be all -100: %r" % (right[:3],)
    if not (mono_mix.tolist() == mono_left.tolist() == mono_right.tolist() == [42] * n):
        return False, "stereo_to_mono must be ignored for mono input"
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


def _t_wav_decoding():
    # this WAV was generated and encoded by:
    # ```bash
    # LC_ALL=C printf '\x1f\xa6\xde\xba\xcc\x13\x7d\x74%s' TEST > test.cas &&
    #     python3 cas2wav.py test.cas test.wav &&
    #     gzip -9 < test.wav | recode l1..l1/b64
    # ```
    wav = gzip.decompress(
        base64.b64decode(
            "H4sIAO6xZWoCA+3UvQnCUBSA0as4gBMEcYuAjaCChY2FWtgIIqSwc5e3QMDGKpUDSKawySYmO1gY"
            "OV+6cx/JC/nZrler92AU+/luebneJuOIGLTHdBexeEQMYxzn0+30bNdEpHx2SJsyOzZF9boXNSGE"
            "EEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBC"
            "CCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQggh"
            "hBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQ"
            "QgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEII"
            "IYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGE"
            "EEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBC"
            "CCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQggh"
            "hBBCCCGEEEIIIYQQQgghhJB+SpmlPG1mh85TXmYR7azupu28Kupj870139pPH8/zr/fex3ejj9fy"
            "XfhH/cKzkCRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJ"
            "kiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJ"
            "kiRJkiRJkiRJkiRJkiRJkiRJkiT9cx9DHaLo5AEEAA=="
        )
    )
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        temp_file.write(wav)
        temp_file.flush()
        with wave.open(temp_file.name, "rb") as wf:
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
            if nchannels != 1:
                return False, "channel count should be 1 but was %r" % nchannels
            if sampwidth != 2:
                return (
                    False,
                    "sample width should be 2 bytes but was %r bytes" % sampwidth,
                )
            if framerate != 22050:
                return False, "frame rate should be 22050 but was %r" % framerate
            if nframes != 131292:
                return False, "frame count should be 131292 but was %r" % nframes
            if len(raw) != sampwidth * nchannels * nframes:
                return False, "raw sample data had the wrong length"
            samples = _unpack_samples(raw, sampwidth)
            edges_gen = find_all_edges(samples, 0.2)
            half_periods_gen = edges_to_half_periods(edges_gen)
            half_periods = list(half_periods_gen)
            blocks = decode(half_periods, framerate)
            if len(blocks) != 1:
                return (
                    False,
                    "got wrong number of blocks: should be 1, instead %r" % len(blocks),
                )
            if blocks[0][0] != 1200:
                return (
                    False,
                    "got wrong baud rate for block 0: should be 1200, instead %r"
                    % blocks[0][0],
                )
            if blocks[0][1] != b"TEST":
                return False, "got wrong data for block 0: should be %r, instead %r" % (
                    b"TEST",
                    blocks[0][1],
                )
            if blocks[0][2] < 0.9:
                return (
                    False,
                    "too little confidence for block 0: should be at least 0.9, instead %r"
                    % blocks[0][2],
                )

    return True, "ok"


def _t_flac_decoding():
    # this FLAC was generated and encoded by:
    # ```bash
    # LC_ALL=C printf '\x1f\xa6\xde\xba\xcc\x13\x7d\x74%s' TEST > test.cas &&
    #     python3 cas2wav.py test.cas test.wav &&
    #     sox test.wav test.flac &&
    #     gzip -9 < test.flac | recode l1..l1/b64
    # ```
    flac = gzip.decompress(
        base64.b64decode(
            "H4sIAFGoZWoCA+3cCUyTZxgH8Lccs4pHQUCECR0KEzmkKCAbTKhSJBanQByCOiwUOWWDkYEYVpka"
            "DhE5BijRKYcnLnIahkNnuD1xylAEdSAaPFGHGFD2vcW44TaNBpfs89+EfM/XH+8XePo2f0r61E+8"
            "Yg4hRJ9HeMxBhXD7lSX8e0SBXKo9Vt1lMMXfxCg8MOxY7hePFAlRIy/ceGQdIXP5TBUm9ZOGSVf5"
            "SPnBARKR2GEOX2BmaWbOtzC3sDS3EAg4zPdoMV9zQkNCpKu+slsYFuojDQ+X+vIlUXy3UI+B3hou"
            "OSskxCazVDCWOZRy9YsDeY5+5/gJJ/c4aGQvjDW22qm0OETILbi4TnUJGAwGg8H/U5aJHtLU45wX"
            "2mR62MpTzxDdAYPBYDA7Wbb6Y5p6CpeEHrYug6/1TNEVMBgMBrOTz7bS0FNsEbpk6ubJQ28aGgMG"
            "g8FgdrJM2Y+mntIpoW7e1aXy1DNHZ8BgMBjMTpZl19LUU64XXl1aFCBPPX10BwwGg8HsZJmnNk29"
            "944KiwJ2V8lT7wM0DAwGg8Hs5IZiGnojaoW7q2oGX+oZoFdgMBgMZiffsqGhx+0S1gQQIg89KzQF"
            "DAaDwezkRgkNvZG3MaEOBoPB4HdgQj2Cpt6oDkyog8FgMJj9E+piR5p6Ku2YUAeDwWAw6zlkOQ29"
            "0T2YUAeDwWAw+yfUl3fS1BvTjQl1MBgMBr8DE+oimnpjezGhDgaDwWD2T6gH+9LUGzeACXUwGAwG"
            "s54jzGno8XIwoQ4Gg8Fg1rPbGBp6qvmYUAeDwWAw67nCnYaeWgEm1MFgMBjM/gl1px6aeuP3Y0Id"
            "DAaDweyfUI+WfwSneiUm1MFgMBjMeo4soqGnUb6ATqjTyQX6lhb6z86SnLKPlDfrkQ8fcKa3J1k9"
            "tv7eSHxtv8eVTSsSvTra1uTPyDjgE2uzwTQ0Lep++inP0k3cTq3SSpfrDW42gU2zJmdJml6yZM1w"
            "XmzokuMGw3ixF5YM68WGLmkYzosNXZL3Nto8uMT4gszYduxB7kXTJf5e5brN8ZPVo/d+I0lTr83V"
            "O9z9nmOAoltlo2F8quizaMmUjYvu5D9JELvpbox5UDVRI2/Mgr65Ww2dJy67HZcqcPoxs8dIIF4t"
            "6DvN83VXGs9xapfOG9FRXikq+zm4urkktD3mfseJjCbm7KjG9hEeSiGaBSWml5KD56Yk2J07Uuic"
            "4NOY3Bb5rf4Oftao2TrT7GcbnLxvWabiZ+Jp3X0kKsxvfpCXd5n3qNOKBQpWiYs9Ne84lenPq9bt"
            "+rw5r1pwo/aurUPKQp6L96LeC66cx2ahKQcOegkkjUeS+o6nWXh5tigXk8WRk7xVSzyLzhwKsytU"
            "E0195Pq02tAoWTQ76rfRCbJY76n9wXtJf6Z1XVOzaVah/+WaJ9fqt5geCIm9kKiZ1B/atnRL6sVn"
            "dPjPM4XiozdJUs75jPUVKa9uRKLoqfVdM3F9SFCw5gnOkisRMlFj4HFha11PlbNWbuv8PvWsXXZa"
            "T67HWUxTGCnL7S4q4U1Kvd1tmTHSb1bJZdHv15y3epUyJ0RxpUzRg7js6KoVLqu/FZfuo7vPqzm9"
            "qrCjNnSmQ045x237mV3rUw3dovUNtGr3ZfQYCEyiBC0nef4/2LtfLzHa6Ruf3lYXLxJ05f8ULw55"
            "3kCxsl7Sd3y91M7H2kzLNE11ei8fLp9hUMp001x55domjo7jflOVrye3qE6RJEdMv+UqyXro82g7"
            "v24P11AhuaZh87j68DVJ/9Qy2s3sIWeDvX35BrR3MFePsXKXxczP7qAt8/jrRhrazefU+sreGs/T"
            "3tYTP1OcalHxmr/IkEc78rV+2jff9nuHPvZvZSOYvWkTXnfL97yt5/7ftu7DpJsNaTMPBZmonfgy"
            "oPJfTgghShMI0b8X9Kmr9Jc98bmEr0RkPAViz+W8qiR6A4O3q8+OT3HHf3jHFXsu/dtH8yDzGAaf"
            "peWEQ0xJ9tFSaytTdcrf8jRxG1MmyD+0VTudKT95QkudNKZ0XU/L9zcw5a83aDlpLVNWbKalbhxT"
            "jtOipV4yU/p4D/T6cfktq+jV+v4A6NNW2yOcAAA="
        )
    )
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        temp_file.write(flac)
        temp_file.flush()
        with open_flac_as_wav(temp_file.name) as wf:
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
            if nchannels != 1:
                return False, "channel count should be 1 but was %r" % nchannels
            if sampwidth != 2:
                return (
                    False,
                    "sample width should be 2 bytes but was %r bytes" % sampwidth,
                )
            if framerate != 22050:
                return False, "frame rate should be 22050 but was %r" % framerate
            if nframes != 131292:
                return False, "frame count should be 131292 but was %r" % nframes
            if len(raw) != sampwidth * nchannels * nframes:
                return False, "raw sample data had the wrong length"
            samples = _unpack_samples(raw, sampwidth)
            edges_gen = find_all_edges(samples, 0.2)
            half_periods_gen = edges_to_half_periods(edges_gen)
            half_periods = list(half_periods_gen)
            blocks = decode(half_periods, framerate)
            if len(blocks) != 1:
                return (
                    False,
                    "got wrong number of blocks: should be 1, instead %r" % len(blocks),
                )
            if blocks[0][0] != 1200:
                return (
                    False,
                    "got wrong baud rate for block 0: should be 1200, instead %r"
                    % blocks[0][0],
                )
            if blocks[0][1] != b"TEST":
                return False, "got wrong data for block 0: should be %r, instead %r" % (
                    b"TEST",
                    blocks[0][1],
                )
            if blocks[0][2] < 0.9:
                return (
                    False,
                    "too little confidence for block 0: should be at least 0.9, instead %r"
                    % blocks[0][2],
                )

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
    ("polarity invariance", _t_polarity_invariance),
    ("stereo-to-mono channel selection", _t_stereo_to_mono),
    ("CAS --pad alignment", _t_pad_alignment),
    ("WAV decoding", _t_wav_decoding),
    ("FLAC decoding", _t_flac_decoding),
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
        description="Decode MSX or MSX-like cassette audio (WAV) into a .CAS file."
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
        default=80,
        help="consecutive pilot half-cycles required to lock onto a baud rate "
        "(default: 80 - each is a half-cycle, so this is ~40 full pilot cycles)",
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
        "--no-native-formats",
        action="store_true",
        help="skip the built-in FLAC/ffmpeg support and read the input as plain WAV only via "
        "the stdlib `wave` module - no `subprocess` call is made. Use this if you'd rather "
        "convert a non-WAV source to WAV yourself first (e.g. with your own ffmpeg command)",
    )
    parser.add_argument(
        "--no-ffmpeg",
        action="store_true",
        help="do not use ffpmeg, but still allow the built-in FLAC decoder, which is very slow",
    )
    parser.add_argument(
        "--stereo-to-mono",
        choices=("mix", "left", "right"),
        default="mix",
        help="how to collapse a multi-channel input to mono: average all channels together "
        "(default: mix), or use only the left or right channel. Ignored for mono input",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Apply built-in noise filter for noisy tapes",
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
    samples, framerate = read_wav_mono(
        args.input,
        allow_native_formats=not args.no_native_formats,
        allow_ffmpeg=not args.no_ffmpeg,
        stereo_to_mono=args.stereo_to_mono,
    )
    print("%d samples at %d Hz" % (len(samples), framerate), file=sys.stderr)

    if args.filter:
        samples = apply_band_filter(samples, framerate)

    # Use generators to chain the processing pipeline and save memory
    edges_gen = find_all_edges(
        samples,
        args.threshold_ratio,
        agc=not args.no_agc,
        agc_attack=args.agc_attack,
        agc_release=args.agc_release,
        agc_floor_ratio=args.agc_floor_ratio,
    )
    half_periods_gen = edges_to_half_periods(edges_gen)

    # decode() requires random access/indexing, so we consume the generator here
    half_periods = list(half_periods_gen)
    print("%d half-cycles detected" % len(half_periods), file=sys.stderr)

    blocks = decode(
        half_periods,
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
