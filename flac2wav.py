#!/usr/bin/env python3
"""flac2wav.py - A pure Python FLAC to WAV converter."""

import argparse
import array
import base64
import lzma
import os
import struct
import sys
import tempfile
import wave


class PurePythonFlacDecoder:
    """A pure Python FLAC decoder that streams from disk to save memory."""

    def __init__(self, file_path):
        self.f = open(file_path, "rb")
        self.data_len = os.path.getsize(file_path)
        self.offset = 0

        self.sample_rate = 44100
        self.channels = 2
        self.bits_per_sample = 16
        self.total_samples = 0

        # Bit stream buffer state
        self.bit_buffer = 0
        self.bit_count = 0

        self._parse_flac()

    def _read_bytes(self, n):
        res = self.f.read(n)
        if not res or len(res) < n:
            return None
        self.offset += len(res)
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

    def _parse_streaminfo(self, data):
        if len(data) < 34:
            return
        # Extract metadata from binary block
        self.sample_rate = (data[10] << 12) | (data[11] << 4) | (data[12] >> 4)
        self.channels = ((data[12] >> 1) & 0x07) + 1
        bps = ((data[12] & 0x01) << 4) | (data[13] >> 4)
        self.bits_per_sample = bps + 1
        self.total_samples = ((data[13] & 0x0F) << 32) | int.from_bytes(
            data[14:18], "big"
        )

    def stream_samples(self):
        """Generator yielding decoded 16-bit PCM samples with progress reports."""
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
                self.f.seek(-2, os.SEEK_CUR)
                self.offset -= 2
                yield from self._parse_frame()
            else:
                self.f.seek(-1, os.SEEK_CUR)
                self.offset -= 1

    def _parse_frame(self):
        self.bit_count = 0
        self.bit_buffer = 0

        header_start = self._read_bytes(2)
        if not header_start or len(header_start) < 2:
            return

        bs_sr = self._read_bytes(1)
        if not bs_sr:
            return
        block_size_type = (bs_sr[0] >> 4) & 0x0F
        sample_rate_code = bs_sr[0] & 0x0F

        ch_bps = self._read_bytes(1)
        if not ch_bps:
            return
        channel_assignment = (ch_bps[0] >> 4) & 0x0F

        # variable-length UTF-8-style encoded frame/sample number
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

        for i in range(block_size):
            for ch in range(self.channels):
                if i < len(subframe_samples[ch]):
                    sample = subframe_samples[ch][i]

                    if self.bits_per_sample > 16:
                        sample >>= self.bits_per_sample - 16
                    elif self.bits_per_sample < 16:
                        sample <<= 16 - self.bits_per_sample

                    sample = max(-32768, min(32767, sample))
                    yield sample

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
                for i in range(1, block_size):
                    samples[i] += samples[i - 1]
            elif order == 2:
                for i in range(2, block_size):
                    samples[i] += 2 * samples[i - 1] - samples[i - 2]
            elif order == 3:
                for i in range(3, block_size):
                    samples[i] += (
                        3 * samples[i - 1] - 3 * samples[i - 2] + samples[i - 3]
                    )
            elif order == 4:
                for i in range(4, block_size):
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

            prec = self._read_bits(4) + 1
            shift = self._read_bits(5)
            coeffs = [self._read_signed_bits(prec) for _ in range(order)]

            samples = self._parse_residual(block_size, order, samples)

            # Apply LPC predictors
            for i in range(order, block_size):
                pred = sum(coeffs[j] * samples[i - 1 - j] for j in range(order))
                samples[i] += pred >> shift

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

            if param == escape:
                bps = self._read_bits(5)
            else:
                bps = 0

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
            b = self._read_bytes(1)
            if b is None:
                return None
            self.bit_buffer = (self.bit_buffer << 8) | b[0]
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

    def close(self):
        sys.stdout.write("\rDecoding FLAC: 100.0% - Complete!\n")
        sys.stdout.flush()
        self.f.close()


class FlacStreamReader:
    """Wraps FlacDecoder to mimic wave.Wave_read without memory buffers."""

    def __init__(self, file_path):
        self.decoder = PurePythonFlacDecoder(file_path)
        self.sample_gen = self.decoder.stream_samples()

    def getnchannels(self):
        return self.decoder.channels

    def getsampwidth(self):
        return 2

    def getframerate(self):
        return self.decoder.sample_rate

    def getnframes(self):
        return self.decoder.total_samples

    def readframes(self, n):
        arr = array.array("h")
        for _ in range(n * self.decoder.channels):
            try:
                arr.append(next(self.sample_gen))
            except StopIteration:
                break
        if not arr:
            return b""
        if sys.byteorder != "little":
            arr.byteswap()
        return arr.tobytes()

    def close(self):
        self.decoder.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def convert_flac_to_wav(in_path, out_path):
    """Converts a FLAC file to WAV."""
    with FlacStreamReader(in_path) as fr:
        with wave.open(out_path, "wb") as fw:
            fw.setnchannels(fr.getnchannels())
            fw.setsampwidth(fr.getsampwidth())
            fw.setframerate(fr.getframerate())
            if fr.getnframes():
                fw.setnframes(fr.getnframes())

            chunk_size = 65536
            while True:
                frames = fr.readframes(chunk_size)
                if not frames:
                    break
                fw.writeframes(frames)


def run_tests():
    print("Running self-test...")
    flac_data = lzma.decompress(
        base64.b64decode(
            "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"
            "raUDZQjgH0Q8lX1lDuT+Z8Cf"
        )
    )

    wav_data = lzma.decompress(
        base64.b64decode(
            "XQAAAAT//////////wApEkTrmN6NWD9/VTXn4ecOY4FtT/55rcJhTYtnpJCd4o6aYb7RT0LxXkXn"
            "eXkXdZsVx3Y2qc66oczkz6UyaM5IWk9BKZhm+OBqFKV+D0txo48GT9zniR2UxYDvOBBxDFxj2A9H"
            "cutnvsNI9xkHHnLrtIStSTt+FVNDooHcDhNSB+jp5vOCIwiriVOwEjLd6gTJqK9WzqK1+bGapwXe"
            "FYlB3hCkARratbya1H/bNy2SHnF14rcKm0n4nJ1Zdv3LGRoQI4Fmv+0q7rxZUVprFnw7oIUs4qv2"
            "3CaO2eTG1kOfZXK9Ut/G41lfBUEe+4JokA=="
        )
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        test_flac = os.path.join(temp_dir, "test.flac")
        test_wav = os.path.join(temp_dir, "test.wav")
        reference_wav = os.path.join(temp_dir, "reference.wav")

        with open(test_flac, "wb") as f:
            f.write(flac_data)

        with open(reference_wav, "wb") as f:
            f.write(wav_data)

        # Decode the FLAC to WAV
        convert_flac_to_wav(test_flac, test_wav)

        # Validate bit-accuracy
        with wave.open(test_wav, "rb") as tw, wave.open(reference_wav, "rb") as rw:
            assert tw.getnchannels() == rw.getnchannels(), "Channel count mismatch"
            assert tw.getsampwidth() == rw.getsampwidth(), "Sample width mismatch"
            assert tw.getframerate() == rw.getframerate(), "Framerate mismatch"
            assert tw.getnframes() == rw.getnframes(), "Frame count mismatch"

            test_frames = tw.readframes(tw.getnframes())
            ref_frames = rw.readframes(rw.getnframes())

            if test_frames == ref_frames:
                print("TEST PASSED: Decoded FLAC matches reference WAV perfectly!")
            else:
                print("TEST FAILED: Decoded raw PCM data differs from reference WAV.")
                sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert FLAC to WAV using a pure Python implementation."
    )
    parser.add_argument("input", nargs="?", help="Input .flac file")
    parser.add_argument("output", nargs="?", help="Output .wav file")
    parser.add_argument(
        "--test", action="store_true", help="Run the internal verification test"
    )

    args = parser.parse_args()

    if args.test:
        run_tests()
        sys.exit(0)

    if not args.input or not args.output:
        parser.error("Input and output files are required unless --test is specified.")

    print(f"Converting '{args.input}' to '{args.output}'...")
    convert_flac_to_wav(args.input, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
