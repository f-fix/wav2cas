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

    _CRC8_TABLE = None

    @classmethod
    def _get_crc8_table(cls):
        """Memoized computation of the CRC-8 table as a bytearray."""
        if cls._CRC8_TABLE is None:
            # FLAC frame header CRC-8 table.
            # Polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x07).
            # This polynomial is defined in the FLAC specification (RFC 9639).
            poly = 0x07
            table = bytearray(256)
            for i in range(256):
                crc = i
                for _ in range(8):
                    if crc & 0x80:
                        crc = (crc << 1) ^ poly
                    else:
                        crc <<= 1
                table[i] = crc & 0xFF
            cls._CRC8_TABLE = table
        return cls._CRC8_TABLE

    def __init__(self, file_path):
        self.f = open(file_path, "rb")
        self.data_len = os.path.getsize(file_path)
        self.offset = 0

        self.sample_rate = 44100
        self.channels = 2
        self.bits_per_sample = 16
        self.total_samples = 0
        self.samples_yielded = 0

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
        table = self._get_crc8_table()
        while self.offset < self.data_len:
            if self.samples_yielded >= self.total_samples:
                break
            percent = min(100.0, (self.offset / self.data_len) * 100.0)
            sys.stdout.write(f"\rDecoding FLAC: {percent:5.1f}%")
            sys.stdout.flush()

            # Fast search for sync byte
            sync_byte = self.f.read(1)
            if not sync_byte:
                break
            self.offset += 1

            if sync_byte[0] == 0xFF:
                next_byte = self.f.read(1)
                if not next_byte:
                    break
                self.offset += 1
                if (next_byte[0] & 0xFE) == 0xF8:
                    # Potential frame. Validate Header CRC-8 to avoid False Syncs.
                    start_pos = self.f.tell() - 2
                    try:
                        self.bit_buffer = self.bit_count = 0
                        yield from self._parse_frame(next_byte[0], table)
                    except (EOFError, ValueError, IndexError):
                        # Rejection: go back and slide forward by one byte
                        self.f.seek(start_pos + 1)
                        self.offset = start_pos + 1
                else:
                    self.f.seek(-1, os.SEEK_CUR)
                    self.offset -= 1

    def _read_vlc(self, header_bytes=None):
        """Read variable-length frame/sample number from bitstream."""
        v = self._read_bits(8, header_bytes)
        if v < 0x80:
            return v
        elif 0xC0 <= v <= 0xDF:
            n, val = 1, v & 0x1F
        elif 0xE0 <= v <= 0xEF:
            n, val = 2, v & 0x0F
        elif 0xF0 <= v <= 0xF7:
            n, val = 3, v & 0x07
        elif 0xF8 <= v <= 0xFB:
            n, val = 4, v & 0x03
        elif 0xFC <= v <= 0xFD:
            n, val = 5, v & 0x01
        else:
            return 0
        for _ in range(n):
            val = (val << 6) | (self._read_bits(8, header_bytes) & 0x3F)
        return val

    def _parse_frame(self, b2, table):
        header_bytes = bytearray([0xFF, b2])

        # Bits: Res(1), Blocking(1) were in b2.
        # Next: BlockSize(4), SampleRate(4)
        b3 = self._read_bits(8, header_bytes)
        bs_code, sr_code = (b3 >> 4), (b3 & 0x0F)

        # Next: ChAssignment(4), BPS(3), Res(1)
        b4 = self._read_bits(8, header_bytes)
        ch_assignment = (b4 >> 4) & 0x0F

        # Frame/Sample number
        self._read_vlc(header_bytes)

        # Block Size Extra bytes
        if bs_code == 1:
            block_size = 192
        elif 2 <= bs_code <= 5:
            block_size = 576 << (bs_code - 2)
        elif bs_code == 6:
            block_size = self._read_bits(8, header_bytes) + 1
        elif bs_code == 7:
            block_size = self._read_bits(16, header_bytes) + 1
        elif 8 <= bs_code <= 15:
            block_size = 256 << (bs_code - 8)
        else:
            raise ValueError("Invalid block size code")

        # Sample Rate Extra bytes
        if sr_code == 12:
            self._read_bits(8, header_bytes)
        elif sr_code in (13, 14):
            self._read_bits(16, header_bytes)

        # Validate Header CRC-8
        actual_crc = self._read_bits(8)
        calc_crc = 0
        for b in header_bytes:
            calc_crc = table[calc_crc ^ b]
        if calc_crc != actual_crc:
            raise ValueError("CRC-8 mismatch: false sync rejected")

        subframe_samples = []
        for ch in range(self.channels):
            # Side channels have bps + 1 bit precision
            bps = self.bits_per_sample
            if (
                (ch_assignment == 8 and ch == 1)
                or (ch_assignment == 9 and ch == 0)
                or (ch_assignment == 10 and ch == 1)
            ):
                bps += 1
            sub_samples = self._parse_subframe(block_size, bps)
            subframe_samples.append(sub_samples)

        if self.channels == 2:
            l, r = subframe_samples[0], subframe_samples[1]
            if ch_assignment == 8:  # Left/Side: R = L - S
                for i in range(block_size):
                    r[i] = l[i] - r[i]
            elif ch_assignment == 9:  # Side/Right: L = S + R
                for i in range(block_size):
                    l[i] += r[i]
            elif ch_assignment == 10:  # Mid/Side Reconstruction
                for i in range(block_size):
                    m, s = l[i], r[i]
                    l[i] = m + ((s + (s & 1)) >> 1)
                    r[i] = m - (s >> 1)

        for i in range(block_size):
            if self.samples_yielded >= self.total_samples:
                break
            for ch in range(self.channels):
                v = subframe_samples[ch][i]

                # High-quality bit-depth reduction (matches Sox -D rounding)
                if self.bits_per_sample > 16:
                    shift = self.bits_per_sample - 16
                    v = (v + (1 << (shift - 1))) >> shift
                elif self.bits_per_sample < 16:
                    v <<= 16 - self.bits_per_sample

                sample = max(-32768, min(32767, v))
                yield sample
            self.samples_yielded += 1

    def _parse_subframe(self, block_size, bps):
        sub_header = self._read_bits(8)
        stype = (sub_header >> 1) & 0x3F
        wasted_bits = 0
        if (sub_header & 0x01) != 0:
            wasted_bits = self._read_unary() + 1

        bps_eff = bps - wasted_bits
        samples = [0] * block_size

        if stype == 0:  # Constant
            sample = self._read_signed_bits(bps_eff)
            samples = [sample] * block_size
        elif stype == 1:  # Verbatim
            for i in range(block_size):
                samples[i] = self._read_signed_bits(bps_eff)
        elif 8 <= stype <= 12:  # Fixed
            order = stype - 8
            for i in range(order):
                samples[i] = self._read_signed_bits(bps_eff)
            self._parse_residual(block_size, order, samples)
            for i in range(order, block_size):
                if order == 1:
                    samples[i] += samples[i - 1]
                elif order == 2:
                    samples[i] += 2 * samples[i - 1] - samples[i - 2]
                elif order == 3:
                    samples[i] += (
                        3 * samples[i - 1] - 3 * samples[i - 2] + samples[i - 3]
                    )
                elif order == 4:
                    samples[i] += (
                        4 * samples[i - 1]
                        - 6 * samples[i - 2]
                        + 4 * samples[i - 3]
                        - samples[i - 4]
                    )
        elif 32 <= stype <= 63:  # LPC
            order = stype - 31
            for i in range(order):
                samples[i] = self._read_signed_bits(bps_eff)
            prec, shift = self._read_bits(4) + 1, self._read_signed_bits(5)
            coeffs = [self._read_signed_bits(prec) for _ in range(order)]
            self._parse_residual(block_size, order, samples)
            for i in range(order, block_size):
                samples[i] += (
                    sum(c * samples[i - 1 - j] for j, c in enumerate(coeffs)) >> shift
                )

        if wasted_bits > 0:
            for i in range(block_size):
                samples[i] <<= wasted_bits
        return samples

    def _parse_residual(self, block_size, order, samples):
        method = self._read_bits(2)
        p_order = self._read_bits(4)
        num_partitions = 1 << p_order
        p_len = block_size >> p_order
        idx = order
        for p in range(num_partitions):
            param = self._read_bits(4 if method == 0 else 5)
            for _ in range(p_len - (order if p == 0 else 0)):
                samples[idx] = self._read_rice_signed(param)
                idx += 1

    def _read_bits(self, n, header_bytes=None):
        if n == 0:
            return 0
        while self.bit_count < n:
            b = self.f.read(1)
            if not b:
                raise EOFError()
            byte_val = b[0]
            if header_bytes is not None:
                header_bytes.append(byte_val)
            self.bit_buffer = (self.bit_buffer << 8) | byte_val
            self.bit_count += 8
            self.offset += 1
        self.bit_count -= n
        res = (self.bit_buffer >> self.bit_count) & ((1 << n) - 1)
        self.bit_buffer &= (1 << self.bit_count) - 1
        return res

    def _read_signed_bits(self, n):
        v = self._read_bits(n)
        return v if not (v & (1 << (n - 1))) else v - (1 << n)

    def _read_unary(self):
        count = 0
        while not self._read_bits(1):
            count += 1
            if count > 512:
                raise ValueError("Unary malformed")
        return count

    def _read_rice_signed(self, param):
        u = self._read_unary()
        val = (u << param) | self._read_bits(param)
        return (val >> 1) ^ -(val & 1)

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
