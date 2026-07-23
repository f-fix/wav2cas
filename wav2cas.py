#!/usr/bin/env python3
"""
wav2cas.py - Decode MSX cassette audio (WAV) into a .CAS file.

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

import sys
import wave
import struct
import argparse
import math
import random
import tempfile
import os

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
# WAV reading
# --------------------------------------------------------------------------

def read_wav_mono(path):
    """Read a WAV file and return (samples, framerate).

    samples is a flat list of ints/floats, mono (channels averaged if the
    file is stereo/multi-channel), DC offset NOT removed (the edge detector
    below handles that implicitly via zero-crossing + hysteresis)."""
    with wave.open(path, 'rb') as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    samples = _unpack_samples(raw, sampwidth)

    if nchannels > 1:
        mono = []
        for i in range(0, len(samples) - nchannels + 1, nchannels):
            frame = samples[i:i + nchannels]
            mono.append(sum(frame) / nchannels)
        samples = mono

    return samples, framerate


def _unpack_samples(raw, sampwidth):
    if sampwidth == 1:
        # WAV 8-bit PCM is unsigned, centered on 128
        return [b - 128 for b in raw]
    elif sampwidth == 2:
        count = len(raw) // 2
        return list(struct.unpack('<%dh' % count, raw[:count * 2]))
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
        return list(struct.unpack('<%di' % count, raw[:count * 4]))
    else:
        raise ValueError("Unsupported sample width: %d bytes" % sampwidth)


# --------------------------------------------------------------------------
# Edge detection -> pulse (cycle) list
# --------------------------------------------------------------------------

def find_rising_edges(samples, threshold_ratio, agc=True,
                       agc_attack=0.3, agc_release=0.0008, agc_floor_ratio=0.02):
    """Schmitt-trigger rising zero-crossing detector.

    Requires the signal to dip below -threshold before a new rising
    zero-crossing is accepted, which gives noise immunity. Returns a list
    of (fractional, linearly-interpolated) sample indices of each accepted
    rising edge.

    If agc is True, the threshold at each sample is threshold_ratio times
    a local amplitude envelope (fast attack / slow release, like an audio
    compressor's envelope follower) instead of a single value derived from
    the whole file's peak amplitude. This rides out gradual volume changes
    (fade in/out, azimuth wobble, etc.) over the length of a recording.
    A floor (agc_floor_ratio times the file's overall peak) keeps the
    envelope - and therefore the threshold - from collapsing to near zero
    during quiet/silent stretches, which would otherwise let noise trigger
    spurious edges."""
    peak = max((abs(s) for s in samples), default=0) or 1

    edges = []
    armed = True
    prev = samples[0]

    if agc:
        floor = peak * agc_floor_ratio
        envelope = max(abs(prev), floor)
    else:
        fixed_threshold = peak * threshold_ratio

    for i in range(1, len(samples)):
        cur = samples[i]

        if agc:
            absval = abs(cur)
            if absval > envelope:
                envelope = envelope * (1 - agc_attack) + absval * agc_attack
            else:
                envelope = envelope * (1 - agc_release) + absval * agc_release
            if envelope < floor:
                envelope = floor
            local_threshold = envelope * threshold_ratio
        else:
            local_threshold = fixed_threshold

        if not armed:
            if cur < -local_threshold:
                armed = True
        else:
            if prev < 0 <= cur:
                edges.append(_interp_zero(i - 1, prev, cur))
                armed = False
        prev = cur

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
    midpoint classification used by decode() a simple comparison."""
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
    closest relative match to freq, or None if none are within tolerance."""
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

    This targets stray low-confidence bytes that sometimes appear right at
    a block's boundary (e.g. from noise during the pilot/data transition
    or the tail end of a pilot run) without touching an otherwise good
    block's interior - a single bad byte deep inside a block is left
    alone, since trimming only ever eats from the two ends inward."""
    start = 0
    end = len(data)
    while start < end and confidences[start] < threshold:
        start += 1
    while end > start and confidences[end - 1] < threshold:
        end -= 1
    return bytes(data[start:end]), list(confidences[start:end])


def decode(periods, framerate, tolerance=0.30, min_pilot_pulses=40,
           adapt=True, adapt_rate=0.1, adapt_clamp=0.35,
           strict_stop=True, stop_bits=2,
           edge_trim=True, edge_trim_threshold=0.5,
           max_gap_multiple=3.0,
           verbose=False):
    """Decode a list of cycle-length ("period", in samples) values into a
    list of (baud, bytes, confidence) blocks.

    Once a baud rate is locked from the pilot tone, every subsequent pulse
    is classified as either "long" (a whole bit-0 cycle) or "short" (half
    of a bit-1 pair) using a single midpoint threshold between the two
    nominal cycle lengths. If adapt is True, the long/short reference
    lengths (and threshold) slowly track the actually observed pulse
    lengths, clamped to stay within adapt_clamp of nominal.

    If strict_stop is True (the default), a byte is only accepted if it is
    followed by stop_bits genuine "1" bits; otherwise the byte is dropped
    and decoding resyncs via the pilot search. This is what keeps stray
    noise or transition artifacts near a block's boundary from being
    misread as spurious extra bytes. If strict_stop is False, any number
    of stop/mark pulses (including none) is accepted before the next start
    bit, matching the more permissive behavior real MSX BIOS routines use.

    Each pulse also gets a confidence score in [0, 1] (1.0 = squarely on
    one of the reference lengths, 0.0 = sitting right on the decision
    threshold). Per-bit confidence is that pulse's score (or the average
    of two pulses' scores for a "1" bit); per-byte confidence is the mean
    over its bits (start bit + 8 data bits, plus the stop bit(s) in strict
    mode); per-block confidence is the mean over its bytes.

    If edge_trim is True (the default), any run of bytes at the very start
    and/or end of a block whose individual confidence falls below
    edge_trim_threshold is dropped before the block's confidence is
    computed - this catches stray low-confidence bytes right at a block's
    boundary without touching a good block's interior. If trimming would
    remove the entire block, it's discarded outright.

    A period longer than max_gap_multiple times the current long-cycle
    reference is treated as a dropout/silence gap rather than a genuine
    bit-0 cycle - on real tape this is a splice, a dropout, or just a
    quiet gap between blocks, not a single very slow cycle. Encountering
    one forces a full resync through the pilot search rather than letting
    it masquerade as a start bit (which would otherwise let a stretch of
    the *next* block's own pilot tone get swallowed as bogus data bits of
    a phantom byte, since a run of pilot "1" bits looks exactly like
    plausible byte content once you start reading it at the wrong
    alignment)."""

    blocks = []
    i = 0
    n = len(periods)

    state = 'SEARCH'   # SEARCH -> SYNCED -> BYTE -> SYNCED -> ... -> SEARCH
    pilot_count = 0
    candidate = None
    baud = None

    # nominal / adaptive long & short cycle lengths (in samples), and the
    # classification threshold derived from them. Set once locked.
    long_nom = short_nom = None
    long_avg = short_avg = None
    threshold = None

    current = bytearray()
    current_confidences = []

    # Tracks a run of "short" pulses seen while in SYNCED state, so we can
    # tell a genuine new pilot tone (a long run, indicating a new block has
    # started) apart from the couple of "short" pulses that make up the
    # stop/mark gap between two bytes of the *same* block.
    ones_run = 0
    flushed_this_run = False

    def log(msg):
        if verbose:
            print(msg, file=sys.stderr)

    def classify(p):
        if p < threshold:
            return 'short'
        if p <= long_avg * max_gap_multiple:
            return 'long'
        return 'gap'

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
        """Read one encoded bit starting at periods[idx].
        Returns (bit, next_idx, confidence); bit is None on failure/EOF/gap."""
        if idx >= n:
            return None, idx, 0.0
        p = periods[idx]
        c = classify(p)
        if c == 'long':
            conf = pulse_confidence(p)
            update_long(p)
            return 0, idx + 1, conf
        elif c == 'short':
            if idx + 1 < n and classify(periods[idx + 1]) == 'short':
                p2 = periods[idx + 1]
                conf = (pulse_confidence(p) + pulse_confidence(p2)) / 2
                update_short(p)
                update_short(p2)
                return 1, idx + 2, conf
            return None, idx + 1, 0.0
        else:  # 'gap' - a dropout/silence stretch, not a genuine bit-0 cycle
            return None, idx + 1, 0.0

    def flush_block():
        if not current:
            return
        data = bytes(current)
        confs = list(current_confidences)
        if edge_trim:
            trimmed_data, trimmed_confs = _trim_block_edges(data, confs, edge_trim_threshold)
            dropped = len(data) - len(trimmed_data)
            if dropped:
                log("[trim] dropped %d low-confidence byte(s) from block edges "
                    "(kept %d of %d)" % (dropped, len(trimmed_data), len(data)))
            data, confs = trimmed_data, trimmed_confs
        if not data:
            log("[block] entire block trimmed away as low-confidence noise")
            return
        block_conf = sum(confs) / len(confs)
        blocks.append((baud, data, block_conf))
        log("[block] %d bytes decoded, confidence=%.3f" % (len(data), block_conf))

    while i < n:
        period = periods[i]

        if state == 'SEARCH':
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
                    log("[pilot] locked baud=%d (f0=%gHz f1=%gHz) near pulse %d"
                        % (baud, f0, f1, i))
                    state = 'SYNCED'
                    ones_run = 0
                    flushed_this_run = False
            else:
                pilot_count = 0
                i += 1

        elif state == 'SYNCED':
            # Either more pilot/mark ("1" bits, short pulses), the start
            # bit ("0", a long pulse) of a byte, or a dropout/silence gap.
            c = classify(period)
            if c == 'short':
                update_short(period)
                ones_run += 1
                i += 1
                # A long run of short pulses here (beyond what a stop/mark
                # gap would produce) means a genuine new pilot tone has
                # begun, i.e. a new block. Re-check the baud rate against
                # this run (it may differ from the currently locked baud),
                # then flush whatever we've accumulated so far exactly once
                # for this run.
                if ones_run >= min_pilot_pulses and not flushed_this_run:
                    window = periods[max(0, i - min_pilot_pulses):i]
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
                        log("[pilot] re-locked baud=%d (f0=%gHz f1=%gHz) near pulse %d"
                            % (baud, f0, f1, i))
                    flush_block()
                    current = bytearray()
                    current_confidences = []
                    flushed_this_run = True
            elif c == 'long':
                # Start bit of a byte. However much (or little) mark/idle
                # tone preceded it, if a genuine new pilot wasn't already
                # flushed above, current just keeps accumulating.
                ones_run = 0
                flushed_this_run = False
                state = 'BYTE'
                # do not advance i - BYTE state re-reads this pulse as the start bit
            else:
                # 'gap' - a dropout/silence stretch, not a genuine bit-0
                # cycle. Treat it as the end of whatever we were decoding:
                # flush what we have and fully resync via the pilot search,
                # rather than letting the gap (or the start of whatever
                # comes after it) be misread as a start bit. This is what
                # keeps a stretch of a *later* block's own pilot tone from
                # getting swallowed as bogus data of a phantom byte.
                log("[gap] dropout/silence period (%.1f samples) - resyncing" % period)
                flush_block()
                current = bytearray()
                current_confidences = []
                ones_run = 0
                flushed_this_run = False
                state = 'SEARCH'
                pilot_count = 0
                i += 1

        elif state == 'BYTE':
            start_bit, j, sconf = read_bit(i)
            if start_bit != 0:
                state = 'SEARCH'
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
                value |= (bit << bitpos)   # LSB first
                bit_confs.append(c)
                i = j
            if not ok:
                state = 'SEARCH'
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
                    state = 'SEARCH'
                    pilot_count = 0
                    continue

            current.append(value)
            current_confidences.append(sum(bit_confs) / len(bit_confs))
            state = 'SYNCED'
            ones_run = 0
            flushed_this_run = False

    flush_block()

    return blocks


# --------------------------------------------------------------------------
# CAS output
# --------------------------------------------------------------------------

def write_cas(path, blocks, pad=False):
    """Write (baud, data, confidence) blocks to a .cas file, each preceded
    by the standard CAS block marker. If pad is True, 0-7 zero bytes are
    inserted before each marker so it lands at a file offset that's a
    multiple of 8."""
    with open(path, 'wb') as f:
        for _baud, data, _conf in blocks:
            if pad:
                pad_len = (-f.tell()) % 8
                if pad_len:
                    f.write(b'\x00' * pad_len)
            f.write(CAS_HEADER)
            f.write(data)


# --------------------------------------------------------------------------
# Self-tests (run with --test)
# --------------------------------------------------------------------------
#
# These generate synthetic FSK sample streams directly in memory (no actual
# WAV files needed) and run them through find_rising_edges/edges_to_periods/
# decode() exactly as main() would, to catch regressions in future changes.

_TEST_FRAMERATE = 44100


def _test_add_cycle(samples, freq, amp, framerate=_TEST_FRAMERATE,
                     noise_amp=0.0, jitter=0.0, rng=None):
    rng = rng or random
    actual_freq = freq * (1 + rng.uniform(-jitter, jitter))
    n = max(2, int(round(framerate / actual_freq)))
    for k in range(n):
        v = amp * math.sin(2 * math.pi * k / n)
        v += rng.uniform(-noise_amp, noise_amp) * amp
        samples.append(v)


def _test_gen_block(samples, baud, payload, pilot_seconds, amp=20000,
                     noise_amp=0.0, jitter=0.0, stop_bits=2, rng=None,
                     framerate=_TEST_FRAMERATE):
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


def _test_decode_samples(samples, framerate=_TEST_FRAMERATE, threshold_ratio=0.2,
                          agc=True, **decode_kwargs):
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
    _test_add_cycle_pilot = None  # (unused placeholder to keep names local/clear)
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
        stop_n = 1 if idx % 2 == 0 else 3   # irregular: never the strict default of 2
        for _ in range(stop_n):
            _test_add_cycle(samples, f1, 20000, rng=rng)
            _test_add_cycle(samples, f1, 20000, rng=rng)
    samples += [0] * int(_TEST_FRAMERATE * 0.3)

    strict_blocks = _test_decode_samples(samples, strict_stop=True, stop_bits=2)
    lenient_blocks = _test_decode_samples(samples, strict_stop=False)

    problems = []
    if any(b[1] == payload for b in strict_blocks):
        problems.append("strict mode unexpectedly decoded the full irregular-stop payload")
    if not any(b[1] == payload for b in lenient_blocks):
        problems.append("lenient mode failed to decode the irregular-stop payload: got %r"
                         % ([b[1] for b in lenient_blocks],))
    return (not problems), "; ".join(problems) or "ok"


def _t_noise_jitter_tolerance():
    rng = random.Random(42)
    samples = []
    _test_gen_block(samples, 1200, b"HEADERID", 1.0, noise_amp=0.15, jitter=0.08, rng=rng)
    _test_gen_block(samples, 1200, bytes(range(256)), 0.3, noise_amp=0.15, jitter=0.08, rng=rng)
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
        return False, ("no-agc unexpectedly recovered the quiet payload too - "
                        "test isn't discriminating, needs a bigger amplitude gap")
    return True, "ok"


def _t_gap_does_not_produce_spurious_byte():
    # A substantial silent gap (splice/dropout) between two blocks must not
    # get misread as a start bit, which would otherwise let the following
    # block's own pilot tone get partially swallowed as a bogus extra byte
    # tacked onto the end of the preceding block.
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
        return False, ("block 1 wasn't decoded cleanly (a gap-induced spurious byte would "
                        "show up here): got %r" % (payloads,))
    if payload2 not in payloads:
        return False, "block 2 wasn't decoded cleanly: got %r" % (payloads,)
    return True, "ok"


def _t_confidence_filtering():
    rng = random.Random(99)
    samples = []
    _test_gen_block(samples, 1200, b"CLEANMSG", 0.5, noise_amp=0.02, jitter=0.01, rng=rng)
    samples += [0] * 300
    _test_gen_block(samples, 1200, b"GARBLED!", 0.5, noise_amp=0.35, jitter=0.25, rng=rng)
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
        return False, ("garbled block confidence (%.3f) not lower than clean block's (%.3f)"
                        % (garbled[2], clean[2]))
    kept = [b for b in blocks if b[2] >= 0.85]
    if garbled in kept:
        return False, "garbled block incorrectly passed a 0.85 confidence threshold"
    return True, "ok"


def _t_edge_trim_mixed_confidence():
    data = bytes([0xFF, 0xFE, ord('H'), ord('I'), 0xFD])
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
        return False, "expected an empty result, got %r / %r" % (trimmed_data, trimmed_confs)
    return True, "ok"


def _t_edge_trim_untouched_interior():
    # a low-confidence byte in the *middle* must NOT be removed - trimming
    # only ever eats inward from the two ends.
    data = bytes([ord('A'), 0x00, ord('B')])
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
        with open(path, 'rb') as f:
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
    ("gap/dropout doesn't produce a spurious byte", _t_gap_does_not_produce_spurious_byte),
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
        print("[%s] %s%s" % ("PASS" if ok else "FAIL", name,
                              "" if (not msg or msg == "ok") else (" - " + msg)))
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
        description="Decode MSX cassette audio (WAV) into a .CAS file.")
    parser.add_argument('input', nargs='?', help="input .wav file")
    parser.add_argument('output', nargs='?', help="output .cas file")
    parser.add_argument('--tolerance', type=float, default=0.30,
                         help="relative frequency tolerance used only for initial pilot-tone "
                              "baud detection, as a fraction (default: 0.30)")
    parser.add_argument('--min-pilot-pulses', type=int, default=40,
                         help="consecutive pilot pulses required to lock onto a baud rate, "
                              "and to distinguish a genuine new pilot tone from ordinary "
                              "inter-byte mark time (default: 40)")
    parser.add_argument('--threshold-ratio', type=float, default=0.2,
                         help="Schmitt-trigger threshold as a fraction of the (local, if AGC "
                              "is enabled) amplitude envelope (default: 0.2)")
    parser.add_argument('--no-agc', action='store_true',
                         help="disable local amplitude tracking (AGC) and use one fixed "
                              "threshold based on the whole file's peak amplitude instead")
    parser.add_argument('--agc-attack', type=float, default=0.3,
                         help="how quickly the AGC envelope rises to follow increasing "
                              "amplitude, 0-1 (default: 0.3)")
    parser.add_argument('--agc-release', type=float, default=0.0008,
                         help="how quickly the AGC envelope falls to follow decreasing "
                              "amplitude, 0-1 (default: 0.0008)")
    parser.add_argument('--agc-floor-ratio', type=float, default=0.02,
                         help="floor for the AGC envelope, as a fraction of the file's overall "
                              "peak amplitude, so quiet/silent stretches don't let the "
                              "threshold collapse and trigger on noise (default: 0.02)")
    parser.add_argument('--no-adapt', action='store_true',
                         help="disable the adaptive long/short threshold tracking and use "
                              "fixed nominal cycle lengths for the whole file")
    parser.add_argument('--adapt-rate', type=float, default=0.1,
                         help="how quickly the adaptive threshold tracks measured pulse "
                              "lengths, 0-1 (default: 0.1)")
    parser.add_argument('--adapt-clamp', type=float, default=0.35,
                         help="maximum fractional deviation from the nominal cycle length "
                              "the adaptive threshold is allowed to track (default: 0.35)")
    parser.add_argument('--lenient-stop-bits', action='store_true',
                         help="don't require a fixed number of stop bits after each byte - "
                              "just accept whatever mark/idle tone (if any) precedes the next "
                              "start bit. Default is strict (see --stop-bits)")
    parser.add_argument('--stop-bits', type=int, default=2,
                         help="number of stop bits required per byte in strict mode "
                              "(default: 2, matching the MSX BIOS cassette routines; ignored "
                              "if --lenient-stop-bits is given)")
    parser.add_argument('--min-confidence', type=float, default=0.8,
                         help="blocks with an average confidence below this (0-1) are left "
                              "out of the output file (default: 0.8)")
    parser.add_argument('--no-edge-trim', action='store_true',
                         help="disable automatic trimming of low-confidence bytes from the "
                              "start/end of an otherwise good block (see --edge-trim-threshold)")
    parser.add_argument('--edge-trim-threshold', type=float, default=0.5,
                         help="bytes at the very start/end of a block with confidence below "
                              "this (0-1) are dropped, stopping at the first byte that meets "
                              "it - catches stray bytes from noise at a block boundary without "
                              "touching the block's interior (default: 0.5)")
    parser.add_argument('--max-gap-multiple', type=float, default=3.0,
                         help="a pulse longer than this many times the current long-cycle "
                              "reference is treated as a dropout/silence gap rather than a "
                              "genuine bit-0 cycle, forcing a resync via the pilot search "
                              "instead of risking it being misread as a start bit "
                              "(default: 3.0)")
    parser.add_argument('--pad', action='store_true',
                         help="insert 0-7 zero-byte padding before each CAS block header so "
                              "the header starts at a file offset that's a multiple of 8 "
                              "(some MSX tools expect this; default: off)")
    parser.add_argument('--test', action='store_true',
                         help="run the internal self-test suite and exit (no input/output "
                              "files needed) - use this after making changes to check for "
                              "regressions")
    parser.add_argument('-v', '--verbose', action='store_true',
                         help="print decoding progress to stderr")
    args = parser.parse_args()

    if args.test:
        sys.exit(0 if run_self_tests() else 1)

    if args.input is None or args.output is None:
        parser.error("input and output are required (unless --test is given)")

    print("Reading %s ..." % args.input, file=sys.stderr)
    samples, framerate = read_wav_mono(args.input)
    print("%d samples at %d Hz" % (len(samples), framerate), file=sys.stderr)

    edges = find_rising_edges(samples, args.threshold_ratio,
                               agc=not args.no_agc,
                               agc_attack=args.agc_attack,
                               agc_release=args.agc_release,
                               agc_floor_ratio=args.agc_floor_ratio)
    print("%d rising edges detected" % len(edges), file=sys.stderr)

    periods = edges_to_periods(edges)

    blocks = decode(periods, framerate,
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
                     verbose=args.verbose)

    if not blocks:
        print("No data blocks decoded - try adjusting --tolerance, --min-pilot-pulses, "
              "--threshold-ratio, or --lenient-stop-bits", file=sys.stderr)

    kept = [b for b in blocks if b[2] >= args.min_confidence]

    if not kept:
        print("No blocks met the --min-confidence threshold (%.2f) - nothing written to %s"
              % (args.min_confidence, args.output), file=sys.stderr)
        for idx, (baud, data, conf) in enumerate(blocks):
            print("  block %d: baud=%d bytes=%d confidence=%.3f [DROPPED (below threshold)]"
                  % (idx, baud, len(data), conf), file=sys.stderr)
        sys.exit(1)

    write_cas(args.output, kept, pad=args.pad)

    print("Decoded %d block(s), wrote %d (min-confidence=%.2f):"
          % (len(blocks), len(kept), args.min_confidence), file=sys.stderr)
    for idx, (baud, data, conf) in enumerate(blocks):
        status = "kept" if conf >= args.min_confidence else "DROPPED (below threshold)"
        print("  block %d: baud=%d bytes=%d confidence=%.3f [%s]"
              % (idx, baud, len(data), conf, status), file=sys.stderr)

    print("Wrote %s" % args.output, file=sys.stderr)


if __name__ == '__main__':
    main()
