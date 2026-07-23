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
    scoring below --min-confidence (default 0.0, i.e. no filtering) are
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


def decode(periods, framerate, tolerance=0.30, min_pilot_pulses=40,
           adapt=True, adapt_rate=0.1, adapt_clamp=0.35,
           strict_stop=True, stop_bits=2, verbose=False):
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
    mode); per-block confidence is the mean over its bytes."""

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
        return 'long' if p >= threshold else 'short'

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
        Returns (bit, next_idx, confidence); bit is None on failure/EOF."""
        if idx >= n:
            return None, idx, 0.0
        p = periods[idx]
        if classify(p) == 'long':
            conf = pulse_confidence(p)
            update_long(p)
            return 0, idx + 1, conf
        else:
            if idx + 1 < n and classify(periods[idx + 1]) == 'short':
                p2 = periods[idx + 1]
                conf = (pulse_confidence(p) + pulse_confidence(p2)) / 2
                update_short(p)
                update_short(p2)
                return 1, idx + 2, conf
            return None, idx + 1, 0.0

    def flush_block():
        if current:
            block_conf = sum(current_confidences) / len(current_confidences)
            blocks.append((baud, bytes(current), block_conf))
            log("[block] %d bytes decoded, confidence=%.3f" % (len(current), block_conf))

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
            # Either more pilot/mark ("1" bits, short pulses) or the start
            # bit ("0", a long pulse) of a byte.
            if classify(period) == 'short':
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
            else:
                # Start bit of a byte. However much (or little) mark/idle
                # tone preceded it, if a genuine new pilot wasn't already
                # flushed above, current just keeps accumulating.
                ones_run = 0
                flushed_this_run = False
                state = 'BYTE'
                # do not advance i - BYTE state re-reads this pulse as the start bit

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
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Decode MSX cassette audio (WAV) into a .CAS file.")
    parser.add_argument('input', help="input .wav file")
    parser.add_argument('output', help="output .cas file")
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
                              "out of the output file (default: 0.0, i.e. no filtering)")
    parser.add_argument('--pad', action='store_true',
                         help="insert 0-7 zero-byte padding before each CAS block header so "
                              "the header starts at a file offset that's a multiple of 8 "
                              "(some MSX tools expect this; default: off)")
    parser.add_argument('-v', '--verbose', action='store_true',
                         help="print decoding progress to stderr")
    args = parser.parse_args()

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
                     verbose=args.verbose)

    if not blocks:
        print("No data blocks decoded - try adjusting --tolerance, --min-pilot-pulses, "
              "--threshold-ratio, or --lenient-stop-bits", file=sys.stderr)

    kept = [b for b in blocks if b[2] >= args.min_confidence]

    assert kept  # give up if no blocks survived

    with open(args.output, 'wb') as f:
        for _baud, data, _conf in kept:
            if args.pad:
                pad_len = (-f.tell()) % 8
                if pad_len:
                    f.write(b'\x00' * pad_len)
            f.write(CAS_HEADER)
            f.write(data)

    print("Decoded %d block(s), wrote %d (min-confidence=%.2f):"
          % (len(blocks), len(kept), args.min_confidence), file=sys.stderr)
    for idx, (baud, data, conf) in enumerate(blocks):
        status = "kept" if conf >= args.min_confidence else "DROPPED (below threshold)"
        print("  block %d: baud=%d bytes=%d confidence=%.3f [%s]"
              % (idx, baud, len(data), conf, status), file=sys.stderr)

    print("Wrote %s" % args.output, file=sys.stderr)


if __name__ == '__main__':
    main()
