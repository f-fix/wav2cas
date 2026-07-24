# wav2cas
decode MSX and MSX-like cassette audio from WAV files into CAS format

Pure Python, standard library only.

## Usage

```
python3 wav2cas.py [--filter] input.wav output.cas [options]
```

Run `python3 wav2cas.py --help` for the full list of options (baud rate is
auto-detected in the 600-9600 range; strict/lenient stop-bit handling, AGC,
adaptive bit-length tracking, per-block confidence scoring/filtering, edge
trimming, and CAS block padding are all configurable).

## Self-tests

```
python3 wav2cas.py --test
```

Runs the built-in regression suite (synthetic audio generated in memory, no
files needed) and exits non-zero on failure. Run this after making any
changes to the decoder.
