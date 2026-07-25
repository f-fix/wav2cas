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

# Note on the code

Parts of this code were written with assistance from LLM-integrated coding tools. If you don't like it, feel free to use other software or rewrite parts you dislike. PR's are welcome!

## How did I end up using those? Don't I dislike slop?

Yes I hate it. This project began because I wanted a tape audio conversion tools where the conversion steps were all clearly documented and readable code, but which also performed well enough in terms of accuracy to actually be the tool I use. My manual attempts hadn't yielded comparable accuracy to existing closed-source tools so I started using the tools to help find the bugs and suggest improvements, and IMO the result is now good enough to actually be useful. In terms of slop, the tool-generated code doesn't closely resemble any existing solutions I have found. Rather it's a fairly passable translation of my requests into Python.
