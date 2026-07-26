# wav2cas
decode MSX and MSX-like cassette audio from WAV files into CAS format

Pure Python, standard library only. Inputs are converted to 16-bit signed PCM WAV first using external `ffmpeg` if you have it; you can turn that feature off using `--no-native-formats`. There is also a fallback pure-python FLAC-to-WAV converter for execution environments without `ffmpeg` but it's quite slow, and likewise will be turned off by `--no-native-formats`.

## Usage

```bash
python3 wav2cas.py [--filter] input.wav output.cas [options]
```

Run `python3 wav2cas.py --help` for the full list of options (baud rate is
auto-detected in the 600-9600 range; strict/lenient stop-bit handling, AGC,
adaptive bit-length tracking, per-block confidence scoring/filtering, edge
trimming, and CAS block padding are all configurable).

You can run it using `pypy3` for a noticeable speed improvement.

## Self-tests

```
python3 wav2cas.py --test
```

Runs the built-in regression suite (synthetic audio generated in memory or as temporary files, no
pre-existing external test files needed) and exits non-zero on failure. Run this after making any
changes to the decoder.

## Output of `--help`

`````
usage: wav2cas.py [-h] [--tolerance TOLERANCE]
                  [--min-pilot-pulses MIN_PILOT_PULSES]
                  [--threshold-ratio THRESHOLD_RATIO] [--no-agc]
                  [--agc-attack AGC_ATTACK] [--agc-release AGC_RELEASE]
                  [--agc-floor-ratio AGC_FLOOR_RATIO] [--no-adapt]
                  [--adapt-rate ADAPT_RATE] [--adapt-clamp ADAPT_CLAMP]
                  [--lenient-stop-bits] [--stop-bits STOP_BITS]
                  [--min-confidence MIN_CONFIDENCE] [--no-edge-trim]
                  [--edge-trim-threshold EDGE_TRIM_THRESHOLD]
                  [--max-gap-multiple MAX_GAP_MULTIPLE] [--no-native-formats]
                  [--no-ffmpeg] [--stereo-to-mono {mix,left,right}] [--filter]
                  [--pad] [--test] [-v]
                  [input] [output]

Decode MSX or MSX-like cassette audio (WAV) into a .CAS file.

positional arguments:
  input                 input .wav file
  output                output .cas file

options:
  -h, --help            show this help message and exit
  --tolerance TOLERANCE
                        relative frequency tolerance used only for initial
                        pilot-tone baud detection, as a fraction (default:
                        0.30)
  --min-pilot-pulses MIN_PILOT_PULSES
                        consecutive pilot half-cycles required to lock onto a
                        baud rate (default: 80 - each is a half-cycle, so this
                        is ~40 full pilot cycles)
  --threshold-ratio THRESHOLD_RATIO
                        Schmitt-trigger threshold as a fraction of the
                        amplitude envelope (default: 0.2)
  --no-agc              disable local amplitude tracking (AGC)
  --agc-attack AGC_ATTACK
                        AGC attack rate, 0-1 (default: 0.3)
  --agc-release AGC_RELEASE
                        AGC release rate, 0-1 (default: 0.005)
  --agc-floor-ratio AGC_FLOOR_RATIO
                        floor for the AGC envelope (default: 0.02)
  --no-adapt            disable adaptive threshold tracking
  --adapt-rate ADAPT_RATE
                        adaptive threshold tracking rate, 0-1 (default: 0.1)
  --adapt-clamp ADAPT_CLAMP
                        maximum fractional deviation for adaptation (default:
                        0.35)
  --lenient-stop-bits   don't require a fixed number of stop bits after each
                        byte
  --stop-bits STOP_BITS
                        number of stop bits required per byte in strict mode
                        (default: 2)
  --min-confidence MIN_CONFIDENCE
                        minimum block confidence threshold (default: 0.8)
  --no-edge-trim        disable automatic trimming of low-confidence bytes
  --edge-trim-threshold EDGE_TRIM_THRESHOLD
                        edge trim confidence threshold (default: 0.5)
  --max-gap-multiple MAX_GAP_MULTIPLE
                        maximum gap multiple for dropout detection (default:
                        5.0)
  --no-native-formats   skip the built-in FLAC/ffmpeg support and read the
                        input as plain WAV only via the stdlib `wave` module -
                        no `subprocess` call is made. Use this if you'd rather
                        convert a non-WAV source to WAV yourself first (e.g.
                        with your own ffmpeg command)
  --no-ffmpeg           do not use ffpmeg, but still allow the built-in FLAC
                        decoder, which is very slow
  --stereo-to-mono {mix,left,right}
                        how to collapse a multi-channel input to mono: average
                        all channels together (default: mix), or use only the
                        left or right channel. Ignored for mono input
  --filter              Apply built-in noise filter for noisy tapes
  --pad                 insert 0-7 zero-byte padding before each CAS block
                        header
  --test                run internal self-tests and exit
  -v, --verbose         print decoding progress to stderr
`````

# cas2wav

Converts MSX CAS to WAV. Generates 22050 Hz monaural 16-bit signed linear PCM with 1200 baud FSK.

## Usage

```bash
python cas2wav.py input.cas output.wav
```

# Note on the code and the tools used to write it

Parts of this code were written with assistance from LLM-integrated coding tools. If you don't like it, feel free to use other software or rewrite parts you dislike. PR's are welcome!

## How did I end up using those? Don't I dislike slop?

Yes I hate it. This project began because I wanted tape audio conversion tools where the conversion steps were all clearly documented and readable code, but which also performed well enough in terms of accuracy to actually be the tool I use. I started out writing the tool myself, but my manual attempts hadn't yielded comparable accuracy to existing closed-source tools so I started using the tools to help find the bugs and suggest improvements, and IMO the result is now good enough to actually be useful. In terms of slop, the tool-generated code doesn't closely resemble any existing solutions I have found. Rather it's a fairly passable translation of my requests into Python.
