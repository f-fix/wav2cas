# wav2cas, cas2wav, flac2wav, cmt_filter, cassette_model

decode MSX and MSX-like cassette audio from WAV/FLAC files into CAS format; also encode CAS to WAV; also decode FLAC to WAV; also simulate CMT input/output audio effects and cassette tape audio effects

## wav2cas
decode MSX and MSX-like cassette audio from WAV files into CAS format

Pure Python, standard library only. Inputs are converted to 16-bit signed PCM WAV first using external `ffmpeg` if you have it; you can turn that feature off using `--no-native-formats`. There is also a fallback pure-python FLAC-to-WAV converter for execution environments without `ffmpeg` but it's quite slow, and likewise will be turned off by `--no-native-formats`.

### Usage

```bash
python3 wav2cas.py [--filter] input.wav output.cas [options]
```

Run `python3 wav2cas.py --help` for the full list of options (baud rate is
auto-detected in the 600-9600 range; strict/lenient stop-bit handling, AGC,
adaptive bit-length tracking, per-block confidence scoring/filtering, edge
trimming, and CAS block padding are all configurable).

You can run it using `pypy3` for a noticeable speed improvement.

### Self-tests

```
python3 wav2cas.py --test
```

Runs the built-in regression suite (synthetic audio generated in memory or as temporary files, no
pre-existing external test files needed) and exits non-zero on failure. Run this after making any
changes to the decoder.

### Output of `--help`

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

## cas2wav

Converts MSX CAS to WAV. Generates 22050 Hz monaural 16-bit signed linear PCM with 1200 baud FSK.

### Usage

```
cas2wav.py [-h] [--test] [input] [output]
input and output are required (unless --test is given)
```

### Output of `--help`

```
usage: cas2wav.py [-h] [--test] [input] [output]

Convert MSX .CAS cassette file to .WAV audio.

positional arguments:
  input       input .cas file
  output      output .wav file

options:
  -h, --help  show this help message and exit
  --test      run internal self-tests and exit
```

## flac2wav

Decodes FLAC to WAV in pure Python. It's no speed demon, but it's fast enough to use.

### Usage

```
flac2wav.py [-h] [--test] [input] [output]
```
Input and output files are required unless --test is specified.

### Output of `--help`

```
usage: flac2wav.py [-h] [--test] [input] [output]

Convert FLAC to WAV using a pure Python implementation.

positional arguments:
  input       Input .flac file
  output      Output .wav file

options:
  -h, --help  show this help message and exit
  --test      Run the internal verification test
```

## cmt_filter

Simulates CMT input and output filter circuit effects

### Output of `--help`

```
usage: cmt_filter.py [-h] [--test] [-m MODE] [-i INPUT_FILE] [-o OUTPUT_FILE]
                     [-c CHUNK_SIZE] [--tape-gain-db TAPE_GAIN_DB]

CMT Audio Shaping Circuits Streaming WAV Filter

options:
  -h, --help            show this help message and exit
  --test                Run the test suite
  -m, --mode MODE       Filter mode: 'input' (CMT-IN -> IOA7), 'output' (PC5
                        -> CMT OUT), or a chain such as 'output+input' /
                        'output,input' to simulate a full record->playback
                        round trip through a cassette deck.
  -i, --input-file INPUT_FILE
                        Path to input WAV file
  -o, --output-file OUTPUT_FILE
                        Path to output WAV file
  -c, --chunk-size CHUNK_SIZE
                        Chunk size in frames for streaming processing
                        (default: 1024)
  --tape-gain-db TAPE_GAIN_DB
                        OPTIONAL: inserts a gain stage right after each
                        'output' stage to simulate real-world tape
                        recording/playback level loss or gain (e.g. a weak
                        recording, worn tape, or misadjusted deck). Not needed
                        for normal use -- the output filter already normalizes
                        its own electrical peak to full WAV scale. Omit to
                        leave levels as-is (default: no extra stage added).

MSX and MSX-like Cassette Magnetic Tape Audio Input and Output Shaping
======================================================================

1. Circuit Overview
-------------------
The Cassette Magnetic Tape (CMT) interface circuits in MSX and
MSX-like computers handle signal conversion between analog audio
signals on magnetic tape and digital logic signals inside the MSX
or MSX-like system.

2. Input Circuit Analysis: CMT-IN -> IOA7 (Fig. 5-5-9)
------------------------------------------------------
This circuit takes the raw, noisy analog audio signal coming from a cassette player
(CMT-IN / CN4-5) and converts it into a clean digital 0V / 5V square wave fed into
IOA7 (Bit 7 of I/O Port A on the AY-3-8910 / YM2149 PSG sound chip).

Component & Circuit Effects:
- High-Pass Filter (C31 = 0.1 uF, R21 = 2.7 kOhm):
  Cutoff frequency fc_hp = 1 / (2 * pi * R21 * C31) ~= 589.5 Hz.
  Blocks DC offsets and eliminates low-frequency power supply hum (50/60 Hz) and
  cassette motor rumble.

- Low-Pass Filter (R20 = 12 kOhm, C29 = 0.0015 uF / 1.5 nF):
  Cutoff frequency fc_lp = 1 / (2 * pi * R20 * C29) ~= 8841.9 Hz.
  Suppresses high-frequency tape hiss, noise spikes, and tape bias tones.

- Combined Bandpass Transfer Function H_in(s):
  H_in(s) = (C31 * R21 * s) / [ (C29 * C31 * R20 * R21) * s^2 + (C29*R20 + C29*R21 + C31*R21) * s + 1 ]
  Peak center frequency is tuned to ~2263 Hz, providing maximum passband gain
  right between the cassette carrier FSK tones (1200 Hz space / 2400 Hz mark).

- Diode Clamping (Diodes D1, D2):
  Anti-parallel diodes connected to ground clamp signal amplitude to ~ +/-0.6V
  to protect the comparator and normalize playback levels.

- Comparator with Hysteresis (R33 = 4.7 kOhm, JRC-311B):
  Acts as a zero-crossing Schmitt trigger comparator, converting the filtered
  AC waveform into a sharp 0V / 5V square wave for digital reading by IOA7.

3. Output Circuit Analysis: PC5 -> CMT OUT (Fig. 5-4-10)
--------------------------------------------------------
This circuit takes digital 0V / 5V pulse streams from PPI (8255) pin PC5, smooths
the sharp transitions into a band-limited wave, attenuates the voltage to
microphone/line level, and sends it to CN4-4 CMT OUT. It also includes a relay motor
control circuit driven by PC4 via driver 75451.

Component & Circuit Effects:
- Logic Inverter (74LS14 4B, R37 = 10 kOhm pull-up):
  Inverts logic state from PPI PC5 and provides clean driving capability.

- DC Blocking / High-Pass Filter (C31 = 3.3 uF):
  fc_hp ~= 8.0 Hz (with ~6 kOhm load impedance). Removes +2.5V DC offset,
  centering the output waveform around 0V AC.

- Low-Pass Waveform Shaping (R41 = 1.2 kOhm, C32 = 0.022 uF / 22 nF):
  fc_lp = 1 / (2 * pi * R41 * C32) ~= 6028.6 Hz.
  Rounds off sharp square-wave edges into a pseudo-sine wave, eliminating
  high harmonics that would saturate magnetic tape.

- Attenuator Voltage Divider (R40 = 4.7 kOhm, R39 = 100 Ohm):
  Attenuation factor = 100 / (4700 + 100) ~= 0.0208 (-33.6 dB).
  Reduces 5V peak-to-peak TTL output down to ~100 mV peak-to-peak MIC/LINE level.

- Combined Exact Transfer Function H_out(s):
  H_out(s) = (C31 * R39 * s) / [ (C31 * C32 * R41 * (R39 + R40)) * s^2 + (C31*(R39+R40+R41) + C32*(R39+R40)) * s + 1 ]
```

## cassette_model

simulate cassette tape audio effects

### Output of `--help`

```
usage: cassette_model.py [-h] [-i INPUT] [-o OUTPUT] [-m MODE] [--drive DRIVE]
                         [--hiss HISS] [--test]

Audio Cassette Modeler with Explicit Filter Chaining (+)

options:
  -h, --help           show this help message and exit
  -i, --input INPUT    Path to input WAV file
  -o, --output OUTPUT  Path to output WAV file
  -m, --mode MODE      Filter sequence separated by '+' (e.g.,
                       'record+playback', 'playback+record',
                       'record+playback+record+playback')
  --drive DRIVE        Tape saturation drive intensity (default: 1.2)
  --hiss HISS          Tape hiss noise level (default: 0.001)
  --test               Run embedded unit test suite
```

# Note on the code and the tools used to write it

Parts of this code were written with assistance from LLM-integrated coding tools. If you don't like it, feel free to use other software or rewrite parts you dislike. PR's are welcome!

## How did I end up using those? Don't I dislike slop?

Yes I hate it. This project began because I wanted tape audio conversion tools where the conversion steps were all clearly documented and readable code, but which also performed well enough in terms of accuracy to actually be the tool I use. I started out writing the tool myself, but my manual attempts hadn't yielded comparable accuracy to existing closed-source tools so I started using the tools to help find the bugs and suggest improvements, and IMO the result is now good enough to actually be useful. In terms of slop, the tool-generated code doesn't closely resemble any existing solutions I have found. Rather it's a fairly passable translation of my requests into Python.
