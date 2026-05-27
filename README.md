# WhisperX Audio Transcriber

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](#requirements)
[![WhisperX](https://img.shields.io/badge/whisperx-3.2.0-brightgreen.svg)](#requirements)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)](#macos-apple-silicon-setup)
[![License](https://img.shields.io/badge/license-TBD-lightgrey.svg)](#license)

Audio transcription and speaker diarization using [WhisperX](https://github.com/m-bain/whisperX), Hugging Face gated diarization models, and ffmpeg. The project is tuned for a simple local workflow on macOS Apple Silicon.

## Features

- Transcribes audio files with WhisperX
- Aligns word-level timestamps
- Performs speaker diarization
- Prints readable speaker-labeled transcript output
- Uses `HF_TOKEN` from environment variables
- Supports Apple Silicon using CPU execution for WhisperX compatibility

## Quick Start

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd Transcriber

# 2. Create and activate a Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install ffmpeg
brew install ffmpeg

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Export your Hugging Face token
export HF_TOKEN="hf_your_token_here"

# 6. Add audio files to input/ and run transcription
python transcribe.py
```

## Requirements

- macOS on Apple Silicon
- Python 3.12
- Homebrew
- ffmpeg
- pkg-config
- Hugging Face account and access token
- Access accepted for the required Hugging Face diarization models

Recommended dependency pins:

```txt
whisperx==3.2.0
torch==2.2.2
torchaudio==2.2.2
faster-whisper==1.0.0
ctranslate2==4.4.0
numpy<2
```

## macOS Apple Silicon Setup

Install Homebrew if needed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Install Python 3.12, ffmpeg, and pkg-config:

```bash
brew install python@3.12 ffmpeg pkg-config
```

Create a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
```

Confirm the version is Python 3.12.x.

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify ffmpeg is available:

```bash
ffmpeg -version
```

## Hugging Face Setup

Speaker diarization requires authenticated access to gated Hugging Face models.

1. Create or sign in to a Hugging Face account.
2. Create an access token at `https://huggingface.co/settings/tokens`.
3. Accept the gated model terms for the diarization models used by WhisperX, commonly:
   - `pyannote/speaker-diarization-3.1`
   - `pyannote/segmentation-3.0`
4. Copy the example environment file and add your token:

```bash
cp default.env .env
```

Edit `.env`:

```env
HF_TOKEN=hf_your_token_here
```

The script loads `.env` automatically. You can still use an exported environment variable instead:

```bash
export HF_TOKEN="hf_your_token_here"
```

## Usage

Add one or more audio files to `input/`:

```text
input/
└── Recording 14.mp3
```

Run transcription:

```bash
python transcribe.py
```

Each audio file gets its own output directory based on the file name:

```text
output/
└── Recording 14/
    ├── Recording 14.mp3
    ├── Recording 14.txt
    ├── Recording 14.json
    ├── Recording 14.srt
    ├── Recording 14.tsv
    └── Recording 14.vtt
```

After a successful run, the original audio file is copied into its output folder. Future runs skip input files that already have their source audio copied there, so completed files are not processed twice and the original input file is left untouched.

Example output:

```text
[SPEAKER_00] Thanks for joining the call today.
[SPEAKER_01] Of course. I wanted to review the project timeline.
[SPEAKER_00] Great, let's start with the transcription workflow.
```

## Project Structure

```text
.
├── transcribe.py       # Main transcription and diarization script
├── requirements.txt    # Python dependency pins
├── README.md           # Project documentation
├── input/              # Audio files to transcribe
└── output/             # Generated transcripts grouped by recording name
```

## Troubleshooting

### Python 3.14 incompatibility

WhisperX and its machine learning dependencies may not support Python 3.14 yet. Use Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
```

If your virtual environment was created with the wrong Python version, remove and recreate it:

```bash
deactivate
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### ctranslate2 version issues

If you see errors from `ctranslate2`, pin it to the known compatible version:

```bash
pip install "ctranslate2==4.4.0"
```

Then reinstall the project requirements:

```bash
pip install -r requirements.txt
```

### NumPy 2 compatibility errors

If startup fails with an error like:

```text
AttributeError: `np.NaN` was removed in the NumPy 2.0 release
```

or:

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

downgrade NumPy inside the virtual environment:

```bash
python -m pip install "numpy<2"
python -m pip install -r requirements.txt
```

### Missing ffmpeg

If transcription fails with an ffmpeg-related error, install ffmpeg:

```bash
brew install ffmpeg
```

Confirm it is on your path:

```bash
which ffmpeg
ffmpeg -version
```

### PyAV requires pkg-config

If installation fails while building `av` / `PyAV` with:

```text
pkg-config is required for building PyAV
```

Install `pkg-config` and make sure ffmpeg is installed:

```bash
brew install pkg-config ffmpeg
python -m pip install -r requirements.txt
```

If `av==11.*` fails with an error like:

```text
use of undeclared identifier 'AV_OPT_TYPE_CHANNEL_LAYOUT'
```

you are likely building PyAV 11 against a newer FFmpeg release. Install an older Homebrew FFmpeg formula and point `pkg-config` at it while installing:

```bash
brew install ffmpeg@6 pkg-config
export PKG_CONFIG_PATH="$(brew --prefix ffmpeg@6)/lib/pkgconfig"
python -m pip install --no-cache-dir --force-reinstall "av==11.0.0"
python -m pip install -r requirements.txt
```

If `ffmpeg@6` is not available from Homebrew, the most reliable fallback is to recreate the project environment with Python 3.11 and reinstall the requirements.

### Hugging Face gated repo access errors

Common symptoms include `401 Unauthorized`, `403 Forbidden`, or messages saying access to a gated repo is restricted.

WhisperX currently tries to download:

```text
pyannote/speaker-diarization-3.1
```

Fix checklist:

- Confirm `HF_TOKEN` is exported in the same terminal session.
- Confirm the token has read access.
- Accept the model terms on Hugging Face while signed in:
  - `https://huggingface.co/pyannote/speaker-diarization-3.1`
  - `https://huggingface.co/pyannote/segmentation-3.0`
- Retry after a few minutes if access was just approved.

```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('HF_TOKEN set:', bool(os.getenv('HF_TOKEN')))"
```

### Unsupported device mps

If startup fails with:

```text
ValueError: unsupported device mps
```

use CPU mode. `faster-whisper` runs through `ctranslate2`, which does not support the `mps` device.

The script currently defaults to:

```python
DEVICE = "cpu"
compute_type = "int8"
```

## Roadmap

- Add command-line arguments for audio file, model size, and output format
- Save transcripts to `.txt`, `.json`, or `.srt`
- Add batch transcription for folders of audio files
- Add automatic audio file discovery from `sample_audio/`
- Add optional CPU/MPS device selection via CLI flags
- Add tests for transcript formatting and environment validation
- Add structured logging and clearer runtime error messages

## License

TBD. Add a license before publishing or sharing the repository publicly.
