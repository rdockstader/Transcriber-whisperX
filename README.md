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
- Supports Apple Silicon via PyTorch MPS with CPU fallback options

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
   - `pyannote/speaker-diarization`
   - `pyannote/segmentation`
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
    ├── Recording 14.txt
    ├── Recording 14.json
    ├── Recording 14.srt
    ├── Recording 14.tsv
    └── Recording 14.vtt
```

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

### Hugging Face gated repo access errors

Common symptoms include `401 Unauthorized`, `403 Forbidden`, or messages saying access to a gated repo is restricted.

Fix checklist:

- Confirm `HF_TOKEN` is exported in the same terminal session.
- Confirm the token has read access.
- Accept the model terms on Hugging Face while signed in.
- Retry after a few minutes if access was just approved.

```bash
echo $HF_TOKEN
```

### Apple Silicon device issues

The script currently uses:

```python
device = "mps"
compute_type = "int8"
```

If you encounter MPS-related runtime errors, try CPU mode:

```python
device = "cpu"
compute_type = "int8"
```

CPU mode is slower, but it is useful for debugging dependency or hardware acceleration issues.

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
