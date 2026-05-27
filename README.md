# WhisperX Audio Transcriber

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#requirements)
[![WhisperX](https://img.shields.io/badge/whisperx-3.8%2B-brightgreen.svg)](#requirements)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

Audio transcription and speaker diarization using [WhisperX](https://github.com/m-bain/whisperX), Hugging Face diarization models, and ffmpeg.

## Features

- Transcribes audio files with WhisperX
- Aligns word-level timestamps
- Performs speaker diarization
- Prints readable speaker-labeled transcript output
- Supports `.aac`, `.aiff`, `.flac`, `.m4a`, `.mp3`, `.ogg`, `.wav`, `.wma`

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/rdockstader/Transcriber-whisperX
cd transcriber-whisperx

# 2. Create and activate a virtual environment (Python 3.10–3.13)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Add audio files to input/ and run
python transcribe.py
```

Speaker diarization works without a Hugging Face token using the default community model. If you want to use gated models, see [Hugging Face Setup](#hugging-face-setup).

## Requirements

- Python 3.10, 3.11, 3.12, or 3.13
- ffmpeg (system package)
- A C compiler (only needed if a dependency falls back to source build — usually not required)

### macOS (Homebrew)

```bash
brew install python ffmpeg
```

### Linux (apt)

```bash
sudo apt install python3 python3-venv ffmpeg
```

## Hugging Face Setup

The default diarization model (`pyannote/speaker-diarization-community-1`) does not require a Hugging Face account. For access to gated research models like `pyannote/speaker-diarization-3.1`, you need a token.

1. Create or sign in to a [Hugging Face](https://huggingface.co) account.
2. Create an access token at `https://huggingface.co/settings/tokens`.
3. Accept the gated model terms for the models you want to use.
4. Copy the example environment file and add your token:

```bash
cp default.env .env
```

Edit `.env`:

```env
HF_TOKEN=hf_your_token_here
```

The script loads `.env` automatically. You can also export the variable directly:

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

Each audio file gets its own output directory:

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

After a successful run, the original audio file is copied into its output folder. Future runs skip input files that already have their source audio copied there.

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
├── default.env         # Example environment file
├── README.md           # Project documentation
├── input/              # Audio files to transcribe
└── output/             # Generated transcripts grouped by recording name
```

## Troubleshooting

### torchcodec warning about FFmpeg version

You may see a warning from `pyannote.audio` that torchcodec can't find FFmpeg shared libraries. This is harmless — whisperx pre-loads audio using soundfile/ffmpeg CLI and passes it to pyannote as a waveform tensor, so torchcodec is never actually used for audio decoding.

### `av` fails to build from source

If you see an error like:

```text
error: 'AV_OPT_TYPE_CHANNEL_LAYOUT' undeclared
```

Your system ffmpeg is too new for the old PyAV source. This is fixed by upgrading to `whisperx>=3.8.0`, which pulls a pre-built `av` wheel that supports ffmpeg 6–8. Make sure your `requirements.txt` does not pin `whisperx==3.2.0` or older.

### NumPy 2 compatibility errors

WhisperX 3.8+ requires NumPy 2.1+. If you have an older environment with `numpy<2`, recreate the venv:

```bash
deactivate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Missing ffmpeg

If transcription fails with an ffmpeg-related error:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

Confirm it is on your path:

```bash
ffmpeg -version
```

### Hugging Face gated repo access errors

Common symptoms include `401 Unauthorized`, `403 Forbidden`, or messages saying access to a gated repo is restricted.

Fix checklist:

- Confirm `HF_TOKEN` is set in `.env` or exported in the same terminal session.
- Confirm the token has read access.
- Accept the model terms on Hugging Face while signed in to the account that owns the token.
- Retry after a few minutes if access was just approved.

```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('HF_TOKEN set:', bool(os.getenv('HF_TOKEN')))"
```

### Unsupported device mps

WhisperX runs through `ctranslate2`/`faster-whisper`, which does not support the Apple Silicon `mps` device. The script defaults to `cpu`:

```python
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
```

## Roadmap

- Add command-line arguments for audio file, model size, and output format
- Add optional CPU/GPU device selection via CLI flags
- Add tests for transcript formatting and environment validation
- Add structured logging and clearer runtime error messages

## License

MIT License — see [LICENSE](LICENSE) for details.
