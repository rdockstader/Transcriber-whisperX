import json
import os
import shutil
from pathlib import Path

import whisperx
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError


load_dotenv()

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
MODEL_SIZE = "medium"
DEVICE = "cpu"  # ctranslate2/faster-whisper does not support "mps".
COMPUTE_TYPE = "int8"
SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".wav",
    ".wma",
}
GATED_HF_REPOS = {
    "pyannote/speaker-diarization-3.1": "config.yaml",
    "pyannote/segmentation-3.0": "config.yaml",
}


def check_hugging_face_access(hf_token):
    for repo_id, filename in GATED_HF_REPOS.items():
        try:
            hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token)
        except HfHubHTTPError as exc:
            raise RuntimeError(
                "Hugging Face access check failed for "
                f"https://huggingface.co/{repo_id}.\n\n"
                "Make sure you accepted that model's user conditions while "
                "signed into the same Hugging Face account that owns your "
                "HF_TOKEN. If you use a fine-grained token, make sure this "
                "repo is included in the token's allowed repositories."
            ) from exc


def format_timestamp(seconds, always_include_hours=False, decimal_marker="."):
    milliseconds = round(seconds * 1000)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    seconds = milliseconds // 1000
    milliseconds %= 1000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}"
        f"{decimal_marker}{milliseconds:03d}"
    )


def speaker_text(segment):
    speaker = segment.get("speaker", "UNKNOWN")
    text = segment.get("text", "").strip()
    return f"[{speaker}]: {text}"


def make_json_safe(value):
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def find_audio_files(input_dir):
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )


def output_audio_path(audio_path, output_dir):
    return output_dir / audio_path.name


def transcription_already_completed(audio_path, output_dir):
    return output_audio_path(audio_path, output_dir).exists()


def copy_source_audio(audio_path, output_dir):
    destination = output_audio_path(audio_path, output_dir)
    if not destination.exists():
        shutil.copy2(audio_path, destination)
    return destination


def write_txt(segments, output_path):
    lines = [speaker_text(segment) for segment in segments]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tsv(segments, output_path):
    lines = ["start\tend\tspeaker\ttext"]
    for segment in segments:
        start = round(segment.get("start", 0) * 1000)
        end = round(segment.get("end", 0) * 1000)
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "").strip().replace("\t", " ")
        lines.append(f"{start}\t{end}\t{speaker}\t{text}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_srt(segments, output_path):
    blocks = []
    for index, segment in enumerate(segments, start=1):
        start = format_timestamp(
            segment.get("start", 0), always_include_hours=True, decimal_marker=","
        )
        end = format_timestamp(
            segment.get("end", 0), always_include_hours=True, decimal_marker=","
        )
        blocks.append(f"{index}\n{start} --> {end}\n{speaker_text(segment)}")

    output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def write_vtt(segments, output_path):
    blocks = ["WEBVTT"]
    for segment in segments:
        start = format_timestamp(segment.get("start", 0))
        end = format_timestamp(segment.get("end", 0))
        blocks.append(f"{start} --> {end}\n{speaker_text(segment)}")

    output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def write_outputs(result, audio_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    segments = result["segments"]

    write_txt(segments, output_dir / f"{stem}.txt")
    write_tsv(segments, output_dir / f"{stem}.tsv")
    write_srt(segments, output_dir / f"{stem}.srt")
    write_vtt(segments, output_dir / f"{stem}.vtt")

    json_path = output_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(make_json_safe(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def transcribe_audio(audio_path, output_dir, model, diarize_model, align_models):
    print(f"\nTranscribing {audio_path}")

    result = model.transcribe(str(audio_path))
    language = result["language"]

    if language not in align_models:
        align_models[language] = whisperx.load_align_model(
            language_code=language, device=DEVICE
        )

    model_a, metadata = align_models[language]
    result = whisperx.align(
        result["segments"], model_a, metadata, str(audio_path), DEVICE
    )

    diarize_segments = diarize_model(str(audio_path))
    result = whisperx.assign_word_speakers(diarize_segments, result)

    write_outputs(result, audio_path, output_dir)
    copied_audio_path = copy_source_audio(audio_path, output_dir)

    for segment in result["segments"]:
        print(speaker_text(segment))

    print(f"Saved output files to {output_dir}")
    print(f"Copied source audio to {copied_audio_path}")


def main():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token or hf_token == "hf_your_token_here":
        raise RuntimeError(
            "HF_TOKEN is not set. Add your Hugging Face token to .env before "
            "running transcribe.py."
        )

    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    audio_files = find_audio_files(INPUT_DIR)
    if not audio_files:
        print(f"No supported audio files found in {INPUT_DIR.resolve()}")
        return

    check_hugging_face_access(hf_token)

    model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
    try:
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token, device=DEVICE
        )
    except AttributeError as exc:
        raise RuntimeError(
            "Could not load the Hugging Face diarization pipeline. Confirm your "
            "HF_TOKEN has read access and that you accepted the user conditions "
            "for https://huggingface.co/pyannote/speaker-diarization-3.1 and "
            "https://huggingface.co/pyannote/segmentation-3.0"
        ) from exc

    align_models = {}

    for audio_path in audio_files:
        file_output_dir = OUTPUT_DIR / audio_path.stem
        if transcription_already_completed(audio_path, file_output_dir):
            print(f"Skipping {audio_path}: source audio already exists in {file_output_dir}")
            continue

        transcribe_audio(
            audio_path=audio_path,
            output_dir=file_output_dir,
            model=model,
            diarize_model=diarize_model,
            align_models=align_models,
        )


if __name__ == "__main__":
    main()
