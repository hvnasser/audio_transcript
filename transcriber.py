"""
Offline multilingual audio transcriber using OpenAI Whisper.
Detects language automatically and handles multiple languages in the same audio.
"""

import argparse
import sys
from pathlib import Path

import whisper


WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]


def transcribe(
    audio_path: str,
    model_name: str = "base",
    language: str | None = None,
    task: str = "transcribe",
    detect_per_segment: bool = False,
) -> dict:
    """
    Transcribe an audio file with automatic language detection.

    Args:
        audio_path:          Path to the audio file.
        model_name:          Whisper model size. Larger = more accurate but slower.
        language:            Force a specific language (ISO-639-1, e.g. 'en', 'es').
                             Leave None to auto-detect.
        task:                'transcribe' to keep original language,
                             'translate' to translate everything to English.
        detect_per_segment:  Print per-segment language probability info.

    Returns:
        dict with keys:
          - text:     Full transcription string.
          - segments: List of timed segments with per-segment metadata.
          - language: Detected (or forced) language code.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    print(f"Transcribing: {audio_path.name}")
    result = model.transcribe(
        str(audio_path),
        language=language,
        task=task,
        verbose=False,
    )

    if detect_per_segment:
        _print_segment_details(result["segments"])

    return result


def _print_segment_details(segments: list) -> None:
    print("\n--- Segment details ---")
    for seg in segments:
        start = _fmt_time(seg["start"])
        end = _fmt_time(seg["end"])
        text = seg["text"].strip()
        # language probability map is available when Whisper returns it
        lang_probs = seg.get("language_probs")
        if lang_probs:
            top = sorted(lang_probs.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join(f"{l}={p:.2f}" for l, p in top)
            print(f"[{start} -> {end}] ({top_str}) {text}")
        else:
            print(f"[{start} -> {end}] {text}")
    print("--- End segments ---\n")


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline multilingual audio transcriber (Whisper)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("audio", help="Path to the audio file (mp3, wav, m4a, ogg, flac, ...)")
    parser.add_argument(
        "--model",
        default="base",
        choices=WHISPER_MODELS,
        help=(
            "Whisper model to use (default: base).\n"
            "  tiny   ~39M  – fastest, least accurate\n"
            "  base   ~74M  – good balance for quick use\n"
            "  small  ~244M – better accuracy\n"
            "  medium ~769M – recommended for multilingual\n"
            "  large  ~1.5G – best accuracy\n"
        ),
    )
    parser.add_argument(
        "--language",
        default=None,
        metavar="LANG",
        help="Force language (ISO-639-1 code, e.g. en, es, fr). Auto-detect if omitted.",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="'transcribe' keeps original language; 'translate' converts to English.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Save transcript to a text file instead of (or in addition to) stdout.",
    )
    parser.add_argument(
        "--segments",
        action="store_true",
        help="Print per-segment timestamps and language probabilities.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    result = transcribe(
        audio_path=args.audio,
        model_name=args.model,
        language=args.language,
        task=args.task,
        detect_per_segment=args.segments,
    )

    detected_language = result.get("language", "unknown")
    full_text = result["text"].strip()

    print(f"\nDetected language: {detected_language}")
    print("\n=== Transcript ===")
    print(full_text)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(full_text, encoding="utf-8")
        print(f"\nTranscript saved to: {out_path}")


if __name__ == "__main__":
    main()
