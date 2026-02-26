"""
Merge keyboard mic audio into a ShadowPlay clip as a second audio track.

Usage:
    python merge.py <video.mp4> <mic_audio.wav>
    python merge.py <video.mp4> <mic_audio.wav> -o output.mp4
    python merge.py --auto          # Auto-match latest clip + latest mic audio
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from config import CLIPS_DIR


def find_latest_file(directory: Path, pattern: str) -> Path | None:
    """Find the most recently modified file matching a glob pattern."""
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def merge_audio(video: Path, mic_audio: Path, output: Path | None = None):
    """Merge mic audio as a second audio stream into the video file using FFmpeg."""
    if output is None:
        stem = video.stem
        output = video.parent / f"{stem}_merged.mp4"

    # ffmpeg: keep original video + audio, add mic as second audio track
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-i", str(mic_audio),
        "-map", "0:v",       # video from clip
        "-map", "0:a",       # original game audio
        "-map", "1:a",       # keyboard mic audio
        "-c:v", "copy",      # don't re-encode video
        "-c:a:0", "copy",    # don't re-encode game audio
        "-c:a:1", "aac",     # encode mic audio as AAC
        "-b:a:1", "128k",
        "-shortest",         # trim to shorter of the two
        "-metadata:s:a:0", "title=Game Audio",
        "-metadata:s:a:1", "title=Keyboard",
        str(output),
    ]

    print(f"[merge] Video: {video.name}")
    print(f"[merge] Mic:   {mic_audio.name}")
    print(f"[merge] Output: {output.name}")
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[error] FFmpeg failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"[done] Merged -> {output}")
    print(f"[done] Size: {output.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Merge keyboard mic audio into a video clip")
    parser.add_argument("video", nargs="?", help="Path to the video file (.mp4)")
    parser.add_argument("audio", nargs="?", help="Path to the mic audio file (.wav)")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-match latest ShadowPlay clip with latest mic recording")
    args = parser.parse_args()

    if args.auto:
        # Find latest ShadowPlay DVR clip — common default paths
        shadowplay_dirs = [
            Path.home() / "Videos" / "Valorant",
            Path.home() / "Videos",
            Path("C:/Users") / Path.home().name / "Videos" / "Valorant",
        ]
        video = None
        for d in shadowplay_dirs:
            if d.exists():
                video = find_latest_file(d, "*.mp4")
                if video:
                    break

        mic_audio = find_latest_file(CLIPS_DIR, "mic_*.wav")

        if not video:
            print("[error] No ShadowPlay clip found in Videos folder", file=sys.stderr)
            sys.exit(1)
        if not mic_audio:
            print("[error] No mic clip found in clips/ folder", file=sys.stderr)
            sys.exit(1)

        merge_audio(video, mic_audio)
    else:
        if not args.video or not args.audio:
            parser.error("Provide both video and audio paths, or use --auto")
        output = Path(args.output) if args.output else None
        merge_audio(Path(args.video), Path(args.audio), output)


if __name__ == "__main__":
    main()
