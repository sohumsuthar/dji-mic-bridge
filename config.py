"""Configuration for DJI Mic Bridge."""

from pathlib import Path

# Audio device — set to None to auto-detect DJI Mic, or set exact name
# Run `python -m sounddevice` to list devices
DEVICE_NAME: str | None = "DJI Mic"  # Auto-detect DJI Mic 3 TX Bluetooth

# Buffer duration in seconds
BUFFER_SECONDS: int = 90

# Audio settings — DirectSound at 44100Hz for max quality
SAMPLE_RATE: int = 44100
CHANNELS: int = 1

# Output — save to S: drive
CLIPS_DIR: Path = Path("S:/dji-mic-clips")
AUDIO_FORMAT: str = "flac"  # lossless compression

# HTTP trigger server
HOST: str = "127.0.0.1"
PORT: int = 9090
