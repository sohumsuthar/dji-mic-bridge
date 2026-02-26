"""Configuration for DJI Mic Bridge."""

from pathlib import Path

# Buffer duration in seconds (shared)
BUFFER_SECONDS: int = 90
AUDIO_FORMAT: str = "flac"

# --- Audio Sources ---
SOURCES = {
    "mic": {
        "device_name": "DJI Mic",       # BT keyboard mic
        "sample_rate": 44100,
        "channels": 1,
        "clips_dir": Path("S:/dji-mic-clips"),
        "prefer_api": "directsound",     # best quality for BT
    },
    "cam": {
        "device_name": "OsmoAction5pro", # wired action camera
        "sample_rate": 48000,
        "channels": 2,
        "clips_dir": Path("S:/dji-cam-clips"),
        "prefer_api": "wasapi",          # best for USB
    },
}

# HTTP trigger server
HOST: str = "127.0.0.1"
PORT: int = 9090
