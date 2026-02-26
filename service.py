"""
DJI Mic Bridge — Background service that continuously buffers mic audio
and exposes an HTTP endpoint to clip the last 90 seconds.

Features:
    - Rolling 90s audio buffer from DJI Mic via Bluetooth
    - HTTP trigger at /clip for Stream Deck plugin
    - Alt+F10 global hotkey auto-clips alongside ShadowPlay
    - Auto-reconnects when BT mic disconnects/reconnects
    - Autostart-friendly (see autostart.py)

Usage:
    python service.py              # Start with auto-detected DJI Mic
    python service.py --device 3   # Start with specific device index
    python service.py --list       # List available audio devices
"""

import argparse
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from flask import Flask, jsonify

from config import (
    AUDIO_FORMAT,
    BUFFER_SECONDS,
    CHANNELS,
    CLIPS_DIR,
    DEVICE_NAME,
    HOST,
    PORT,
    SAMPLE_RATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dji-mic")


class AudioBuffer:
    """Thread-safe circular buffer for continuous audio recording."""

    def __init__(self, duration: int, sample_rate: int, channels: int):
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_frames = duration * sample_rate
        self.buffer = np.zeros((self.max_frames, channels), dtype=np.float32)
        self.write_pos = 0
        self.total_written = 0
        self.lock = threading.Lock()

    def write(self, data: np.ndarray):
        """Write incoming audio data into the circular buffer."""
        frames = len(data)
        with self.lock:
            if frames >= self.max_frames:
                self.buffer[:] = data[-self.max_frames:]
                self.write_pos = 0
                self.total_written += frames
                return

            end_pos = self.write_pos + frames
            if end_pos <= self.max_frames:
                self.buffer[self.write_pos:end_pos] = data
            else:
                first = self.max_frames - self.write_pos
                self.buffer[self.write_pos:] = data[:first]
                self.buffer[:frames - first] = data[first:]

            self.write_pos = end_pos % self.max_frames
            self.total_written += frames

    def read(self) -> np.ndarray:
        """Read the entire buffer in chronological order."""
        with self.lock:
            if self.total_written < self.max_frames:
                return self.buffer[:self.write_pos].copy()
            return np.concatenate([
                self.buffer[self.write_pos:],
                self.buffer[:self.write_pos],
            ])

    def reset(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer[:] = 0
            self.write_pos = 0
            self.total_written = 0

    @property
    def seconds_buffered(self) -> float:
        with self.lock:
            frames = min(self.total_written, self.max_frames)
        return frames / self.sample_rate


# --- Globals ---
audio_buf: AudioBuffer | None = None
stream: sd.InputStream | None = None
mic_connected = False
_args = None  # parsed CLI args, stored for reconnect


def find_dji_device(name_hint: str | None = None) -> int | None:
    """Find the DJI Mic device index by name substring match."""
    sd._terminate()
    sd._initialize()
    devices = sd.query_devices()
    candidates = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] < 1:
            continue
        dev_name = dev["name"].lower()
        if name_hint and name_hint.lower() in dev_name:
            candidates.append((i, dev))
        elif not name_hint and "dji" in dev_name:
            candidates.append((i, dev))

    if not candidates:
        return None
    # Prefer DirectSound endpoint (highest quality for BT audio)
    for idx, dev in candidates:
        hostapi = sd.query_hostapis(dev["hostapi"])["name"]
        if "directsound" in hostapi.lower():
            return idx
    # Fallback: prefer highest sample rate
    candidates.sort(key=lambda x: x[1]["default_samplerate"], reverse=True)
    return candidates[0][0]


def audio_callback(indata, frames, time_info, status):
    """Called by sounddevice for each audio block."""
    if status:
        log.warning(f"audio: {status}")
    if audio_buf is not None:
        audio_buf.write(indata.copy())


def start_stream(device_idx: int, rate: int, channels: int) -> bool:
    """Start the audio input stream. Returns True on success."""
    global stream, mic_connected
    try:
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

        dev_info = sd.query_devices(device_idx)
        actual_channels = min(channels, dev_info["max_input_channels"])

        stream = sd.InputStream(
            device=device_idx,
            samplerate=rate,
            channels=actual_channels,
            dtype="float32",
            callback=audio_callback,
            blocksize=1024,
        )
        stream.start()
        mic_connected = True
        log.info(f"stream started: [{device_idx}] {dev_info['name']} @ {rate}Hz")
        return True
    except Exception as e:
        mic_connected = False
        log.error(f"stream start failed: {e}")
        return False


def stop_stream():
    """Stop the audio input stream."""
    global stream, mic_connected
    if stream is not None:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        stream = None
    mic_connected = False


def reconnect_loop():
    """Background thread: monitor mic connection and auto-reconnect."""
    global mic_connected
    while True:
        time.sleep(5)
        if mic_connected:
            # Check if stream is still alive
            if stream is not None and not stream.active:
                log.warning("stream died, attempting reconnect...")
                mic_connected = False

        if not mic_connected:
            device_idx = _args.device if _args and _args.device is not None else None
            if device_idx is None:
                device_idx = find_dji_device(DEVICE_NAME)

            if device_idx is not None:
                rate = _args.rate if _args else SAMPLE_RATE
                log.info(f"mic detected at [{device_idx}], reconnecting...")
                if audio_buf:
                    audio_buf.reset()
                start_stream(device_idx, rate, CHANNELS)
            # If not found, just keep polling silently


def save_clip() -> Path:
    """Save the current buffer to a timestamped audio file."""
    if audio_buf is None or audio_buf.seconds_buffered == 0:
        raise RuntimeError("Buffer is empty — no audio recorded yet")

    data = audio_buf.read()
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"mic_{timestamp}.{AUDIO_FORMAT}"
    filepath = CLIPS_DIR / filename

    sf.write(str(filepath), data, audio_buf.sample_rate)
    duration = len(data) / audio_buf.sample_rate
    log.info(f"clip saved: {duration:.1f}s -> {filepath}")
    return filepath


# --- Flask HTTP trigger ---
app = Flask(__name__)
app.logger.setLevel(logging.WARNING)


@app.route("/clip", methods=["GET", "POST"])
def clip_endpoint():
    """Trigger a clip save."""
    try:
        path = save_clip()
        return jsonify({
            "status": "ok",
            "file": str(path),
            "seconds": audio_buf.seconds_buffered,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/status")
def status_endpoint():
    """Check service health and buffer status."""
    return jsonify({
        "status": "running",
        "mic_connected": mic_connected,
        "buffered_seconds": round(audio_buf.seconds_buffered, 1) if audio_buf else 0,
        "device": stream.device if stream else None,
        "sample_rate": _args.rate if _args else SAMPLE_RATE,
    })


def list_devices():
    """Print all available audio input devices."""
    devices = sd.query_devices()
    print("\nAvailable input devices:\n")
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = " <-- DJI?" if "dji" in dev["name"].lower() else ""
            print(f"  [{i}] {dev['name']} "
                  f"(ch={dev['max_input_channels']}, "
                  f"sr={dev['default_samplerate']:.0f}){marker}")
    print()


def setup_hotkey():
    """Register Alt+F10 global hotkey to auto-clip alongside ShadowPlay."""
    try:
        import keyboard
        def on_alt_f10():
            log.info("Alt+F10 detected — clipping mic audio")
            try:
                save_clip()
            except Exception as e:
                log.error(f"hotkey clip failed: {e}")

        keyboard.add_hotkey("alt+f10", on_alt_f10, suppress=False)
        log.info("hotkey registered: Alt+F10 -> clip")
    except ImportError:
        log.warning("keyboard module not installed — Alt+F10 hotkey disabled")
    except Exception as e:
        log.warning(f"hotkey setup failed: {e}")


def main():
    global audio_buf, _args

    parser = argparse.ArgumentParser(description="DJI Mic Bridge Service")
    parser.add_argument("--list", action="store_true", help="List audio devices and exit")
    parser.add_argument("--device", type=int, default=None, help="Audio device index (from --list)")
    parser.add_argument("--rate", type=int, default=SAMPLE_RATE, help="Sample rate (Hz)")
    parser.add_argument("--duration", type=int, default=BUFFER_SECONDS, help="Buffer duration (seconds)")
    parser.add_argument("--port", type=int, default=PORT, help="HTTP server port")
    args = parser.parse_args()
    _args = args

    if args.list:
        list_devices()
        sys.exit(0)

    # Resolve device
    device_idx = args.device
    if device_idx is None:
        device_idx = find_dji_device(DEVICE_NAME)

    actual_rate = args.rate

    log.info(f"buffer: {args.duration}s rolling @ {actual_rate}Hz")
    log.info(f"clips dir: {CLIPS_DIR.resolve()}")
    log.info(f"HTTP trigger: http://{HOST}:{args.port}/clip")

    # Init buffer
    audio_buf = AudioBuffer(args.duration, actual_rate, CHANNELS)

    # Start audio stream (or wait for mic to connect)
    if device_idx is not None:
        start_stream(device_idx, actual_rate, CHANNELS)
    else:
        log.warning("DJI Mic not found — waiting for Bluetooth connection...")

    # Start reconnect monitor
    reconnect_thread = threading.Thread(target=reconnect_loop, daemon=True)
    reconnect_thread.start()

    # Register Alt+F10 hotkey
    setup_hotkey()

    # Run Flask in a thread
    import werkzeug.serving
    werkzeug_log = logging.getLogger("werkzeug")
    werkzeug_log.setLevel(logging.ERROR)

    server_thread = threading.Thread(
        target=lambda: app.run(host=HOST, port=args.port, threaded=True, use_reloader=False),
        daemon=True,
    )
    server_thread.start()

    log.info("service running — press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("shutting down...")
        stop_stream()


if __name__ == "__main__":
    main()
