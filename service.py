"""
DJI Mic Bridge — Multi-source audio buffer service.

Buffers audio from multiple DJI devices (mic + camera) and exposes
HTTP endpoints to clip them. Alt+F10 clips all sources at once.

Usage:
    python service.py              # Start with auto-detected devices
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

from config import AUDIO_FORMAT, BUFFER_SECONDS, HOST, PORT, SOURCES

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dji-bridge")


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
        with self.lock:
            if self.total_written < self.max_frames:
                return self.buffer[:self.write_pos].copy()
            return np.concatenate([
                self.buffer[self.write_pos:],
                self.buffer[:self.write_pos],
            ])

    def reset(self):
        with self.lock:
            self.buffer[:] = 0
            self.write_pos = 0
            self.total_written = 0

    @property
    def seconds_buffered(self) -> float:
        with self.lock:
            frames = min(self.total_written, self.max_frames)
        return frames / self.sample_rate


class AudioSource:
    """Manages one audio device: stream, buffer, and reconnection."""

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.device_name = cfg["device_name"]
        self.sample_rate = cfg["sample_rate"]
        self.channels = cfg["channels"]
        self.clips_dir = cfg["clips_dir"]
        self.prefer_api = cfg.get("prefer_api", "")
        self.buffer = AudioBuffer(BUFFER_SECONDS, self.sample_rate, self.channels)
        self.stream: sd.InputStream | None = None
        self.connected = False
        self.device_idx: int | None = None

    def find_device(self, refresh: bool = False) -> int | None:
        if refresh:
            try:
                sd._terminate()
                sd._initialize()
            except Exception:
                pass
        devices = sd.query_devices()
        candidates = []
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] < 1:
                continue
            if self.device_name.lower() in dev["name"].lower():
                candidates.append((i, dev))
        if not candidates:
            return None
        # Prefer the specified host API
        if self.prefer_api:
            for idx, dev in candidates:
                hostapi = sd.query_hostapis(dev["hostapi"])["name"]
                if self.prefer_api.lower() in hostapi.lower():
                    return idx
        # Fallback: highest sample rate
        candidates.sort(key=lambda x: x[1]["default_samplerate"], reverse=True)
        return candidates[0][0]

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.warning(f"[{self.name}] {status}")
        self.buffer.write(indata.copy())

    def start(self, device_idx: int | None = None) -> bool:
        try:
            self.stop()
            if device_idx is None:
                device_idx = self.find_device()
            if device_idx is None:
                return False
            dev_info = sd.query_devices(device_idx)
            ch = min(self.channels, dev_info["max_input_channels"])
            self.stream = sd.InputStream(
                device=device_idx,
                samplerate=self.sample_rate,
                channels=ch,
                dtype="float32",
                callback=self._callback,
                blocksize=1024,
            )
            self.stream.start()
            self.connected = True
            self.device_idx = device_idx
            log.info(f"[{self.name}] started: [{device_idx}] {dev_info['name']} @ {self.sample_rate}Hz ch={ch}")
            return True
        except Exception as e:
            self.connected = False
            log.error(f"[{self.name}] start failed: {e}")
            return False

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.connected = False

    def check_alive(self) -> bool:
        if not self.connected:
            return False
        try:
            if self.stream is not None and not self.stream.active:
                self.connected = False
                return False
        except Exception:
            self.connected = False
            return False
        return True

    def save_clip(self) -> Path:
        if self.buffer.seconds_buffered == 0:
            raise RuntimeError(f"[{self.name}] buffer empty")
        data = self.buffer.read()
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{self.name}_{ts}.{AUDIO_FORMAT}"
        filepath = self.clips_dir / filename
        sf.write(str(filepath), data, self.buffer.sample_rate)
        dur = len(data) / self.buffer.sample_rate
        log.info(f"[{self.name}] clip: {dur:.1f}s -> {filepath}")
        return filepath

    def status_dict(self) -> dict:
        return {
            "name": self.name,
            "connected": self.connected,
            "buffered_seconds": round(self.buffer.seconds_buffered, 1),
            "device": self.device_idx,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }


# --- Globals ---
sources: dict[str, AudioSource] = {}


def reconnect_loop():
    """Monitor all sources and auto-reconnect."""
    while True:
        time.sleep(5)
        disconnected = [src for src in sources.values() if not src.check_alive()]
        if not disconnected:
            continue
        for src in disconnected:
            idx = src.find_device(refresh=True)
            if idx is not None:
                log.info(f"[{src.name}] reconnecting...")
                src.start(idx)


def clip_all() -> dict:
    """Clip all connected sources. Returns results per source."""
    results = {}
    for name, src in sources.items():
        try:
            if src.connected:
                path = src.save_clip()
                results[name] = {"status": "ok", "file": str(path)}
            else:
                results[name] = {"status": "skipped", "message": "not connected"}
        except Exception as e:
            results[name] = {"status": "error", "message": str(e)}
    return results


# --- Flask ---
app = Flask(__name__)
app.logger.setLevel(logging.WARNING)


@app.route("/clip", methods=["GET", "POST"])
def clip_all_endpoint():
    """Clip all sources."""
    results = clip_all()
    return jsonify({"status": "ok", "sources": results})


@app.route("/clip/<source_name>", methods=["GET", "POST"])
def clip_one_endpoint(source_name):
    """Clip a single source by name."""
    src = sources.get(source_name)
    if not src:
        return jsonify({"status": "error", "message": f"unknown source: {source_name}"}), 404
    try:
        path = src.save_clip()
        return jsonify({"status": "ok", "file": str(path), "seconds": src.buffer.seconds_buffered})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/status")
def status_endpoint():
    """Status of all sources."""
    src_status = {name: src.status_dict() for name, src in sources.items()}
    mic_src = sources.get("mic")
    mic_connected = mic_src.connected if mic_src else False
    return jsonify({
        "status": "running",
        "mic_connected": mic_connected,
        "buffered_seconds": round(mic_src.buffer.seconds_buffered, 1) if mic_src and mic_connected else 0,
        "sources": src_status,
    })


def list_devices():
    devices = sd.query_devices()
    print("\nAvailable input devices:\n")
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            hostapi = sd.query_hostapis(dev["hostapi"])["name"]
            markers = []
            for name, cfg in SOURCES.items():
                if cfg["device_name"].lower() in dev["name"].lower():
                    markers.append(name)
            tag = f" <-- {','.join(markers)}" if markers else ""
            print(f"  [{i}] {dev['name']} (ch={dev['max_input_channels']}, sr={dev['default_samplerate']:.0f}, api={hostapi}){tag}")
    print()


def setup_hotkey():
    try:
        import keyboard

        def on_alt_f10():
            log.info("Alt+F10 -> clipping all sources")
            clip_all()

        keyboard.add_hotkey("alt+f10", on_alt_f10, suppress=False)
        log.info("hotkey: Alt+F10 -> clip all")
    except ImportError:
        log.warning("keyboard module not installed — hotkey disabled")
    except Exception as e:
        log.warning(f"hotkey setup failed: {e}")


def main():
    global sources

    parser = argparse.ArgumentParser(description="DJI Audio Bridge Service")
    parser.add_argument("--list", action="store_true", help="List audio devices and exit")
    parser.add_argument("--port", type=int, default=PORT, help="HTTP server port")
    args = parser.parse_args()

    if args.list:
        list_devices()
        sys.exit(0)

    # Create sources
    for name, cfg in SOURCES.items():
        sources[name] = AudioSource(name, cfg)
        log.info(f"[{name}] {cfg['device_name']} -> {cfg['clips_dir']}")

    # Start all sources
    for src in sources.values():
        src.start()
        if not src.connected:
            log.warning(f"[{src.name}] not found — waiting for connection...")

    # Reconnect thread
    threading.Thread(target=reconnect_loop, daemon=True).start()

    # Hotkey
    setup_hotkey()

    # HTTP server
    import werkzeug.serving
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    threading.Thread(
        target=lambda: app.run(host=HOST, port=args.port, threaded=True, use_reloader=False),
        daemon=True,
    ).start()

    log.info(f"HTTP: http://{HOST}:{args.port}/clip")
    log.info("service running — Ctrl+C to stop")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("shutting down...")
        for src in sources.values():
            src.stop()


if __name__ == "__main__":
    main()
