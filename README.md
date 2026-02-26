# DJI Mic Bridge

Records keyboard audio from a DJI Mic 2 (Bluetooth, no receiver) as a separate audio track for Valorant clips. Runs as a background service with a 90-second rolling buffer — triggered via Stream Deck or hotkey.

## Setup

### 1. Pair DJI Mic 2 via Bluetooth

1. On the DJI Mic 2 **transmitter**: hold the power button until the LED flashes rapidly (pairing mode)
2. On Windows: **Settings > Bluetooth & devices > Add device > Bluetooth**
3. Select "DJI MIC-2" (or similar) from the list
4. Once paired, it appears as an audio input device

> If Windows connects it as a "Hands-Free" device with poor quality, go to **Sound Settings > Input** and make sure it's set as the input device. The Bluetooth HFP profile gives 16kHz mono — fine for keyboard sounds.

### 2. Install

```bat
install.bat
```

Or manually:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python service.py --list    # find your device index
```

### 3. Run

```bash
# Start the service (replace <index> with your device number)
python service.py --device <index>

# Or use the batch file
start.bat --device <index>
```

Once you know your device index, edit `config.py` and set `DEVICE_NAME` to a substring of the device name for auto-detection.

### 4. Stream Deck

Add a button with **System > Open** action:
- **App/File**: Full path to `clip.bat` (e.g., `P:\sohum\dji-mic-bridge\clip.bat`)

When pressed, it saves the last 90 seconds of keyboard audio to `clips/`.

### 5. Merge with ShadowPlay Clip

```bash
# Auto-match latest ShadowPlay clip + latest mic recording
python merge.py --auto

# Or specify files manually
python merge.py "C:\Users\sohum\Videos\Valorant\clip.mp4" "clips\mic_2026-02-24_15-30-00.wav"
```

This adds the keyboard audio as a second audio track (game audio = track 1, keyboard = track 2). Video editors like Premiere Pro or DaVinci Resolve can then mix/adjust levels per track.

### 6. Autostart (Optional)

```bash
python autostart.py --enable    # Start on Windows login
python autostart.py --disable   # Remove from startup
python autostart.py --status    # Check
```

## Files

| File | Purpose |
|------|---------|
| `service.py` | Main service — audio buffer + HTTP trigger server |
| `clip.py` | CLI trigger to save a clip (calls the service) |
| `merge.py` | FFmpeg merge: keyboard audio into video as 2nd track |
| `config.py` | Settings (device, buffer duration, sample rate, ports) |
| `autostart.py` | Register/unregister Windows startup |
| `clip.bat` | Stream Deck shortcut — triggers clip save |
| `start.bat` | Start the service (for shortcuts/startup) |
| `install.bat` | One-click setup |

## API

While the service is running:

- `GET http://127.0.0.1:9090/clip` — save clip, returns JSON with file path
- `GET http://127.0.0.1:9090/status` — check buffer status and device info
