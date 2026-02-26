@echo off
echo === DJI Mic Bridge Setup ===
echo.

cd /d "%~dp0"

echo [1/3] Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

echo [2/3] Installing dependencies...
pip install -r requirements.txt

echo [3/3] Listing audio devices...
python service.py --list

echo.
echo === Setup complete ===
echo.
echo Next steps:
echo   1. Pair your DJI Mic 2 via Bluetooth (see README)
echo   2. Find your device index from the list above
echo   3. Run:  python service.py --device ^<index^>
echo   4. In Stream Deck, add a System action that runs: clip.bat
pause
