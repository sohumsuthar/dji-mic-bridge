"""
Register/unregister the DJI Mic Bridge service to run at Windows startup.

Usage:
    python autostart.py --enable     # Add to startup
    python autostart.py --disable    # Remove from startup
    python autostart.py --status     # Check if registered
"""

import argparse
import sys
import winreg
from pathlib import Path

APP_NAME = "DJIMicBridge"
START_BAT = Path(__file__).parent / "start.bat"
REG_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"


def set_autostart(enable: bool):
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_PATH, 0, winreg.KEY_SET_VALUE)
        if enable:
            # Run minimized via cmd /c start /min
            cmd = f'cmd /c start /min "" "{START_BAT}"'
            winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, cmd)
            print(f"[ok] Added {APP_NAME} to startup")
            print(f"     Command: {cmd}")
        else:
            try:
                winreg.DeleteValue(key, APP_NAME)
                print(f"[ok] Removed {APP_NAME} from startup")
            except FileNotFoundError:
                print(f"[info] {APP_NAME} was not in startup")
        winreg.CloseKey(key)
    except Exception as e:
        print(f"[error] Registry access failed: {e}", file=sys.stderr)
        sys.exit(1)


def check_status():
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_PATH, 0, winreg.KEY_READ)
        try:
            value, _ = winreg.QueryValueEx(key, APP_NAME)
            print(f"[enabled] {APP_NAME} is in startup")
            print(f"          Command: {value}")
        except FileNotFoundError:
            print(f"[disabled] {APP_NAME} is not in startup")
        winreg.CloseKey(key)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Manage DJI Mic Bridge autostart")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--enable", action="store_true", help="Enable autostart")
    group.add_argument("--disable", action="store_true", help="Disable autostart")
    group.add_argument("--status", action="store_true", help="Check autostart status")
    args = parser.parse_args()

    if args.enable:
        set_autostart(True)
    elif args.disable:
        set_autostart(False)
    else:
        check_status()


if __name__ == "__main__":
    main()
