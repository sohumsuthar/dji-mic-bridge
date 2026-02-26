"""
Clip trigger script — call this from Stream Deck or a hotkey.
Sends a request to the running service to save the last 90 seconds.

Usage:
    python clip.py              # Save clip
    python clip.py --status     # Check service status
"""

import argparse
import sys
import urllib.request
import json

from config import HOST, PORT


def trigger_clip():
    url = f"http://{HOST}:{PORT}/clip"
    try:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            if data["status"] == "ok":
                print(f"Clipped {data['seconds']:.0f}s -> {data['file']}")
                return 0
            else:
                print(f"Error: {data.get('message', 'unknown')}", file=sys.stderr)
                return 1
    except Exception as e:
        print(f"Failed to reach service at {url}: {e}", file=sys.stderr)
        print("Is the service running? Start it with: python service.py", file=sys.stderr)
        return 1


def check_status():
    url = f"http://{HOST}:{PORT}/status"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            print(f"Service: {data['status']}")
            print(f"Buffered: {data['buffered_seconds']}s")
            print(f"Device: {data['device']}")
            print(f"Sample rate: {data['sample_rate']} Hz")
    except Exception as e:
        print(f"Service not reachable: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Trigger a mic clip save")
    parser.add_argument("--status", action="store_true", help="Check service status")
    args = parser.parse_args()

    if args.status:
        check_status()
    else:
        sys.exit(trigger_clip())


if __name__ == "__main__":
    main()
