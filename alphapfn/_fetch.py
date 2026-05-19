"""HTTP download helper shared by checkpoints.py and data.py.

Streams a URL to a file with a single-line progress indicator on stderr.
The progress line is suppressed when stderr is not a TTY (e.g. CI logs).
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


_NETWORK_TIMEOUT = 60  # seconds; prevents an indefinite hang on stalled connections


def _stream_download(url: str, dest: Path, label: str = "alphapfn") -> None:
    """Stream `url` to `dest` with a progress indicator on stderr.

    Args:
        url:   HTTP(S) URL to fetch.
        dest:  Local file path to write to. Parent dir must exist.
        label: Prefix used on the progress lines (e.g. "alphapfn",
               "alphapfn-data"). Lets the caller distinguish what's
               being fetched when logs interleave.
    """
    print(f"{label}: downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=_NETWORK_TIMEOUT) as response:
        total = int(response.headers.get("Content-Length", 0))
        chunk = 1 << 20  # 1 MiB
        written = 0
        with open(dest, "wb") as out:
            while True:
                buf = response.read(chunk)
                if not buf:
                    break
                out.write(buf)
                written += len(buf)
                if total > 0 and sys.stderr.isatty():
                    pct = 100.0 * written / total
                    print(
                        f"\r{label}:   {_human_size(written)}/{_human_size(total)} ({pct:5.1f}%)",
                        end="",
                        file=sys.stderr,
                    )
        if sys.stderr.isatty():
            print("", file=sys.stderr)  # newline after progress line
