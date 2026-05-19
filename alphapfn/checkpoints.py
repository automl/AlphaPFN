"""Checkpoint cache and lazy fetch.

Public surface:
    `ensure_checkpoints(version)` -> Path to <cache>/<version>/, downloading
    and extracting the published bundle on first use.

Resolution order:
    1. `path=` argument to `AlphaPFN.from_pretrained` (handled in loader.py)
    2. `$ALPHAPFN_CACHE_DIR/<version>/`
    3. Platform-aware user cache: `platformdirs.user_cache_dir("alphapfn")`
       (Linux: ~/.cache/alphapfn, macOS: ~/Library/Caches/alphapfn,
       Windows: %LOCALAPPDATA%\\alphapfn)

If the cache is empty, the bundle is fetched from
`$ALPHAPFN_BASE_URL/alpha_pfn_<version>.zip` (default base URL is the
ML Freiburg artifact host) and extracted atomically.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from platformdirs import user_cache_dir


DEFAULT_BASE_URL = (
    "https://ml.informatik.uni-freiburg.de/research-artifacts/rakotoah/alpha_pfn"
)

# All four predictors live under <cache>/<version>/. We probe ppd as the
# canary; if it's there we trust the bundle was extracted fully.
_CANARY_PREDICTOR = "ppd"
_CANARY_FILE = "weights.safetensors"


def get_cache_dir() -> Path:
    """Resolve the alphapfn cache root.

    Honors $ALPHAPFN_CACHE_DIR; falls back to the platform user cache.
    """
    env = os.environ.get("ALPHAPFN_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    return Path(user_cache_dir("alphapfn"))


def _bundle_url(version: str) -> str:
    base = os.environ.get("ALPHAPFN_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    return f"{base}/alpha_pfn_{version}.zip"


def _is_populated(version_dir: Path) -> bool:
    return (version_dir / _CANARY_PREDICTOR / _CANARY_FILE).exists()


def _download(url: str, dest: Path) -> None:
    """Stream a URL to `dest`. Back-compat wrapper around _fetch._stream_download."""
    from alphapfn._fetch import _stream_download
    _stream_download(url, dest, label="alphapfn")


def _download_and_extract(version: str, version_dir: Path) -> None:
    """Download + atomically install the bundle into <cache>/<version>/."""
    version_dir.parent.mkdir(parents=True, exist_ok=True)
    url = _bundle_url(version)

    with tempfile.TemporaryDirectory(
        prefix=f"alphapfn-{version}-", dir=str(version_dir.parent)
    ) as tmpdir:
        tmp = Path(tmpdir)
        zip_path = tmp / "bundle.zip"
        _download(url, zip_path)

        extract_dir = tmp / "extracted"
        extract_dir.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        # The published bundle's top-level layout is `v1/<predictor>/...`.
        # Find the directory that contains predictors and move it into place.
        roots = [p for p in extract_dir.iterdir() if p.is_dir()]
        if len(roots) == 1 and (roots[0] / _CANARY_PREDICTOR).is_dir():
            src = roots[0]
        elif (extract_dir / _CANARY_PREDICTOR).is_dir():
            src = extract_dir
        else:
            raise RuntimeError(
                f"alphapfn: unexpected bundle layout in {zip_path}. "
                f"Expected a top-level dir containing {_CANARY_PREDICTOR}/. "
                f"Found: {[p.name for p in extract_dir.iterdir()]}"
            )

        # Atomic publish: rename a sibling temp dir into place, so a
        # concurrent loader either sees the final dir or doesn't.
        staged = version_dir.parent / f".{version_dir.name}.staging-{os.getpid()}"
        if staged.exists():
            shutil.rmtree(staged)
        shutil.move(str(src), str(staged))
        try:
            os.replace(str(staged), str(version_dir))
        except OSError:
            # Another process may have published first; clean up.
            shutil.rmtree(staged, ignore_errors=True)
            if not _is_populated(version_dir):
                raise


def ensure_checkpoints(version: str) -> Path:
    """Return <cache>/<version>/, downloading and extracting if missing."""
    version_dir = get_cache_dir() / version
    if _is_populated(version_dir):
        return version_dir

    try:
        _download_and_extract(version, version_dir)
    except Exception as e:
        raise RuntimeError(
            f"alphapfn: failed to download checkpoints for version={version!r} "
            f"from {_bundle_url(version)}: {e}\n"
            f"You can pre-populate the cache manually by extracting the bundle "
            f"into {version_dir} (so that {version_dir}/{_CANARY_PREDICTOR}/"
            f"{_CANARY_FILE} exists). The cache root can be overridden via "
            f"$ALPHAPFN_CACHE_DIR."
        ) from e

    if not _is_populated(version_dir):
        raise RuntimeError(
            f"alphapfn: download appeared to succeed but the cache is still "
            f"missing {_CANARY_PREDICTOR}/{_CANARY_FILE} under {version_dir}."
        )
    return version_dir
