"""Training-data corpus cache and lazy fetch.

Public surface:
    `ensure_prior_corpus(corpus, version)` → Path to <cache>/<version>/<corpus>/,
    downloading and extracting the published tar.xz bundle on first use.

Resolution order:
    1. <repo>/data/<version>/<corpus>/                                   (dev convenience)
    2. $ALPHAPFN_DATA_CACHE_DIR/<version>/<corpus>/
    3. $ALPHAPFN_CACHE_DIR/data/<version>/<corpus>/
    Otherwise → raises. We deliberately do NOT fall back to
    `platformdirs.user_cache_dir`: each corpus expands to ~250 GB, which
    would fill `$HOME` on Freiburg clusters (94 GB on NEMO).

If the cache location is set but empty, the bundle is fetched from
`$ALPHAPFN_BASE_URL/alpha_pfn_data_<version>_<corpus>.tar.xz` (default
base URL is the ML Freiburg artifact host) and extracted atomically.
SHA-256 is verified against the `.sha256` sidecar when reachable.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from alphapfn._fetch import _stream_download


# Timeout (seconds) for sidecar / archive URL fetches. Without this,
# urllib hangs forever on a stalled connection.
_NETWORK_TIMEOUT = 60


DEFAULT_BASE_URL = (
    "https://ml.informatik.uni-freiburg.de/research-artifacts/rakotoah/alpha_pfn"
)

ALLOWED_CORPORA = {"ppd", "pes", "mes", "jes"}
ALLOWED_DATA_VERSIONS = {"v1"}

# Canary lives at the top of each extracted corpus dir.
_CANARY_FILE = "manifest.json"

# Repo dev override: alpha-pfn/data/<version>/<corpus>/
_REPO_ROOT = Path(__file__).resolve().parent.parent
_REPO_DATA_ROOT = _REPO_ROOT / "data"


def get_data_cache_dir() -> Path:
    """Resolve the data cache root.

    Order: $ALPHAPFN_DATA_CACHE_DIR, then $ALPHAPFN_CACHE_DIR/data.
    Raises if neither is set — see module docstring for rationale.
    """
    env = os.environ.get("ALPHAPFN_DATA_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    cache = os.environ.get("ALPHAPFN_CACHE_DIR")
    if cache:
        return Path(cache).expanduser() / "data"
    raise RuntimeError(
        "alphapfn-data: no cache location configured. "
        "Set $ALPHAPFN_DATA_CACHE_DIR to a workspace path "
        "(e.g. /work/.../alphapfn_data on a Freiburg cluster) — "
        "each corpus is ~250 GB, do not put it under $HOME."
    )


def _bundle_url(corpus: str, version: str) -> str:
    base = os.environ.get("ALPHAPFN_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    return f"{base}/alpha_pfn_data_{version}_{corpus}.tar.xz"


def _sha256_url(corpus: str, version: str) -> str:
    return _bundle_url(corpus, version)[:-len(".tar.xz")] + ".sha256"


def _is_populated(corpus_dir: Path) -> bool:
    return (corpus_dir / _CANARY_FILE).exists()


def _repo_path(corpus: str, version: str) -> Path:
    return _REPO_DATA_ROOT / version / corpus


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_expected_sha(corpus: str, version: str) -> str | None:
    """Return the hex digest from the .sha256 sidecar, or None if unreachable."""
    url = _sha256_url(corpus, version)
    try:
        with urllib.request.urlopen(url, timeout=_NETWORK_TIMEOUT) as response:
            content = response.read().decode("ascii", errors="replace").strip()
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(
            f"alphapfn-data: sidecar {url} unreachable ({e}); "
            "skipping SHA-256 check (HTTPS still provides transport integrity).",
            file=sys.stderr,
        )
        return None
    # Sidecar may be "<hex>" alone or "<hex>  filename"; take first token.
    return content.split()[0] if content else None


def _verify_sha(archive: Path, expected: str) -> None:
    got = _hash_file(archive)
    if got != expected:
        raise RuntimeError(
            f"alphapfn-data: SHA-256 mismatch for {archive.name}: "
            f"got {got}, expected {expected}"
        )


def _extract_tarxz(archive: Path, dest_root: Path) -> Path:
    """Extract `archive` (tar.xz) into a tempdir under `dest_root`.

    Returns the path of the top-level dir inside the archive (assumed
    to be a single dir whose name matches the public slug, e.g.
    `alpha_pfn_data_v1_ppd/`).

    Uses `filter='data'` to defend against path-traversal entries
    (CVE-2007-4559). Available on Python 3.10+ via the `tarfile.data_filter`
    callable; passing the string `'data'` selects it by name.
    """
    extract_dir = dest_root / "extracted"
    extract_dir.mkdir()
    with tarfile.open(str(archive), mode="r:xz") as tf:
        try:
            tf.extractall(str(extract_dir), filter="data")
        except TypeError:
            # Python < 3.10's tarfile didn't accept `filter=`. We declare
            # >=3.10 in pyproject, so this branch is defensive only.
            tf.extractall(str(extract_dir))
    roots = [p for p in extract_dir.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(
            f"alphapfn-data: unexpected archive layout in {archive.name}. "
            f"Expected a single top-level dir; found {[p.name for p in extract_dir.iterdir()]}"
        )
    src = roots[0]
    if not (src / _CANARY_FILE).exists():
        raise RuntimeError(
            f"alphapfn-data: extracted dir {src.name!r} missing {_CANARY_FILE}"
        )
    return src


def _read_manifest(corpus_dir: Path) -> dict:
    """Load and parse `<corpus_dir>/manifest.json`. Raises on missing/bad."""
    path = corpus_dir / _CANARY_FILE
    with open(path) as f:
        return json.load(f)


def _verify_manifest(manifest: dict, corpus: str, version: str) -> None:
    """Defensive check that the cached corpus matches what we asked for.

    Catches the unusual case where a user manually populated the cache
    with the wrong archive (e.g. extracted the PPD archive into the
    JES slot, or pre-populated from an older v2 we never published).
    """
    actual_version = manifest.get("alpha_pfn_version")
    actual_corpus = manifest.get("corpus")
    if actual_version != version:
        raise RuntimeError(
            f"alphapfn-data: manifest version mismatch for {corpus!r}: "
            f"requested {version!r}, manifest says {actual_version!r}"
        )
    if actual_corpus != corpus:
        raise RuntimeError(
            f"alphapfn-data: manifest corpus mismatch: requested {corpus!r}, "
            f"manifest says {actual_corpus!r}"
        )


def _download_and_extract(corpus: str, version: str, corpus_dir: Path) -> None:
    """Download + atomically install the corpus into <cache>/<version>/<corpus>/."""
    corpus_dir.parent.mkdir(parents=True, exist_ok=True)

    expected_sha = _fetch_expected_sha(corpus, version)
    url = _bundle_url(corpus, version)

    with tempfile.TemporaryDirectory(
        prefix=f"alphapfn-data-{version}-{corpus}-", dir=str(corpus_dir.parent)
    ) as tmpdir:
        tmp = Path(tmpdir)
        archive = tmp / "bundle.tar.xz"
        _stream_download(url, archive, label="alphapfn-data")

        if expected_sha is not None:
            _verify_sha(archive, expected_sha)

        src = _extract_tarxz(archive, tmp)

        # Atomic publish: stage next to the target, then os.replace.
        #
        # Note: `os.replace` of a directory onto an already-populated
        # directory raises OSError(EEXIST) on Linux. Under concurrent
        # downloads (two processes racing past the _is_populated check),
        # the first to finish publishes; the second hits EEXIST, falls
        # into the except branch, removes its own staging dir, and
        # accepts the already-populated target.
        staged = corpus_dir.parent / f".{corpus_dir.name}.staging-{os.getpid()}"
        if staged.exists():
            shutil.rmtree(staged)
        shutil.move(str(src), str(staged))
        try:
            os.replace(str(staged), str(corpus_dir))
        except OSError:
            shutil.rmtree(staged, ignore_errors=True)
            if not _is_populated(corpus_dir):
                raise


def ensure_prior_corpus(corpus: str, version: str = "v1") -> Path:
    """Return the local path to a prior corpus, downloading if missing.

    Resolution order (returns the first populated location):
      1. <repo>/data/<version>/<corpus>/        (dev convenience)
      2. $ALPHAPFN_DATA_CACHE_DIR/<version>/<corpus>/
      3. $ALPHAPFN_CACHE_DIR/data/<version>/<corpus>/

    If neither env var is set, raises (we never default to ~/.cache for
    ~250 GB payloads — see module docstring).
    """
    if corpus not in ALLOWED_CORPORA:
        raise ValueError(
            f"corpus={corpus!r} is not supported. "
            f"Allowed: {sorted(ALLOWED_CORPORA)}"
        )
    if version not in ALLOWED_DATA_VERSIONS:
        raise ValueError(
            f"version={version!r} is not supported. "
            f"Allowed: {sorted(ALLOWED_DATA_VERSIONS)}"
        )

    # 1) Repo dev override.
    repo = _repo_path(corpus, version)
    if _is_populated(repo):
        _verify_manifest(_read_manifest(repo), corpus, version)
        return repo

    # 2/3) User-cache (raises if neither env is set).
    cache_dir = get_data_cache_dir()  # may raise
    corpus_dir = cache_dir / version / corpus
    if _is_populated(corpus_dir):
        _verify_manifest(_read_manifest(corpus_dir), corpus, version)
        return corpus_dir

    try:
        _download_and_extract(corpus, version, corpus_dir)
    except Exception as e:
        raise RuntimeError(
            f"alphapfn-data: failed to download corpus {corpus!r} "
            f"version={version!r} from {_bundle_url(corpus, version)}: {e}\n"
            f"To pre-populate the cache manually, extract the bundle "
            f"into {corpus_dir} (so that {corpus_dir}/{_CANARY_FILE} exists)."
        ) from e

    if not _is_populated(corpus_dir):
        raise RuntimeError(
            f"alphapfn-data: download appeared to succeed but the cache "
            f"is still missing {_CANARY_FILE} under {corpus_dir}."
        )
    _verify_manifest(_read_manifest(corpus_dir), corpus, version)
    return corpus_dir
