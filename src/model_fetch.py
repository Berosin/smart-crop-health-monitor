"""Model weight bootstrap for deployment.

Trained model weights (disease_model_*/model.keras, embedding_stats.npz,
environment_model/model.joblib) are deliberately gitignored — see
.gitignore's comment on why (repo size; weights belong in release/LFS/
external storage, not the normal git history). That's fine for local
development (train them yourself via src/model_training.py /
src/environment_model.py, or unzip a Colab-trained bundle straight into
models/), but a fresh clone on a hosting platform like Streamlit Community
Cloud has an empty models/ directory and no way to run Colab itself.

This module bridges that gap: on first boot, if models/ looks empty, it
downloads and extracts a single zip archive (built by zipping the whole
models/ folder after training — see README's Deployment section) from a
URL supplied via secrets/env, and never touches the network again once
that's done — every subsequent app.py rerun's ensure_models_available()
call is a cheap filesystem check.

Deliberately NOT wired to any specific host: `MODELS_ZIP_URL` can point at
a GitHub Release asset, an S3/GCS public URL, a Google Drive direct-download
link, or anything else a plain HTTP GET can fetch — this module doesn't
care, it only needs bytes that unzip into a models/ directory tree.
"""

from __future__ import annotations

import os
import zipfile
from io import BytesIO
from pathlib import Path

import requests

from config import DISEASE_MODELS

MODELS_DIR = Path("models")
REQUEST_TIMEOUT_S = 60  # generous — model bundles can be tens to hundreds of MB
DOWNLOAD_CHUNK_BYTES = 1024 * 1024  # 1 MB


def resolve_models_zip_url() -> str | None:
    """Same env-var-first, then st.secrets resolution order src.weather's
    resolve_api_key() uses, for consistency across the app's two deployment
    secrets (OPENWEATHERMAP_API_KEY and this one)."""
    env_url = os.environ.get("MODELS_ZIP_URL")
    if env_url:
        return env_url

    try:
        import streamlit as st
        secrets_url = st.secrets.get("MODELS_ZIP_URL")
        if secrets_url:
            return secrets_url
    except Exception:
        # No secrets.toml configured at all — st.secrets raises in that
        # case rather than returning empty; expected locally, not an error.
        pass

    return None


def models_present() -> bool:
    """True if at least one crop's trained disease model is already on
    disk. Deliberately a cheap, approximate check (not verifying every
    file for every crop) — get_trained_crops() elsewhere in the app already
    handles a partially-populated models/ directory gracefully (a crop
    with a missing model just doesn't show up as trained), so this only
    needs to decide "is a download worth attempting at all", not validate
    a complete set.
    """
    return any(Path(paths["model_path"]).exists() for paths in DISEASE_MODELS.values())


def download_and_extract_models(url: str, progress_callback=None) -> None:
    """Stream-download a zip from `url` and extract it into models/.

    Args:
        url: direct-download URL to a .zip of the models/ directory tree
            (i.e. the zip's top level should contain disease_model_tomato/,
            environment_model/, etc. — the same layout `cd models && zip -r
            ../models.zip .` produces; see README's Deployment section).
        progress_callback: optional callable(bytes_downloaded, total_bytes)
            — total_bytes is None when the server doesn't send a
            Content-Length header. Used by ensure_models_available() to
            drive a Streamlit progress bar; safe to omit for non-UI use
            (e.g. a CLI warm-up script).

    Raises:
        requests.RequestException: network/HTTP failure.
        zipfile.BadZipFile: the downloaded bytes aren't a valid zip
            (wrong URL, an HTML error/interstitial page instead of the
            actual file, etc.).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()

    total = response.headers.get("Content-Length")
    total_bytes = int(total) if total is not None else None
    downloaded = 0
    buffer = BytesIO()
    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
        if not chunk:
            continue
        buffer.write(chunk)
        downloaded += len(chunk)
        if progress_callback:
            progress_callback(downloaded, total_bytes)

    buffer.seek(0)
    with zipfile.ZipFile(buffer) as zf:
        zf.extractall(MODELS_DIR)


def ensure_models_available() -> bool:
    """Idempotent startup hook: download+extract models/ once if it looks
    empty and a source URL is configured. Call this once near the top of
    app.py's main(), before render_sidebar()/get_trained_crops() run.

    Returns:
        True if at least one trained disease model is present on disk
        after this call (whether it was already there or was just
        downloaded) — callers can use this to decide whether to show a
        "no models configured" hint vs. proceeding normally. Never raises:
        a failed download is reported via st.error() (when Streamlit is
        available) and this simply returns models_present()'s (likely
        False) result, so one bad deploy config degrades to the same
        friendly "no trained model found" empty states every page already
        has, rather than crashing the whole app.
    """
    if models_present():
        return True

    url = resolve_models_zip_url()
    if not url:
        # No URL configured — most likely local dev where models/ is
        # meant to be populated by training or manually unzipping, not
        # downloaded. Silently proceed; every page's own "no trained
        # model found" empty state already covers this.
        return False

    try:
        import streamlit as st
        with st.spinner("Downloading trained models (first run only)…"):
            progress_bar = st.progress(0.0)
            status = st.empty()

            def _progress(done: int, total: int | None) -> None:
                if total:
                    progress_bar.progress(min(done / total, 1.0))
                    status.caption(f"{done / 1_048_576:.0f} / {total / 1_048_576:.0f} MB")
                else:
                    status.caption(f"{done / 1_048_576:.0f} MB downloaded…")

            download_and_extract_models(url, progress_callback=_progress)
            progress_bar.empty()
            status.empty()
    except ImportError:
        # Streamlit not installed in this context (e.g. a CLI warm-up
        # script) — download without any UI feedback.
        download_and_extract_models(url)
    except Exception as e:
        try:
            import streamlit as st
            st.error(
                f"Couldn't download trained models from the configured "
                f"MODELS_ZIP_URL. The app will run with disease detection "
                f"unavailable until this is fixed. ({type(e).__name__}: {e})"
            )
        except ImportError:
            raise

    return models_present()


if __name__ == "__main__":
    # CLI warm-up: `python -m src.model_fetch` — useful for a Dockerfile
    # RUN step or a pre-deploy sanity check, without needing Streamlit at
    # all (see the ImportError fallback above).
    import sys

    url = resolve_models_zip_url()
    if not url:
        print("MODELS_ZIP_URL not set (env var or .streamlit/secrets.toml) — nothing to do.")
        sys.exit(0)

    if models_present():
        print("Models already present — skipping download.")
        sys.exit(0)

    print(f"Downloading models from {url} ...")
    download_and_extract_models(url)
    print("Done." if models_present() else "Download completed but no trained models were found afterward — check the zip's structure.")