"""Build models.zip for the GitHub Release upload — run this instead of
Compress-Archive / zip. Only includes the files pages actually load at
runtime (not training artifacts like logs/, checkpoints, or plots), and
always writes forward-slash paths inside the zip regardless of host OS,
which is what src/model_fetch.py's zipfile.extractall() expects on
Streamlit Cloud's Linux containers.

Usage (from the repo root, same folder this script sits in):
    python make_models_zip.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

MODELS_DIR = Path("models")
OUTPUT_ZIP = Path("models.zip")

# Only these — everything else under models/ (logs/, best_model.keras,
# confusion_matrix.png, training_curves.png, metrics.json, ...) is a
# training-time artifact the app never reads at runtime.
INCLUDE_FILENAMES = {"model.keras", "labels.json", "embedding_stats.npz", "model.joblib", "metadata.json"}


def build() -> None:
    if not MODELS_DIR.exists():
        raise SystemExit(f"'{MODELS_DIR}' not found — run this from the repo root.")

    included = []
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(MODELS_DIR.rglob("*")):
            if not path.is_file() or path.name not in INCLUDE_FILENAMES:
                continue
            # Path *relative to MODELS_DIR*, forward slashes always —
            # this is the piece Compress-Archive gets wrong on Windows.
            arcname = path.relative_to(MODELS_DIR).as_posix()
            zf.write(path, arcname)
            included.append((arcname, path.stat().st_size))

    if not included:
        OUTPUT_ZIP.unlink(missing_ok=True)
        raise SystemExit(
            "No matching files found under models/ — check that you've "
            "trained the models and they're named model.keras / "
            "labels.json / embedding_stats.npz / model.joblib as expected."
        )

    total_mb = sum(size for _, size in included) / (1024 * 1024)
    print(f"Wrote {OUTPUT_ZIP} — {len(included)} files, {total_mb:.1f} MB total:\n")
    for arcname, size in included:
        print(f"  {arcname}  ({size / (1024 * 1024):.1f} MB)")


if __name__ == "__main__":
    build()