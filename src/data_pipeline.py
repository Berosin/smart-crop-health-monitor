"""Reusable agricultural data preprocessing pipeline.

Pure pandas + numpy. No Streamlit, no ML training. This module exposes the
individual steps of a preprocessing pipeline so they can be composed into a
`build_pipeline` (or called individually, e.g. from model-training scripts):

- load_dataset                : read CSV / Excel into a DataFrame
- clean_data                  : strip whitespace, drop dupes, fix dtypes
- handle_missing_values       : drop or impute (mean/median/mode/constant)
- encode_categorical          : label-encode or one-hot encode columns
- split_train_test            : stratified train/test split with reproducibility
- scale_features              : standardize / min-max / robust scaling (fit on train)

Every mutating function is **non-destructive**: it returns a new DataFrame and
accepts an optional `inplace` flag, so callers can chain steps or keep the raw
source untouched.

Layout conventions
------------------
- raw/      : original, unmodified datasets (downloaded / exported)
- processed/: cleaned & transformed datasets ready for model training
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Loading datasets
# ---------------------------------------------------------------------------
def load_dataset(
    path: str | Path,
    sheet_name: str | int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load a dataset from CSV, Excel, or Parquet.

    Args:
        path:       file path.
        sheet_name: optional Excel sheet to read (name or index).
        **kwargs:   forwarded to the underlying pandas reader.

    Returns:
        The loaded DataFrame.

    Raises:
        ValueError: if the file extension is unsupported.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p, **kwargs)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(p, sheet_name=sheet_name, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(p, **kwargs)
    raise ValueError(
        f"Unsupported file type '{suffix}'. Use .csv, .xlsx/.xls, or .parquet."
    )


def save_dataset(
    df: pd.DataFrame,
    path: str | Path,
    index: bool = False,
    **kwargs: Any,
) -> Path:
    """Persist a DataFrame to CSV/Parquet. Creates parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        df.to_csv(p, index=index, **kwargs)
    elif suffix == ".parquet":
        df.to_parquet(p, index=index, **kwargs)
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use .csv or .parquet.")
    return p


# ---------------------------------------------------------------------------
# 2. Cleaning data
# ---------------------------------------------------------------------------
def clean_data(
    df: pd.DataFrame,
    strip_strings: bool = True,
    drop_duplicates: bool = True,
    drop_columns: list[str] | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Basic cleaning: whitespace, duplicates, and dropped columns.

    Args:
        strip_strings:  strip leading/trailing whitespace from object columns.
        drop_duplicates: drop fully-duplicated rows (keeps first).
        drop_columns:   column names to remove.
        inplace:        modify `df` instead of returning a copy.

    Returns:
        The cleaned DataFrame (same object if inplace=True).
    """
    out = df if inplace else df.copy()

    if strip_strings:
        for col in out.select_dtypes(include="object").columns:
            out[col] = out[col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})

    if drop_duplicates:
        out.drop_duplicates(inplace=True)

    if drop_columns:
        out.drop(columns=[c for c in drop_columns if c in out.columns],
                 inplace=True)

    return out


# ---------------------------------------------------------------------------
# 3. Handling missing values
# ---------------------------------------------------------------------------
MISSING_STRATEGIES = {"drop", "mean", "median", "mode", "constant"}


def handle_missing_values(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    strategy: str = "drop",
    fill_value: Any = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Impute or remove missing values.

    Args:
        columns:   columns to operate on. None = all columns.
        strategy:  "drop" | "mean" | "median" | "mode" | "constant".
                   Only "drop" removes rows; the rest impute in place.
                   mean/median use _only numeric columns; mode works on any;
                   constant uses `fill_value`.
        fill_value: value used by the "constant" strategy.

    Returns:
        The processed DataFrame.

    Raises:
        ValueError: for an unknown strategy or constant with no fill_value.
    """
    if strategy not in MISSING_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. "
                         f"Choose from {sorted(MISSING_STRATEGIES)}.")

    out = df if inplace else df.copy()
    targets = columns or list(out.columns)

    if strategy == "drop":
        out.dropna(subset=targets, inplace=True)
        return out

    if strategy == "constant":
        if fill_value is None:
            raise ValueError("strategy='constant' requires fill_value.")
        for col in targets:
            out[col] = out[col].fillna(fill_value)
        return out

    for col in targets:
        if out[col].isna().sum() == 0:
            continue
        if strategy in ("mean", "median"):
            if not pd.api.types.is_numeric_dtype(out[col]):
                warnings.warn(
                    f"Skipping non-numeric column '{col}' for strategy"
                    f" '{strategy}'. Use 'mode' or 'constant'.",
                    UserWarning,
                )
                continue
            value = out[col].mean() if strategy == "mean" else out[col].median()
        else:  # mode
            value = out[col].mode()
            value = value.iloc[0] if not value.empty else np.nan
        out[col] = out[col].fillna(value)

    return out


# ---------------------------------------------------------------------------
# 4. Encoding categorical values
# ---------------------------------------------------------------------------
def encode_categorical(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "label",
    inplace: bool = False,
) -> pd.DataFrame:
    """Encode categorical columns.

    Args:
        columns:      columns to encode. None = all object/category columns.
        method:       "label" (ordinal integers, 0..n-1) or
                      "onehot" (pandas get_dummies, new columns, originals kept).
        inplace:      for "label" only; one-hot always returns a copy.

    Returns:
        The encoded DataFrame.

    Raises:
        ValueError: for an unknown method.
    """
    if method not in ("label", "onehot"):
        raise ValueError(f"Unknown encoding method '{method}'. "
                         "Choose 'label' or 'onehot'.")

    out = df if inplace else df.copy()
    if columns is None:
        columns = list(out.select_dtypes(include=["object", "category"]).columns)

    if method == "label":
        for col in columns:
            if col in out.columns:
                out[col] = out[col].astype("category").cat.codes
        return out

    # one-hot
    out = pd.get_dummies(out, columns=columns, dtype=int, drop_first=False)
    return out


# ---------------------------------------------------------------------------
# 5. Train / test splitting
# ---------------------------------------------------------------------------
def split_train_test(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | None = None,
    test_size: float = 0.2,
    stratify: pd.Series | np.ndarray | None = None,
    random_state: int | None = 42,
    shuffle: bool = True,
) -> tuple:
    """Split features (and optional target) into train/test subsets.

    Returns
    -------
    If y is given:
        X_train, X_test, y_train, y_test
    Otherwise:
        X_train, X_test

    The split is reproducible (default random_state=42) and stratifiable via
    the `stratify` argument (e.g. the class labels).
    """
    from sklearn.model_selection import train_test_split

    if y is None:
        return train_test_split(
            X,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
    if stratify is None:
        stratify = y
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
    )


# ---------------------------------------------------------------------------
# 6. Feature scaling
# ---------------------------------------------------------------------------
SCALING_METHODS = {"standard", "minmax", "robust"}


def scale_features(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray | None = None,
    method: str = "standard",
    columns: list[str] | None = None,
) -> tuple:
    """Scale numeric features. Fits the scaler on X_train only.

    Args:
        X_train: train features.
        X_test:  optional test features to transform with the SAME scaler.
        method:  "standard" (z-score) | "minmax" (0-1) | "robust" (median/IQR).
        columns: optional numeric columns to scale (otherwise all numeric).

    Returns:
        (scaled_train, scaled_test_or_None, scaler). For a DataFrame input,
        scaled outputs are DataFrames preserving columns; for arrays, arrays.
    """
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    if method not in SCALING_METHODS:
        raise ValueError(f"Unknown scaling method '{method}'. "
                         f"Choose from {sorted(SCALING_METHODS)}.")

    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    scaler = scalers[method]()

    is_df = isinstance(X_train, pd.DataFrame)
    cols = columns
    if is_df and cols is None:
        cols = list(X_train.select_dtypes(include=np.number).columns)

    # Select numeric columns to scale (DataFrame path).
    if is_df:
        base_train = X_train.drop(columns=cols, errors="ignore")
        target_train = X_train[cols]
    else:
        base_train, target_train = None, X_train

    scaled_train = scaler.fit_transform(target_train)

    # __ REPACKAGING ------------------------------------------------
    def repack(scaled_array, df):
        if is_df:
            out = df.copy()
            out[cols] = scaled_array
            return out
        return scaled_array
    # ---------------------------------------------------------------

    result_train = repack(scaled_train, X_train)

    if X_test is None:
        return result_train, None, scaler

    if is_df:
        scaled_test = scaler.transform(X_test[cols])
        result_test = repack(scaled_test, X_test)
    else:
        result_test = scaler.transform(X_test)

    return result_train, result_test, scaler


# ---------------------------------------------------------------------------
# Composable pipeline helper
# ---------------------------------------------------------------------------
def build_pipeline(
    df: pd.DataFrame,
    *,
    clean: bool = True,
    missing: str = "drop",
    fill_value: Any = None,
    encode: str | None = "label",
    cat_columns: list[str] | None = None,
    drop_columns: list[str] | None = None,
    target: str | None = None,
    test_size: float = 0.2,
    scale: str | None = None,
    scale_columns: list[str] | None = None,
    random_state: int | None = 42,
) -> dict:
    """Run the full preprocessing chain and return a dict of artifacts.

    Returns keys: "X_train", "X_test", "y_train" (if target given),
    "y_test", "scaler" (if scaling), and "data" (the fully processed frame).
    This is a convenience wrapper over the granular functions above.
    """
    data = df
    if clean:
        data = clean_data(data, drop_columns=drop_columns)
    if missing:
        data = handle_missing_values(
            data, strategy=missing, fill_value=fill_value
        )

    cols_to_encode = cat_columns
    if cols_to_encode is None:
        cols_to_encode = list(data.select_dtypes(include=["object", "category"]).columns)
    # Never encode the target.
    if target and target in cols_to_encode:
        cols_to_encode.remove(target)
    if encode and cols_to_encode:
        data = encode_categorical(data, columns=cols_to_encode, method=encode)

    artifacts: dict[str, Any] = {}

    if target:
        y = data[target]
        X = data.drop(columns=[target])
        X_train, X_test, y_train, y_test = split_train_test(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        artifacts.update(X_train=X_train, X_test=X_test,
                         y_train=y_train, y_test=y_test)
    else:
        X_train, X_test = split_train_test(
            data, test_size=test_size, random_state=random_state
        )
        artifacts.update(X_train=X_train, X_test=X_test)

    if scale:
        result_train, result_test, scaler = scale_features(
            X_train, X_test, method=scale, columns=scale_columns
        )
        artifacts["X_train"] = result_train
        if X_test is not None:
            artifacts["X_test"] = result_test
        artifacts["scaler"] = scaler

    artifacts["data"] = data
    return artifacts