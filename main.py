#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean OpenADMET PXR pEC50 baseline using TabPFN API.

This script keeps the project simple:
  - read official train/test CSV files
  - canonicalize SMILES when needed
  - build molecular tabular features from Morgan/ECFP fingerprints + RDKit 2D descriptors
  - train one TabPFNRegressor through tabpfn-client
  - evaluate on a random validation split
  - refit on all train data and write a final submission

Important:
  Do not hard-code your TabPFN token in this file. Set it as an environment variable:

  PowerShell:
    $env:TABPFN_CLIENT_TOKEN="your_token_here"

  CMD:
    set TABPFN_CLIENT_TOKEN=your_token_here

Run:
  python main_tabpfn.py

Outputs:
  outputs/validation_metrics_tabpfn.csv
  outputs/validation_predictions_tabpfn.csv
  outputs/test_predictions_tabpfn.csv
  submissions/submission_tabpfn.csv
"""

from __future__ import annotations

import argparse
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
SUB_DIR = ROOT / "submissions"

TRAIN_CSV = DATA_DIR / "openadmet_train_clean.csv"
TEST_CSV = DATA_DIR / "openadmet_test_clean.csv"

ID_COL = "Molecule Name"
SMILES_COL = "canonical_smiles"
RAW_SMILES_COL = "SMILES"
TARGET_COL = "pEC50"

DESC_LIST = Descriptors._descList


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def canonicalize_smiles(smiles: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def pick_smiles_column(df: pd.DataFrame) -> str:
    if SMILES_COL in df.columns:
        return SMILES_COL
    if RAW_SMILES_COL in df.columns:
        return RAW_SMILES_COL
    raise ValueError(f"Cannot find SMILES column. Expected {SMILES_COL!r} or {RAW_SMILES_COL!r}.")


def clean_input(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = df.copy()
    smiles_col = pick_smiles_column(df)

    if smiles_col != SMILES_COL:
        df[SMILES_COL] = df[smiles_col].map(canonicalize_smiles)

    if RAW_SMILES_COL not in df.columns:
        df[RAW_SMILES_COL] = df[SMILES_COL]

    if "is_valid_mol" in df.columns:
        df = df[df["is_valid_mol"] == True].copy()

    required = [ID_COL, RAW_SMILES_COL, SMILES_COL]
    if is_train:
        required.append(TARGET_COL)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    subset = [SMILES_COL] + ([TARGET_COL] if is_train else [])
    df = df.dropna(subset=subset).reset_index(drop=True)
    return df


def calc_morgan_bits(smiles: str, radius: int, n_bits: int) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def calc_rdkit2d(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    values = []
    for _, func in DESC_LIST:
        try:
            values.append(float(func(mol)))
        except Exception:
            values.append(np.nan)
    return np.asarray(values, dtype=np.float32)


def build_features(df: pd.DataFrame, radius: int, n_bits: int) -> tuple[pd.DataFrame, np.ndarray]:
    features = []
    keep_idx = []

    for i, smi in enumerate(df[SMILES_COL].values):
        fp = calc_morgan_bits(smi, radius=radius, n_bits=n_bits)
        desc = calc_rdkit2d(smi)
        if fp is None or desc is None:
            continue
        features.append(np.concatenate([fp, desc]).astype(np.float32))
        keep_idx.append(i)

    if not keep_idx:
        raise RuntimeError("No valid molecules after feature generation.")

    df2 = df.iloc[keep_idx].reset_index(drop=True)
    X = np.vstack(features).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    return df2, X


def preprocess_fit_transform(X_train: np.ndarray, X_other: np.ndarray) -> tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler, np.ndarray]:
    """
    TabPFN is a tabular model. We keep preprocessing conservative:
      1. drop constant columns based only on X_train
      2. median-impute based only on X_train
      3. standardize based only on X_train
    """
    X0 = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    keep = X0.std(axis=0) > 1e-12

    X_train2 = X_train[:, keep]
    X_other2 = X_other[:, keep]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train2 = imputer.fit_transform(X_train2)
    X_other2 = imputer.transform(X_other2)

    X_train2 = scaler.fit_transform(X_train2)
    X_other2 = scaler.transform(X_other2)

    return X_train2.astype(np.float32), X_other2.astype(np.float32), imputer, scaler, keep


def preprocess_transform(X: np.ndarray, imputer: SimpleImputer, scaler: StandardScaler, keep: np.ndarray) -> np.ndarray:
    X2 = X[:, keep]
    X2 = imputer.transform(X2)
    X2 = scaler.transform(X2)
    return X2.astype(np.float32)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    baseline = np.full_like(y_true, np.mean(y_true), dtype=float)
    denom = np.sum(np.abs(y_true - baseline))

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "RAE": float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else np.nan,
        "Spearman": float(spearmanr(y_true, y_pred).correlation),
        "Kendall": float(kendalltau(y_true, y_pred).correlation),
    }


def get_tabpfn_regressor(args):
    try:
        import tabpfn_client
        from tabpfn_client import TabPFNRegressor
    except Exception as e:
        raise ImportError(
            "Cannot import tabpfn-client. Install it with: pip install -U tabpfn-client"
        ) from e

    token = args.token or os.getenv("TABPFN_CLIENT_TOKEN") or os.getenv("PRIORLABS_API_KEY")
    if token:
        tabpfn_client.set_access_token(token)
    else:
        print("[auth] No TABPFN_CLIENT_TOKEN/PRIORLABS_API_KEY found. tabpfn-client may open/login interactively.")

    kwargs = {}
    if args.model_path:
        kwargs["model_path"] = args.model_path
    if args.thinking:
        kwargs["thinking_mode"] = True
        kwargs["thinking_timeout_s"] = args.thinking_timeout_s
        kwargs["thinking_metric"] = args.thinking_metric

    return TabPFNRegressor(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default=str(TRAIN_CSV))
    parser.add_argument("--test_csv", default=str(TEST_CSV))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--n_bits", type=int, default=1024, help="Morgan fingerprint bits. Keep <= 1024 for a compact TabPFN input.")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--clip_min", type=float, default=2.0)
    parser.add_argument("--clip_max", type=float, default=8.0)
    parser.add_argument("--token", default=None, help="Optional TabPFN token. Safer: use TABPFN_CLIENT_TOKEN env var instead.")
    parser.add_argument("--model_path", default=None, help="Optional API model path, e.g. v2.5_real. Default lets API choose.")
    parser.add_argument("--thinking", action="store_true", help="Enable TabPFN API thinking mode if supported by the selected model.")
    parser.add_argument("--thinking_timeout_s", type=float, default=600)
    parser.add_argument("--thinking_metric", default="mean_absolute_error")
    args = parser.parse_args()

    seed_everything(args.seed)
    OUT_DIR.mkdir(exist_ok=True)
    SUB_DIR.mkdir(exist_ok=True)

    train_raw = pd.read_csv(args.train_csv)
    test_raw = pd.read_csv(args.test_csv)

    train_df = clean_input(train_raw, is_train=True)
    test_df = clean_input(test_raw, is_train=False)

    print(f"[data] train={len(train_df)} test={len(test_df)}")

    all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    all_df, X_all_raw = build_features(all_df, radius=args.radius, n_bits=args.n_bits)

    n_train = len(train_df)
    train_df = all_df.iloc[:n_train].reset_index(drop=True)
    test_df = all_df.iloc[n_train:].reset_index(drop=True)

    X_train_raw = X_all_raw[:n_train]
    X_test_raw = X_all_raw[n_train:]
    y_all = train_df[TARGET_COL].values.astype(float)

    tr_idx, va_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True,
    )

    X_tr, X_va, _, _, _ = preprocess_fit_transform(X_train_raw[tr_idx], X_train_raw[va_idx])
    y_tr, y_va = y_all[tr_idx], y_all[va_idx]

    print(f"[features] validation X_tr={X_tr.shape} X_va={X_va.shape}")
    print("[fit] TabPFN validation model")

    val_model = get_tabpfn_regressor(args)
    val_model.fit(X_tr, y_tr)
    pred_va = np.asarray(val_model.predict(X_va), dtype=float)
    pred_va = np.clip(pred_va, args.clip_min, args.clip_max)

    val_metrics = metrics(y_va, pred_va)
    print(
        "[validation] "
        f"MAE={val_metrics['MAE']:.4f} "
        f"R2={val_metrics['R2']:.4f} "
        f"Spearman={val_metrics['Spearman']:.4f}"
    )

    metric_df = pd.DataFrame([{ "model": "tabpfn", **val_metrics }])
    metric_df.to_csv(OUT_DIR / "validation_metrics_tabpfn.csv", index=False)

    val_out = train_df.iloc[va_idx][[ID_COL, RAW_SMILES_COL, TARGET_COL]].copy()
    val_out = val_out.rename(columns={TARGET_COL: "y_true"})
    val_out["pred_tabpfn"] = pred_va
    val_out.to_csv(OUT_DIR / "validation_predictions_tabpfn.csv", index=False)

    print("[fit] TabPFN full-train model")
    X_train_full, X_test_full, _, _, _ = preprocess_fit_transform(X_train_raw, X_test_raw)
    print(f"[features] full X_train={X_train_full.shape} X_test={X_test_full.shape}")

    full_model = get_tabpfn_regressor(args)
    full_model.fit(X_train_full, y_all)
    pred_test = np.asarray(full_model.predict(X_test_full), dtype=float)
    pred_test = np.clip(pred_test, args.clip_min, args.clip_max)

    sub = test_df[[RAW_SMILES_COL, ID_COL]].copy()
    sub[TARGET_COL] = pred_test
    sub = sub[[RAW_SMILES_COL, ID_COL, TARGET_COL]]
    sub.to_csv(SUB_DIR / "submission_tabpfn.csv", index=False)

    detail = test_df[[ID_COL, RAW_SMILES_COL]].copy()
    detail["pred_tabpfn"] = pred_test
    detail.to_csv(OUT_DIR / "test_predictions_tabpfn.csv", index=False)

    print(f"[saved] {OUT_DIR / 'validation_metrics_tabpfn.csv'}")
    print(f"[saved] {OUT_DIR / 'validation_predictions_tabpfn.csv'}")
    print(f"[saved] {OUT_DIR / 'test_predictions_tabpfn.csv'}")
    print(f"[saved] {SUB_DIR / 'submission_tabpfn.csv'}")


if __name__ == "__main__":
    main()
