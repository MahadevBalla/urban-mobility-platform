"""
13_model_stgnn.py

Spatio-Temporal Graph Neural Network (ST-GNN) for travel demand forecasting.

Architecture:
    - Two GCN layers for spatial message passing over the H3 adjacency graph
    - Multi-head self-attention over the time-step axis
    - Huber loss for robustness to outlier trip counts
    - Early stopping on validation loss
    - Normalised graph Laplacian using H3 k-ring=1 adjacency

Graph construction:
    Edges between H3 hexes sharing a k-ring=1 boundary.
    Edge weights: Gaussian kernel of Haversine distance between centroids.
    Symmetric degree normalisation as per Kipf & Welling (2017).

Temporal split:
    Identical boundaries to 11_model_nb.py and 12_model_xgboost.py.
    MODEL_TRAIN_END / MODEL_VALID_END / MODEL_TEST_START from config.

Sequence generation:
    Pivot the ENTIRE dataset once, make sequences once, then split sequences
    by target timestamp. This preserves full historical context for every
    validation and test sequence — no artificial warmup restart per fold.

Input : features_master.parquet
Output: outputs/models/stgnn_model.pt
        outputs/models/stgnn_metadata.json
        outputs/tables/stgnn_predictions.parquet
        outputs/tables/stgnn_metrics.csv
        outputs/figures/stgnn_results.png
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    EARTH_R_M,
    FEATURES_MASTER,
    FIGURES,
    H3_RESOLUTION,
    MODEL_TEST_START,
    MODEL_TRAIN_END,
    MODEL_VALID_END,
    MODELS_DIR,
    STGNN_PARAMS,
    TABLES_DIR,
)

RANDOM_SEED = 42
TARGET_COL = "trip_count"

# Feature sets
TEMPORAL_FEATURES: list[str] = [
    "trip_count",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "is_peak",
    "is_weekend",
    "is_monsoon",
    "is_pre_monsoon",
    "is_winter",
    "dist_cbd_km",
    "rolling_24h_mean",
    "rolling_24h_std",
]

WEATHER_FEATURES: list[str] = [
    "log_precip",
    "precip_3h",
    "weather_severity",
    "heat_stress",
    "relative_humidity_2m",
    "wind_speed_10m",
]

# Shared helpers
def temporal_split(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    valid_end: pd.Timestamp,
    test_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, validation, test) using strict temporal boundaries."""
    assert train_end < valid_end < test_start, (
        f"Split dates must satisfy train_end < valid_end < test_start. "
        f"Got: {train_end.date()} / {valid_end.date()} / {test_start.date()}"
    )
    t = df["time_bin_30min"]
    return (
        df[t <= train_end].copy(),
        df[(t > train_end) & (t <= valid_end)].copy(),
        df[t >= test_start].copy(),
    )


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
) -> dict[str, float]:
    """MAE, RMSE, sMAPE, R2, Pearson r -- identical to 11 and 12."""
    mae = float(np.abs(y_true - y_pred).mean())
    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))

    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = float(
        np.where(denom > 0, np.abs(y_true - y_pred) / denom, 0.0).mean() * 100
    )

    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    try:
        r, _ = stats.pearsonr(y_true, y_pred)
        r = float(r)
    except Exception:
        r = np.nan

    print(f"\n  [{label}]")
    print(f"    MAE     : {mae:.4f}")
    print(f"    RMSE    : {rmse:.4f}")
    print(f"    sMAPE   : {smape:.2f} %")
    print(f"    R2      : {r2:.4f}")
    print(f"    Pearson : {r:.4f}")

    return {
        "split": label,
        "model": "ST-GNN",
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE": smape,
        "R2": r2,
        "Pearson_r": r,
    }

# Geometry helper
def _haversine_km_vec(
    lat1: float,
    lng1: float,
    lats2: np.ndarray,
    lngs2: np.ndarray,
) -> np.ndarray:
    r = EARTH_R_M / 1_000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lats2)
    dphi = np.radians(lats2 - lat1)
    dlam = np.radians(lngs2 - lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))

# Graph construction
def build_h3_graph(hex_list: list[str]) -> torch.Tensor:
    """
    Build L_tilde = D_tilde^(-1/2) A_tilde D_tilde^(-1/2),  A_tilde = A + I.
    Edge weights: Gaussian kernel of centroid Haversine distance.
    Returns dense (N, N) float32 tensor.
    """
    N = len(hex_list)
    hex_to_idx = {hx: i for i, hx in enumerate(hex_list)}
    coords = {hx: h3.cell_to_latlng(hx) for hx in hex_list}

    rows, cols, weights = [], [], []
    for hx in hex_list:
        i = hex_to_idx[hx]
        lat1, lng1 = coords[hx]
        active_nb = [nb for nb in h3.grid_disk(hx, 1) - {hx} if nb in hex_to_idx]
        if not active_nb:
            continue
        nb_lats = np.array([coords[nb][0] for nb in active_nb], dtype=np.float64)
        nb_lngs = np.array([coords[nb][1] for nb in active_nb], dtype=np.float64)
        dists = _haversine_km_vec(lat1, lng1, nb_lats, nb_lngs)
        for nb, d in zip(active_nb, dists):
            rows.append(i)
            cols.append(hex_to_idx[nb])
            weights.append(float(np.exp(-(d**2) / 2.0)))

    A = torch.zeros(N, N, dtype=torch.float32)
    if rows:
        A[rows, cols] = torch.tensor(weights, dtype=torch.float32)

    A_tilde = A + torch.eye(N, dtype=torch.float32)

    degree = A_tilde.sum(1)
    degree = torch.clamp(degree, min=1e-12)

    d_inv_sqrt = degree.pow(-0.5)
    D = torch.diag(d_inv_sqrt)

    return D @ A_tilde @ D

# Data helpers
def build_pivot(df: pd.DataFrame, col: str, hex_list: list[str]) -> np.ndarray:
    """
    Pivot (time_bin_30min x origin_h3) for one column.
    Missing cells are forward-filled then zero-filled. Returns shape (T, N).
    """
    p = df.pivot_table(
        values=col, index="time_bin_30min", columns="origin_h3", aggfunc="mean"
    )
    return p.reindex(columns=hex_list).ffill().fillna(0.0).values.astype(np.float32)


def make_sequences_with_timestamps(
    X: np.ndarray,
    y: np.ndarray,
    time_index: pd.DatetimeIndex,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Slide a window of length seq_len over the ENTIRE time axis.
    Each sequence is tagged with the TARGET timestamp (the step being predicted).

    X : (T, N, F)  ->  Xs: (T-seq_len, seq_len, N, F)
    y : (T, N)     ->  ys: (T-seq_len, N)
    returns (Xs, ys, target_timestamps)
    """
    Xs, ys, ttimes = [], [], []
    for t in range(seq_len, len(X)):
        Xs.append(X[t - seq_len : t])
        ys.append(y[t])
        ttimes.append(time_index[t])
    return (
        np.array(Xs, dtype=np.float32),
        np.array(ys, dtype=np.float32),
        pd.DatetimeIndex(ttimes),
    )

# Model
class GCNLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.W = nn.Linear(in_f, out_f, bias=True)

    def forward(self, H: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        return F.relu(self.W(torch.matmul(L, H)))


class TemporalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)


class STGNN(nn.Module):
    def __init__(self, in_feat: int, hidden: int, n_heads: int, dropout: float = 0.2):
        super().__init__()
        self.gcn1 = GCNLayer(in_feat, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.temporal = TemporalAttention(hidden, n_heads, dropout)
        self.fc1 = nn.Linear(hidden, hidden // 2)
        self.fc_out = nn.Linear(hidden // 2, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """x: (B, T, N, F) | L: (N, N) | returns: (B, N)"""
        B, T, N, _ = x.shape
        sp = torch.stack(
            [self.gcn2(self.gcn1(x[:, t], L), L) for t in range(T)], dim=1
        )  # (B, T, N, hidden)
        sp = sp.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        sp = self.temporal(sp)[:, -1, :]  # (B*N, hidden)
        return self.fc_out(F.relu(self.fc1(self.drop(sp)))).reshape(B, N)


def _batch_predict(
    model: STGNN,
    X: torch.Tensor,
    L: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for b in range(0, len(X), batch_size):
            preds.append(model(X[b : b + batch_size].to(device), L).cpu().numpy())
    return np.concatenate(preds, axis=0)

# Main
def train_stgnn(features_path: Path) -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Load & sort
    df = pd.read_parquet(features_path)
    df["time_bin_30min"] = pd.to_datetime(df["time_bin_30min"], utc=True)
    df = df.sort_values("time_bin_30min")

    if "origin_h3" not in df.columns:
        raise ValueError("'origin_h3' missing -- run 09_feature_engineering.py first.")

    # Temporal split boundaries
    train_end = pd.Timestamp(MODEL_TRAIN_END, tz="UTC")
    valid_end = pd.Timestamp(MODEL_VALID_END, tz="UTC")
    test_start = pd.Timestamp(MODEL_TEST_START, tz="UTC")

    train, valid, test = temporal_split(df, train_end, valid_end, test_start)
    for name, fold in [("train", train), ("valid", valid), ("test", test)]:
        lo = fold["time_bin_30min"].min().date()
        hi = fold["time_bin_30min"].max().date()
        print(f"  {name:<6}: {len(fold):>10,} rows  ({lo} - {hi})")

    # Feature filtering & NaN drop on the FULL dataframe
    all_feat_cols = TEMPORAL_FEATURES + WEATHER_FEATURES
    feat_cols = [c for c in all_feat_cols if c in df.columns]
    missing = set(all_feat_cols) - set(feat_cols)
    if missing:
        print(f"  Feature columns not in data (skipped): {sorted(missing)}")
    print(f"  Feature columns used: {len(feat_cols)}")

    required = feat_cols + [TARGET_COL]
    n_before = len(df)
    df = df.dropna(subset=required)
    print(f"  Rows after dropna: {len(df):,}  (dropped {n_before - len(df):,})")

    # Re-derive clean per-fold references after NaN drop (metadata only)
    train = df[df["time_bin_30min"] <= train_end]
    valid = df[(df["time_bin_30min"] > train_end) & (df["time_bin_30min"] <= valid_end)]
    test = df[df["time_bin_30min"] >= test_start]

    # Build H3 graph
    hex_list = sorted(df["origin_h3"].dropna().unique())
    N = len(hex_list)
    print(f"\nH3 hexes: {N}  |  Building adjacency graph ...")
    L_norm = build_h3_graph(hex_list)

    # Pivot ENTIRE dataset once, make sequences once
    print("Building global pivot tensors ...")
    feature_arrays = [build_pivot(df, c, hex_list) for c in feat_cols]
    X_all = np.stack(feature_arrays, axis=-1)  # (T, N, F)
    y_all = build_pivot(df, TARGET_COL, hex_list)  # (T, N)
    F_dim = X_all.shape[-1]

    time_index = df.groupby("time_bin_30min").size().sort_index().index
    assert len(time_index) == X_all.shape[0], (
        f"time_index length {len(time_index)} != X_all.shape[0] {X_all.shape[0]}"
    )

    seq_len = STGNN_PARAMS["seq_len"]
    Xs, ys, target_times = make_sequences_with_timestamps(
        X_all, y_all, time_index, seq_len
    )
    print(f"  Total sequences: {len(Xs):,}  (first target: {target_times[0].date()})")

    # Split sequences by target timestamp
    tr_mask = target_times <= train_end
    va_mask = (target_times > train_end) & (target_times <= valid_end)
    te_mask = target_times >= test_start

    X_tr = torch.FloatTensor(Xs[tr_mask])
    y_tr = torch.FloatTensor(ys[tr_mask])
    X_va = torch.FloatTensor(Xs[va_mask])
    y_va = torch.FloatTensor(ys[va_mask])
    X_te = torch.FloatTensor(Xs[te_mask])
    y_te = torch.FloatTensor(ys[te_mask])
    target_times_va = target_times[va_mask]
    target_times_te = target_times[te_mask]
    print(f"  Sequences  train={len(X_tr):,}  valid={len(X_va):,}  test={len(X_te):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    L_norm = L_norm.to(device)

    n_heads = 4  # must divide hidden_dim evenly
    assert STGNN_PARAMS["hidden_dim"] % n_heads == 0, (
        "hidden_dim must be divisible by n_heads"
    )
    model = STGNN(
        in_feat=F_dim,
        hidden=STGNN_PARAMS["hidden_dim"],
        n_heads=n_heads,
        dropout=STGNN_PARAMS["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ST-GNN parameters: {n_params:,}")

    opt = torch.optim.Adam(
        model.parameters(),
        lr=STGNN_PARAMS["lr"],
        weight_decay=STGNN_PARAMS["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit = nn.HuberLoss(delta=0.5)

    epochs = STGNN_PARAMS["epochs"]
    batch_size = STGNN_PARAMS["batch_size"]
    patience = 15

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    pat_cnt = 0
    train_losses, val_losses = [], []

    print(f"\nTraining for up to {epochs} epochs (patience={patience}) ...")
    for epoch in range(epochs):
        model.train()
        ep_losses = []
        for b in range(0, len(X_tr), batch_size):
            xb = X_tr[b : b + batch_size].to(device)
            yb = y_tr[b : b + batch_size].to(device)
            opt.zero_grad()
            loss = crit(model(xb, L_norm), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_losses.append(loss.item())

        ep_loss = float(np.mean(ep_losses))

        model.eval()
        with torch.no_grad():
            vl = crit(model(X_va.to(device), L_norm), y_va.to(device)).item()

        train_losses.append(ep_loss)
        val_losses.append(vl)
        sched.step(vl)

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            pat_cnt = 0
        else:
            pat_cnt += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch + 1:3d} | train: {ep_loss:.5f} | "
                f"val: {vl:.5f} | lr: {opt.param_groups[0]['lr']:.2e}"
            )

        if pat_cnt >= patience:
            print(f"  Early stop at epoch {epoch + 1}  (best: {best_epoch})")
            break

    if best_state is None:
        raise RuntimeError(
            "best_state is None — training produced no valid checkpoint."
        )
    model.load_state_dict(best_state)
    print(f"  Best val loss: {best_val_loss:.5f}  at epoch {best_epoch}")

    # Evaluate on validated sequence outputs
    print("\nEvaluation —")
    all_metrics: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for split_name, X_split, y_split, t_times in [
        ("validation", X_va, y_va, target_times_va),
        ("test", X_te, y_te, target_times_te),
    ]:
        y_pred_arr = _batch_predict(model, X_split, L_norm, batch_size, device)
        y_true_flat = y_split.numpy().flatten()
        y_pred_flat = y_pred_arr.flatten()

        metrics = evaluate(y_true_flat, y_pred_flat, split_name)
        all_metrics.append(metrics)

        pf = pd.DataFrame(
            {
                "target_time": np.repeat(t_times, N),
                "origin_h3": np.tile(hex_list, len(t_times)),
                "y_true": y_true_flat,
                "stgnn_pred": y_pred_flat,
                "split": split_name,
            }
        )
        prediction_frames.append(pf)

    # Save artefacts
    pt_path = MODELS_DIR / "stgnn_model.pt"
    torch.save(model.state_dict(), pt_path)

    pred_path = TABLES_DIR / "stgnn_predictions.parquet"
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        pred_path, index=False, compression="zstd"
    )

    metrics_path = TABLES_DIR / "stgnn_metrics.csv"
    pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)

    meta = {
        "model": "ST-GNN",
        "random_seed": RANDOM_SEED,
        "architecture": {
            "in_feat": F_dim,
            "hidden_dim": STGNN_PARAMS["hidden_dim"],
            "n_heads": n_heads,
            "dropout": STGNN_PARAMS["dropout"],
            "n_params": n_params,
        },
        "training": {
            "seq_len": seq_len,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": STGNN_PARAMS["lr"],
            "weight_decay": STGNN_PARAMS["weight_decay"],
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "patience": patience,
        },
        "graph": {
            "h3_resolution": H3_RESOLUTION,
            "n_hexes": N,
        },
        "train_period": {
            "start": str(train["time_bin_30min"].min().date()),
            "end": str(train["time_bin_30min"].max().date()),
            "n_rows": int(len(train)),
        },
        "validation_period": {
            "start": str(valid["time_bin_30min"].min().date()),
            "end": str(valid["time_bin_30min"].max().date()),
            "n_rows": int(len(valid)),
        },
        "test_period": {
            "start": str(test["time_bin_30min"].min().date()),
            "end": str(test["time_bin_30min"].max().date()),
            "n_rows": int(len(test)),
        },
        "n_sequences": {
            "train": int(tr_mask.sum()),
            "valid": int(va_mask.sum()),
            "test": int(te_mask.sum()),
        },
        "features": feat_cols,
        "n_features": len(feat_cols),
        "metrics": all_metrics,
    }
    meta_path = MODELS_DIR / "stgnn_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    # Figures
    test_metrics = next(m for m in all_metrics if m["split"] == "test")
    test_preds = prediction_frames[-1]
    y_true_test = test_preds["y_true"].values
    y_pred_test = test_preds["stgnn_pred"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label="Train")
    axes[0].plot(val_losses, label="Validation")
    axes[0].axvline(
        best_epoch - 1,
        color="grey",
        linestyle="--",
        linewidth=0.8,
        label=f"Best epoch {best_epoch}",
    )
    axes[0].set(title="ST-GNN Training Curve", xlabel="Epoch", ylabel="Huber Loss")
    axes[0].legend()

    idx_sample = np.random.choice(
        len(y_true_test), min(2_000, len(y_true_test)), replace=False
    )
    axes[1].scatter(y_true_test[idx_sample], y_pred_test[idx_sample], alpha=0.3, s=6)
    lims = [float(y_true_test.min()), float(y_true_test.max())]
    axes[1].plot(lims, lims, "r--", linewidth=1.5)
    axes[1].set(
        title=f"ST-GNN: Actual vs Predicted (R2={test_metrics['R2']:.3f})",
        xlabel="Actual trip count",
        ylabel="Predicted trip count",
    )

    plt.suptitle(
        f"ST-GNN Results -- Cityflo Travel Demand\n"
        f"Test R2={test_metrics['R2']:.3f}  "
        f"MAE={test_metrics['MAE']:.3f}  "
        f"sMAPE={test_metrics['sMAPE']:.1f}%",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(FIGURES / "stgnn_results.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nModel       -> {pt_path}")
    print(f"Predictions -> {pred_path}")
    print(f"Metrics     -> {metrics_path}")
    print(f"Metadata    -> {meta_path}")
    print(f"Figure      -> {FIGURES / 'stgnn_results.png'}")
    print("\nST-GNN complete.")


if __name__ == "__main__":
    train_stgnn(FEATURES_MASTER)
