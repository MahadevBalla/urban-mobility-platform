"""
14_analysis_reporting.py

Produces all EDA outputs and model evaluation visualisations from the pipeline.

Figures saved to outputs/figures/:
    eda_demand_dashboard.png        — 6-panel demand EDA
    vehicle_trajectories.png        — top-N vehicle GPS paths on H3 grid
    od_matrix_heatmap.png           — top-N origin x destination trip count heatmap
    od_flow_map.png                 — spatial flow arrows origin -> destination
    od_origin_dest_choropleth.png   — side-by-side origin and destination density
    od_peak_vs_offpeak.png          — net flow (arrivals - departures) by period
    nb_results.png                  — NB actual vs predicted dashboard (if available)
    xgboost_results.png             — XGBoost actual vs predicted dashboard (if available)
    stgnn_results.png               — ST-GNN actual vs predicted dashboard (if available)
    model_comparison.png            — side-by-side metric comparison (if >=2 models exist)

Tables saved to outputs/tables/:
    hex_demand_summary.csv          — per-H3-cell demand statistics
    model_comparison.csv            — MAE / RMSE / R2 / sMAPE / Pearson across models

Other outputs:
    outputs/tables/network_summary.json — study period, #hexes, #stops, #OD pairs, etc.

Input:
    data/processed/pings_clean.parquet
    data/processed/od_agg.parquet
    data/processed/features_master.parquet

Optional (model evaluation):
    outputs/tables/nb_predictions.parquet
    outputs/tables/xgboost_predictions.parquet
    outputs/tables/stgnn_predictions.parquet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import h3
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from shapely.geometry import LineString, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FEATURES_MASTER,
    FIGURES,
    H3_RESOLUTION,
    MUMBAI_BBOX,
    NB_PREDICTIONS,
    OD_AGG,
    PINGS_CLEAN,
    STGNN_PREDICTIONS,
    TABLES_DIR,
    XGB_PREDICTIONS,
)

CBD_LAT = 18.9256
CBD_LNG = 72.8242

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    }
)

# Geometry
def build_hex_geodataframe(hex_cells: list[str]) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame with H3 polygon geometry for a list of cell ids."""
    records = []
    for hx in hex_cells:
        lat_c, lng_c = h3.cell_to_latlng(hx)
        boundary = h3.cell_to_boundary(hx)
        poly = Polygon([(lng, lat) for lat, lng in boundary])
        records.append(
            {
                "h3_index": hx,
                "lat_center": lat_c,
                "lng_center": lng_c,
                "geometry": poly,
            }
        )
    return gpd.GeoDataFrame(records, crs="EPSG:4326")

# Shared metric helper
def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae  = float(np.abs(y_true - y_pred).mean())
    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = float(np.where(denom > 0, np.abs(y_true - y_pred) / denom, 0.0).mean() * 100)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    try:
        r, _ = stats.pearsonr(y_true, y_pred)
        r = float(r)
    except Exception:
        r = np.nan
    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape, "R2": r2, "Pearson_r": r}

# EDA functions
def eda_demand_dashboard(features: pd.DataFrame, out_path: Path) -> None:
    """
    6-panel demand EDA dashboard.

    Panels:
        1. Mean trip_count by hour of day (with peak shading)
        2. Mean trip_count by day of week
        3. trip_count distribution
        4. Mean daily trip_count over study window
        5. Precipitation vs trip_count scatter (coloured by hour)
        6. Top-20 origin hexes: trip_count heatmap by hour
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    hourly = features.groupby("hour")["trip_count"].mean()
    axes[0, 0].bar(hourly.index, hourly.values, color="#185FA5", alpha=0.85)
    axes[0, 0].axvspan(6, 10, alpha=0.08, color="green", label="AM peak")
    axes[0, 0].axvspan(17, 21, alpha=0.08, color="orange", label="PM peak")
    axes[0, 0].set_title("Mean Trip Count by Hour")
    axes[0, 0].set_xlabel("Hour of day (IST)")
    axes[0, 0].legend(fontsize=9)

    dow_d  = features.groupby("dow")["trip_count"].mean()
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    colors = ["#185FA5"] * 5 + ["#EF9F27"] * 2
    axes[0, 1].bar(
        labels[: len(dow_d)], dow_d.values, color=colors[: len(dow_d)], alpha=0.85
    )
    axes[0, 1].set_title("Mean Trip Count by Day of Week")

    axes[0, 2].hist(
        features["trip_count"].clip(upper=features["trip_count"].quantile(0.99)),
        bins=60,
        color="#185FA5",
        alpha=0.8,
        edgecolor="white",
    )
    axes[0, 2].set_title("Trip Count Distribution (clipped at p99)")
    axes[0, 2].set_xlabel("Trip count")

    daily = (
        features.assign(_date=features["time_bin_30min"].dt.date)
        .groupby("_date")["trip_count"]
        .mean()
    )
    axes[1, 0].plot(range(len(daily)), daily.values, color="#185FA5", lw=1.5)
    axes[1, 0].set_title("Mean Trip Count per Day")
    axes[1, 0].set_xlabel("Day index")
    axes[1, 0].set_ylabel("Mean trip count")

    if "precipitation" in features.columns:
        s = features.sample(min(4000, len(features)), random_state=42)
        sc = axes[1, 1].scatter(
            s["precipitation"], s["trip_count"],
            c=s["hour"], cmap="viridis", alpha=0.35, s=6,
        )
        plt.colorbar(sc, ax=axes[1, 1], label="Hour")
        axes[1, 1].set_title("Precipitation vs Trip Count")
        axes[1, 1].set_xlabel("Precipitation (mm/h)")
    else:
        axes[1, 1].text(
            0.5, 0.5, "Weather data not joined",
            ha="center", va="center", transform=axes[1, 1].transAxes,
        )

    if "origin_h3" in features.columns:
        top_hexes = (
            features.groupby("origin_h3")["trip_count"].mean().nlargest(20).index
        )
        hx_hour = features[features["origin_h3"].isin(top_hexes)].pivot_table(
            values="trip_count", index="origin_h3", columns="hour", aggfunc="mean"
        )
        if not hx_hour.empty:
            sns.heatmap(
                hx_hour, ax=axes[1, 2], cmap="YlOrRd",
                cbar_kws={"label": "Mean trip count"},
            )
            axes[1, 2].set_title("Top-20 Origin Hexes: Trip Count by Hour")
            axes[1, 2].set_xlabel("Hour")
            axes[1, 2].set_ylabel("")
            axes[1, 2].tick_params(axis="y", labelsize=6)

    plt.suptitle("Cityflo Demand — EDA Dashboard", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def vehicle_trajectory_map(
    pings: pd.DataFrame,
    hex_gdf: gpd.GeoDataFrame,
    out_path: Path,
    top_n: int = 20,
) -> None:
    """
    Plot GPS trajectories for the top_n most active vehicles on the H3 grid.
    Each line is one vehicle's path coloured by vehicle identity.
    """
    top_vehicles = pings["vehicle_id"].value_counts().head(top_n).index.tolist()
    traj_df = pings[pings["vehicle_id"].isin(top_vehicles)].copy()

    vehicle_lines = []
    for vid, grp in traj_df.sort_values("timestamp_ist").groupby("vehicle_id"):
        grp = grp.sort_values("timestamp_ist")
        if len(grp) >= 2:
            line = LineString(zip(grp["lng"], grp["lat"]))
            vehicle_lines.append({"vehicle_id": vid, "n_pings": len(grp), "geometry": line})

    if not vehicle_lines:
        print("  No vehicle trajectories to plot")
        return

    traj_gdf = gpd.GeoDataFrame(vehicle_lines, crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(13, 13))
    hex_gdf.plot(ax=ax, facecolor="#F0F4FA", edgecolor="#BDC9E8", linewidth=0.3, alpha=0.6)

    cmap    = plt.cm.get_cmap("tab20", len(traj_gdf))
    handles = []
    for i, (_, row) in enumerate(traj_gdf.iterrows()):
        color  = cmap(i)
        xs, ys = row["geometry"].xy
        ax.plot(xs, ys, lw=1.2, color=color, alpha=0.8)
        handles.append(
            mpatches.Patch(color=color, label=f"v{row['vehicle_id']} ({row['n_pings']} pings)")
        )

    ax.plot(CBD_LNG, CBD_LAT, "k*", markersize=14, label="CBD (Nariman Point)", zorder=5)
    ax.set_xlim(MUMBAI_BBOX["lng_min"], MUMBAI_BBOX["lng_max"])
    ax.set_ylim(MUMBAI_BBOX["lat_min"], MUMBAI_BBOX["lat_max"])
    ax.set_title(
        f"Cityflo Bus Trajectories — Top {top_n} Vehicles\nEach coloured line = one vehicle's path",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def od_matrix_heatmap(od: pd.DataFrame, out_path: Path, top_n: int = 15) -> None:
    """
    Heatmap of trip_count between the top_n origin and top_n destination stops.
    Rows = origins, columns = destinations, cells = total trip count.
    """
    top_origins = od["origin_stop_id"].value_counts().head(top_n).index
    top_dests   = od["dest_stop_id"].value_counts().head(top_n).index

    sub = od[
        od["origin_stop_id"].isin(top_origins) & od["dest_stop_id"].isin(top_dests)
    ]

    def _label(stop_id: int, od_df: pd.DataFrame) -> str:
        row = od_df[od_df["origin_stop_id"] == stop_id].head(1)
        if len(row) and pd.notna(row.iloc[0].get("origin_name")):
            return str(row.iloc[0]["origin_name"])[:20]
        return str(stop_id)

    od_pivot = (
        sub.groupby(["origin_stop_id", "dest_stop_id"])["trip_count"]
        .sum()
        .unstack(fill_value=0)
    )
    orig_labels = [_label(i, od) for i in od_pivot.index]
    dest_labels = [
        str(od[od["dest_stop_id"] == j].head(1)["dest_name"].values[0])[:20]
        if len(od[od["dest_stop_id"] == j])
        and pd.notna(od[od["dest_stop_id"] == j].head(1)["dest_name"].values[0])
        else str(j)
        for j in od_pivot.columns
    ]
    od_pivot.index   = orig_labels
    od_pivot.columns = dest_labels

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        od_pivot, cmap="YlOrRd", ax=ax, linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Total vehicle trips"},
        annot=(top_n <= 12), fmt=".0f", annot_kws={"size": 8},
    )
    ax.set_title(
        f"Origin-Destination Flow Matrix — Cityflo Mumbai\n"
        f"Top {top_n} origins × {top_n} destinations | cell = trip count",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Destination stop")
    ax.set_ylabel("Origin stop")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def od_flow_map(
    od: pd.DataFrame,
    hex_gdf: gpd.GeoDataFrame,
    out_path: Path,
    top_n: int = 30,
) -> None:
    """
    Spatial flow map: arrows from origin to destination for the top_n O-D pairs
    by total trip count. Arrow thickness proportional to flow.
    """
    od_flow = (
        od.groupby(
            ["origin_stop_id", "dest_stop_id",
             "origin_lat", "origin_lng", "dest_lat", "dest_lng"]
        )["trip_count"]
        .sum()
        .reset_index(name="flow_count")
        .sort_values("flow_count", ascending=False)
        .head(top_n)
    )

    hex_demand = (
        od.groupby("origin_h3")["trip_count"].mean().reset_index()
        if "origin_h3" in od.columns else None
    )

    fig, ax = plt.subplots(figsize=(13, 13))

    if hex_demand is not None:
        hex_plot = hex_gdf.merge(hex_demand, on="h3_index", how="left")
        hex_plot["trip_count"] = hex_plot["trip_count"].fillna(0)
        hex_plot.plot(
            column="trip_count", ax=ax, cmap="Blues", alpha=0.6,
            vmin=0, vmax=hex_plot["trip_count"].quantile(0.95),
            edgecolor="#99AABF", linewidth=0.3,
        )
    else:
        hex_gdf.plot(ax=ax, facecolor="#F0F4FA", edgecolor="#BDC9E8", linewidth=0.3, alpha=0.6)

    flow_max = od_flow["flow_count"].max()
    for _, row in od_flow.iterrows():
        lw    = 0.5 + 3.5 * (row["flow_count"] / flow_max)
        alpha = 0.4 + 0.5 * (row["flow_count"] / flow_max)
        ax.annotate(
            "", xy=(row["dest_lng"], row["dest_lat"]),
            xytext=(row["origin_lng"], row["origin_lat"]),
            arrowprops=dict(
                arrowstyle="->", lw=lw, color="#C0392B", alpha=alpha,
                connectionstyle="arc3,rad=0.2",
            ),
        )

    origins_plot = od_flow.drop_duplicates("origin_stop_id")
    ax.scatter(origins_plot["origin_lng"], origins_plot["origin_lat"],
               s=70, c="#27AE60", zorder=5, label="Origins")
    dests_plot = od_flow.drop_duplicates("dest_stop_id")
    ax.scatter(dests_plot["dest_lng"], dests_plot["dest_lat"],
               s=70, c="#E74C3C", zorder=5, marker="s", label="Destinations")
    ax.plot(CBD_LNG, CBD_LAT, "k*", markersize=16, label="CBD", zorder=6)

    ax.set_xlim(MUMBAI_BBOX["lng_min"], MUMBAI_BBOX["lng_max"])
    ax.set_ylim(MUMBAI_BBOX["lat_min"], MUMBAI_BBOX["lat_max"])
    ax.set_title(
        f"Origin-Destination Flow Map — Cityflo Mumbai\n"
        f"Top {top_n} O-D flows | Arrow thickness ∝ trip count",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def od_origin_dest_choropleth(
    od: pd.DataFrame, hex_gdf: gpd.GeoDataFrame, out_path: Path
) -> None:
    """
    Side-by-side choropleth maps showing origin density (green) and
    destination density (red) per H3 hex.
    """
    if "origin_h3" not in od.columns:
        print("  Skipping choropleth: origin_h3 not in OD table")
        return

    origin_counts = (
        od.groupby("origin_h3")["trip_count"].sum().rename("origin_count").reset_index()
    )
    dest_counts = (
        od.groupby("dest_h3")["trip_count"].sum().rename("dest_count").reset_index()
        if "dest_h3" in od.columns
        else pd.DataFrame(columns=["dest_h3", "dest_count"])
    )

    hex_od = hex_gdf.copy()
    hex_od = hex_od.merge(
        origin_counts.rename(columns={"origin_h3": "h3_index"}), on="h3_index", how="left"
    )
    hex_od = hex_od.merge(
        dest_counts.rename(columns={"dest_h3": "h3_index"}), on="h3_index", how="left"
    )
    hex_od["origin_count"] = hex_od["origin_count"].fillna(0)
    hex_od["dest_count"]   = hex_od["dest_count"].fillna(0)

    vmax = max(
        hex_od["origin_count"].quantile(0.95),
        hex_od["dest_count"].quantile(0.95),
    )

    fig, axes = plt.subplots(1, 2, figsize=(20, 11))
    for ax, col, title, cmap, label in [
        (axes[0], "origin_count", "Origin Density (where trips start)",  "Greens", "Trip starts"),
        (axes[1], "dest_count",   "Destination Density (where trips end)", "Reds",   "Trip ends"),
    ]:
        hex_od.plot(
            column=col, ax=ax, cmap=cmap, legend=True,
            vmin=0, vmax=vmax, edgecolor="white", linewidth=0.3, alpha=0.85,
            legend_kwds={"label": label, "shrink": 0.6},
        )
        ax.plot(CBD_LNG, CBD_LAT, "b*", markersize=14, label="CBD", zorder=5)
        ax.set_xlim(MUMBAI_BBOX["lng_min"], MUMBAI_BBOX["lng_max"])
        ax.set_ylim(MUMBAI_BBOX["lat_min"], MUMBAI_BBOX["lat_max"])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.legend(fontsize=9)

    plt.suptitle(
        "Cityflo Mumbai — Origin vs Destination Density",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def od_peak_vs_offpeak(
    od: pd.DataFrame, hex_gdf: gpd.GeoDataFrame, out_path: Path
) -> None:
    """
    Net trip flow (arrivals - departures) per hex for AM peak vs off-peak.
    Green hex = net departure zone; red hex = net arrival zone.
    """
    if "origin_h3" not in od.columns or "dest_h3" not in od.columns:
        print("  Skipping peak map: H3 columns not in OD table")
        return

    od = od.copy()
    od["is_peak"] = od["period"].isin(["AM_peak", "PM_peak"]).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    for ax, peak_val, title in [
        (axes[0], 1, "Peak Hours (AM 06–10, PM 17–21)"),
        (axes[1], 0, "Off-Peak Hours"),
    ]:
        sub   = od[od["is_peak"] == peak_val]
        o_cnt = sub.groupby("origin_h3")["trip_count"].sum().rename("origin_count").reset_index()
        d_cnt = sub.groupby("dest_h3")["trip_count"].sum().rename("dest_count").reset_index()

        h_sub = hex_gdf.copy()
        h_sub = h_sub.merge(o_cnt.rename(columns={"origin_h3": "h3_index"}), on="h3_index", how="left")
        h_sub = h_sub.merge(d_cnt.rename(columns={"dest_h3": "h3_index"}),   on="h3_index", how="left")
        h_sub["origin_count"] = h_sub["origin_count"].fillna(0)
        h_sub["dest_count"]   = h_sub["dest_count"].fillna(0)
        h_sub["net_flow"]     = h_sub["dest_count"] - h_sub["origin_count"]

        h_sub.plot(
            column="net_flow", ax=ax, cmap="RdYlGn_r", legend=True,
            edgecolor="white", linewidth=0.3, alpha=0.85,
            legend_kwds={"label": "Net flow (arrivals − departures)", "shrink": 0.6},
        )
        ax.plot(CBD_LNG, CBD_LAT, "b*", markersize=14, label="CBD", zorder=5)
        ax.set_xlim(MUMBAI_BBOX["lng_min"], MUMBAI_BBOX["lng_max"])
        ax.set_ylim(MUMBAI_BBOX["lat_min"], MUMBAI_BBOX["lat_max"])
        ax.set_title(f"Net Trip Flow — {title}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.legend(fontsize=9)

    plt.suptitle(
        "Cityflo Mumbai — Trip Flow: Peak vs Off-Peak",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def hex_demand_summary(features: pd.DataFrame, out_path: Path) -> None:
    """Write per-H3-cell demand statistics to CSV."""
    if "origin_h3" not in features.columns:
        return
    summary = (
        features.groupby("origin_h3")
        .agg(
            lat_center=("origin_lat", "first"),
            lng_center=("origin_lng", "first"),
            mean_demand=("trip_count", "mean"),
            peak_demand=("trip_count", "max"),
            n_observations=("trip_count", "count"),
        )
        .reset_index()
        .rename(columns={"origin_h3": "h3_index"})
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.name}  ({len(summary):,} hexes)")

# Network summary JSON
def save_network_summary(
    features: pd.DataFrame,
    od: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Write a network_summary.json with key study facts for use in the report.
    Fields: study period, #hexes, #stops, #OD pairs, #trips,
            mean/peak daily demand, peak hour.
    """
    t = features["time_bin_30min"]

    hourly_mean  = features.groupby("hour")["trip_count"].mean()
    peak_hour    = int(hourly_mean.idxmax())
    n_trips_total = int(od["trip_count"].sum()) if "trip_count" in od.columns else None

    daily_demand = (
        features.assign(_date=t.dt.date)
        .groupby("_date")["trip_count"]
        .mean()
    )

    summary = {
        "study_period": {
            "start": str(t.min().date()),
            "end":   str(t.max().date()),
            "n_days": int((t.max() - t.min()).days) + 1,
        },
        "spatial": {
            "h3_resolution":    H3_RESOLUTION,
            "n_origin_hexes":   int(features["origin_h3"].nunique())
                                if "origin_h3" in features.columns else None,
            "n_origin_stops":   int(od["origin_stop_id"].nunique())
                                if "origin_stop_id" in od.columns else None,
            "n_dest_stops":     int(od["dest_stop_id"].nunique())
                                if "dest_stop_id" in od.columns else None,
            "n_od_pairs":       int(
                od.groupby(["origin_stop_id", "dest_stop_id"]).ngroups
            ) if {"origin_stop_id", "dest_stop_id"}.issubset(od.columns) else None,
        },
        "demand": {
            "total_trips":           n_trips_total,
            "mean_daily_trip_count": round(float(daily_demand.mean()), 3),
            "peak_daily_trip_count": round(float(daily_demand.max()),  3),
            "peak_hour_of_day":      peak_hour,
            "n_feature_rows":        len(features),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Saved: {out_path.name}")

# Model prediction dashboard
def _model_prediction_dashboard(
    pred_df: pd.DataFrame,
    model_name: str,
    pred_col: str,
    out_path: Path,
) -> dict[str, float]:
    """
    3-panel figure for one model:
        1. Actual vs Predicted scatter (test split)
        2. Residual histogram
        3. Residual vs Actual

    Each panel includes an inset textbox summarising:
        — MAE
        — RMSE
        — R²
        — sMAPE
        — Pearson correlation

    Returns metric dict for aggregation.
    """
    test_df = pred_df[pred_df["split"] == "test"] if "split" in pred_df.columns else pred_df

    if len(test_df) == 0:
        return None

    if "y_true" in test_df.columns:
        y_true = test_df["y_true"].values
    else:
        y_true = test_df["trip_count"].values

    y_pred  = test_df[pred_col].values

    m = _compute_metrics(y_true, y_pred)
    resid = y_pred - y_true

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Scatter
    idx = np.random.default_rng(42).choice(len(y_true), min(3_000, len(y_true)), replace=False)
    axes[0].scatter(y_true[idx], y_pred[idx], alpha=0.3, s=6, color="#185FA5")
    lims = [float(y_true.min()), float(y_true.max())]
    axes[0].plot(lims, lims, "r--", lw=1.5)
    axes[0].set(title="Actual vs Predicted", xlabel="Actual trip count",
                ylabel="Predicted trip count")

    # 2. Residual histogram
    axes[1].hist(resid, bins=60, color="#185FA5", alpha=0.8, edgecolor="white")
    axes[1].axvline(0, color="red", lw=1.5, linestyle="--")
    axes[1].set(title="Residual Distribution", xlabel="Residual (pred − actual)",
                ylabel="Count")

    # 3. Residual vs Actual
    axes[2].scatter(y_true[idx], resid[idx], alpha=0.3, s=6, color="#E6861A")
    axes[2].axhline(0, color="red", lw=1.5, linestyle="--")
    axes[2].set(title="Residual vs Actual", xlabel="Actual trip count",
                ylabel="Residual (pred − actual)")

    metrics_text = (
        f"MAE   : {m['MAE']:.4f}\n"
        f"RMSE  : {m['RMSE']:.4f}\n"
        f"R²    : {m['R2']:.4f}\n"
        f"sMAPE : {m['sMAPE']:.2f}%\n"
        f"Pearson: {m['Pearson_r']:.4f}"
    )
    for ax in axes:
        ax.text(
            0.02, 0.97, metrics_text,
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      alpha=0.85, edgecolor="grey"),
        )

    plt.suptitle(
        f"{model_name} — Test Set Evaluation  (n={len(y_true):,})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")

    return {"Model": model_name, **m}


def model_prediction_dashboard(figures_dir: Path, tables_dir: Path) -> None:
    """
    Auto-detect which prediction files exist, produce per-model dashboards,
    and emit model_comparison.csv + model_comparison.png when >=2 models found.
    """
    PRED_REGISTRY = [
        (NB_PREDICTIONS,    "Naive Bayes",  "nb_pred",      "nb_results.png"),
        (XGB_PREDICTIONS,   "XGBoost",      "xgb_pred",     "xgboost_results.png"),
        (STGNN_PREDICTIONS, "ST-GNN",       "stgnn_pred",   "stgnn_results.png"),
    ]

    all_metrics: list[dict] = []

    for pred_path, model_name, pred_col, fig_name in PRED_REGISTRY:
        if not pred_path.exists():
            print(f"  Skipping {model_name}: {pred_path.name} not found")
            continue

        pred_df = pd.read_parquet(pred_path)
        if pred_col not in pred_df.columns:
            print(f"  Skipping {model_name}: column '{pred_col}' not in {pred_path.name}")
            continue

        metrics = _model_prediction_dashboard(
            pred_df, model_name, pred_col, figures_dir / fig_name
        )
        if metrics is not None:
            all_metrics.append(metrics)

    if not all_metrics:
        print("  No prediction files found — skipping model comparison.")
        return

    # Save model comparison CSV
    comp_df = pd.DataFrame(all_metrics)
    comp_path = tables_dir / "model_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"  Saved: {comp_path.name}")

    if len(all_metrics) < 2:
        return

    # Comparison bar chart (3-panel: MAE / RMSE / R2)
    metrics_to_plot = [
        ("MAE",   "Mean Absolute Error",         False),
        ("RMSE",  "Root Mean Squared Error",      False),
        ("R2",    "R² (coefficient of determination)", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    palette   = ["#185FA5", "#E6861A", "#27AE60", "#9B59B6"]
    models    = comp_df["Model"].tolist()

    for ax, (col, label, higher_is_better) in zip(axes, metrics_to_plot):
        vals   = comp_df[col].tolist()
        colors = [palette[i % len(palette)] for i in range(len(models))]
        bars   = ax.bar(models, vals, color=colors, alpha=0.85, edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        best_idx = int(np.argmax(vals) if higher_is_better else np.argmin(vals))
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle(
        "Model Comparison — Cityflo Travel Demand Forecasting",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    cmp_fig_path = figures_dir / "model_comparison.png"
    plt.savefig(cmp_fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {cmp_fig_path.name}")

# Entry point
def main() -> None:
    print(f"H3 resolution: {H3_RESOLUTION}")

    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    od       = pd.read_parquet(OD_AGG)
    features = pd.read_parquet(FEATURES_MASTER)
    features["time_bin_30min"] = pd.to_datetime(features["time_bin_30min"], utc=True)

    if "hour" not in features.columns:
        features["hour"] = features["time_bin_30min"].dt.hour
    if "dow" not in features.columns:
        features["dow"] = features["time_bin_30min"].dt.dayofweek

    # Build H3 GeoDataFrame from all origin hexes
    all_hexes: set[str] = set()
    if "origin_h3" in od.columns:
        all_hexes.update(od["origin_h3"].dropna().unique())
    if "origin_h3" in features.columns:
        all_hexes.update(features["origin_h3"].dropna().unique())

    print(f"Building H3 GeoDataFrame: {len(all_hexes):,} hexes ...")
    hex_gdf = build_hex_geodataframe(list(all_hexes))

    print("Loading GPS pings (trajectory map) ...")
    pings = pd.read_parquet(
        PINGS_CLEAN,
        columns=["vehicle_id", "lat", "lng", "timestamp_ist"],
    )

    # EDA figures
    print("\nnGenerating EDA figures ...")
    eda_demand_dashboard(features,           FIGURES / "eda_demand_dashboard.png")
    vehicle_trajectory_map(pings, hex_gdf,  FIGURES / "vehicle_trajectories.png")
    od_matrix_heatmap(od,                   FIGURES / "od_matrix_heatmap.png")
    od_flow_map(od, hex_gdf,                FIGURES / "od_flow_map.png")
    od_origin_dest_choropleth(od, hex_gdf,  FIGURES / "od_origin_dest_choropleth.png")
    od_peak_vs_offpeak(od, hex_gdf,         FIGURES / "od_peak_vs_offpeak.png")

    # Tables
    print("\nnSaving tables ...")
    hex_demand_summary(features, TABLES_DIR / "hex_demand_summary.csv")
    save_network_summary(features, od, TABLES_DIR / "network_summary.json")

    # Model evaluation
    print("\nnGenerating model prediction dashboards ...")
    model_prediction_dashboard(FIGURES, TABLES_DIR)

    print("\n14_analysis_reporting.py complete.")


if __name__ == "__main__":
    main()