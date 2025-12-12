#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SPEC_LIMIT_DBC = -60.0


# -----------------------------
# Helpers
# -----------------------------

def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _to_bool_or_na(x: Any):
    # Returns True/False/pd.NA
    if x is None:
        return pd.NA
    if isinstance(x, float) and np.isnan(x):
        return pd.NA
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "t", "yes", "y", "1"):
            return True
        if s in ("false", "f", "no", "n", "0"):
            return False
    return pd.NA


def load_sweep_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_top_contributor(tc: Any) -> Dict[str, Any]:
    """
    Flatten predictor "top_contributor" dict into CSV/plot-friendly columns.
    """
    out: Dict[str, Any] = {
        "top_origin": None,
        "top_stage_path": None,
        "top_dataset_id": None,
        "top_kind": None,
        "top_expression": None,
        "top_m": None,
        "top_n": None,
        "top_sign": None,
        "top_dbc": None,
        "top_lineage_row_indices": None,
    }

    if not isinstance(tc, dict):
        return out

    out["top_origin"] = tc.get("origin", None)
    out["top_stage_path"] = tc.get("stage_path", None)
    out["top_dataset_id"] = tc.get("dataset_id", None)
    out["top_dbc"] = tc.get("dbc", None)
    out["top_lineage_row_indices"] = (
        ",".join(str(i) for i in tc.get("lineage_row_indices", []) if i is not None)
        if isinstance(tc.get("lineage_row_indices", None), list)
        else None
    )

    fam = tc.get("family_id", None)
    if isinstance(fam, dict):
        out["top_kind"] = fam.get("kind", None)
        out["top_expression"] = fam.get("expression", None)
        out["top_m"] = fam.get("m", None)
        out["top_n"] = fam.get("n", None)
        out["top_sign"] = fam.get("sign", None)

    return out


def _make_spur_type_label(row: pd.Series, mode: str) -> str:
    """
    Derive a spur "type" label for coding/legend.
    mode:
      - "expression": origin|kind|expression
      - "kind":       origin|kind
      - "stage":      origin|stage_path
      - "dataset":    dataset_id|origin|kind|expression
    """
    origin = row.get("top_origin", None)
    kind = row.get("top_kind", None)
    expr = row.get("top_expression", None)
    stage = row.get("top_stage_path", None)
    ds = row.get("top_dataset_id", None)

    def s(x: Any) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "?"
        t = str(x).strip()
        return t if t else "?"

    if mode == "kind":
        return f"{s(origin)} | {s(kind)}"
    if mode == "stage":
        return f"{s(origin)} | {s(stage)}"
    if mode == "dataset":
        return f"{s(ds)} | {s(origin)} | {s(kind)} | {s(expr)}"
    # default: expression
    return f"{s(origin)} | {s(kind)} | {s(expr)}"


def sweep_df(payload: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = payload.get("worst_by_if", [])
    df = pd.DataFrame(rows)

    needed = [
        "if1_hz",
        "worst_tone_freq_hz",
        "worst_tone_level_dbc",
        "margin_to_spec_db",
        "coincident",
        "coincident_with_carrier_bin",
        "has_unknown_contributors",
        "unknown_contributors_count",
        "unknown_combining_mode",
        "top_contributor",
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan

    # Numeric conversions
    df["if1_hz"] = df["if1_hz"].apply(_to_float)
    df["worst_tone_freq_hz"] = df["worst_tone_freq_hz"].apply(_to_float)
    df["worst_tone_level_dbc"] = df["worst_tone_level_dbc"].apply(_to_float)
    df["margin_to_spec_db"] = df["margin_to_spec_db"].apply(_to_float)

    # Booleans (clean dtype)
    for bcol in ["coincident", "coincident_with_carrier_bin", "has_unknown_contributors"]:
        df[bcol] = df[bcol].apply(_to_bool_or_na).astype("boolean")

    df["unknown_contributors_count"] = pd.to_numeric(df["unknown_contributors_count"], errors="coerce")

    # Convenience units
    df["if1_mhz"] = df["if1_hz"] / 1e6
    df["worst_tone_freq_ghz"] = df["worst_tone_freq_hz"] / 1e9

    # Margin fill where possible
    missing_margin = df["margin_to_spec_db"].isna() & np.isfinite(df["worst_tone_level_dbc"])
    df.loc[missing_margin, "margin_to_spec_db"] = SPEC_LIMIT_DBC - df.loc[missing_margin, "worst_tone_level_dbc"]

    # Flatten top_contributor
    top_flat = df["top_contributor"].apply(_flatten_top_contributor).apply(pd.Series)
    for c in top_flat.columns:
        if c not in df.columns:
            df[c] = top_flat[c]
        else:
            df[c + "_top"] = top_flat[c]

    return df.sort_values("if1_hz").reset_index(drop=True)


# -----------------------------
# Styling / coding of spur types
# -----------------------------

def _default_color_cycle() -> List[str]:
    colors = plt.rcParams.get("axes.prop_cycle", None)
    if colors is None:
        return ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    by_key = colors.by_key()
    return list(by_key.get("color", ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]))


def _marker_cycle() -> List[str]:
    # Common, distinct markers
    return ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h", "8", "p"]


def choose_top_spur_types(
    df: pd.DataFrame,
    spur_type_col: str,
    max_types: int,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Choose the top spur types for legend/styles.
    Ordering:
      1) severity (max worst_tone_level_dbc, i.e. least negative is worst)
      2) frequency (count)
    Returns (selected_types, stats_df)
    """
    work = df.copy()
    work = work[np.isfinite(work["worst_tone_level_dbc"])].copy()
    if work.empty:
        return [], pd.DataFrame(columns=[spur_type_col, "count", "max_level_dbc"])

    g = work.groupby(spur_type_col, dropna=False)
    stats = g.agg(
        count=("worst_tone_level_dbc", "size"),
        max_level_dbc=("worst_tone_level_dbc", "max"),
    ).reset_index()

    stats = stats.sort_values(["max_level_dbc", "count"], ascending=[False, False]).reset_index(drop=True)

    selected = stats[spur_type_col].head(max_types).astype(str).tolist()
    return selected, stats


def assign_styles(spur_types: List[str]) -> Dict[str, Dict[str, Any]]:
    colors = _default_color_cycle()
    markers = _marker_cycle()

    styles: Dict[str, Dict[str, Any]] = {}
    for i, t in enumerate(spur_types):
        styles[t] = {
            "color": colors[i % len(colors)],
            "marker": markers[i % len(markers)],
        }
    styles["Other"] = {
        "color": colors[len(spur_types) % len(colors)],
        "marker": markers[len(spur_types) % len(markers)],
    }
    styles["Unknown"] = {
        "color": colors[(len(spur_types) + 1) % len(colors)],
        "marker": markers[(len(spur_types) + 1) % len(markers)],
    }
    return styles


def apply_spur_type_coding(
    df: pd.DataFrame,
    spur_type_mode: str,
    max_types: int,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Adds:
      - df["spur_type_raw"]
      - df["spur_type"] (top-N else Other; Unknown if missing)
    Returns updated df, style map, and stats table.
    """
    mode = spur_type_mode.strip().lower()
    if mode not in ("expression", "kind", "stage", "dataset"):
        mode = "expression"

    df = df.copy()
    df["spur_type_raw"] = df.apply(lambda r: _make_spur_type_label(r, mode), axis=1)

    # If top info missing, label as Unknown
    unknown_mask = df["top_origin"].isna() & df["top_kind"].isna() & df["top_expression"].isna()
    df.loc[unknown_mask, "spur_type_raw"] = "Unknown"

    selected, stats = choose_top_spur_types(df, "spur_type_raw", max_types=max_types)
    styles = assign_styles(selected)

    df["spur_type"] = df["spur_type_raw"].astype(str)
    df.loc[~df["spur_type"].isin(selected + ["Unknown"]), "spur_type"] = "Other"

    return df, styles, stats


# -----------------------------
# Plotting
# -----------------------------

def _savefig(outpath: Path, dpi: int = 160) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()


def _plot_overall_line(df: pd.DataFrame, xcol: str, ycol: str) -> None:
    # A single overall trend line in default styling.
    plt.plot(df[xcol], df[ycol], linestyle="-", linewidth=1)


def _overlay_scatter_by_type(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    styles: Dict[str, Dict[str, Any]],
    legend: bool = True,
    legend_max_items: int = 20,
) -> None:
    # Overlay per-type scatter points (colored + marker-coded) and build a legend.
    # Plot in a stable order: selected types first, then Other, then Unknown.
    order = [k for k in styles.keys() if k not in ("Other", "Unknown")] + ["Other", "Unknown"]
    handles = []
    labels = []

    for t in order:
        sub = df[df["spur_type"] == t]
        sub = sub[np.isfinite(sub[ycol]) & np.isfinite(sub[xcol])]
        if sub.empty:
            continue
        st = styles.get(t, styles.get("Other", {"marker": "o", "color": "C0"}))
        h = plt.scatter(sub[xcol], sub[ycol], marker=st["marker"], c=st["color"])
        handles.append(h)
        labels.append(f"{t} (n={len(sub)})")

    if legend and handles:
        # Cap legend size to avoid giant legends if user chooses a large max_types.
        if len(handles) > legend_max_items:
            handles = handles[:legend_max_items]
            labels = labels[:legend_max_items]
            labels[-1] = labels[-1] + "  (legend truncated)"
        plt.legend(handles, labels, fontsize="small", loc="best", framealpha=0.85)


def plot_worst_level_vs_if(df: pd.DataFrame, styles: Dict[str, Dict[str, Any]], outdir: Path, title_prefix: str) -> None:
    plt.figure()
    _plot_overall_line(df, "if1_mhz", "worst_tone_level_dbc")
    _overlay_scatter_by_type(df, "if1_mhz", "worst_tone_level_dbc", styles=styles, legend=True)
    plt.axhline(SPEC_LIMIT_DBC, linestyle="--")
    plt.xlabel("IF1 (MHz)")
    plt.ylabel("Worst spur level (dBc)")
    plt.title(f"{title_prefix} Worst spur level vs IF1")
    plt.grid(True)
    _savefig(outdir / "01_worst_level_vs_if1.png")


def plot_margin_vs_if(df: pd.DataFrame, styles: Dict[str, Dict[str, Any]], outdir: Path, title_prefix: str) -> None:
    plt.figure()
    _plot_overall_line(df, "if1_mhz", "margin_to_spec_db")
    _overlay_scatter_by_type(df, "if1_mhz", "margin_to_spec_db", styles=styles, legend=True)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("IF1 (MHz)")
    plt.ylabel("Margin to -60 dBc (dB)  (positive = pass)")
    plt.title(f"{title_prefix} Margin vs IF1")
    plt.grid(True)
    _savefig(outdir / "02_margin_vs_if1.png")


def plot_worst_freq_vs_if(df: pd.DataFrame, styles: Dict[str, Dict[str, Any]], outdir: Path, title_prefix: str) -> None:
    plt.figure()
    _plot_overall_line(df, "if1_mhz", "worst_tone_freq_ghz")
    _overlay_scatter_by_type(df, "if1_mhz", "worst_tone_freq_ghz", styles=styles, legend=True)
    plt.xlabel("IF1 (MHz)")
    plt.ylabel("Worst spur frequency (GHz)")
    plt.title(f"{title_prefix} Worst spur frequency vs IF1")
    plt.grid(True)
    _savefig(outdir / "03_worst_freq_vs_if1.png")


def plot_unknowns_vs_if(df: pd.DataFrame, outdir: Path, title_prefix: str) -> None:
    # No spur-type coding here; it’s a property of the worst cluster, not a spur family/type itself.
    if df["unknown_contributors_count"].notna().any():
        plt.figure()
        plt.plot(df["if1_mhz"], df["unknown_contributors_count"], marker=".", linestyle="-")
        plt.xlabel("IF1 (MHz)")
        plt.ylabel("Unknown contributors in worst cluster (count)")
        plt.title(f"{title_prefix} Unknown contributors vs IF1")
        plt.grid(True)
        _savefig(outdir / "04_unknown_contributors_vs_if1.png")


def plot_scatter_freq_vs_level(df: pd.DataFrame, styles: Dict[str, Dict[str, Any]], outdir: Path, title_prefix: str) -> None:
    good = df[np.isfinite(df["worst_tone_freq_ghz"]) & np.isfinite(df["worst_tone_level_dbc"])].copy()
    if good.empty:
        return
    plt.figure()
    _overlay_scatter_by_type(good, "worst_tone_freq_ghz", "worst_tone_level_dbc", styles=styles, legend=True)
    plt.axhline(SPEC_LIMIT_DBC, linestyle="--")
    plt.xlabel("Worst spur frequency (GHz)")
    plt.ylabel("Worst spur level (dBc)")
    plt.title(f"{title_prefix} Worst spur freq vs level (scatter)")
    plt.grid(True)
    _savefig(outdir / "05_scatter_freq_vs_level.png")


# -----------------------------
# CSV summary
# -----------------------------

def write_spur_summary_csv(payload: Dict[str, Any], df: pd.DataFrame, outdir: Path) -> Path:
    """
    Write a spur summary CSV ordered by severity (worst first).
    Severity = highest (least negative) worst spur level (dBc) across IF points.
    """
    lo2_hz = _to_float(payload.get("lo2_hz", float("nan")))
    lo2_ghz = lo2_hz / 1e9 if np.isfinite(lo2_hz) else np.nan

    tmp = df.copy()
    tmp["severity_score_dbc"] = tmp["worst_tone_level_dbc"]
    tmp = tmp.sort_values("severity_score_dbc", ascending=False, na_position="last").reset_index(drop=True)
    tmp["severity_rank"] = np.arange(1, len(tmp) + 1, dtype=int)

    cols = [
        "severity_rank",
        "if1_hz",
        "if1_mhz",
        "worst_tone_freq_hz",
        "worst_tone_freq_ghz",
        "worst_tone_level_dbc",
        "margin_to_spec_db",
        "coincident",
        "coincident_with_carrier_bin",
        "has_unknown_contributors",
        "unknown_contributors_count",
        "unknown_combining_mode",
        "spur_type",         # coded (top-N/Other/Unknown)
        "spur_type_raw",     # full label before top-N grouping
        "top_origin",
        "top_stage_path",
        "top_dataset_id",
        "top_kind",
        "top_expression",
        "top_m",
        "top_n",
        "top_sign",
        "top_dbc",
        "top_lineage_row_indices",
    ]
    for c in cols:
        if c not in tmp.columns:
            tmp[c] = np.nan

    tmp.insert(1, "lo2_hz", lo2_hz)
    tmp.insert(2, "lo2_ghz", lo2_ghz)

    outpath = outdir / "spur_summary_by_severity.csv"
    tmp[["lo2_hz", "lo2_ghz"] + cols].to_csv(outpath, index=False)
    return outpath


def write_spur_type_stats_csv(stats: pd.DataFrame, outdir: Path, max_types: int) -> Path:
    outpath = outdir / "spur_type_stats.csv"
    stats = stats.copy()
    stats.insert(0, "note", f"Top spur types selected for legend: max_types={max_types}")
    stats.to_csv(outpath, index=False)
    return outpath


# -----------------------------
# Console summary
# -----------------------------

def print_summary(payload: Dict[str, Any], df: pd.DataFrame) -> None:
    lo2 = payload.get("lo2_hz", None)
    worst_overall = payload.get("worst_overall", None)

    print("=== Sweep summary ===")
    if lo2 is not None and np.isfinite(_to_float(lo2)):
        print(f"LO2: {_to_float(lo2)/1e9:.3f} GHz")

    if isinstance(worst_overall, dict) and worst_overall:
        if1 = worst_overall.get("if1_hz", None)
        f = worst_overall.get("tone_freq_hz", None) or worst_overall.get("worst_tone_freq_hz", None)
        lvl = worst_overall.get("tone_level_dbc", None) or worst_overall.get("worst_tone_level_dbc", None)
        m = worst_overall.get("margin_to_spec_db", None)

        print("Worst overall:")
        if if1 is not None and np.isfinite(_to_float(if1)):
            print(f"  IF1: {_to_float(if1)/1e6:.3f} MHz")
        if f is not None and np.isfinite(_to_float(f)):
            print(f"  Spur freq: {_to_float(f)/1e9:.6f} GHz")
        if lvl is not None and np.isfinite(_to_float(lvl)):
            print(f"  Spur level: {_to_float(lvl):.2f} dBc")
        if m is not None and np.isfinite(_to_float(m)):
            print(f"  Margin: {_to_float(m):.2f} dB")

    finite_levels = df[np.isfinite(df["worst_tone_level_dbc"])]["worst_tone_level_dbc"]
    if not finite_levels.empty:
        print(f"Min/Max worst spur level across sweep: {finite_levels.min():.2f} / {finite_levels.max():.2f} dBc")

    finite_margin = df[np.isfinite(df["margin_to_spec_db"])]["margin_to_spec_db"]
    if not finite_margin.empty:
        print(f"Min/Max margin across sweep: {finite_margin.min():.2f} / {finite_margin.max():.2f} dB")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot BUC spur sweep JSON results with spur-type coding + write severity CSV summary"
    )
    ap.add_argument("--in", dest="in_path", required=True, help="Path to sweep JSON (e.g., out/sweep_lo2_21.json)")
    ap.add_argument("--outdir", default=None, help="Output directory for plots/CSV (default: <json_dir>/plots)")
    ap.add_argument("--title", default=None, help="Title prefix (default: derived from LO2)")

    ap.add_argument(
        "--spur-type-mode",
        default="expression",
        choices=["expression", "kind", "stage", "dataset"],
        help="How to define spur 'type' for coding/legend",
    )
    ap.add_argument(
        "--max-spur-types",
        type=int,
        default=12,
        help="Max distinct spur types to show in legend; the rest are grouped into 'Other'",
    )

    args = ap.parse_args()

    in_path = Path(args.in_path)
    payload = load_sweep_json(in_path)
    df = sweep_df(payload)

    if df["if1_hz"].isna().all():
        raise ValueError("No valid IF1 points found in worst_by_if.")

    lo2_hz = _to_float(payload.get("lo2_hz", float("nan")))
    default_title = f"LO2 {lo2_hz/1e9:.1f} GHz" if np.isfinite(lo2_hz) else "BUC Sweep"
    title_prefix = args.title or default_title

    outdir = Path(args.outdir) if args.outdir else (in_path.parent / "plots")
    outdir.mkdir(parents=True, exist_ok=True)

    # Spur-type coding for plots + CSV
    df_coded, styles, stats = apply_spur_type_coding(
        df,
        spur_type_mode=str(args.spur_type_mode),
        max_types=int(max(1, args.max_spur_types)),
    )

    print_summary(payload, df_coded)

    # Plots (no extra "failure marks" beyond reference lines)
    plot_worst_level_vs_if(df_coded, styles, outdir, title_prefix)
    plot_margin_vs_if(df_coded, styles, outdir, title_prefix)
    plot_worst_freq_vs_if(df_coded, styles, outdir, title_prefix)
    plot_unknowns_vs_if(df_coded, outdir, title_prefix)
    plot_scatter_freq_vs_level(df_coded, styles, outdir, title_prefix)

    # CSV summary ordered by severity
    csv_path = write_spur_summary_csv(payload, df_coded, outdir)
    stats_path = write_spur_type_stats_csv(stats, outdir, max_types=int(max(1, args.max_spur_types)))

    print(f"\nWrote plots to: {outdir.resolve()}")
    print(f"Wrote spur summary CSV to: {csv_path.resolve()}")
    print(f"Wrote spur type stats CSV to: {stats_path.resolve()}")


if __name__ == "__main__":
    main()
