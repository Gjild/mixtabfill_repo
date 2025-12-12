#!/usr/bin/env python3
"""
BUC Spur/System Prediction Tool (Revised)
=========================================

Two-stage BUC spur prediction / verification engine driven by measured truth:

- Measured filter attenuation curves (IF2 and RF filters)
- Measured mixer spur datasets (Mixer1 CSV; Mixer2 CSVs per LO2)
- Spur-family modeling:
    * Frequency: primary uses expected_corr_freq_Hz (or configured equivalent) vs IF with missing-point fill;
                 falls back to analytic |m*LO ± n*IF| (absolute value to avoid negative Hz).
    * Level: uses rel_dBc_vs_desired (or configured equivalent) vs IF with missing-point fill and noise-limited handling.

Architecture / Nodes:
  IF1 -> Mixer1 -> (Node A) -> IF2 Filter -> (Node B) -> Mixer2 translation -> (Node C)
      -> Mixer2 intrinsic spurs injected at Node C (default) OR Node D (if measured post RF filter)
      -> RF Filter -> (Node D) -> coincidence clustering + linear power summation

This tool does NOT perform any LO2 optimization or recommendation.

Revisions applied per review:
- Filter out-of-range behavior is configurable and implemented (default endpoint clamp, never optimistic unless explicitly enabled).
- Mixer CSV column flexibility improved (candidates for level, measured freq, expected-corr freq).
- Mixer2 LO sanity check added.
- Coincidence clusters explicitly flag unknown contributors; optional conservative combining mode available.
- Report includes both "known-only" combined dBc and "conservative (unknown floor)" combined dBc when unknowns exist.
- "Required attenuation attribution" renamed to avoid implying true measured filter-shape differential attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import math
import json
import argparse

import numpy as np
import pandas as pd

SPEC_LIMIT_DBC = -60.0


# -----------------------------
# Utilities
# -----------------------------

def db_to_lin_ratio(dbc: float) -> float:
    """Convert dBc (relative dB) to linear relative power ratio."""
    return 10.0 ** (dbc / 10.0)


def lin_ratio_to_db(r: float) -> float:
    """Convert linear relative power ratio to dBc."""
    if r <= 0.0:
        return -np.inf
    return 10.0 * math.log10(r)


def is_finite(x: Any) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def sideband_map(lo_hz: float, if_hz: float, sideband: str) -> float:
    """Desired mapping per mixer stage: sum or diff."""
    sb = sideband.lower()
    if sb == "sum":
        return lo_hz + if_hz
    if sb == "diff":
        return abs(lo_hz - if_hz)
    raise ValueError(f"Unknown sideband '{sideband}', expected 'sum' or 'diff'.")


def clamp_endpoint(x: float, x_min: float, x_max: float) -> Tuple[float, Optional[str]]:
    """Clamp x to [x_min, x_max], returning (clamped_x, side_flag low/high/None)."""
    if x < x_min:
        return x_min, "low"
    if x > x_max:
        return x_max, "high"
    return x, None


def in_range(x: float, lo: float, hi: float) -> bool:
    return (x >= lo) and (x <= hi)


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class Config:
    # Core frequencies
    lo1_hz: float = 5.55e9

    # Mixer desired mapping (system plan)
    mixer1_sideband: str = "sum"
    mixer2_sideband: str = "sum"

    # Coincidence combining
    freq_bin_tolerance_hz: float = 200e3
    cluster_using_measured_freq_if_available: bool = True

    # Noise-limited handling
    noise_mode: str = "optimistic"  # optimistic / conservative
    noise_margin_db: float = 6.0

    # Optional: how to mark noise-limited on interpolation
    # "any_source" (default): if any endpoint/source row used is noise_limited -> mark
    # "all_sources": mark only if all used source rows are noise_limited
    noise_flag_policy: str = "any_source"

    # Unknown contributors handling in coincidence combining:
    # - "ignore": sum finite contributors only; flag unknowns; also compute conservative_dbc with floor for reporting
    # - "conservative_floor": treat unknown contributors as unknown_floor_dbc in the actual sum
    unknown_combining_mode: str = "ignore"
    unknown_floor_dbc: float = SPEC_LIMIT_DBC

    # Output pruning (SAFE): applied after clustering, never before summation
    prune_clusters_below_dbc: Optional[float] = None  # e.g. -140 to shrink output; None = no prune

    # Missing-point policy (fixed order per spec)
    missing_point_policy: str = "interpolate_then_nearest_then_unknown"

    # Filter interpolation / out-of-range handling
    allow_extrapolate_filters: bool = False  # default per spec: no optimistic extrapolation
    # When out-of-range:
    # - "endpoint_clamp" (default): use nearest endpoint attenuation
    # - "pessimistic_min"          : use minimum attenuation observed in curve (worst-case for compliance)
    # - "pessimistic_zero"         : assume 0 dB attenuation out-of-range (very conservative)
    # If allow_extrapolate_filters=True and filter_oob_policy=="linear_extrapolation", linear extrapolation is used.
    filter_oob_policy: str = "endpoint_clamp"
    filter_eval_uses: str = "model"  # "model" | "measured_if_available"

    # Optional IF power scaling heuristic
    enable_if_power_scaling: bool = False
    p_if_meas_dbm: float = 0.0
    compression_warning: str = "heuristic_small_signal_only"

    # Mixer2 measurement reference point:
    # If True, treat Mixer2 intrinsic CSV levels as already post RF filter and inject at Node D (skip RF filter).
    mixer2_intrinsic_levels_are_post_rf_filter: bool = False

    # Dataset parsing (family keys)
    family_key_fields: Tuple[str, ...] = ("mode", "kind", "m", "n", "sign", "expression")

    mixer_if_col_candidates: Tuple[str, ...] = ("if_model_Hz", "if_set_Hz", "IF_Hz", "if_Hz")
    mixer_lo_col_candidates: Tuple[str, ...] = ("lo_model_Hz", "lo_set_Hz", "LO_Hz", "lo_Hz")

    mixer_level_col_candidates: Tuple[str, ...] = ("rel_dBc_vs_desired", "rel_dBc", "rel_dbc", "rel_level_dBc")
    mixer_measured_freq_col_candidates: Tuple[str, ...] = ("measured_freq_Hz", "meas_freq_Hz", "measured_frequency_Hz")
    mixer_expected_corr_freq_col_candidates: Tuple[str, ...] = (
        "expected_corr_freq_Hz", "expected_corrected_freq_Hz", "expected_freq_corr_Hz", "expected_freq_Hz"
    )

    # Exact-hit tolerance for matching query IF to dataset IF grid
    exact_hit_atol_hz: float = 1.0

    # LO2 channel validity checks (based on spec)
    enforce_lo2_channel_ranges: bool = True

    # Reporting helpers
    top_contributors_per_cluster: int = 25

    def validate(self) -> None:
        mp = self.missing_point_policy
        if mp != "interpolate_then_nearest_then_unknown":
            raise ValueError(
                f"missing_point_policy='{mp}' not supported (fixed order per spec)."
            )
        nm = self.noise_mode.lower()
        if nm not in ("optimistic", "conservative"):
            raise ValueError("noise_mode must be 'optimistic' or 'conservative'.")
        nfp = self.noise_flag_policy.lower()
        if nfp not in ("any_source", "all_sources"):
            raise ValueError("noise_flag_policy must be 'any_source' or 'all_sources'.")
        um = self.unknown_combining_mode.lower()
        if um not in ("ignore", "conservative_floor"):
            raise ValueError("unknown_combining_mode must be 'ignore' or 'conservative_floor'.")
        pol = self.filter_oob_policy.lower()
        if pol not in ("endpoint_clamp", "pessimistic_min", "pessimistic_zero", "linear_extrapolation"):
            raise ValueError(
                "filter_oob_policy must be one of endpoint_clamp, pessimistic_min, pessimistic_zero, linear_extrapolation."
            )
        if pol == "linear_extrapolation" and not self.allow_extrapolate_filters:
            # Not an error, but clearly inconsistent. We treat as endpoint_clamp.
            pass
        fe = self.filter_eval_uses.lower()
        if fe not in ("model", "measured_if_available"):
            raise ValueError("filter_eval_uses must be 'model' or 'measured_if_available'.")


# -----------------------------
# Filter curve model
# -----------------------------

@dataclass
class FilterQueryFlag:
    filter_name: str
    requested_hz: float
    used_hz: float
    oob: bool = False
    oob_side: Optional[str] = None  # low/high
    oob_handling: Optional[str] = None  # endpoint_clamp/pessimistic_min/pessimistic_zero/linear_extrapolation
    note: Optional[str] = None


@dataclass
class FilterEffect:
    """Records differential filter behavior used to update dBc."""
    filter_name: str
    desired_freq_hz: float
    spur_freq_requested_hz: float
    spur_freq_used_hz: float
    desired_att_db: float
    spur_att_db: float
    differential_db: float
    desired_query_flag: Dict[str, Any]
    spur_query_flag: Dict[str, Any]


@dataclass
class FilterCurve:
    name: str
    freq_hz: np.ndarray
    att_db: np.ndarray

    @staticmethod
    def from_csv(
        path: str,
        name: str,
        freq_col: Optional[str] = None,
        att_col: Optional[str] = None,
        freq_scale: float = 1.0,
    ) -> "FilterCurve":
        df = pd.read_csv(path)
        if freq_col is None:
            freq_col = df.columns[0]
        if att_col is None:
            att_col = df.columns[1]
        f = pd.to_numeric(df[freq_col], errors="coerce").astype(float).to_numpy() * float(freq_scale)
        a = pd.to_numeric(df[att_col], errors="coerce").astype(float).to_numpy()

        ok = np.isfinite(f) & np.isfinite(a)
        f = f[ok]
        a = a[ok]
        if f.size < 2:
            raise ValueError(f"FilterCurve '{name}' needs at least 2 finite points after parsing.")

        idx = np.argsort(f)
        return FilterCurve(name=name, freq_hz=f[idx], att_db=a[idx])

    def _linear_extrapolate(self, f_hz: float, side: str) -> float:
        # Use endpoint slope from first two or last two points.
        if side == "low":
            x0, x1 = float(self.freq_hz[0]), float(self.freq_hz[1])
            y0, y1 = float(self.att_db[0]), float(self.att_db[1])
        else:
            x0, x1 = float(self.freq_hz[-2]), float(self.freq_hz[-1])
            y0, y1 = float(self.att_db[-2]), float(self.att_db[-1])
        if x1 == x0:
            return y0
        m = (y1 - y0) / (x1 - x0)
        return float(y0 + m * (f_hz - x0))

    def attenuation_db(self, f_hz: float, cfg: Config) -> Tuple[float, FilterQueryFlag]:
        cfg.validate()
        f_hz = float(f_hz)
        fmin = float(self.freq_hz[0])
        fmax = float(self.freq_hz[-1])

        flag = FilterQueryFlag(
            filter_name=self.name,
            requested_hz=f_hz,
            used_hz=f_hz,
            oob=False,
            oob_side=None,
            oob_handling=None,
            note=None,
        )

        if (f_hz >= fmin) and (f_hz <= fmax):
            att = float(np.interp(f_hz, self.freq_hz, self.att_db))
            return att, flag

        # Out-of-range handling
        side = "low" if f_hz < fmin else "high"
        flag.oob = True
        flag.oob_side = side

        pol = cfg.filter_oob_policy.lower()
        if pol == "linear_extrapolation" and cfg.allow_extrapolate_filters:
            att = float(self._linear_extrapolate(f_hz, side))
            flag.used_hz = f_hz
            flag.oob_handling = "linear_extrapolation"
            flag.note = "Linear extrapolation enabled by user."
            return att, flag

        # Default / non-extrapolating policies (never optimistic without explicit enable)
        if pol == "pessimistic_zero":
            att = 0.0
            flag.used_hz = float(clamp_endpoint(f_hz, fmin, fmax)[0])
            flag.oob_handling = "pessimistic_zero"
            flag.note = "Out-of-range: assumed 0 dB attenuation (very conservative)."
            return float(att), flag

        if pol == "pessimistic_min":
            att = float(np.min(self.att_db))
            flag.used_hz = float(clamp_endpoint(f_hz, fmin, fmax)[0])
            flag.oob_handling = "pessimistic_min"
            flag.note = "Out-of-range: used minimum observed attenuation (conservative)."
            return float(att), flag

        # endpoint_clamp (default) or fallback
        used, _ = clamp_endpoint(f_hz, fmin, fmax)
        flag.used_hz = float(used)
        flag.oob_handling = "endpoint_clamp"
        if pol == "linear_extrapolation" and not cfg.allow_extrapolate_filters:
            flag.note = "Linear extrapolation requested but disabled; used endpoint clamp."
        att = float(np.interp(used, self.freq_hz, self.att_db))
        return att, flag


# -----------------------------
# Spur-family modeling
# -----------------------------

@dataclass
class EvalProvenance:
    kind: str  # measured / interpolated / nearest_filled / unknown
    source_row_indices: List[int] = field(default_factory=list)
    used_x_points_hz: List[float] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class SpurFamilyID:
    mode: str
    kind: str
    m: int
    n: int
    sign: int
    expression: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpurFamily:
    dataset_id: str
    family_id: SpurFamilyID

    lo_hz: float  # fixed per dataset/family
    x_if_hz: np.ndarray  # independent variable axis (IF1 for mixer1; IF2 for mixer2)
    y_dbc: np.ndarray  # rel dBc vs desired (may contain nan)

    noise_limited: np.ndarray  # bool aligned to x_if_hz

    measured_freq_hz: np.ndarray  # may contain nan
    expected_corr_freq_hz: np.ndarray  # may contain nan

    row_indices: np.ndarray  # original df indices aligned to x_if_hz

    def __post_init__(self):
        order = np.argsort(self.x_if_hz.astype(float))
        self.x_if_hz = self.x_if_hz[order]
        self.y_dbc = self.y_dbc[order]
        self.noise_limited = self.noise_limited[order]
        self.measured_freq_hz = self.measured_freq_hz[order]
        self.expected_corr_freq_hz = self.expected_corr_freq_hz[order]
        self.row_indices = self.row_indices[order]

    def analytic_frequency_hz(self, if_hz: float) -> float:
        """Analytic fallback frequency, absolute value to avoid negative Hz."""
        m = int(self.family_id.m)
        n = int(self.family_id.n)
        s = int(self.family_id.sign)
        return float(abs(m * self.lo_hz + s * n * float(if_hz)))

    def _eval_series_with_provenance(
        self,
        x_query: float,
        y: np.ndarray,
        valid_mask: np.ndarray,
        cfg: Config,
    ) -> Tuple[float, EvalProvenance]:
        """
        Evaluate y(x_query) using:
          - exact point (np.isclose) => measured
          - between nearest valid neighbors => interpolated
          - else nearest valid => nearest_filled
          - if no valid => unknown

        Adds provenance notes if query is outside the measured span of valid points.
        """
        cfg.validate()
        prov = EvalProvenance(kind="unknown", source_row_indices=[], used_x_points_hz=[])

        if not np.any(valid_mask):
            prov.kind = "unknown"
            prov.notes.append("No valid points in family for this field.")
            return float("nan"), prov

        x = self.x_if_hz.astype(float)
        x_valid = x[valid_mask]
        x_min = float(np.min(x_valid))
        x_max = float(np.max(x_valid))

        x_query = float(x_query)
        if x_query < x_min:
            prov.notes.append(f"Query IF below measured span (query={x_query:.6e} Hz, min_valid={x_min:.6e} Hz).")
        elif x_query > x_max:
            prov.notes.append(f"Query IF above measured span (query={x_query:.6e} Hz, max_valid={x_max:.6e} Hz).")

        # Exact hit using tolerance
        exact = np.where(np.isclose(x, x_query, rtol=0.0, atol=float(cfg.exact_hit_atol_hz)) & valid_mask)[0]
        if exact.size > 0:
            i = int(exact[0])
            prov.kind = "measured"
            prov.source_row_indices = [int(self.row_indices[i])]
            prov.used_x_points_hz = [float(x[i])]
            return float(y[i]), prov

        idx = int(np.searchsorted(x, x_query))
        left_candidates = range(idx - 1, -1, -1)
        right_candidates = range(idx, len(x))

        left = next((i for i in left_candidates if valid_mask[i]), None)
        right = next((i for i in right_candidates if valid_mask[i]), None)

        if left is not None and right is not None:
            x0, x1 = float(x[left]), float(x[right])
            y0, y1 = float(y[left]), float(y[right])
            if x1 == x0:
                use = left
                prov.kind = "nearest_filled"
                prov.source_row_indices = [int(self.row_indices[use])]
                prov.used_x_points_hz = [float(x[use])]
                prov.notes.append("Degenerate interpolation interval; used nearest.")
                return float(y[use]), prov

            t = (x_query - x0) / (x1 - x0)
            val = y0 + t * (y1 - y0)
            prov.kind = "interpolated"
            prov.source_row_indices = [int(self.row_indices[left]), int(self.row_indices[right])]
            prov.used_x_points_hz = [x0, x1]
            return float(val), prov

        use = left if left is not None else right
        if use is not None:
            prov.kind = "nearest_filled"
            prov.source_row_indices = [int(self.row_indices[use])]
            prov.used_x_points_hz = [float(x[use])]
            if left is None:
                prov.notes.append("No valid left neighbor; used right nearest.")
            if right is None:
                prov.notes.append("No valid right neighbor; used left nearest.")
            return float(y[use]), prov

        prov.kind = "unknown"
        prov.notes.append("No valid neighbor found (unexpected).")
        return float("nan"), prov

    def _noise_meta_for_sources(self, prov: EvalProvenance, cfg: Config) -> Dict[str, Any]:
        """
        Decide whether this evaluated value should be marked noise-limited.
        Policy:
          - any_source (default): if any source rows are noise_limited -> mark
          - all_sources: mark only if all source rows are noise_limited (when there are sources)
        """
        cfg.validate()
        if not prov.source_row_indices:
            return {
                "noise_limited": False,
                "noise_sources_row_indices": [],
                "noise_mode_applied": None,
                "noise_margin_db": None,
                "policy": cfg.noise_flag_policy,
            }

        flags: List[bool] = []
        noise_sources: List[int] = []
        for ridx in prov.source_row_indices:
            pos = np.where(self.row_indices == ridx)[0]
            if pos.size:
                p = int(pos[0])
                nl = bool(self.noise_limited[p])
                flags.append(nl)
                if nl:
                    noise_sources.append(int(ridx))

        if cfg.noise_flag_policy.lower() == "all_sources":
            noise_flag = bool(flags) and all(flags)
        else:
            noise_flag = any(flags)

        return {
            "noise_limited": bool(noise_flag),
            "noise_sources_row_indices": noise_sources,
            "noise_mode_applied": None,
            "noise_margin_db": None,
            "policy": cfg.noise_flag_policy,
        }

    def evaluate_level_dbc(self, if_hz: float, cfg: Config) -> Tuple[float, EvalProvenance, Dict[str, Any]]:
        """Evaluate level in dBc vs desired with missing-point rules + noise-limited handling."""
        cfg.validate()
        y = self.y_dbc.astype(float)
        valid = np.isfinite(y)

        val, prov = self._eval_series_with_provenance(if_hz, y, valid, cfg)
        noise_meta = self._noise_meta_for_sources(prov, cfg)

        if noise_meta["noise_limited"] and np.isfinite(val):
            nm = cfg.noise_mode.lower()
            if nm == "optimistic":
                noise_meta["noise_mode_applied"] = "optimistic_upper_bound"
                noise_meta["noise_margin_db"] = 0.0
            elif nm == "conservative":
                val = float(val + cfg.noise_margin_db)
                noise_meta["noise_mode_applied"] = "conservative_uplift"
                noise_meta["noise_margin_db"] = float(cfg.noise_margin_db)

        return float(val), prov, noise_meta

    def evaluate_measured_freq_hz(self, if_hz: float, cfg: Config) -> Tuple[float, EvalProvenance]:
        """Evaluate measured_freq_hz for diagnostics / coincidence. Uses same fill rules."""
        y = self.measured_freq_hz.astype(float)
        valid = np.isfinite(y)
        return self._eval_series_with_provenance(if_hz, y, valid, cfg)

    def evaluate_expected_corr_freq_hz(self, if_hz: float, cfg: Config) -> Tuple[float, EvalProvenance]:
        """Evaluate expected_corr_freq_hz as primary frequency model reference."""
        y = self.expected_corr_freq_hz.astype(float)
        valid = np.isfinite(y)
        return self._eval_series_with_provenance(if_hz, y, valid, cfg)

    def frequency_model_hz(self, if_hz: float, cfg: Config) -> Tuple[float, Dict[str, Any]]:
        """
        Primary: expected_corr_freq_Hz vs IF with fill rules.
        Fallback: analytic |m*LO ± n*IF| if expected_corr unavailable.
        """
        f_corr, prov = self.evaluate_expected_corr_freq_hz(if_hz, cfg)
        if np.isfinite(f_corr):
            return float(f_corr), {
                "model": "expected_corr_interp",
                "provenance": asdict(prov),
            }
        return float(self.analytic_frequency_hz(if_hz)), {
            "model": "analytic_fallback_abs",
            "provenance": asdict(prov),
        }


# -----------------------------
# Dataset loading + validation
# -----------------------------

def _pick_first_present(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the candidate columns exist: {list(candidates)}")


def _infer_dataset_sideband(
    df: pd.DataFrame,
    lo_col: str,
    if_col: str,
) -> Optional[str]:
    """
    Infer whether the dataset desired mapping is SUM or DIFF based on 'desired' rows.

    Returns:
      "sum", "diff", or None (if cannot infer).
    """
    if "kind" not in df.columns:
        return None
    d = df[df["kind"].astype(str).str.lower() == "desired"].copy()
    if d.empty:
        return None

    if all(c in d.columns for c in ["m", "n", "sign", "expected_freq_Hz"]):
        d["m"] = pd.to_numeric(d["m"], errors="coerce")
        d["n"] = pd.to_numeric(d["n"], errors="coerce")
        d["sign"] = pd.to_numeric(d["sign"], errors="coerce")
        d = d[(d["m"] == 1) & (d["n"] == 1)]
        if d.empty:
            return None

    lo = pd.to_numeric(d[lo_col], errors="coerce").astype(float).median()
    iff = pd.to_numeric(d[if_col], errors="coerce").astype(float).median()
    if "expected_freq_Hz" not in d.columns:
        return None
    exp = pd.to_numeric(d["expected_freq_Hz"], errors="coerce").astype(float).median()

    if not (np.isfinite(lo) and np.isfinite(iff) and np.isfinite(exp)):
        return None

    sum_f = lo + iff
    diff_f = abs(lo - iff)

    if abs(exp - sum_f) <= abs(exp - diff_f):
        return "sum"
    return "diff"


def load_mixer_families(
    path: str,
    dataset_id: str,
    cfg: Config,
    expect_if_range_hz: Optional[Tuple[float, float]] = None,
    expected_sideband: Optional[str] = None,
    expected_lo_hz: Optional[float] = None,
    expected_lo_tol_hz: float = 50e6,
) -> Tuple[List[SpurFamily], Dict[str, Any]]:
    """
    Load a mixer CSV, group rows into spur families, and return SpurFamily objects.

    Returns: (families, dataset_summary)

    Required:
      family keys: mode, kind, m, n, sign, expression
      IF, LO (picked from candidates)
      level column (picked from candidates)
      noise_limited
      measured freq column (picked from candidates)
      expected-corr freq column (picked from candidates)
    """
    cfg.validate()
    df = pd.read_csv(path)

    if_col = _pick_first_present(df, cfg.mixer_if_col_candidates)
    lo_col = _pick_first_present(df, cfg.mixer_lo_col_candidates)
    lvl_col = _pick_first_present(df, cfg.mixer_level_col_candidates)
    meas_f_col = _pick_first_present(df, cfg.mixer_measured_freq_col_candidates)
    exp_f_col = _pick_first_present(df, cfg.mixer_expected_corr_freq_col_candidates)

    inferred = _infer_dataset_sideband(df, lo_col, if_col)
    if expected_sideband is not None and inferred is not None:
        if inferred.lower() != expected_sideband.lower():
            raise ValueError(
                f"{dataset_id}: dataset desired mapping inferred '{inferred}', but config expected '{expected_sideband}'. "
                f"Check mixer sideband setting or CSV selection."
            )

    required = list(cfg.family_key_fields) + [
        if_col, lo_col, lvl_col, "noise_limited", meas_f_col, exp_f_col
    ]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' in {path}")

    # Basic numeric coercions
    df["m"] = pd.to_numeric(df["m"], errors="coerce").fillna(0).astype(int)
    df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0).astype(int)
    df["sign"] = pd.to_numeric(df["sign"], errors="coerce").fillna(1).astype(int)
    df["noise_limited"] = pd.to_numeric(df["noise_limited"], errors="coerce").fillna(0).astype(int)

    dataset_summary: Dict[str, Any] = {
        "path": path,
        "dataset_id": dataset_id,
        "if_col": if_col,
        "lo_col": lo_col,
        "level_col": lvl_col,
        "measured_freq_col": meas_f_col,
        "expected_corr_freq_col": exp_f_col,
        "inferred_sideband": inferred,
    }

    # IF sanity check
    if expect_if_range_hz is not None:
        lo_, hi_ = expect_if_range_hz
        finite_if = pd.to_numeric(df[if_col], errors="coerce")
        finite_if = finite_if[np.isfinite(finite_if.astype(float))]
        if finite_if.notna().any():
            med = float(finite_if.median())
            dataset_summary["if_median_hz"] = med
            if not (lo_ <= med <= hi_):
                raise ValueError(
                    f"{dataset_id}: median IF column '{if_col}' = {med:.3e} Hz outside expected "
                    f"range [{lo_:.3e}, {hi_:.3e}] Hz. Likely wrong IF axis selection."
                )

    # LO sanity check (especially important for Mixer2 LO2 CSVs)
    finite_lo = pd.to_numeric(df[lo_col], errors="coerce").astype(float)
    finite_lo = finite_lo[np.isfinite(finite_lo)]
    if finite_lo.size > 0:
        lo_med = float(np.median(finite_lo))
        dataset_summary["lo_median_hz"] = lo_med
        if expected_lo_hz is not None and abs(lo_med - float(expected_lo_hz)) > float(expected_lo_tol_hz):
            raise ValueError(
                f"{dataset_id}: LO median {lo_med:.6e} Hz does not match expected {float(expected_lo_hz):.6e} Hz "
                f"within tol {float(expected_lo_tol_hz):.3e} Hz. Likely wrong CSV for this LO2."
            )

    families: List[SpurFamily] = []
    group_cols = list(cfg.family_key_fields)

    for key, g in df.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, key))
        fid = SpurFamilyID(
            mode=str(key_dict.get("mode", "")),
            kind=str(key_dict.get("kind", "")),
            m=int(key_dict.get("m", 0)),
            n=int(key_dict.get("n", 0)),
            sign=int(key_dict.get("sign", 1)),
            expression=str(key_dict.get("expression", "")),
        )

        lo_vals = pd.to_numeric(g[lo_col], errors="coerce").astype(float)
        lo_hz = float(lo_vals.median()) if np.isfinite(lo_vals.median()) else float("nan")

        x = pd.to_numeric(g[if_col], errors="coerce").astype(float).to_numpy()
        y = pd.to_numeric(g[lvl_col], errors="coerce").astype(float).to_numpy()
        noise = pd.to_numeric(g["noise_limited"], errors="coerce").fillna(0).astype(int).to_numpy().astype(bool)

        meas_f = pd.to_numeric(g[meas_f_col], errors="coerce").astype(float).to_numpy()
        exp_f = pd.to_numeric(g[exp_f_col], errors="coerce").astype(float).to_numpy()

        row_idx = g.index.to_numpy()

        tmp = pd.DataFrame({
            "x": x, "y": y, "noise": noise, "meas_f": meas_f, "exp_f": exp_f, "row_idx": row_idx
        })

        # Drop non-finite IF-axis rows (critical for safe interpolation/searchsorted)
        tmp = tmp[np.isfinite(tmp["x"].astype(float))]
        if tmp.empty:
            continue

        # If multiple rows share the same x, keep the WORST (highest dBc) among finite (conservative for compliance)
        rows = []
        for xv, gg in tmp.groupby("x", dropna=False):
            finite = gg[np.isfinite(gg["y"].to_numpy())]
            if len(finite) > 0:
                chosen = finite.iloc[int(np.argmax(finite["y"].to_numpy()))]
            else:
                chosen = gg.iloc[0]
            rows.append(chosen)
        dfg = pd.DataFrame(rows)

        families.append(SpurFamily(
            dataset_id=dataset_id,
            family_id=fid,
            lo_hz=lo_hz,
            x_if_hz=dfg["x"].astype(float).to_numpy(),
            y_dbc=dfg["y"].astype(float).to_numpy(),
            noise_limited=dfg["noise"].astype(bool).to_numpy(),
            measured_freq_hz=dfg["meas_f"].astype(float).to_numpy(),
            expected_corr_freq_hz=dfg["exp_f"].astype(float).to_numpy(),
            row_indices=dfg["row_idx"].astype(int).to_numpy(),
        ))

    dataset_summary["num_families"] = int(len(families))
    return families, dataset_summary


# -----------------------------
# Tone objects + propagation
# -----------------------------

@dataclass
class ToneContributor:
    node: str                      # A/B/C/D
    freq_hz: float                 # model/corrected frequency at this node (for reporting)
    dbc: float                     # relative to desired carrier (system reference) (may be NaN -> unknown)

    # Traceability / origin
    origin: str                    # mixer1_family / translated_residual / mixer2_family / etc.
    dataset_id: Optional[str] = None
    family_id: Optional[Dict[str, Any]] = None
    lineage_row_indices: List[int] = field(default_factory=list)

    # Quality flags
    provenance: Optional[Dict[str, Any]] = None
    noise: Optional[Dict[str, Any]] = None
    if_power_scaled: Optional[Dict[str, Any]] = None

    # Filter flags and effects
    filter_flags: List[Dict[str, Any]] = field(default_factory=list)
    filter_effects: List[Dict[str, Any]] = field(default_factory=list)

    # Processing history
    processing_steps: List[str] = field(default_factory=list)

    # Diagnostics
    freq_model_hz: Optional[float] = None      # predicted/corrected model freq at this node
    freq_measured_hz: Optional[float] = None   # measured freq at this node (if available)

    def summarize(self) -> str:
        if self.family_id:
            return f"{self.origin}:{self.family_id.get('kind','')}:{self.family_id.get('expression','')}"
        return self.origin


@dataclass
class ToneCluster:
    node: str
    freq_hz: float

    # Combined level actually used for compliance reporting (depends on cfg.unknown_combining_mode)
    dbc: float

    # Always computed:
    dbc_known_only: float
    dbc_conservative_unknown_floor: Optional[float]

    contributors: List[ToneContributor]
    coincident: bool = False
    has_unknown_contributors: bool = False
    unknown_contributors_count: int = 0
    unknown_floor_dbc: Optional[float] = None
    unknown_combining_mode: str = "ignore"

    def top_contributors(self, n: int = 5) -> List[ToneContributor]:
        return sorted(
            self.contributors,
            key=lambda c: c.dbc if np.isfinite(c.dbc) else -np.inf,
            reverse=True
        )[:n]


def _tone_freq_for_filter_eval(t: ToneContributor, cfg: Config) -> float:
    """
    Choose which frequency to use for filter attenuation evaluation.
    - "model": always use t.freq_hz
    - "measured_if_available": prefer t.freq_measured_hz when finite
    """
    if cfg.filter_eval_uses.lower() == "measured_if_available":
        if is_finite(t.freq_measured_hz):
            return float(t.freq_measured_hz)  # type: ignore[arg-type]
    return float(t.freq_hz)


def apply_differential_filter(
    tones: List[ToneContributor],
    filt: FilterCurve,
    desired_freq_hz: float,
    cfg: Config,
    out_node: str,
    step_name: str,
) -> List[ToneContributor]:
    """
    Apply differential attenuation to each tone relative to desired frequency:

      dBc_after = dBc_before + (A(desired) - A(spur))

    Records filter effects and out-of-range flags.
    """
    a_des, flag_des = filt.attenuation_db(float(desired_freq_hz), cfg)

    out: List[ToneContributor] = []
    for t in tones:
        spur_req = _tone_freq_for_filter_eval(t, cfg)
        a_spur, flag_sp = filt.attenuation_db(float(spur_req), cfg)

        dbc_after = float(t.dbc + (a_des - a_spur)) if np.isfinite(t.dbc) else float("nan")

        effect = FilterEffect(
            filter_name=filt.name,
            desired_freq_hz=float(desired_freq_hz),
            spur_freq_requested_hz=float(spur_req),
            spur_freq_used_hz=float(flag_sp.used_hz),
            desired_att_db=float(a_des),
            spur_att_db=float(a_spur),
            differential_db=float(a_des - a_spur),
            desired_query_flag=asdict(flag_des),
            spur_query_flag=asdict(flag_sp),
        )

        new = ToneContributor(
            node=out_node,
            freq_hz=float(t.freq_hz),
            dbc=dbc_after,
            origin=t.origin,
            dataset_id=t.dataset_id,
            family_id=t.family_id,
            lineage_row_indices=list(t.lineage_row_indices),
            provenance=t.provenance,
            noise=t.noise,
            if_power_scaled=t.if_power_scaled,
            filter_flags=list(t.filter_flags),
            filter_effects=list(t.filter_effects) + [asdict(effect)],
            processing_steps=list(t.processing_steps) + [step_name],
            freq_model_hz=t.freq_model_hz,
            freq_measured_hz=t.freq_measured_hz,
        )

        # Preserve both desired and spur query flags for traceability.
        if flag_des.oob:
            new.filter_flags.append(asdict(flag_des))
        if flag_sp.oob:
            new.filter_flags.append(asdict(flag_sp))

        out.append(new)
    return out


def translate_tones_mixer2(
    tones_b: List[ToneContributor],
    lo2_hz: float,
    mixer2_sideband: str,
    cfg: Config,
) -> List[ToneContributor]:
    """
    Translate Node B tones to Node C via Mixer2 LO2 mapping.
    dBc is preserved (pre-RF-filter). Frequencies AND diagnostic measured/model frequencies are translated.
    """
    out: List[ToneContributor] = []
    for t in tones_b:
        f_rf = float(sideband_map(float(lo2_hz), float(t.freq_hz), mixer2_sideband))

        meas = None
        if cfg.cluster_using_measured_freq_if_available and is_finite(t.freq_measured_hz):
            meas = float(sideband_map(float(lo2_hz), float(t.freq_measured_hz), mixer2_sideband))

        fmod = None
        if is_finite(t.freq_model_hz):
            fmod = float(sideband_map(float(lo2_hz), float(t.freq_model_hz), mixer2_sideband))

        out.append(ToneContributor(
            node="C",
            freq_hz=f_rf,
            dbc=float(t.dbc),
            origin="translated_residual" if t.origin != "translated_residual" else t.origin,
            dataset_id=t.dataset_id,
            family_id=t.family_id,
            lineage_row_indices=list(t.lineage_row_indices),
            provenance=t.provenance,
            noise=t.noise,
            if_power_scaled=t.if_power_scaled,
            filter_flags=list(t.filter_flags),
            filter_effects=list(t.filter_effects),
            processing_steps=list(t.processing_steps) + ["mixer2_translate"],
            freq_model_hz=fmod,
            freq_measured_hz=meas,
        ))
    return out


def coincidence_cluster(
    contributors: List[ToneContributor],
    cfg: Config,
    node: str,
) -> List[ToneCluster]:
    """
    Cluster by frequency within tolerance and sum linear power ratios in-bin.

    IMPORTANT: Anchor-based clustering (no chaining).
    Handles unknown contributors:
      - always computes known-only combined dBc
      - if unknowns exist, also computes conservative dBc assuming unknown_floor_dbc for each unknown
      - depending on cfg.unknown_combining_mode, chooses which combined dBc becomes cluster.dbc
    """
    cfg.validate()
    if not contributors:
        return []

    def cluster_freq(c: ToneContributor) -> float:
        if cfg.cluster_using_measured_freq_if_available and is_finite(c.freq_measured_hz):
            return float(c.freq_measured_hz)  # type: ignore[arg-type]
        return float(c.freq_hz)

    items = sorted(contributors, key=cluster_freq)
    tol = float(cfg.freq_bin_tolerance_hz)

    clusters: List[List[ToneContributor]] = []
    cur: List[ToneContributor] = [items[0]]
    anchor_f = cluster_freq(items[0])

    for c in items[1:]:
        f = cluster_freq(c)
        if abs(f - anchor_f) <= tol:
            cur.append(c)
        else:
            clusters.append(cur)
            cur = [c]
            anchor_f = f
    clusters.append(cur)

    out: List[ToneCluster] = []
    for group in clusters:
        finite = [c for c in group if np.isfinite(c.dbc)]
        unknown = [c for c in group if not np.isfinite(c.dbc)]

        # known-only sum
        ratios_known = [db_to_lin_ratio(float(c.dbc)) for c in finite]
        dbc_known = lin_ratio_to_db(float(np.sum(ratios_known))) if ratios_known else float("nan")

        dbc_cons = None
        if unknown:
            # conservative (unknown floor)
            ratios_cons = list(ratios_known) + [db_to_lin_ratio(float(cfg.unknown_floor_dbc))] * len(unknown)
            dbc_cons = lin_ratio_to_db(float(np.sum(ratios_cons))) if ratios_cons else float("nan")

        # which combined is used as primary
        if cfg.unknown_combining_mode.lower() == "conservative_floor" and unknown:
            dbc_use = float(dbc_cons) if dbc_cons is not None else float("nan")
        else:
            dbc_use = float(dbc_known)

        # Representative frequency: use the worst (highest dBc) *finite* contributor if possible, else first unknown.
        if finite:
            top = max(finite, key=lambda x: x.dbc)
        else:
            top = group[0]
        rep = cluster_freq(top)

        out.append(ToneCluster(
            node=node,
            freq_hz=float(rep),
            dbc=float(dbc_use),
            dbc_known_only=float(dbc_known),
            dbc_conservative_unknown_floor=(float(dbc_cons) if dbc_cons is not None else None),
            contributors=group,
            coincident=(len(group) > 1),
            has_unknown_contributors=bool(len(unknown) > 0),
            unknown_contributors_count=int(len(unknown)),
            unknown_floor_dbc=(float(cfg.unknown_floor_dbc) if unknown else None),
            unknown_combining_mode=str(cfg.unknown_combining_mode),
        ))
    return out


# -----------------------------
# System Evaluator
# -----------------------------

@dataclass
class SystemInputs:
    if1_hz: float
    lo2_hz: float
    if1_power_dbm: Optional[float] = None


@dataclass
class PointReport:
    inputs: Dict[str, Any]
    desired: Dict[str, Any]
    validations: Dict[str, Any]
    tones_node_d: List[Dict[str, Any]]  # includes desired + clustered spurs (including carrier-bin spurs)
    worst: Optional[Dict[str, Any]] = None
    dataset_summaries: Optional[Dict[str, Any]] = None


def _lo2_channel_bounds(lo2_hz: float) -> Optional[Tuple[float, float]]:
    """
    Per spec:
      LO2=21 GHz -> 27.5–29.0 GHz
      LO2=22 GHz -> 28.5–30.0 GHz
      LO2=23 GHz -> 29.5–31.0 GHz
    """
    ghz = int(round(float(lo2_hz) / 1e9))
    if ghz == 21:
        return 27.5e9, 29.0e9
    if ghz == 22:
        return 28.5e9, 30.0e9
    if ghz == 23:
        return 29.5e9, 31.0e9
    return None


def _stage_path(origin: str, steps: List[str]) -> str:
    base = origin
    if steps:
        return " -> ".join([base] + steps)
    return base


def _serialize_contributor(c: ToneContributor) -> Dict[str, Any]:
    return {
        "node": c.node,
        "freq_hz": float(c.freq_hz),
        "dbc": float(c.dbc) if np.isfinite(c.dbc) else None,
        "origin": c.origin,
        "stage_path": _stage_path(c.origin, c.processing_steps),
        "dataset_id": c.dataset_id,
        "family_id": c.family_id,
        "lineage_row_indices": c.lineage_row_indices,
        "provenance": c.provenance,
        "noise": c.noise,
        "if_power_scaled": c.if_power_scaled,
        "filter_flags": c.filter_flags,
        "filter_effects": c.filter_effects,
        "processing_steps": c.processing_steps,
        "freq_model_hz": float(c.freq_model_hz) if is_finite(c.freq_model_hz) else None,
        "freq_measured_hz": float(c.freq_measured_hz) if is_finite(c.freq_measured_hz) else None,
    }


class BUCSpurPredictor:
    def __init__(
        self,
        cfg: Config,
        if2_filter: FilterCurve,
        rf_filter: FilterCurve,
        mixer1_families: List[SpurFamily],
        mixer2_families_by_lo2: Dict[float, List[SpurFamily]],
        dataset_summaries: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = cfg
        self.cfg.validate()
        self.if2_filter = if2_filter
        self.rf_filter = rf_filter
        self.mixer1_families = mixer1_families
        self.mixer2_families_by_lo2 = mixer2_families_by_lo2
        self.dataset_summaries = dataset_summaries or {}

    def _apply_if_power_scaling(
        self,
        dbc: float,
        n_if: int,
        cfg: Config,
        if1_power_dbm: Optional[float],
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        Optional small-signal spur-order heuristic:
          dBc_new = dBc_meas + (|n|-1)*ΔP_IF
        where n is the IF coefficient in the spur expression.
        """
        if not cfg.enable_if_power_scaling:
            return dbc, None
        if if1_power_dbm is None:
            return dbc, {
                "enabled": True,
                "applied": False,
                "reason": "if1_power_dbm not provided",
                "formula": "dBc_new = dBc_meas + (|n|-1)*ΔP_IF",
            }

        delta = float(if1_power_dbm - cfg.p_if_meas_dbm)
        scaled = float(dbc + (abs(int(n_if)) - 1) * delta)
        meta = {
            "enabled": True,
            "applied": True,
            "p_if_meas_dbm": float(cfg.p_if_meas_dbm),
            "p_if_user_dbm": float(if1_power_dbm),
            "delta_p_db": float(delta),
            "n_if": int(n_if),
            "compression_warning": cfg.compression_warning,
            "formula": "dBc_new = dBc_meas + (|n|-1)*ΔP_IF",
        }
        return scaled, meta

    @staticmethod
    def _compute_required_extra_attenuation_for_group(
        contributors: List[ToneContributor],
        group_selector: Callable[[ToneContributor], bool],
        target_dbc: float = SPEC_LIMIT_DBC,
    ) -> Dict[str, Any]:
        """
        Compute required EXTRA attenuation (dB) applied only to a subset of contributors
        to bring the combined tone to <= target_dbc.

        NOTE: This is a diagnostic "attenuate at this node" metric; it is not the same as
        "measured filter differential attenuation vs frequency".

        Returns:
          {
            "possible": bool,
            "required_extra_db": float|None,
            "reason": str|None,
            "p_target": float,
            "p_affected": float,
            "p_unaffected": float
          }
        """
        finite = [c for c in contributors if np.isfinite(c.dbc)]
        if not finite:
            return {
                "possible": False,
                "required_extra_db": None,
                "reason": "No finite contributor levels.",
                "p_target": db_to_lin_ratio(target_dbc),
                "p_affected": 0.0,
                "p_unaffected": 0.0,
            }

        p_target = db_to_lin_ratio(float(target_dbc))

        affected = [c for c in finite if bool(group_selector(c))]
        unaffected = [c for c in finite if not bool(group_selector(c))]

        p_aff = float(np.sum([db_to_lin_ratio(float(c.dbc)) for c in affected])) if affected else 0.0
        p_unaff = float(np.sum([db_to_lin_ratio(float(c.dbc)) for c in unaffected])) if unaffected else 0.0

        if p_aff <= 0.0:
            return {
                "possible": False,
                "required_extra_db": None,
                "reason": "No affected contributors (p_affected=0).",
                "p_target": p_target,
                "p_affected": p_aff,
                "p_unaffected": p_unaff,
            }

        if p_unaff >= p_target:
            return {
                "possible": False,
                "required_extra_db": None,
                "reason": "Unaffected contributors alone exceed target; cannot fix via this attenuation point.",
                "p_target": p_target,
                "p_affected": p_aff,
                "p_unaffected": p_unaff,
            }

        rhs = (p_target - p_unaff) / p_aff
        rhs = max(min(rhs, 1.0), 1e-300)
        required = float(-10.0 * math.log10(rhs))
        required = max(0.0, required)

        return {
            "possible": True,
            "required_extra_db": required,
            "reason": None,
            "p_target": p_target,
            "p_affected": p_aff,
            "p_unaffected": p_unaff,
        }

    def evaluate_point(self, inputs: SystemInputs) -> PointReport:
        cfg = self.cfg
        cfg.validate()

        if1 = float(inputs.if1_hz)
        lo2 = float(inputs.lo2_hz)
        if1_p = inputs.if1_power_dbm

        # Desired frequencies
        if2_des = float(sideband_map(cfg.lo1_hz, if1, cfg.mixer1_sideband))
        rf_des = float(sideband_map(lo2, if2_des, cfg.mixer2_sideband))

        # Validations (channel sanity)
        validations: Dict[str, Any] = {}
        bounds = _lo2_channel_bounds(lo2)
        if cfg.enforce_lo2_channel_ranges and bounds is not None:
            lo_b, hi_b = bounds
            validations["rf_desired_in_lo2_channel"] = bool(in_range(rf_des, lo_b, hi_b))
            validations["lo2_channel_bounds_hz"] = {"lo_hz": float(lo_b), "hi_hz": float(hi_b)}
        else:
            validations["rf_desired_in_lo2_channel"] = None
            validations["lo2_channel_bounds_hz"] = None

        # ---------------- Node A: Mixer1 spurs (exclude desired) ----------------
        node_a: List[ToneContributor] = []
        for fam in self.mixer1_families:
            if fam.family_id.kind.lower() == "desired":
                continue

            lvl, prov_lvl, noise_meta = fam.evaluate_level_dbc(if1, cfg)
            ifp_meta = None
            if np.isfinite(lvl):
                lvl, ifp_meta = self._apply_if_power_scaling(lvl, fam.family_id.n, cfg, if1_p)

            f_model, f_model_meta = fam.frequency_model_hz(if1, cfg)
            f_meas, prov_f = fam.evaluate_measured_freq_hz(if1, cfg)

            node_a.append(ToneContributor(
                node="A",
                freq_hz=float(f_model),
                dbc=float(lvl) if np.isfinite(lvl) else float("nan"),
                origin="mixer1_family",
                dataset_id=fam.dataset_id,
                family_id=fam.family_id.as_dict(),
                lineage_row_indices=list(prov_lvl.source_row_indices),
                provenance={
                    "level": asdict(prov_lvl),
                    "freq_model": f_model_meta,
                    "freq_measured": asdict(prov_f),
                },
                noise=noise_meta,
                if_power_scaled=ifp_meta,
                filter_flags=[],
                filter_effects=[],
                processing_steps=[],
                freq_model_hz=float(f_model),
                freq_measured_hz=float(f_meas) if np.isfinite(f_meas) else None,
            ))

        # ---------------- Node B: IF2 filter differential ----------------
        node_b = apply_differential_filter(
            tones=node_a,
            filt=self.if2_filter,
            desired_freq_hz=if2_des,
            cfg=cfg,
            out_node="B",
            step_name="if2_filter",
        )

        # ---------------- Node C: translated residuals ----------------
        translated = translate_tones_mixer2(node_b, lo2, cfg.mixer2_sideband, cfg)

        # ---------------- Mixer2 intrinsic spurs (exclude desired) ----------------
        mixer2_fams = self.mixer2_families_by_lo2.get(lo2)
        if mixer2_fams is None:
            raise ValueError(f"No mixer2 dataset loaded for LO2={lo2:.6e} Hz")

        intrinsic_c: List[ToneContributor] = []
        intrinsic_d: List[ToneContributor] = []

        for fam in mixer2_fams:
            if fam.family_id.kind.lower() == "desired":
                continue

            # Mixer2 intrinsic families are evaluated vs IF2_des (dataset IF axis)
            lvl, prov_lvl, noise_meta = fam.evaluate_level_dbc(if2_des, cfg)

            # Optional IF power scaling:
            # If enabled, this uses IF1 delta as a heuristic proxy for IF2 drive (explicitly flagged).
            ifp_meta = None
            if np.isfinite(lvl) and cfg.enable_if_power_scaling:
                lvl, ifp_meta = self._apply_if_power_scaling(lvl, fam.family_id.n, cfg, if1_p)
                if ifp_meta:
                    ifp_meta["note"] = "Heuristic: applied IF1 power delta as proxy for IF2 drive in Mixer2 dataset."

            f_model, f_model_meta = fam.frequency_model_hz(if2_des, cfg)
            f_meas, prov_f = fam.evaluate_measured_freq_hz(if2_des, cfg)

            t = ToneContributor(
                node="C" if not cfg.mixer2_intrinsic_levels_are_post_rf_filter else "D",
                freq_hz=float(f_model),
                dbc=float(lvl) if np.isfinite(lvl) else float("nan"),
                origin="mixer2_family",
                dataset_id=fam.dataset_id,
                family_id=fam.family_id.as_dict(),
                lineage_row_indices=list(prov_lvl.source_row_indices),
                provenance={
                    "level": asdict(prov_lvl),
                    "freq_model": f_model_meta,
                    "freq_measured": asdict(prov_f),
                    "intrinsic_levels_are_post_rf_filter": bool(cfg.mixer2_intrinsic_levels_are_post_rf_filter),
                },
                noise=noise_meta,
                if_power_scaled=ifp_meta,
                filter_flags=[],
                filter_effects=[],
                processing_steps=[],
                freq_model_hz=float(f_model),
                freq_measured_hz=float(f_meas) if np.isfinite(f_meas) else None,
            )

            if cfg.mixer2_intrinsic_levels_are_post_rf_filter:
                intrinsic_d.append(t)
            else:
                intrinsic_c.append(t)

        # Node C contributors are translated residuals + intrinsic pre-RF-filter spurs
        node_c = translated + intrinsic_c

        # ---------------- Node D: RF filter differential (skip if already post-filter) ----------------
        node_d_from_c = apply_differential_filter(
            tones=node_c,
            filt=self.rf_filter,
            desired_freq_hz=rf_des,
            cfg=cfg,
            out_node="D",
            step_name="rf_filter",
        )

        node_d_contrib = node_d_from_c + intrinsic_d  # intrinsic_d already post RF filter per config

        # ---------------- Coincidence combining at Node D ----------------
        clusters = coincidence_cluster(node_d_contrib, cfg=cfg, node="D")

        # Optional SAFE pruning: after clustering only (affects output size, not sums or worst-case)
        if cfg.prune_clusters_below_dbc is not None:
            pr = float(cfg.prune_clusters_below_dbc)
            clusters = [c for c in clusters if (not np.isfinite(c.dbc)) or (float(c.dbc) >= pr)]

        # Build report lines:
        # Include desired as a separate line at 0 dBc (never power-summed).
        report_lines: List[Dict[str, Any]] = []
        report_lines.append({
            "node": "D",
            "tone_freq_hz": float(rf_des),
            "tone_level_dbc": 0.0,
            "margin_to_spec_db": None,  # spec applies to spurs, not the carrier
            "required_additional_attenuation_db": 0.0,
            "coincident": False,
            "num_contributors": 0,
            "contributors": [],
            "contributors_top": [],
            "is_desired": True,
            "coincident_with_carrier_bin": False,
            "has_unknown_contributors": False,
            "unknown_contributors_count": 0,
            "tone_level_dbc_known_only": 0.0,
            "tone_level_dbc_conservative_unknown_floor": None,
            "unknown_combining_mode": cfg.unknown_combining_mode,
            "unknown_floor_dbc": float(cfg.unknown_floor_dbc),
        })

        worst: Optional[Dict[str, Any]] = None

        def is_carrier_bin(f_hz: float) -> bool:
            return abs(float(f_hz) - float(rf_des)) <= float(cfg.freq_bin_tolerance_hz)

        # For each cluster, compute compliance and diagnostic metrics
        for cl in clusters:
            dbc_used = float(cl.dbc)
            dbc_known = float(cl.dbc_known_only)
            dbc_cons = float(cl.dbc_conservative_unknown_floor) if cl.dbc_conservative_unknown_floor is not None else None

            margin = float(SPEC_LIMIT_DBC - dbc_used) if np.isfinite(dbc_used) else float("nan")
            req = float(max(0.0, dbc_used - SPEC_LIMIT_DBC)) if np.isfinite(dbc_used) else float("nan")

            # Also compute conservative margin/req for reporting (even if unknown_combining_mode == ignore)
            margin_cons = None
            req_cons = None
            if dbc_cons is not None and np.isfinite(dbc_cons):
                margin_cons = float(SPEC_LIMIT_DBC - dbc_cons)
                req_cons = float(max(0.0, dbc_cons - SPEC_LIMIT_DBC))

            coinc_with_carrier = bool(is_carrier_bin(cl.freq_hz))

            def has_step(step: str) -> Callable[[ToneContributor], bool]:
                return lambda c: (c.processing_steps is not None) and (step in c.processing_steps)

            # Renamed: avoid implying this is true measured differential-filter attribution.
            attenuation_node_diagnostics = {
                "target_dbc": float(SPEC_LIMIT_DBC),
                "required_extra_attenuation_if_applied_at_if2_filter_db": None,
                "required_extra_attenuation_if_applied_at_rf_filter_db": None,
                "unattributed_minimum_db": None,
                "notes": [],
            }

            # Only attempt node-attenuation diagnostics if all contributors are finite.
            all_finite = all(np.isfinite(c.dbc) for c in cl.contributors if c.dbc is not None)
            if all_finite and np.isfinite(dbc_used):
                rf_sol = self._compute_required_extra_attenuation_for_group(
                    cl.contributors,
                    group_selector=has_step("rf_filter"),
                    target_dbc=SPEC_LIMIT_DBC,
                )
                if rf_sol["possible"]:
                    attenuation_node_diagnostics["required_extra_attenuation_if_applied_at_rf_filter_db"] = float(rf_sol["required_extra_db"])
                else:
                    attenuation_node_diagnostics["notes"].append(f"RF-node diagnostic not possible: {rf_sol['reason']}")

                if2_sol = self._compute_required_extra_attenuation_for_group(
                    cl.contributors,
                    group_selector=has_step("if2_filter"),
                    target_dbc=SPEC_LIMIT_DBC,
                )
                if if2_sol["possible"]:
                    attenuation_node_diagnostics["required_extra_attenuation_if_applied_at_if2_filter_db"] = float(if2_sol["required_extra_db"])
                else:
                    attenuation_node_diagnostics["notes"].append(f"IF2-node diagnostic not possible: {if2_sol['reason']}")

                post_only = [c for c in cl.contributors if not ("rf_filter" in c.processing_steps)]
                if post_only:
                    p_post = float(np.sum([db_to_lin_ratio(float(c.dbc)) for c in post_only if np.isfinite(c.dbc)]))
                    if p_post > 0.0:
                        dbc_post = lin_ratio_to_db(p_post)
                        if dbc_post > SPEC_LIMIT_DBC:
                            attenuation_node_diagnostics["unattributed_minimum_db"] = float(dbc_post - SPEC_LIMIT_DBC)
                            attenuation_node_diagnostics["notes"].append("Post-RF contributors alone violate spec; filters cannot fix entirely.")
            else:
                if cl.has_unknown_contributors:
                    attenuation_node_diagnostics["notes"].append("Node-attenuation diagnostics skipped due to unknown contributor level(s).")
                else:
                    attenuation_node_diagnostics["notes"].append("Node-attenuation diagnostics skipped due to non-finite contributor level(s).")

            # Spec requirement: report ALL contributors
            contributors_all = [_serialize_contributor(c) for c in cl.contributors]
            contributors_top = [_serialize_contributor(c) for c in cl.top_contributors(int(cfg.top_contributors_per_cluster))]

            line = {
                "node": "D",
                "tone_freq_hz": float(cl.freq_hz),
                "tone_level_dbc": dbc_used,
                "margin_to_spec_db": margin,
                "required_additional_attenuation_db": req,
                "coincident": bool(cl.coincident),
                "num_contributors": int(len(cl.contributors)),
                "is_desired": False,
                "coincident_with_carrier_bin": coinc_with_carrier,

                # Unknown visibility
                "has_unknown_contributors": bool(cl.has_unknown_contributors),
                "unknown_contributors_count": int(cl.unknown_contributors_count),
                "unknown_combining_mode": str(cl.unknown_combining_mode),
                "unknown_floor_dbc": float(cfg.unknown_floor_dbc),

                # Always-provided alternate sums for traceability/compliance review
                "tone_level_dbc_known_only": dbc_known,
                "tone_level_dbc_conservative_unknown_floor": dbc_cons,
                "margin_to_spec_db_conservative_unknown_floor": margin_cons,
                "required_additional_attenuation_db_conservative_unknown_floor": req_cons,

                "required_attenuation_node_diagnostics": attenuation_node_diagnostics,
                "contributors": contributors_all,
                "contributors_top": contributors_top,
            }
            report_lines.append(line)

            # Worst spur selection: use dbc_used (which may already include conservative unknown floor
            # if unknown_combining_mode=conservative_floor). Always carry unknown flags for visibility.
            if np.isfinite(dbc_used):
                if worst is None or dbc_used > worst["tone_level_dbc"]:
                    top = cl.top_contributors(1)[0] if cl.contributors else None
                    worst = {
                        "tone_freq_hz": float(cl.freq_hz),
                        "tone_level_dbc": float(dbc_used),
                        "margin_to_spec_db": float(margin),
                        "required_additional_attenuation_db": float(req),
                        "coincident": bool(cl.coincident),
                        "coincident_with_carrier_bin": bool(coinc_with_carrier),
                        "num_contributors": int(len(cl.contributors)),
                        "has_unknown_contributors": bool(cl.has_unknown_contributors),
                        "unknown_contributors_count": int(cl.unknown_contributors_count),
                        "unknown_combining_mode": str(cl.unknown_combining_mode),
                        "unknown_floor_dbc": float(cfg.unknown_floor_dbc),
                        "tone_level_dbc_known_only": float(cl.dbc_known_only),
                        "tone_level_dbc_conservative_unknown_floor": float(cl.dbc_conservative_unknown_floor)
                        if cl.dbc_conservative_unknown_floor is not None else None,
                        "top_contributor": {
                            "origin": top.origin if top else None,
                            "stage_path": _stage_path(top.origin, top.processing_steps) if top else None,
                            "dataset_id": top.dataset_id if top else None,
                            "family_id": top.family_id if top else None,
                            "lineage_row_indices": top.lineage_row_indices if top else None,
                            "dbc": float(top.dbc) if (top and np.isfinite(top.dbc)) else None,
                        } if top else None,
                    }

        # Sort: desired first, then worst spurs first
        desired_line = report_lines[0]
        spur_lines = report_lines[1:]
        spur_lines.sort(
            key=lambda r: (r["tone_level_dbc"] if np.isfinite(r["tone_level_dbc"]) else -1e9),
            reverse=True
        )
        report_lines = [desired_line] + spur_lines

        return PointReport(
            inputs={
                "if1_hz": float(if1),
                "if1_power_dbm": float(if1_p) if if1_p is not None else None,
                "lo1_hz": float(cfg.lo1_hz),
                "lo2_hz": float(lo2),
                "mixer1_sideband": cfg.mixer1_sideband,
                "mixer2_sideband": cfg.mixer2_sideband,
                "noise_mode": cfg.noise_mode,
                "noise_margin_db": float(cfg.noise_margin_db),
                "noise_flag_policy": cfg.noise_flag_policy,
                "freq_bin_tolerance_hz": float(cfg.freq_bin_tolerance_hz),
                "cluster_using_measured_freq_if_available": bool(cfg.cluster_using_measured_freq_if_available),
                "mixer2_intrinsic_levels_are_post_rf_filter": bool(cfg.mixer2_intrinsic_levels_are_post_rf_filter),
                "filter_eval_uses": cfg.filter_eval_uses,
                "allow_extrapolate_filters": bool(cfg.allow_extrapolate_filters),
                "filter_oob_policy": cfg.filter_oob_policy,
                "unknown_combining_mode": cfg.unknown_combining_mode,
                "unknown_floor_dbc": float(cfg.unknown_floor_dbc),
                "prune_clusters_below_dbc": float(cfg.prune_clusters_below_dbc) if cfg.prune_clusters_below_dbc is not None else None,
                "enforce_lo2_channel_ranges": bool(cfg.enforce_lo2_channel_ranges),
                "top_contributors_per_cluster": int(cfg.top_contributors_per_cluster),
            },
            desired={
                "if2_desired_hz": float(if2_des),
                "rf_desired_hz": float(rf_des),
            },
            validations=validations,
            tones_node_d=report_lines,
            worst=worst,
            dataset_summaries=self.dataset_summaries,
        )

    def sweep_if1(
        self,
        lo2_hz: float,
        if1_start_hz: float = 950e6,
        if1_stop_hz: float = 2450e6,
        step_hz: float = 10e6,
        if1_power_dbm: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Sweep IF1 and return worst-case spur report."""
        lo2_hz = float(lo2_hz)
        ifs = np.arange(float(if1_start_hz), float(if1_stop_hz) + 0.5 * float(step_hz), float(step_hz))

        worst_overall = None
        worst_by_if: List[Dict[str, Any]] = []

        for if1 in ifs:
            rpt = self.evaluate_point(SystemInputs(if1_hz=float(if1), lo2_hz=lo2_hz, if1_power_dbm=if1_power_dbm))
            w = rpt.worst
            worst_by_if.append({
                "if1_hz": float(if1),
                "worst_tone_freq_hz": w["tone_freq_hz"] if w else None,
                "worst_tone_level_dbc": w["tone_level_dbc"] if w else None,
                "margin_to_spec_db": w["margin_to_spec_db"] if w else None,
                "coincident": w["coincident"] if w else None,
                "coincident_with_carrier_bin": w["coincident_with_carrier_bin"] if w else None,
                "has_unknown_contributors": w["has_unknown_contributors"] if w else None,
                "unknown_contributors_count": w["unknown_contributors_count"] if w else None,
                "unknown_combining_mode": w["unknown_combining_mode"] if w else None,
                "top_contributor": w["top_contributor"] if w else None,
            })

            if w and (worst_overall is None or w["tone_level_dbc"] > worst_overall["tone_level_dbc"]):
                worst_overall = {
                    "if1_hz": float(if1),
                    **w,
                }

        return {
            "lo2_hz": float(lo2_hz),
            "if1_start_hz": float(if1_start_hz),
            "if1_stop_hz": float(if1_stop_hz),
            "step_hz": float(step_hz),
            "worst_overall": worst_overall,
            "worst_by_if": worst_by_if,
        }


# -----------------------------
# CLI
# -----------------------------

def build_predictor_from_args(args: argparse.Namespace) -> BUCSpurPredictor:
    cfg = Config(
        lo1_hz=float(args.lo1),
        mixer1_sideband=args.mixer1_sideband,
        mixer2_sideband=args.mixer2_sideband,
        freq_bin_tolerance_hz=float(args.bin_tol),
        noise_mode=args.noise_mode,
        noise_margin_db=float(args.noise_margin),
        noise_flag_policy=str(args.noise_flag_policy),
        enable_if_power_scaling=bool(args.enable_if_power_scaling),
        p_if_meas_dbm=float(args.p_if_meas),
        mixer2_intrinsic_levels_are_post_rf_filter=bool(args.mixer2_post_rf),
        exact_hit_atol_hz=float(args.exact_hit_atol_hz),
        filter_eval_uses=str(args.filter_eval_uses),
        allow_extrapolate_filters=bool(args.allow_filter_extrapolation),
        filter_oob_policy=str(args.filter_oob_policy),
        unknown_combining_mode=str(args.unknown_mode),
        unknown_floor_dbc=float(args.unknown_floor_dbc),
        prune_clusters_below_dbc=(float(args.prune_clusters_below_dbc) if args.prune_clusters_below_dbc is not None else None),
        enforce_lo2_channel_ranges=bool(args.enforce_lo2_channel_ranges),
        top_contributors_per_cluster=int(args.top_contrib),
    )
    cfg.validate()

    if2_filt = FilterCurve.from_csv(
        args.if2_filter_csv, name="IF2_filter",
        freq_col=args.if2_filter_freq_col, att_col=args.if2_filter_att_col,
        freq_scale=float(args.if2_filter_freq_scale),
    )
    rf_filt = FilterCurve.from_csv(
        args.rf_filter_csv, name="RF_filter",
        freq_col=args.rf_filter_freq_col, att_col=args.rf_filter_att_col,
        freq_scale=float(args.rf_filter_freq_scale),
    )

    dataset_summaries: Dict[str, Any] = {}

    mixer1, sum1 = load_mixer_families(
        args.mixer1_csv,
        dataset_id="mixer1",
        cfg=cfg,
        expect_if_range_hz=(950e6, 2450e6),
        expected_sideband=cfg.mixer1_sideband,
        expected_lo_hz=cfg.lo1_hz,  # LO1 fixed
        expected_lo_tol_hz=50e6,
    )
    dataset_summaries["mixer1"] = sum1

    mix2_map: Dict[float, List[SpurFamily]] = {}
    for lo2_hz, path in [(21e9, args.mixer2_csv_21), (22e9, args.mixer2_csv_22), (23e9, args.mixer2_csv_23)]:
        if path:
            fams, s = load_mixer_families(
                path,
                dataset_id=f"mixer2_lo2_{int(lo2_hz/1e9)}GHz",
                cfg=cfg,
                expect_if_range_hz=(6.0e9, 9.0e9),
                expected_sideband=cfg.mixer2_sideband,
                expected_lo_hz=float(lo2_hz),
                expected_lo_tol_hz=50e6,
            )
            mix2_map[float(lo2_hz)] = fams
            dataset_summaries[f"mixer2_{int(lo2_hz/1e9)}GHz"] = s

    return BUCSpurPredictor(
        cfg=cfg,
        if2_filter=if2_filt,
        rf_filter=rf_filt,
        mixer1_families=mixer1,
        mixer2_families_by_lo2=mix2_map,
        dataset_summaries=dataset_summaries,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="BUC spur/system prediction tool (revised)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(pp: argparse.ArgumentParser) -> None:
        pp.add_argument("--lo1", type=float, default=5.55e9)
        pp.add_argument("--lo2", type=float, required=True, help="LO2 in Hz (21e9/22e9/23e9)")

        pp.add_argument("--mixer1-sideband", dest="mixer1_sideband", type=str, default="sum", choices=["sum", "diff"])
        pp.add_argument("--mixer2-sideband", dest="mixer2_sideband", type=str, default="sum", choices=["sum", "diff"])

        pp.add_argument("--bin-tol", dest="bin_tol", type=float, default=200e3)

        pp.add_argument("--noise-mode", dest="noise_mode", type=str, default="optimistic", choices=["optimistic", "conservative"])
        pp.add_argument("--noise-margin", dest="noise_margin", type=float, default=6.0)
        pp.add_argument("--noise-flag-policy", dest="noise_flag_policy", type=str, default="any_source",
                        choices=["any_source", "all_sources"])

        pp.add_argument("--enable-if-power-scaling", dest="enable_if_power_scaling", action="store_true")
        pp.add_argument("--p-if-meas", dest="p_if_meas", type=float, default=0.0)

        pp.add_argument("--mixer2-post-rf", dest="mixer2_post_rf", action="store_true",
                        help="Treat Mixer2 intrinsic CSV levels as already post RF filter (inject at Node D)")

        pp.add_argument("--exact-hit-atol-hz", dest="exact_hit_atol_hz", type=float, default=1.0,
                        help="Tolerance for treating a query IF as an exact measurement grid hit")

        pp.add_argument("--filter-eval-uses", dest="filter_eval_uses", type=str, default="model",
                        choices=["model", "measured_if_available"],
                        help="Which tone frequency to use when evaluating filter attenuation")

        # Filter out-of-range controls
        pp.add_argument("--allow-filter-extrapolation", dest="allow_filter_extrapolation", action="store_true",
                        help="Allow filter out-of-range extrapolation (disabled by default per spec)")
        pp.add_argument("--filter-oob-policy", dest="filter_oob_policy", type=str, default="endpoint_clamp",
                        choices=["endpoint_clamp", "pessimistic_min", "pessimistic_zero", "linear_extrapolation"],
                        help="Out-of-range filter handling policy (default endpoint_clamp)")

        # Unknown contributor handling
        pp.add_argument("--unknown-mode", dest="unknown_mode", type=str, default="ignore",
                        choices=["ignore", "conservative_floor"],
                        help="How to handle unknown contributor levels in coincidence sums")
        pp.add_argument("--unknown-floor-dbc", dest="unknown_floor_dbc", type=float, default=SPEC_LIMIT_DBC,
                        help="dBc used for unknown contributors when conservative handling is applied/reported")

        pp.add_argument("--prune-clusters-below-dbc", dest="prune_clusters_below_dbc", type=float, default=None,
                        help="SAFE output pruning (after clustering): drop clusters below this dBc")

        pp.add_argument("--enforce-lo2-channel-ranges", dest="enforce_lo2_channel_ranges",
                        action="store_true", default=True,
                        help="Validate rf_desired against spec-defined LO2 channel ranges (default: enabled)")
        pp.add_argument("--no-enforce-lo2-channel-ranges", dest="enforce_lo2_channel_ranges",
                        action="store_false",
                        help="Disable LO2 channel validation checks")

        pp.add_argument("--top-contrib", dest="top_contrib", type=int, default=25,
                        help="How many contributors to repeat in contributors_top (ALL contributors always included).")

        # Filter CSVs
        pp.add_argument("--if2-filter-csv", dest="if2_filter_csv", type=str, required=True)
        pp.add_argument("--rf-filter-csv", dest="rf_filter_csv", type=str, required=True)

        pp.add_argument("--if2-filter-freq-col", dest="if2_filter_freq_col", type=str, default=None)
        pp.add_argument("--if2-filter-att-col", dest="if2_filter_att_col", type=str, default=None)
        pp.add_argument("--if2-filter-freq-scale", dest="if2_filter_freq_scale", type=float, default=1.0)

        pp.add_argument("--rf-filter-freq-col", dest="rf_filter_freq_col", type=str, default=None)
        pp.add_argument("--rf-filter-att-col", dest="rf_filter_att_col", type=str, default=None)
        pp.add_argument("--rf-filter-freq-scale", dest="rf_filter_freq_scale", type=float, default=1.0)

        # Mixer CSVs
        pp.add_argument("--mixer1-csv", dest="mixer1_csv", type=str, required=True)
        pp.add_argument("--mixer2-csv-21", dest="mixer2_csv_21", type=str, default=None)
        pp.add_argument("--mixer2-csv-22", dest="mixer2_csv_22", type=str, default=None)
        pp.add_argument("--mixer2-csv-23", dest="mixer2_csv_23", type=str, default=None)

    pp_point = sub.add_parser("point", help="Evaluate a single operating point")
    add_common(pp_point)
    pp_point.add_argument("--if1", type=float, required=True, help="IF1 frequency in Hz")
    pp_point.add_argument("--if1-power-dbm", dest="if1_power_dbm", type=float, default=None)
    pp_point.add_argument("--out", type=str, default=None, help="Output JSON file (optional)")

    pp_sweep = sub.add_parser("sweep", help="Sweep IF1 and report worst-case spurs")
    add_common(pp_sweep)
    pp_sweep.add_argument("--if1-start", dest="if1_start", type=float, default=950e6)
    pp_sweep.add_argument("--if1-stop", dest="if1_stop", type=float, default=2450e6)
    pp_sweep.add_argument("--step", type=float, default=10e6)
    pp_sweep.add_argument("--if1-power-dbm", dest="if1_power_dbm", type=float, default=None)
    pp_sweep.add_argument("--out", type=str, default=None, help="Output JSON file (optional)")

    args = p.parse_args()
    predictor = build_predictor_from_args(args)

    if args.cmd == "point":
        rpt = predictor.evaluate_point(SystemInputs(
            if1_hz=float(args.if1),
            lo2_hz=float(args.lo2),
            if1_power_dbm=args.if1_power_dbm,
        ))
        payload = asdict(rpt)
        txt = json.dumps(payload, indent=2)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(txt)
        else:
            print(txt)

    elif args.cmd == "sweep":
        res = predictor.sweep_if1(
            lo2_hz=float(args.lo2),
            if1_start_hz=float(args.if1_start),
            if1_stop_hz=float(args.if1_stop),
            step_hz=float(args.step),
            if1_power_dbm=args.if1_power_dbm,
        )
        txt = json.dumps(res, indent=2)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(txt)
        else:
            print(txt)


if __name__ == "__main__":
    main()