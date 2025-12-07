#!/usr/bin/env python3
"""
2D LO/IF mixer spur sweep for R&S FSW + SMBV100A + SMA100B

- Controls:
    * R&S SMBV100A CW generator as IF source
    * R&S SMA100B CW generator as LO source
    * R&S FSW spectrum analyzer for spur measurements
- Sweeps LO and IF according to user-defined ranges
- For each (LO, IF) point:
    * Sets SMA100B LO frequency + level
    * Sets SMBV100A IF frequency + level
    * Runs a narrow-span spur scan on the FSW
    * Writes a per-point spur CSV (optional)
- After the sweep:
    * Aggregates all spur results into one master CSV (written directly from memory)
    * (Optional) runs spur overlap detection and remeasurement pass

Plotting has been intentionally omitted; only CSV output is generated.

Requires:
    pip install RsFsw RsSmbv RsSmab
"""

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import statistics

from RsFsw import RsFsw, enums as fsw_enums, repcap
from RsSmbv import RsSmbv, enums as smbv_enums
from RsSmab import RsSmab, enums as smab_enums


# ---------------------------------------------------------------------------
# Data classes for spur definitions / results
# ---------------------------------------------------------------------------

@dataclass
class SpurDef:
    """Definition of a mixer product to be measured."""
    kind: str            # "desired", "image", "LO_leak", "LO_harmonic",
                         # "IF_leak", "IF_harmonic", "spur"
    m: int
    n: int
    sign: int            # +1 or -1 for ±n·IF
    expression: str      # e.g. "1*LO - 2*IF"
    expected_freq_hz: float


@dataclass
class SpurResult:
    """Measurement result of one mixer product."""
    kind: str
    m: int
    n: int
    sign: int
    expression: str
    expected_freq_hz: float
    measured_freq_hz: float
    power_dbm: float
    rel_dbc: Optional[float]        # 0.0 for desired, None if measurement failed
    error: Optional[str] = None     # Error message if measurement failed
    noise_limited: bool = False     # True if flagged as noise-limited


# ---------------------------------------------------------------------------
# Utility functions shared by all measurements
# ---------------------------------------------------------------------------

def dbm_list_to_avg_dbm(powers_dbm: List[float]) -> float:
    """Average in linear (mW), return in dBm."""
    if not powers_dbm:
        return float("nan")
    lin_mw = [10.0 ** (p / 10.0) for p in powers_dbm]
    avg_mw = sum(lin_mw) / len(lin_mw)
    return 10.0 * math.log10(avg_mw)


def _dedupe_spurs_by_freq(spurs: List[SpurDef], dedupe_hz: float) -> List[SpurDef]:
    """
    Optional frequency de-duplication.

    If dedupe_hz > 0, group spurs into frequency bins of width dedupe_hz and keep
    a single representative in each bin, preferring more 'important' kinds:

        desired > image > leakage > harmonics > spur
    """
    if dedupe_hz is None or dedupe_hz <= 0.0:
        return spurs

    priority = {
        "desired": 5,
        "image": 4,
        "LO_leak": 3,
        "IF_leak": 3,
        "LO_harmonic": 2,
        "IF_harmonic": 2,
        "spur": 1,
    }

    bins: Dict[int, SpurDef] = {}
    for s in spurs:
        key = int(round(s.expected_freq_hz / dedupe_hz))
        existing = bins.get(key)
        if existing is None:
            bins[key] = s
        else:
            if priority.get(s.kind, 0) > priority.get(existing.kind, 0):
                bins[key] = s

    return list(bins.values())


def build_spur_list(
    f_lo_hz: float,
    f_if_hz: float,
    m_max: int,
    n_max: int,
    desired_mode: str,
    f_min_hz: float,
    f_max_hz: float,
    dedupe_hz: float = 0.0,
) -> List[SpurDef]:
    """
    Build list of mixer products m·LO ± n·IF within [f_min, f_max].

    desired_mode: "lo+if" or "lo-if" (desired product is 1*LO ± 1*IF).
    """
    spur_list: List[SpurDef] = []

    desired_mode_l = desired_mode.lower()
    if desired_mode_l == "lo+if":
        desired_sign = +1
    elif desired_mode_l == "lo-if":
        desired_sign = -1
    else:
        raise ValueError("desired_mode must be 'lo+if' or 'lo-if'")

    # Desired product: 1*LO ± 1*IF
    f_desired = abs(1 * f_lo_hz + desired_sign * 1 * f_if_hz)
    if f_min_hz <= f_desired <= f_max_hz:
        spur_list.append(
            SpurDef(
                kind="desired",
                m=1,
                n=1,
                sign=desired_sign,
                expression=f"1*LO {'+' if desired_sign > 0 else '-'} 1*IF",
                expected_freq_hz=f_desired,
            )
        )

    # All other products
    for m in range(0, m_max + 1):
        for n in range(0, n_max + 1):
            # Skip DC (0*LO ± 0*IF)
            if m == 0 and n == 0:
                continue

            # For pure LO or pure IF terms, ± yields the same |freq|, so only sign = +1
            if m == 0 or n == 0:
                sign_values = (+1,)
            else:
                sign_values = (+1, -1)

            for sign in sign_values:
                # Desired product already added explicitly
                if m == 1 and n == 1 and sign == desired_sign:
                    continue

                freq = abs(m * f_lo_hz + sign * n * f_if_hz)
                if freq < f_min_hz or freq > f_max_hz:
                    continue

                # Classification
                if m == 1 and n == 1 and sign == -desired_sign:
                    kind = "image"
                elif m == 1 and n == 0:
                    kind = "LO_leak"
                elif m > 1 and n == 0:
                    kind = "LO_harmonic"
                elif m == 0 and n == 1:
                    kind = "IF_leak"
                elif m == 0 and n > 1:
                    kind = "IF_harmonic"
                else:
                    kind = "spur"

                expr = f"{m}*LO {'+' if sign > 0 else '-'} {n}*IF"
                spur_list.append(
                    SpurDef(
                        kind=kind,
                        m=m,
                        n=n,
                        sign=sign,
                        expression=expr,
                        expected_freq_hz=freq,
                    )
                )

    # Optional de-duplication by frequency
    spur_list = _dedupe_spurs_by_freq(spur_list, dedupe_hz)

    # Sort by expected frequency
    spur_list.sort(key=lambda s: s.expected_freq_hz)

    # Ensure desired, if present, is first
    for i, s in enumerate(spur_list):
        if s.kind == "desired":
            spur_list.insert(0, spur_list.pop(i))
            break

    return spur_list


def configure_span_and_bw(
    fsw: RsFsw,
    center_hz: float,
    span_hz: float,
    rbw_hz: float,
    vbw_hz: Optional[float],
) -> None:
    """
    Set frequency span and RBW/VBW around a given center frequency.
    Uses start/stop so it works even if center/span are not used in your current setup.

    Enforces:
        span_hz >= 1 Hz
        rbw_hz  >= 1 Hz
    """
    # Clamp to sensible minimums
    span_hz = max(span_hz, 1.0)
    rbw_hz = max(rbw_hz, 1.0)

    f_start = max(center_hz - span_hz / 2.0, 0.0)
    f_stop = center_hz + span_hz / 2.0

    fsw.sense.frequency.start.set(f_start)  # SENSe:FREQuency:STARt
    fsw.sense.frequency.stop.set(f_stop)    # SENSe:FREQuency:STOP

    # RBW
    fsw.sense.bandwidth.resolution.set(rbw_hz)  # [SENS]:BANDwidth:RESolution

    # Optional VBW (if provided, we set it; otherwise leave instrument's setting)
    if vbw_hz is not None and vbw_hz > 0:
        fsw.sense.bandwidth.video.set(vbw_hz)   # [SENS]:BANDwidth:VIDeo


def _freq_error_limit(span_hz: float, rbw_hz: float) -> float:
    """
    Frequency error limit for sanity checks.

    We allow the larger of:
        - 10 * RBW
        - 10% of span

    This ties the allowed deviation to resolution as well as zoom.
    """
    return max(10.0 * rbw_hz, 0.1 * span_hz)


def measure_single_tone(
    fsw: RsFsw,
    center_hz: float,
    span_hz: float,
    rbw_hz: float,
    vbw_hz: Optional[float],
    avg_sweeps: int,
    sweep_timeout_ms: Optional[int],
    marker_mode: str = "peak",
) -> Tuple[float, float]:
    """
    Zoom in on 'center_hz', run 'avg_sweeps' sweeps,
    and return (marker_freq_hz, avg_power_dBm).

    marker_mode:
        "peak"    - marker to maximum in span
        "at-freq" - marker at 'center_hz' (useful when the spur of interest
                    is not the biggest in the span)

    Averaging is done in Python (linear units) to avoid having to configure
    the FSW’s averaging mode in detail.
    """
    marker_mode = marker_mode.lower()
    if marker_mode not in ("peak", "at-freq"):
        raise ValueError("marker_mode must be 'peak' or 'at-freq'")

    configure_span_and_bw(fsw, center_hz, span_hz, rbw_hz, vbw_hz)

    # Estimate sweep timeout if not given, based on instrument sweep time.
    if sweep_timeout_ms is None:
        try:
            sweep_time_s = fsw.sense.sweep.time.get()
            sweep_timeout_ms = int(max(1000.0 * sweep_time_s * 2.0, 1000.0))
        except Exception:
            # Fallback to a conservative default if sweep time is not accessible
            sweep_timeout_ms = 5000

    # Single-sweep mode
    fsw.initiate.continuous.set(False)  # INIT:CONT OFF

    powers_dbm: List[float] = []
    last_marker_freq_hz = center_hz

    # Use Trace 1, Marker 1
    fsw.calculate.marker.trace.set(1, repcap.Window.Nr1, repcap.Marker.Nr1)

    sweeps = max(avg_sweeps, 1)
    for _ in range(sweeps):
        # Trigger one sweep and wait for completion
        fsw.initiate.immediate_with_opc(sweep_timeout_ms)  # INIT:IMM; *OPC?

        if marker_mode == "peak":
            # Peak search (maximum in span)
            fsw.calculate.marker.maximum.peak.set(
                repcap.Window.Nr1, repcap.Marker.Nr1
            )
        else:
            # Place marker at the requested frequency position
            fsw.calculate.marker.x.set(
                center_hz,
                repcap.Window.Nr1,
                repcap.Marker.Nr1,
            )

        last_marker_freq_hz = fsw.calculate.marker.x.get(
            repcap.Window.Nr1,
            repcap.Marker.Nr1,
        )
        marker_power_dbm = fsw.calculate.marker.y.get(
            repcap.Window.Nr1,
            repcap.Marker.Nr1,
        )

        powers_dbm.append(marker_power_dbm)

    avg_power_dbm = dbm_list_to_avg_dbm(powers_dbm)
    return last_marker_freq_hz, avg_power_dbm


def _resolve_marker_mode(global_mode: str, kind: str) -> str:
    """
    Determine the marker mode for a given spur kind.

    global_mode:
        "peak" or "at-freq": force that mode for all kinds.
        "auto": choose per kind:
            - LO_leak / IF_leak: at-freq
            - all others: peak
    """
    gm = global_mode.lower()
    if gm in ("peak", "at-freq"):
        return gm

    # auto mode: choose based on kind
    if kind in ("LO_leak", "IF_leak"):
        return "at-freq"
    # for desired, image, harmonics, generic spurs: peak search
    return "peak"


def _to_optional_float(value: Any) -> Optional[float]:
    """Parse CSV field to float, returning None if empty/NaN/unparseable."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    try:
        s = str(value).strip()
        if s == "":
            return None
        v = float(s)
        if math.isnan(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Core spur scan for a single (LO, IF) operating point (FSW reused)
# ---------------------------------------------------------------------------

def measure_spurs_for_point(
    fsw: RsFsw,
    lo_hz: float,
    if_hz: float,
    mode: str,
    m_max: int,
    n_max: int,
    f_min_hz: float,
    f_max_hz: float,
    dedupe_freq_hz: float,
    span_hz: float,
    rbw_hz: float,
    vbw_hz: Optional[float],
    avg_sweeps: int,
    timeout_ms: Optional[int],
    marker_mode: str,
    min_power_db: Optional[float],
    min_desired_db: Optional[float],
) -> List[SpurResult]:
    """
    Perform a spur scan for a single (LO, IF) mixer operating point using
    an already open FSW session. Returns a list of SpurResult.
    """
    if mode == "lo-if" and lo_hz <= if_hz:
        print(
            "WARNING: LO <= IF while using lo-if (down-conversion). "
            "Check that your LO and IF frequencies are correct."
        )

    # Build list of mixer products
    spur_defs = build_spur_list(
        f_lo_hz=lo_hz,
        f_if_hz=if_hz,
        m_max=m_max,
        n_max=n_max,
        desired_mode=mode,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
        dedupe_hz=dedupe_freq_hz,
    )
    if not spur_defs or spur_defs[0].kind != "desired":
        raise RuntimeError(
            "Desired product is not within the [f_min, f_max] range "
            "after generation / de-duplication."
        )

    print(f"Total products to measure (including desired): {len(spur_defs)}")

    results: List[SpurResult] = []

    # Measure desired product first
    desired_def = spur_defs[0]
    print(
        f"\nMeasuring desired product: {desired_def.expression} "
        f"at ~{desired_def.expected_freq_hz / 1e9:.6f} GHz"
    )

    try:
        d_marker_mode = _resolve_marker_mode(marker_mode, desired_def.kind)
        d_meas_freq, d_power_dbm = measure_single_tone(
            fsw=fsw,
            center_hz=desired_def.expected_freq_hz,
            span_hz=span_hz,
            rbw_hz=rbw_hz,
            vbw_hz=vbw_hz,
            avg_sweeps=avg_sweeps,
            sweep_timeout_ms=timeout_ms,
            marker_mode=d_marker_mode,
        )
    except Exception as ex:
        print(f"ERROR: Failed to measure desired product: {ex}")
        raise

    freq_error_desired_hz = abs(d_meas_freq - desired_def.expected_freq_hz)
    limit_desired_hz = _freq_error_limit(span_hz, rbw_hz)
    print(
        f"  Desired product measured at {d_meas_freq/1e9:.6f} GHz, "
        f"{d_power_dbm:.2f} dBm (Δf = {freq_error_desired_hz/1e3:.1f} kHz, "
        f"limit ≈ {limit_desired_hz/1e3:.1f} kHz)"
    )

    # Frequency sanity check for the desired product
    if freq_error_desired_hz > limit_desired_hz:
        raise RuntimeError(
            "Desired product marker is far from expected frequency "
            f"(Δf = {freq_error_desired_hz/1e6:.3f} MHz, "
            f"limit ≈ {limit_desired_hz/1e6:.3f} MHz). "
            "Check your LO/IF settings, span, RBW, and marker mode."
        )

    # Optional minimum desired level (in dBm)
    if min_desired_db is not None and d_power_dbm < min_desired_db:
        raise RuntimeError(
            f"Desired product level {d_power_dbm:.2f} dBm is below "
            f"--min-desired-db threshold of {min_desired_db:.2f} dBm. "
            "Aborting spur scan for this point."
        )

    desired_result = SpurResult(
        kind=desired_def.kind,
        m=desired_def.m,
        n=desired_def.n,
        sign=desired_def.sign,
        expression=desired_def.expression,
        expected_freq_hz=desired_def.expected_freq_hz,
        measured_freq_hz=d_meas_freq,
        power_dbm=d_power_dbm,
        rel_dbc=0.0,   # reference
        error=None,
        noise_limited=False,
    )
    results.append(desired_result)

    # Measure all other products
    for spur_def in spur_defs[1:]:
        f_nom = spur_def.expected_freq_hz
        kind = spur_def.kind

        print(
            f"\nMeasuring {kind}: {spur_def.expression} "
            f"at ~{f_nom / 1e9:.6f} GHz"
        )

        meas_freq = float("nan")
        p_dbm = float("nan")
        rel_dbc: Optional[float] = None
        error_msg: Optional[str] = None
        noise_limited = False

        try:
            eff_marker_mode = _resolve_marker_mode(marker_mode, kind)
            meas_freq, p_dbm = measure_single_tone(
                fsw=fsw,
                center_hz=f_nom,
                span_hz=span_hz,
                rbw_hz=rbw_hz,
                vbw_hz=vbw_hz,
                avg_sweeps=avg_sweeps,
                sweep_timeout_ms=timeout_ms,
                marker_mode=eff_marker_mode,
            )

            # Frequency sanity check
            freq_error_hz = abs(meas_freq - f_nom)
            limit_hz = _freq_error_limit(span_hz, rbw_hz)
            if freq_error_hz > limit_hz:
                print(
                    f"  WARNING: Marker at {meas_freq/1e9:.6f} GHz is far from "
                    f"expected {f_nom/1e9:.6f} GHz (Δ = {freq_error_hz/1e3:.1f} kHz, "
                    f"limit ≈ {limit_hz/1e3:.1f} kHz). "
                    "Possible wrong peak captured."
                )

            rel_dbc = p_dbm - d_power_dbm

            # Noise-limited flag (optional)
            if min_power_db is not None and p_dbm < min_power_db:
                noise_limited = True

            print(
                f"  Measured at {meas_freq / 1e9:.6f} GHz, "
                f"{p_dbm:.2f} dBm ({rel_dbc:.2f} dBc)"
                + (" [noise-limited]" if noise_limited else "")
            )

        except Exception as ex:
            error_msg = str(ex)
            print(f"  ERROR measuring {spur_def.expression}: {error_msg}")
            print("  -> Logging NaN for power/rel_dBc and continuing.")

        results.append(
            SpurResult(
                kind=kind,
                m=spur_def.m,
                n=spur_def.n,
                sign=spur_def.sign,
                expression=spur_def.expression,
                expected_freq_hz=f_nom,
                measured_freq_hz=meas_freq,
                power_dbm=p_dbm,
                rel_dbc=rel_dbc,
                error=error_msg,
                noise_limited=noise_limited,
            )
        )

    print("Spur scan for this point done.")
    return results


# ---------------------------------------------------------------------------
# Helper(s) for sweeping and CSV handling
# ---------------------------------------------------------------------------

def frange(start: float, stop: float, step: float) -> List[float]:
    """Simple inclusive float range for sweeps."""
    if step <= 0:
        raise ValueError("Step must be > 0")
    vals: List[float] = []
    x = start
    # Avoid floating rounding problems: use a small epsilon
    eps = abs(step) * 1e-6
    while x <= stop + eps:
        vals.append(x)
        x += step
    return vals


def build_csv_row(
    r: SpurResult,
    lo_hz: float,
    if_hz: float,
    mode: str,
    span_hz: float,
    rbw_hz: float,
    vbw_hz: Optional[float],
    avg_sweeps: int,
    marker_mode_global: str,
    lo_index: int,
    if_index: int,
    point_index: int,
) -> Dict[str, Any]:
    """Convert a SpurResult + context into a CSV row dictionary."""
    if math.isnan(r.measured_freq_hz):
        freq_error_hz = float("nan")
    else:
        freq_error_hz = r.measured_freq_hz - r.expected_freq_hz

    return {
        "lo_index": lo_index,
        "if_index": if_index,
        "point_index": point_index,
        "lo_Hz": lo_hz,
        "if_Hz": if_hz,
        "mode": mode,
        "kind": r.kind,
        "m": r.m,
        "n": r.n,
        "sign": r.sign,
        "expression": r.expression,
        "expected_freq_Hz": r.expected_freq_hz,
        "measured_freq_Hz": r.measured_freq_hz,
        "freq_error_Hz": freq_error_hz,
        "power_dBm": r.power_dbm,
        "rel_dBc_vs_desired": "" if r.rel_dbc is None else r.rel_dbc,
        "noise_limited": 1 if r.noise_limited else 0,
        "span_Hz": span_hz,
        "rbw_Hz": rbw_hz,
        "vbw_Hz": "" if vbw_hz is None else vbw_hz,
        "avg_sweeps": avg_sweeps,
        "marker_mode_global": marker_mode_global,
        "marker_mode_effective": _resolve_marker_mode(marker_mode_global, r.kind),
        "error": "" if r.error is None else r.error,
        # New remeasurement metadata (initialized for base sweep)
        "remeasure_applied": 0,
        "remeasure_lo_Hz": "",
        "remeasure_if_Hz": "",
        "remeasure_rf_Hz": "",
        "remeasure_lo_offset_Hz": "",
        "remeasure_if_offset_Hz": "",
        "remeasure_reason": "",
        "remeasure_note": "",
        # Optional originals, filled only if remeasure_applied == 1
        "power_dBm_original": "",
        "rel_dBc_vs_desired_original": "",
    }


def get_csv_fieldnames() -> List[str]:
    """Canonical CSV header used for both per-point and master CSVs."""
    return [
        "lo_index",
        "if_index",
        "point_index",
        "lo_Hz",
        "if_Hz",
        "mode",
        "kind",
        "m",
        "n",
        "sign",
        "expression",
        "expected_freq_Hz",
        "measured_freq_Hz",
        "freq_error_Hz",
        "power_dBm",
        "rel_dBc_vs_desired",
        "noise_limited",
        "span_Hz",
        "rbw_Hz",
        "vbw_Hz",
        "avg_sweeps",
        "marker_mode_global",
        "marker_mode_effective",
        "error",
        # New overlap / remeasurement metadata
        "remeasure_applied",
        "remeasure_lo_Hz",
        "remeasure_if_Hz",
        "remeasure_rf_Hz",
        "remeasure_lo_offset_Hz",
        "remeasure_if_offset_Hz",
        "remeasure_reason",
        "remeasure_note",
        "power_dBm_original",
        "rel_dBc_vs_desired_original",
    ]


# ---------------------------------------------------------------------------
# Overlap detection and remeasurement (Pass 2 + 3)
# ---------------------------------------------------------------------------

def run_overlap_detection_and_remeasure(
    master_rows: List[Dict[str, Any]],
    fsw: RsFsw,
    smab: RsSmab,
    smbv: RsSmbv,
    args: argparse.Namespace,
) -> None:
    """
    Pass 2 + 3:
        - Build 2D planes of rel_dBc_vs_desired for each spur (m,n,sign)
        - Detect local positive outliers for weak spurs
        - Identify overlap with stronger spurs at same (LO,IF) and same RF
        - Compute LO/IF perturbation to separate them
        - Remeasure small spur with marker_mode='at-freq'
        - Overwrite contaminated values in master_rows and annotate metadata
    """
    if not master_rows:
        print("\n[Overlap] No data in master_rows, skipping overlap detection.")
        return

    if fsw is None or smab is None or smbv is None:
        print("\n[Overlap] Instruments not available, skipping overlap detection.")
        return

    print("\n============================================================")
    print("Starting overlap detection and remeasurement pass "
          "(--overlap-detect enabled)")

    # ------------------------------------------------------------------
    # Build per-spur planes and helper maps
    # ------------------------------------------------------------------
    planes: Dict[Tuple[int, int, int], Dict[Tuple[int, int], float]] = {}
    plane_medians: Dict[Tuple[int, int, int], float] = {}

    # point -> list[row]
    rows_by_point: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    # (point, spur_key) -> row
    row_map: Dict[Tuple[int, int, int, int, int], Dict[str, Any]] = {}
    # desired row per (lo_idx, if_idx)
    desired_map: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for row in master_rows:
        lo_idx = int(row["lo_index"])
        if_idx = int(row["if_index"])
        point_key = (lo_idx, if_idx)

        rows_by_point.setdefault(point_key, []).append(row)

        kind = str(row["kind"])
        m = int(row["m"])
        n = int(row["n"])
        sign = int(row["sign"])
        spur_key = (m, n, sign)

        row_map[(lo_idx, if_idx, m, n, sign)] = row

        if kind == "desired":
            desired_map[point_key] = row
            # desired plane is not considered for A/B overlap detection
            continue

        error_str = str(row.get("error") or "")
        if error_str:
            continue

        rel = _to_optional_float(row.get("rel_dBc_vs_desired"))
        if rel is None:
            continue

        planes.setdefault(spur_key, {})[(lo_idx, if_idx)] = rel

    # Compute plane medians
    for spur_key, plane in planes.items():
        vals = list(plane.values())
        if not vals:
            continue
        plane_medians[spur_key] = statistics.median(vals)

    if not planes:
        print("[Overlap] No valid non-desired spurs with rel_dBc found; "
              "skipping overlap detection.")
        return

    # ------------------------------------------------------------------
    # Local positive outlier detection
    # ------------------------------------------------------------------
    local_thr = float(args.overlap_local_thr_db)
    freq_tol = max(3.0 * float(args.rbw), 0.01 * float(args.span))

    candidates: List[Tuple[Tuple[int, int, int], int, int]] = []

    print("[Overlap] Detecting local positive outliers in spur planes ...")

    for spur_key, plane in planes.items():
        for (i, j), P in plane.items():
            neighbors: List[float] = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    coord = (i + di, j + dj)
                    if coord in plane:
                        neighbors.append(plane[coord])
            if len(neighbors) < 3:
                continue

            med_neighbors = statistics.median(neighbors)
            if P - med_neighbors > local_thr:
                candidates.append((spur_key, i, j))

    print(f"[Overlap] Found {len(candidates)} local outlier candidates.")

    if not candidates:
        print("[Overlap] No suspicious local outliers; nothing to correct.")
        return

    # ------------------------------------------------------------------
    # Check for overlapping spurs at same LO/IF
    # ------------------------------------------------------------------
    global_thr = float(args.overlap_global_thr_db)
    equal_thr = float(args.overlap_equal_thr_db)

    confirmed: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], int, int]] = []

    print("[Overlap] Checking each local outlier for overlap with a stronger spur...")

    for spur_key_A, lo_idx, if_idx in candidates:
        point_key = (lo_idx, if_idx)

        row_A = row_map.get((lo_idx, if_idx, spur_key_A[0], spur_key_A[1], spur_key_A[2]))
        if not row_A:
            continue

        rows_at_point = rows_by_point.get(point_key, [])
        if not rows_at_point:
            continue

        P_A = _to_optional_float(row_A.get("rel_dBc_vs_desired"))
        if P_A is None:
            continue

        median_plane_A = plane_medians.get(spur_key_A)
        if median_plane_A is None:
            continue

        f_A_exp = _to_optional_float(row_A.get("expected_freq_Hz"))
        if f_A_exp is None:
            continue

        best_B_key: Optional[Tuple[int, int, int]] = None
        best_B_row: Optional[Dict[str, Any]] = None
        best_B_strength: Optional[float] = None

        for row_B in rows_at_point:
            kind_B = str(row_B["kind"])
            # Desired is not considered as B in this algorithm
            if kind_B == "desired":
                continue

            m_B = int(row_B["m"])
            n_B = int(row_B["n"])
            s_B = int(row_B["sign"])
            spur_key_B = (m_B, n_B, s_B)

            if spur_key_B == spur_key_A:
                continue

            P_B = _to_optional_float(row_B.get("rel_dBc_vs_desired"))
            if P_B is None:
                continue

            median_plane_B = plane_medians.get(spur_key_B)
            if median_plane_B is None:
                continue

            f_B_exp = _to_optional_float(row_B.get("expected_freq_Hz"))
            if f_B_exp is None:
                continue

            # RF-coincident within tolerance
            if abs(f_B_exp - f_A_exp) > freq_tol:
                continue

            # B is typically much stronger than A
            if median_plane_B - median_plane_A <= global_thr:
                continue

            # At this point, A and B read almost the same level
            if abs(P_A - P_B) >= equal_thr:
                continue

            # This looks like strong B pulling weak A up
            if (best_B_strength is None) or (median_plane_B > best_B_strength):
                best_B_strength = median_plane_B
                best_B_key = spur_key_B
                best_B_row = row_B

        if best_B_key is not None and best_B_row is not None:
            confirmed.append((spur_key_A, best_B_key, lo_idx, if_idx))

    print(f"[Overlap] Confirmed {len(confirmed)} overlapping spur cases.")

    if not confirmed:
        print("[Overlap] No confirmed overlaps; nothing to correct.")
        return

    # ------------------------------------------------------------------
    # Remeasure and correct each confirmed overlap
    # ------------------------------------------------------------------
    sep_factor = float(args.overlap_sep_rbw)
    rbw_hz = float(args.rbw)

    max_lo_shift = _to_optional_float(getattr(args, "max_remeasure_lo_shift_hz", None))
    max_if_shift = _to_optional_float(getattr(args, "max_remeasure_if_shift_hz", None))

    lo_start = float(args.lo_start)
    lo_stop = float(args.lo_stop)
    if_start = float(args.if_start)
    if_stop = float(args.if_stop)

    corrections_applied = 0

    print("[Overlap] Starting remeasurement of confirmed overlaps...")

    for spur_key_A, spur_key_B, lo_idx, if_idx in confirmed:
        point_key = (lo_idx, if_idx)
        m1, n1, s1 = spur_key_A
        m2, n2, s2 = spur_key_B

        row_A = row_map.get((lo_idx, if_idx, m1, n1, s1))
        row_B = row_map.get((lo_idx, if_idx, m2, n2, s2))
        desired_row = desired_map.get(point_key)

        if row_A is None or row_B is None or desired_row is None:
            continue

        LO = float(row_A["lo_Hz"])
        IF = float(row_A["if_Hz"])

        # Degenerate algebraic expression: cannot separate
        if (m1 == m2) and (s1 * n1 == s2 * n2):
            row_A["remeasure_applied"] = 0
            row_A["remeasure_reason"] = "degenerate_expression"
            row_A["remeasure_note"] = (
                f"Spur {row_A['expression']} is algebraically identical to "
                f"{row_B['expression']}; cannot separate via LO/IF shift."
            )
            continue

        f_sep = sep_factor * rbw_hz

        candidates: List[Dict[str, Any]] = []

        # IF-only perturbation
        denom_if = (s1 * n1) - (s2 * n2)
        if denom_if != 0:
            base_step_if = f_sep / abs(denom_if)
            for sign_dir in (+1, -1):
                d_if = sign_dir * base_step_if
                new_if = IF + d_if
                new_lo = LO

                if max_if_shift is not None and abs(d_if) > max_if_shift:
                    continue
                if new_if < if_start or new_if > if_stop:
                    continue

                candidates.append(
                    {
                        "mode": "IF-only",
                        "delta_lo": 0.0,
                        "delta_if": d_if,
                        "new_lo": new_lo,
                        "new_if": new_if,
                    }
                )

        # LO-only perturbation
        denom_lo = (m1 - m2)
        if denom_lo != 0:
            base_step_lo = f_sep / abs(denom_lo)
            for sign_dir in (+1, -1):
                d_lo = sign_dir * base_step_lo
                new_lo = LO + d_lo
                new_if = IF

                if max_lo_shift is not None and abs(d_lo) > max_lo_shift:
                    continue
                if new_lo < lo_start or new_lo > lo_stop:
                    continue

                candidates.append(
                    {
                        "mode": "LO-only",
                        "delta_lo": d_lo,
                        "delta_if": 0.0,
                        "new_lo": new_lo,
                        "new_if": new_if,
                    }
                )

        if not candidates:
            row_A["remeasure_applied"] = 0
            row_A["remeasure_reason"] = "no_valid_perturbation"
            row_A["remeasure_note"] = (
                "No valid LO/IF perturbation within configured limits "
                "to separate spur from "
                f"{row_B['expression']} by {sep_factor}*RBW."
            )
            continue

        # Choose candidate with minimal total shift; prefer IF-only on tie
        candidates.sort(
            key=lambda c: (
                abs(c["delta_lo"]) + abs(c["delta_if"]),
                0 if c["mode"] == "IF-only" else 1,
            )
        )
        best = candidates[0]

        delta_lo = best["delta_lo"]
        delta_if = best["delta_if"]
        new_lo = best["new_lo"]
        new_if = best["new_if"]

        print(
            f"[Overlap] Remeasuring spur {row_A['expression']} at "
            f"(lo_idx={lo_idx}, if_idx={if_idx}) with {best['mode']} "
            f"perturbation: ΔLO={delta_lo:.1f} Hz, ΔIF={delta_if:.1f} Hz."
        )

        # Retune generators
        try:
            if abs(delta_lo) > 0.0:
                smab.source.frequency.fixed.set_value(new_lo)
                if args.lo_settle_s and args.lo_settle_s > 0:
                    time.sleep(args.lo_settle_s)

            if abs(delta_if) > 0.0:
                smbv.source.frequency.fixed.set_value(new_if)
                if args.if_settle_s and args.if_settle_s > 0:
                    time.sleep(args.if_settle_s)
        except Exception as ex:
            row_A["remeasure_applied"] = 0
            row_A["remeasure_reason"] = "remeasure_failed"
            row_A["remeasure_note"] = f"Failed to retune generators: {ex}"
            continue

        # Compute new expected RF for spur A
        f_A_new_exp = abs(m1 * new_lo + s1 * n1 * new_if)

        # Measure spur A at new point, marker at-freq
        try:
            remeas_freq_hz, remeas_power_dbm = measure_single_tone(
                fsw=fsw,
                center_hz=f_A_new_exp,
                span_hz=args.span,
                rbw_hz=args.rbw,
                vbw_hz=args.vbw,
                avg_sweeps=args.avg,
                sweep_timeout_ms=args.timeout_ms,
                marker_mode="at-freq",
            )
        except Exception as ex:
            row_A["remeasure_applied"] = 0
            row_A["remeasure_reason"] = "remeasure_failed"
            row_A["remeasure_note"] = f"Remeasurement failed: {ex}"
            continue

        P_desired_base = _to_optional_float(desired_row.get("power_dBm"))
        if P_desired_base is None:
            row_A["remeasure_applied"] = 0
            row_A["remeasure_reason"] = "remeasure_failed"
            row_A["remeasure_note"] = (
                "Desired product power missing at base point; "
                "cannot compute dBc for remeasurement."
            )
            continue

        rel_dBc_new = remeas_power_dbm - P_desired_base

        # Preserve original values if not already preserved
        if str(row_A.get("power_dBm_original", "")).strip() == "":
            row_A["power_dBm_original"] = row_A["power_dBm"]
        if str(row_A.get("rel_dBc_vs_desired_original", "")).strip() == "":
            row_A["rel_dBc_vs_desired_original"] = row_A.get("rel_dBc_vs_desired", "")

        # Update measurement values
        row_A["power_dBm"] = remeas_power_dbm
        row_A["rel_dBc_vs_desired"] = rel_dBc_new
        row_A["measured_freq_Hz"] = remeas_freq_hz

        # Update metadata
        row_A["remeasure_applied"] = 1
        row_A["remeasure_lo_Hz"] = new_lo
        row_A["remeasure_if_Hz"] = new_if
        row_A["remeasure_rf_Hz"] = remeas_freq_hz
        row_A["remeasure_lo_offset_Hz"] = new_lo - LO
        row_A["remeasure_if_offset_Hz"] = new_if - IF
        row_A["remeasure_reason"] = "overlap_with_strong_spur"
        row_A["remeasure_note"] = (
            f"{best['mode']} perturbation, {sep_factor:.2f}*RBW separation "
            f"from spur {row_B['expression']}"
        )

        corrections_applied += 1

    print(f"[Overlap] Remeasurement applied to {corrections_applied} spur rows.")
    print("Finished overlap detection and correction pass.")
    print("============================================================\n")


# ---------------------------------------------------------------------------
# 2D LO/IF sweep main routine
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=(
            "2D LO/IF sweep for mixer spur scan "
            "(FSW + SMBV100A IF source + SMA100B LO source)"
        )
    )

    # Instrument connections
    p.add_argument(
        "--fsw",
        dest="fsw_resource",
        default="TCPIP::192.168.1.101::HISLIP",
        help="FSW VISA resource string.",
    )
    p.add_argument(
        "--smbv",
        dest="smbv_resource",
        default="TCPIP::192.168.1.102::HISLIP",
        help="SMBV100A VISA resource string (IF generator).",
    )
    p.add_argument(
        "--sma",
        dest="sma_resource",
        default="TCPIP::192.168.1.103::HISLIP",
        help="SMA100B VISA resource string (LO generator).",
    )
    p.add_argument(
        "--fsw-reset",
        action="store_true",
        help="Preset (*RST) the FSW on connect.",
    )
    p.add_argument(
        "--smbv-reset",
        action="store_true",
        help="Preset (*RST) the SMBV100A on connect.",
    )
    p.add_argument(
        "--sma-reset",
        action="store_true",
        help="Preset (*RST) the SMA100B on connect.",
    )

    # LO sweep
    p.add_argument(
        "--lo-start",
        type=float,
        required=True,
        help="Start LO frequency in Hz.",
    )
    p.add_argument(
        "--lo-stop",
        type=float,
        required=True,
        help="Stop LO frequency in Hz.",
    )
    p.add_argument(
        "--lo-step",
        type=float,
        required=True,
        help="LO step in Hz.",
    )
    p.add_argument(
        "--lo-level-db",
        type=float,
        default=0.0,
        help="SMA100B LO output level in dBm.",
    )
    p.add_argument(
        "--lo-settle-s",
        type=float,
        default=0.05,
        help="Settling time after changing LO frequency (seconds).",
    )

    # IF sweep
    p.add_argument(
        "--if-start",
        type=float,
        required=True,
        help="Start IF frequency in Hz.",
    )
    p.add_argument(
        "--if-stop",
        type=float,
        required=True,
        help="Stop IF frequency in Hz.",
    )
    p.add_argument(
        "--if-step",
        type=float,
        required=True,
        help="IF step in Hz.",
    )
    p.add_argument(
        "--if-level-db",
        type=float,
        default=-10.0,
        help="SMBV IF output level in dBm.",
    )
    p.add_argument(
        "--if-settle-s",
        type=float,
        default=0.02,
        help="Settling time after changing IF frequency (seconds).",
    )

    # Mixer spur configuration (mirrors single-point scanner)
    p.add_argument(
        "--mode",
        choices=["lo+if", "lo-if"],
        default="lo-if",
        help="Desired mixing product: LO+IF or LO-IF.",
    )
    p.add_argument(
        "--m-max",
        type=int,
        default=5,
        help="Maximum LO order m to consider (0..m_max).",
    )
    p.add_argument(
        "--n-max",
        type=int,
        default=5,
        help="Maximum IF order n to consider (0..n_max).",
    )
    p.add_argument(
        "--f-min",
        type=float,
        default=0.0,
        help="Minimum RF frequency to consider (Hz).",
    )
    p.add_argument(
        "--f-max",
        type=float,
        default=43e9,
        help="Maximum RF frequency to consider (Hz).",
    )
    p.add_argument(
        "--dedupe-freq",
        type=float,
        default=0.0,
        help="Frequency bin for merging coincident products (Hz).",
    )

    # Measurement settings
    p.add_argument(
        "--span",
        type=float,
        default=1e6,
        help="Zoom span around each product (Hz).",
    )
    p.add_argument(
        "--rbw",
        type=float,
        default=1e3,
        help="Resolution bandwidth (Hz).",
    )
    p.add_argument(
        "--vbw",
        type=float,
        default=None,
        help="Video bandwidth (Hz, optional).",
    )
    p.add_argument(
        "--avg",
        type=int,
        default=10,
        help="Sweeps to average per product.",
    )
    p.add_argument(
        "--timeout-ms",
        type=int,
        default=None,
        help="Sweep timeout in ms (for INIT:IMM *OPC).",
    )
    p.add_argument(
        "--marker-mode",
        choices=["auto", "peak", "at-freq"],
        default="auto",
        help="Marker positioning mode for spur scan.",
    )

    # Noise limits / validation
    p.add_argument(
        "--min-power-db",
        type=float,
        default=None,
        help="If tone is below this level (dBm), flag as noise-limited.",
    )
    p.add_argument(
        "--min-desired-db",
        type=float,
        default=None,
        help="Minimum desired mixer product level (dBm) required.",
    )

    # Overlap detection / remeasurement
    p.add_argument(
        "--overlap-detect",
        action="store_true",
        help=(
            "Enable spur overlap detection and automatic remeasurement "
            "of contaminated weak spurs."
        ),
    )
    p.add_argument(
        "--overlap-local-thr-db",
        type=float,
        default=5.0,
        help=(
            "Local positive outlier threshold in dB for detecting "
            "suspicious spur points (default 5 dB)."
        ),
    )
    p.add_argument(
        "--overlap-global-thr-db",
        type=float,
        default=15.0,
        help=(
            "Required median-plane level difference between strong spur B "
            "and weak spur A in dB (default 15 dB)."
        ),
    )
    p.add_argument(
        "--overlap-equal-thr-db",
        type=float,
        default=1.0,
        help=(
            "Max allowed dB difference between measured A and B at the "
            "overlap point (default 1 dB)."
        ),
    )
    p.add_argument(
        "--overlap-sep-rbw",
        type=float,
        default=5.0,
        help=(
            "Target frequency separation as multiple of RBW for remeasurement "
            "perturbation (default 5)."
        ),
    )
    p.add_argument(
        "--max-remeasure-lo-shift-hz",
        type=float,
        default=None,
        help=(
            "Maximum allowed magnitude of LO shift in Hz during remeasurement. "
            "If omitted, only sweep bounds are enforced."
        ),
    )
    p.add_argument(
        "--max-remeasure-if-shift-hz",
        type=float,
        default=None,
        help=(
            "Maximum allowed magnitude of IF shift in Hz during remeasurement. "
            "If omitted, only sweep bounds are enforced."
        ),
    )

    # Output
    p.add_argument(
        "--out-dir",
        default="spur_sweep_results",
        help="Directory for CSVs and outputs.",
    )
    p.add_argument(
        "--master-csv",
        default="spur_sweep_master.csv",
        help="Name of aggregated CSV inside out-dir.",
    )
    p.add_argument(
        "--no-per-point-csv",
        action="store_true",
        help="If set, do not write per-(LO,IF) CSV files, only the master CSV.",
    )

    args = p.parse_args()

    # Validate sweep ranges
    if args.lo_stop < args.lo_start:
        raise RuntimeError("lo-stop must be >= lo-start")
    if args.if_stop < args.if_start:
        raise RuntimeError("if-stop must be >= if-start")

    if args.overlap_detect and args.dedupe_freq != 0.0:
        print(
            "WARNING: --overlap-detect used with non-zero --dedupe-freq.\n"
            "         Overlap detection relies on multiple spur expressions "
            "being present at the same RF. Frequency de-duplication can mask "
            "overlaps by collapsing spurs into a single representative.\n"
            "         Consider re-running with --dedupe-freq 0 for best results."
        )

    lo_values = frange(args.lo_start, args.lo_stop, args.lo_step)
    if_values = frange(args.if_start, args.if_stop, args.if_step)

    print(f"LO sweep points: {len(lo_values)}")
    print(f"IF sweep points: {len(if_values)}")
    total_points = len(lo_values) * len(if_values)
    print(f"Total (LO, IF) operating points: {total_points}")

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make sure FSW driver is recent enough
    RsFsw.assert_minimum_version("5.0.0")

    # Prepare containers
    master_rows: List[Dict[str, Any]] = []

    # Connect to instruments
    smab: Optional[RsSmab] = None
    smbv: Optional[RsSmbv] = None
    fsw: Optional[RsFsw] = None

    try:
        # LO source (SMA100B)
        print(f"Connecting to SMA100B at '{args.sma_resource}' ...")
        smab = RsSmab(args.sma_resource, reset=args.sma_reset, id_query=True)
        print(f"SMA100B IDN: {smab.utilities.idn_string}")

        smab.output.state.set_value(True)
        smab.source.frequency.set_mode(smab_enums.FreqMode.CW)
        smab.source.power.level.immediate.set_amplitude(args.lo_level_db)

        # IF source (SMBV100A)
        print(f"Connecting to SMBV100A at '{args.smbv_resource}' ...")
        smbv = RsSmbv(args.smbv_resource, reset=args.smbv_reset, id_query=True)
        print(f"SMBV100A IDN: {smbv.utilities.idn_string}")

        smbv.output.state.set_value(True)
        smbv.source.frequency.set_mode(smbv_enums.FreqMode.CW)
        smbv.source.power.level.immediate.set_amplitude(args.if_level_db)

        # Spectrum analyzer (FSW)
        print(f"Connecting to FSW at '{args.fsw_resource}' ...")
        fsw = RsFsw(args.fsw_resource, reset=args.fsw_reset, id_query=True)
        print(f"FSW IDN: {fsw.utilities.idn_string}")

        fsw.instrument.select.set(fsw_enums.ChannelType.SpectrumAnalyzer)
        fsw.system.display.update.set(True)
        # Start in continuous mode
        try:
            fsw.initiate.continuous.set(True)
        except Exception:
            pass

        point_index = 0

        for lo_idx, lo_hz in enumerate(lo_values):
            print("\n============================================================")
            print(f"LO = {lo_hz / 1e9:.6f} GHz ({lo_idx + 1}/{len(lo_values)})")

            # Set LO frequency
            smab.source.frequency.fixed.set_value(lo_hz)

            # Optional LO settling
            if args.lo_settle_s and args.lo_settle_s > 0:
                time.sleep(args.lo_settle_s)

            for if_idx, if_hz in enumerate(if_values):
                point_index += 1
                print("\n------------------------------------------------------------")
                print(
                    f"[{point_index}/{total_points}] "
                    f"LO = {lo_hz/1e9:.6f} GHz, IF = {if_hz/1e6:.3f} MHz"
                )

                # Set IF frequency
                smbv.source.frequency.fixed.set_value(if_hz)

                # Optional IF settling
                if args.if_settle_s and args.if_settle_s > 0:
                    time.sleep(args.if_settle_s)

                # Run the spur scan for this point
                try:
                    results = measure_spurs_for_point(
                        fsw=fsw,
                        lo_hz=lo_hz,
                        if_hz=if_hz,
                        mode=args.mode,
                        m_max=args.m_max,
                        n_max=args.n_max,
                        f_min_hz=args.f_min,
                        f_max_hz=args.f_max,
                        dedupe_freq_hz=args.dedupe_freq,
                        span_hz=args.span,
                        rbw_hz=args.rbw,
                        vbw_hz=args.vbw,
                        avg_sweeps=args.avg,
                        timeout_ms=args.timeout_ms,
                        marker_mode=args.marker_mode,
                        min_power_db=args.min_power_db,
                        min_desired_db=args.min_desired_db,
                    )
                except Exception as ex:
                    print(
                        "ERROR during spur scan at "
                        f"LO={lo_hz/1e9:.6f} GHz, IF={if_hz/1e6:.3f} MHz: {ex}"
                    )
                    print("Skipping this point and continuing with sweep.")
                    continue

                # Build rows for this point and append to master list
                rows_for_point: List[Dict[str, Any]] = []
                for r in results:
                    row = build_csv_row(
                        r=r,
                        lo_hz=lo_hz,
                        if_hz=if_hz,
                        mode=args.mode,
                        span_hz=args.span,
                        rbw_hz=args.rbw,
                        vbw_hz=args.vbw,
                        avg_sweeps=args.avg,
                        marker_mode_global=args.marker_mode,
                        lo_index=lo_idx,
                        if_index=if_idx,
                        point_index=point_index,
                    )
                    rows_for_point.append(row)
                    master_rows.append(row)

                # Optionally write per-point CSV
                if not args.no_per_point_csv:
                    lo_ghz = lo_hz / 1e9
                    if_mhz = if_hz / 1e6
                    csv_name = (
                        f"spur_results_lo_{lo_ghz:.6f}GHz_if_{if_mhz:.3f}MHz.csv"
                    )
                    csv_path = out_dir / csv_name

                    print(f"Writing per-point CSV to '{csv_path}' ...")
                    with csv_path.open("w", newline="") as f:
                        writer = csv.DictWriter(
                            f, fieldnames=get_csv_fieldnames()
                        )
                        writer.writeheader()
                        for row in rows_for_point:
                            writer.writerow(row)

        # --------------------------------------------------------------
        # Overlap detection + remeasurement before closing instruments
        # --------------------------------------------------------------
        if args.overlap_detect and master_rows and fsw is not None and smab is not None and smbv is not None:
            run_overlap_detection_and_remeasure(master_rows, fsw, smab, smbv, args)

    finally:
        # Turn outputs off and close sessions
        if smbv is not None:
            print("\nTurning SMBV RF output OFF and closing session.")
            try:
                smbv.output.state.set_value(False)
            except Exception as ex:
                print(f"WARNING: Failed to switch SMBV RF OFF cleanly: {ex}")
            smbv.close()

        if smab is not None:
            print("\nTurning SMA RF output OFF and closing session.")
            try:
                smab.output.state.set_value(False)
            except Exception as ex:
                print(f"WARNING: Failed to switch SMA RF OFF cleanly: {ex}")
            smab.close()

        if fsw is not None:
            print("\nClosing the FSW session.")
            try:
                fsw.initiate.continuous.set(True)
            except Exception:
                pass
            fsw.close()

    # ------------------------------------------------------------------
    # Write master CSV from in-memory rows
    # ------------------------------------------------------------------
    master_csv_path = out_dir / args.master_csv

    if not master_rows:
        print("\nNo data collected - master CSV will not be written.")
        return

    print(f"\nWriting master CSV to '{master_csv_path}' ...")
    with master_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=get_csv_fieldnames())
        writer.writeheader()
        for row in master_rows:
            writer.writerow(row)

    print("Master CSV written with "
          f"{len(master_rows)} rows (all LO/IF points combined).")
    print("All done.")
    print(f"  Output directory : {out_dir}")
    if not args.no_per_point_csv:
        print("  Per-point CSVs   : one file per (LO, IF) point")
    else:
        print("  Per-point CSVs   : disabled (--no-per-point-csv)")
    print(f"  Master CSV       : {master_csv_path}")


if __name__ == "__main__":
    main()
