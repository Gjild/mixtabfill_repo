#!/usr/bin/env python3
"""
Extended 2D LO/IF mixer spur sweep for R&S FSW + SMF100A + HP 83711B

Implements:
    - Full spur list without de-duplication
    - Analytic coincidence groups (expected frequencies vs RBW)
    - Cluster detection (connected components using realistic contamination limits)
    - Per-point Δf correction from desired product (order-aware weighting)
    - Cluster-based trace measurement and software peak assignment
    - Measured coincidence grouping (based on measured frequencies)
    - Extended CSV topology fields
    - Extra metadata for cluster searches (window size, fallback used)

Controls:
    * R&S SMF100A CW generator as LO source
    * HP 83711B CW generator as IF source (via GPIB, pyvisa)
    * R&S FSW spectrum analyzer for spur measurements

Sweeps LO and IF according to user-defined ranges and writes:
    * Optional per-(LO,IF) CSVs
    * One master CSV aggregated from memory

Requires:
    pip install RsFsw RsInstrument pyvisa
"""

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pyvisa

from RsFsw import RsFsw, enums as fsw_enums, repcap
from RsInstrument import RsInstrument


# ---------------------------------------------------------------------------
# Data classes for spur definitions / results
# ---------------------------------------------------------------------------

@dataclass
class SpurDef:
    """Definition of a mixer product to be measured (per (LO, IF) point)."""
    kind: str            # "desired", "image", "LO_leak", "LO_harmonic",
                         # "IF_leak", "IF_harmonic", "spur"
    m: int
    n: int
    sign: int            # +1 or -1 for ±n·IF
    expression: str      # e.g. "1*LO - 2*IF"

    # Frequency model (from calibrated / readback LO/IF)
    expected_freq_hz: float

    # Topology fields (per (LO, IF) point)
    cluster_id: int = -1
    cluster_size: int = 1

    analytic_coincidence_id: int = -1
    analytic_coincidence_size: int = 1

    # After per-point Δf correction from desired product
    expected_corr_freq_hz: float = 0.0


@dataclass
class SpurResult:
    """Measurement result of one mixer product."""
    kind: str
    m: int
    n: int
    sign: int
    expression: str

    expected_freq_hz: float           # analytic model (calibrated)
    expected_corr_freq_hz: float      # per-point corrected value

    measured_freq_hz: float
    power_dbm: float
    rel_dbc: Optional[float]          # 0.0 for desired, None if measurement failed

    error: Optional[str] = None
    noise_limited: bool = False

    # Topology (copied from SpurDef at measurement time)
    cluster_id: int = -1
    cluster_size: int = 1

    analytic_coincidence_id: int = -1
    analytic_coincidence_size: int = 1

    measured_coincidence_id: int = -1
    measured_coincidence_size: int = 1

    # "single"  - measure_single_tone()
    # "cluster" - derived from shared cluster trace
    measurement_scope: str = "single"

    # Cluster-search diagnostics (only meaningful for measurement_scope == "cluster")
    cluster_window_half_width_hz: float = 0.0
    cluster_window_fallback: bool = False


# ---------------------------------------------------------------------------
# Utility functions shared by all measurements
# ---------------------------------------------------------------------------

def dbm_to_mw(p_dbm: float) -> float:
    return 10.0 ** (p_dbm / 10.0)


def mw_to_dbm(p_mw: float) -> float:
    if p_mw <= 0.0:
        return float("-inf")
    return 10.0 * math.log10(p_mw)


def dbm_list_to_avg_dbm(powers_dbm: List[float]) -> float:
    """Average in linear (mW), return in dBm."""
    if not powers_dbm:
        return float("nan")
    lin_mw = [dbm_to_mw(p) for p in powers_dbm]
    avg_mw = sum(lin_mw) / len(lin_mw)
    return mw_to_dbm(avg_mw)


def _dedupe_spurs_by_freq(spurs: List[SpurDef], dedupe_hz: float) -> List[SpurDef]:
    """
    Optional frequency de-duplication (NOT used in the main measurement path).

    If dedupe_hz > 0, group spurs into frequency bins of width dedupe_hz and keep
    a single representative in each bin, preferring more 'important' kinds:

        desired > image > leakage > harmonics > spur

    Kept for debugging / summaries only.
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
) -> List[SpurDef]:
    """
    Build list of mixer products m·LO ± n·IF within [f_min, f_max].

    desired_mode: "lo+if" or "lo-if" (desired product is 1*LO ± 1*IF).

    IMPORTANT:
        - No frequency de-duplication in the main measurement path.
        - All distinct (kind, m, n, sign, expression) entries are kept.
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
                expected_corr_freq_hz=f_desired,
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
                        expected_corr_freq_hz=freq,
                    )
                )

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
    span_hz = max(span_hz, 1.0)
    rbw_hz = max(rbw_hz, 1.0)

    f_start = max(center_hz - span_hz / 2.0, 0.0)
    f_stop = center_hz + span_hz / 2.0

    fsw.sense.frequency.start.set(f_start)
    fsw.sense.frequency.stop.set(f_stop)

    fsw.sense.bandwidth.resolution.set(rbw_hz)

    if vbw_hz is not None and vbw_hz > 0.0:
        fsw.sense.bandwidth.video.set(vbw_hz)


def _freq_error_limit(span_hz: float, rbw_hz: float) -> float:
    """
    Frequency error limit for sanity checks.

    We allow the larger of:
        - 10 * RBW
        - 10% of span
    """
    # return max(10.0 * rbw_hz, 0.1 * span_hz)
    return 0.2e6


def _get_sweep_timeout_ms(fsw: RsFsw, sweep_timeout_ms: Optional[int]) -> int:
    """Helper: determine sweep timeout."""
    if sweep_timeout_ms is not None:
        return sweep_timeout_ms
    try:
        sweep_time_s = fsw.sense.sweep.time.get()
        return int(max(1000.0 * sweep_time_s * 2.0, 1000.0))
    except Exception:
        return 5000


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

    Averaging is done in Python (linear units).
    """
    marker_mode = marker_mode.lower()
    if marker_mode not in ("peak", "at-freq"):
        raise ValueError("marker_mode must be 'peak' or 'at-freq'")

    configure_span_and_bw(fsw, center_hz, span_hz, rbw_hz, vbw_hz)

    sweep_timeout_ms = _get_sweep_timeout_ms(fsw, sweep_timeout_ms)

    fsw.initiate.continuous.set(False)

    powers_dbm: List[float] = []
    last_marker_freq_hz = center_hz

    fsw.calculate.marker.trace.set(1, repcap.Window.Nr1, repcap.Marker.Nr1)

    sweeps = max(avg_sweeps, 1)
    for _ in range(sweeps):
        fsw.initiate.immediate_with_opc(sweep_timeout_ms)

        if marker_mode == "peak":
            fsw.calculate.marker.maximum.peak.set(
                repcap.Window.Nr1, repcap.Marker.Nr1
            )
        else:
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

    if kind in ("LO_leak", "IF_leak"):
        return "at-freq"
    return "peak"


# ---------------------------------------------------------------------------
# Frequency model / Δf weighting / topology helpers
# ---------------------------------------------------------------------------

def spur_weight(
    m: int,
    n: int,
    sign: int,
    low_order_sum: int,
    mid_order_sum: int,
    mid_weight: float,
) -> float:
    """
    Order-aware weight for applying Δf from desired product.

    - m+n <= low_order_sum -> 1.0
    - low_order_sum < m+n <= mid_order_sum -> mid_weight
    - m+n  > mid_order_sum -> 0.0
    """
    order_sum = m + n
    if order_sum <= low_order_sum:
        return 1.0
    if order_sum <= mid_order_sum:
        return float(mid_weight)
    return 0.0


def assign_analytic_coincidence_groups(
    spurs: List[SpurDef],
    rbw_hz: float,
    coincidence_factor: float,
    min_coincidence_hz: float,
) -> None:
    """
    Assign analytic coincidence groups based on expected_freq_hz using
    connected components.

    Tolerance:
        tol_hz = max(rbw_hz * coincidence_factor, min_coincidence_hz)
    """
    if not spurs:
        return

    tol_hz = max(rbw_hz * float(coincidence_factor), float(min_coincidence_hz))
    if tol_hz <= 0.0:
        return

    n = len(spurs)
    freqs = [s.expected_freq_hz for s in spurs]

    # Sort indices by frequency
    indices = list(range(n))
    indices.sort(key=lambda i: freqs[i])

    # Build adjacency list (connect spurs within tol_hz)
    adj: List[List[int]] = [[] for _ in range(n)]
    for idx_pos, i in enumerate(indices):
        for jdx_pos in range(idx_pos + 1, len(indices)):
            j = indices[jdx_pos]
            if abs(freqs[j] - freqs[i]) <= tol_hz:
                adj[i].append(j)
                adj[j].append(i)
            else:
                # Sorted by frequency; beyond tolerance
                break

    visited = [False] * n
    group_id = 0

    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        component: List[int] = []
        while stack:
            u = stack.pop()
            if visited[u]:
                continue
            visited[u] = True
            component.append(u)
            for v in adj[u]:
                if not visited[v]:
                    stack.append(v)
        if len(component) > 1:
            for idx in component:
                spurs[idx].analytic_coincidence_id = group_id
                spurs[idx].analytic_coincidence_size = len(component)
            group_id += 1


def assign_clusters(
    spurs: List[SpurDef],
    span_hz: float,
    rbw_hz: float,
    marker_confusion_factor: float,
    rbw_guard_factor: float,
    max_cluster_distance_hz: float,
) -> None:
    """
    Assign clusters using connected components on frequencies.

    Use expected_corr_freq_hz if set (>0), otherwise expected_freq_hz.

    contamination_limit = min(
        span * marker_confusion_factor,
        rbw * rbw_guard_factor,
        max_cluster_distance_hz (if > 0)
    )
    """
    if not spurs:
        return

    span_hz = max(span_hz, 1.0)
    rbw_hz = max(rbw_hz, 1.0)

    d_marker_hz = span_hz * float(marker_confusion_factor)
    d_rbw_hz = rbw_hz * float(rbw_guard_factor)
    contamination_limit = min(d_marker_hz, d_rbw_hz)

    if max_cluster_distance_hz is not None and max_cluster_distance_hz > 0.0:
        contamination_limit = min(contamination_limit, float(max_cluster_distance_hz))

    if contamination_limit <= 0.0:
        # Everything stays singleton
        for s in spurs:
            s.cluster_id = -1
            s.cluster_size = 1
        return

    n = len(spurs)
    freqs = [
        (s.expected_corr_freq_hz if s.expected_corr_freq_hz > 0.0 else s.expected_freq_hz)
        for s in spurs
    ]

    indices = list(range(n))
    indices.sort(key=lambda i: freqs[i])

    # Build adjacency list (naive O(N^2), fine for small spur counts)
    adj: List[List[int]] = [[] for _ in range(n)]
    for i_idx in range(n):
        i = indices[i_idx]
        for j_idx in range(i_idx + 1, n):
            j = indices[j_idx]
            if abs(freqs[j] - freqs[i]) <= contamination_limit:
                adj[i].append(j)
                adj[j].append(i)
            else:
                # Sorted by frequency; no need to check further
                break

    visited = [False] * n
    cluster_id_counter = 0

    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        component: List[int] = []
        while stack:
            u = stack.pop()
            if visited[u]:
                continue
            visited[u] = True
            component.append(u)
            for v in adj[u]:
                if not visited[v]:
                    stack.append(v)
        if len(component) > 1:
            # Real cluster
            for idx in component:
                spurs[idx].cluster_id = cluster_id_counter
                spurs[idx].cluster_size = len(component)
            cluster_id_counter += 1
        else:
            # Singleton
            spurs[component[0]].cluster_id = -1
            spurs[component[0]].cluster_size = 1


def assign_measured_coincidence_groups(
    results: List[SpurResult],
    rbw_hz: float,
    measured_coincidence_factor: float,
) -> None:
    """
    Assign measured-coincidence groups based on measured_freq_hz.

    Only within a single (LO, IF) point (this function is called per-point).
    """
    if not results:
        return

    tol_hz = max(rbw_hz * float(measured_coincidence_factor), 0.0)

    # Filter to valid measured frequencies
    valid_indices = [
        i for i, r in enumerate(results)
        if not math.isnan(r.measured_freq_hz)
    ]
    if not valid_indices:
        return

    valid_indices.sort(key=lambda i: results[i].measured_freq_hz)

    current_group: List[int] = []
    group_id = 0

    def finalize_group(group: List[int], gid: int) -> int:
        if len(group) > 1:
            for idx in group:
                results[idx].measured_coincidence_id = gid
                results[idx].measured_coincidence_size = len(group)
            return gid + 1
        return gid

    for idx in valid_indices:
        if not current_group:
            current_group = [idx]
            continue

        ref_freq = results[current_group[0]].measured_freq_hz
        f = results[idx].measured_freq_hz
        if abs(f - ref_freq) <= tol_hz:
            current_group.append(idx)
        else:
            group_id = finalize_group(current_group, group_id)
            current_group = [idx]

    if current_group:
        finalize_group(current_group, group_id)


# ---------------------------------------------------------------------------
# Cluster-based measurement
# ---------------------------------------------------------------------------

def measure_cluster(
    fsw: RsFsw,
    cluster_spurs: List[SpurDef],
    rbw_hz: float,
    vbw_hz: Optional[float],
    avg_sweeps: int,
    sweep_timeout_ms: Optional[int],
    delta_desired_hz: float,
    low_order_sum: int,
    mid_order_sum: int,
    mid_weight: float,
    cluster_max_window_rbw: float,
    all_spurs_for_point: Optional[List[SpurDef]] = None,
    cluster_id: Optional[int] = None,
) -> List[Tuple[float, float, float, bool]]:
    """
    Measure all spurs in a cluster with a single trace acquisition and
    assign local peaks by software.

    Returns a list of tuples in the same order as 'cluster_spurs':

        (measured_freq_hz, power_dbm, window_half_width_hz, window_fallback_used)

    If 'all_spurs_for_point' and 'cluster_id' are provided, the cluster span is
    derived from *all* spurs sharing that cluster_id (including the desired),
    even if only a subset is being assigned within this call.
    """
    if not cluster_spurs:
        return []

    # Determine which spurs define the cluster bounds
    if all_spurs_for_point is not None and cluster_id is not None:
        full_cluster_spurs = [
            s for s in all_spurs_for_point if s.cluster_id == cluster_id
        ]
        if not full_cluster_spurs:
            full_cluster_spurs = cluster_spurs
    else:
        full_cluster_spurs = cluster_spurs

    # Compute cluster bounds using corrected frequencies
    freqs_corr_full = [
        (s.expected_corr_freq_hz if s.expected_corr_freq_hz > 0.0 else s.expected_freq_hz)
        for s in full_cluster_spurs
    ]
    f_min = min(freqs_corr_full)
    f_max = max(freqs_corr_full)
    cluster_width = max(f_max - f_min, rbw_hz)
    guard_hz = max(5.0 * rbw_hz, 0.1 * cluster_width)

    span_cluster = cluster_width + 2.0 * guard_hz
    center_cluster = 0.5 * (f_min + f_max)

    configure_span_and_bw(fsw, center_cluster, span_cluster, rbw_hz, vbw_hz)

    sweep_timeout_ms = _get_sweep_timeout_ms(fsw, sweep_timeout_ms)
    fsw.initiate.continuous.set(False)

    # Acquire averaged trace (linear averaging in Python)
    sweeps = max(avg_sweeps, 1)
    x_data: Optional[List[float]] = None
    acc_lin: Optional[List[float]] = None

    for _ in range(sweeps):
        fsw.initiate.immediate_with_opc(sweep_timeout_ms)

        # TRAC:DATA? TRACE1 -> y-data, TRAC:DATA:X? TRACE1 -> x-axis (freq)
        y_values = fsw.utilities.query_float_list("TRAC:DATA? TRACE1")
        if x_data is None:
            x_data = fsw.utilities.query_float_list("TRAC:DATA:X? TRACE1")

        if acc_lin is None:
            acc_lin = [dbm_to_mw(y) for y in y_values]
        else:
            if len(acc_lin) != len(y_values):
                raise RuntimeError(
                    "Trace length changed between sweeps in cluster measurement."
                )
            for i, y in enumerate(y_values):
                acc_lin[i] += dbm_to_mw(y)

    if x_data is None or acc_lin is None:
        raise RuntimeError("Failed to acquire trace data in cluster measurement.")

    avg_lin = [v / sweeps for v in acc_lin]
    avg_dbm = [mw_to_dbm(v) for v in avg_lin]

    # Helper: find index window around target frequency
    def find_peak_near(freq_target: float, half_width_hz: float) -> Tuple[float, float, bool]:
        """
        Returns:
            (peak_freq_hz, peak_power_dbm, used_fallback)
        """
        n = len(x_data)
        if n == 0:
            return float("nan"), float("nan"), True

        used_fallback = False

        f_start = freq_target - half_width_hz
        f_stop = freq_target + half_width_hz

        # Find index range
        i_start = 0
        while i_start < n and x_data[i_start] < f_start:
            i_start += 1
        i_stop = n - 1
        while i_stop >= 0 and x_data[i_stop] > f_stop:
            i_stop -= 1

        if i_start >= n or i_stop < 0 or i_start > i_stop:
            # Fallback to global maximum if window is empty
            used_fallback = True
            i_start = 0
            i_stop = n - 1

        max_idx = i_start
        max_val = avg_dbm[i_start]
        for i in range(i_start + 1, i_stop + 1):
            if avg_dbm[i] > max_val:
                max_val = avg_dbm[i]
                max_idx = i

        return x_data[max_idx], max_val, used_fallback

    # For each spur, search around expected_corr with a window based on Δf
    results: List[Tuple[float, float, float, bool]] = []
    for s in cluster_spurs:
        w = spur_weight(s.m, s.n, s.sign, low_order_sum, mid_order_sum, mid_weight)
        # Base window wide enough to cover Δf scaling + safety
        half_width = max(abs(delta_desired_hz) * w, rbw_hz * 20.0)

        # Optional cap in units of RBW
        if cluster_max_window_rbw is not None and cluster_max_window_rbw > 0.0:
            half_width = min(half_width, rbw_hz * float(cluster_max_window_rbw))

        target_freq = (
            s.expected_corr_freq_hz if s.expected_corr_freq_hz > 0.0 else s.expected_freq_hz
        )
        mfreq, mpower, used_fallback = find_peak_near(target_freq, half_width)
        results.append((mfreq, mpower, half_width, used_fallback))

    return results


# ---------------------------------------------------------------------------
# Core spur scan for a single (LO, IF) operating point (FSW reused)
# ---------------------------------------------------------------------------

def measure_spurs_for_point(
    fsw: RsFsw,
    lo_set_hz: float,
    if_set_hz: float,
    f_lo_model_hz: float,
    f_if_model_hz: float,
    mode: str,
    m_max: int,
    n_max: int,
    f_min_hz: float,
    f_max_hz: float,
    span_hz: float,
    rbw_hz: float,
    vbw_hz: Optional[float],
    avg_sweeps: int,
    timeout_ms: Optional[int],
    marker_mode: str,
    min_power_db: Optional[float],
    min_desired_db: Optional[float],
    coincidence_factor: float,
    measured_coincidence_factor: float,
    min_coincidence_hz: float,
    marker_confusion_factor: float,
    rbw_guard_factor: float,
    max_cluster_distance_hz: float,
    deltaf_low_order_sum: int,
    deltaf_mid_order_sum: int,
    deltaf_mid_weight: float,
    cluster_max_window_rbw: float,
) -> List[SpurResult]:
    """
    Perform a spur scan for a single (LO, IF) mixer operating point using
    an already open FSW session. Returns a list of SpurResult.

    Frequencies for spur generation are based on f_lo_model_hz / f_if_model_hz
    (calibrated/readback model), not the raw setpoints.
    """
    if mode == "lo-if" and f_lo_model_hz <= f_if_model_hz:
        print(
            "WARNING: LO <= IF while using lo-if (down-conversion). "
            "Check that your LO and IF frequencies are correct."
        )

    # Build list of mixer products (no de-duplication)
    spur_defs = build_spur_list(
        f_lo_hz=f_lo_model_hz,
        f_if_hz=f_if_model_hz,
        m_max=m_max,
        n_max=n_max,
        desired_mode=mode,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
    )
    if not spur_defs or spur_defs[0].kind != "desired":
        raise RuntimeError(
            "Desired product is not within the [f_min, f_max] range "
            "after spur generation."
        )

    print(f"Total products to measure (including desired): {len(spur_defs)}")

    # Analytic coincidences based on expected frequencies
    assign_analytic_coincidence_groups(
        spur_defs,
        rbw_hz=rbw_hz,
        coincidence_factor=coincidence_factor,
        min_coincidence_hz=min_coincidence_hz,
    )

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
        f"  Desired product measured at {d_meas_freq / 1e9:.6f} GHz, "
        f"{d_power_dbm:.2f} dBm (Δf = {freq_error_desired_hz / 1e3:.1f} kHz, "
        f"limit ≈ {limit_desired_hz / 1e3:.1f} kHz)"
    )

    if freq_error_desired_hz > limit_desired_hz:
        raise RuntimeError(
            "Desired product marker is far from expected frequency "
            f"(Δf = {freq_error_desired_hz / 1e6:.3f} MHz, "
            f"limit ≈ {limit_desired_hz / 1e6:.3f} MHz). "
            "Check LO/IF settings, span, RBW, and marker mode."
        )

    if min_desired_db is not None and d_power_dbm < min_desired_db:
        raise RuntimeError(
            f"Desired product level {d_power_dbm:.2f} dBm is below "
            f"--min-desired-db threshold of {min_desired_db:.2f} dBm. "
            "Aborting spur scan for this point."
        )

    # Per-point Δf correction
    delta_desired_hz = d_meas_freq - desired_def.expected_freq_hz
    for s in spur_defs:
        w = spur_weight(
            s.m, s.n, s.sign,
            low_order_sum=deltaf_low_order_sum,
            mid_order_sum=deltaf_mid_order_sum,
            mid_weight=deltaf_mid_weight,
        )
        s.expected_corr_freq_hz = s.expected_freq_hz + w * delta_desired_hz

    # Clusters based on corrected frequencies
    assign_clusters(
        spur_defs,
        span_hz=span_hz,
        rbw_hz=rbw_hz,
        marker_confusion_factor=marker_confusion_factor,
        rbw_guard_factor=rbw_guard_factor,
        max_cluster_distance_hz=max_cluster_distance_hz,
    )

    # Record desired result (measured separately, not from cluster)
    desired_result = SpurResult(
        kind=desired_def.kind,
        m=desired_def.m,
        n=desired_def.n,
        sign=desired_def.sign,
        expression=desired_def.expression,
        expected_freq_hz=desired_def.expected_freq_hz,
        expected_corr_freq_hz=desired_def.expected_corr_freq_hz,
        measured_freq_hz=d_meas_freq,
        power_dbm=d_power_dbm,
        rel_dbc=0.0,
        error=None,
        noise_limited=False,
        cluster_id=desired_def.cluster_id,
        cluster_size=desired_def.cluster_size,
        analytic_coincidence_id=desired_def.analytic_coincidence_id,
        analytic_coincidence_size=desired_def.analytic_coincidence_size,
        measurement_scope="single",
        cluster_window_half_width_hz=0.0,
        cluster_window_fallback=False,
    )
    results.append(desired_result)

    # Group non-desired spurs by cluster_id
    cluster_map: Dict[int, List[SpurDef]] = {}
    single_spurs: List[SpurDef] = []

    for s in spur_defs[1:]:
        if s.cluster_size <= 1:
            single_spurs.append(s)
        else:
            cluster_map.setdefault(s.cluster_id, []).append(s)

    # First handle single-spur clusters via marker-based measurement
    for spur_def in single_spurs:
        f_nom_corr = (
            spur_def.expected_corr_freq_hz if spur_def.expected_corr_freq_hz > 0.0
            else spur_def.expected_freq_hz
        )
        kind = spur_def.kind

        print(
            f"\nMeasuring isolated {kind}: {spur_def.expression} "
            f"at ~{f_nom_corr / 1e9:.6f} GHz"
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
                center_hz=f_nom_corr,
                span_hz=span_hz,
                rbw_hz=rbw_hz,
                vbw_hz=vbw_hz,
                avg_sweeps=avg_sweeps,
                sweep_timeout_ms=timeout_ms,
                marker_mode=eff_marker_mode,
            )

            freq_error_hz = abs(meas_freq - f_nom_corr)
            limit_hz = _freq_error_limit(span_hz, rbw_hz)
            if freq_error_hz > limit_hz:
                print(
                    f"  WARNING: Marker at {meas_freq / 1e9:.6f} GHz is far from "
                    f"expected {f_nom_corr / 1e9:.6f} GHz (Δ = {freq_error_hz / 1e3:.1f} kHz, "
                    f"limit ≈ {limit_hz / 1e3:.1f} kHz). "
                    "Possible wrong peak captured."
                )

            rel_dbc = p_dbm - d_power_dbm

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
                expected_freq_hz=spur_def.expected_freq_hz,
                expected_corr_freq_hz=spur_def.expected_corr_freq_hz,
                measured_freq_hz=meas_freq,
                power_dbm=p_dbm,
                rel_dbc=rel_dbc,
                error=error_msg,
                noise_limited=noise_limited,
                cluster_id=spur_def.cluster_id,
                cluster_size=spur_def.cluster_size,
                analytic_coincidence_id=spur_def.analytic_coincidence_id,
                analytic_coincidence_size=spur_def.analytic_coincidence_size,
                measurement_scope="single",
                cluster_window_half_width_hz=0.0,
                cluster_window_fallback=False,
            )
        )

    # Now handle multi-spur clusters via shared trace
    for cid, cluster_spurs in cluster_map.items():
        if not cluster_spurs:
            continue
        print(
            f"\nMeasuring cluster {cid} with {len(cluster_spurs)} non-desired spurs "
            f"(span-based cluster measurement)"
        )

        try:
            cluster_results = measure_cluster(
                fsw=fsw,
                cluster_spurs=cluster_spurs,
                rbw_hz=rbw_hz,
                vbw_hz=vbw_hz,
                avg_sweeps=avg_sweeps,
                sweep_timeout_ms=timeout_ms,
                delta_desired_hz=delta_desired_hz,
                low_order_sum=deltaf_low_order_sum,
                mid_order_sum=deltaf_mid_order_sum,
                mid_weight=deltaf_mid_weight,
                cluster_max_window_rbw=cluster_max_window_rbw,
                all_spurs_for_point=spur_defs,
                cluster_id=cid,
            )
        except Exception as ex:
            print(f"  ERROR measuring cluster {cid}: {ex}")
            print("  -> Logging NaN for all cluster spurs.")
            cluster_results = [
                (float("nan"), float("nan"), 0.0, True)
            ] * len(cluster_spurs)

        for spur_def, (meas_freq, p_dbm, window_half_width, used_fallback) in zip(
            cluster_spurs, cluster_results
        ):
            kind = spur_def.kind
            rel_dbc: Optional[float] = None
            error_msg: Optional[str] = None
            noise_limited = False

            if math.isnan(p_dbm) or math.isinf(p_dbm):
                error_msg = "Cluster measurement failed or invalid power"
                print(f"  ERROR for {spur_def.expression}: {error_msg}")
            else:
                rel_dbc = p_dbm - d_power_dbm
                if min_power_db is not None and p_dbm < min_power_db:
                    noise_limited = True

                f_nom_corr = (
                    spur_def.expected_corr_freq_hz if spur_def.expected_corr_freq_hz > 0.0
                    else spur_def.expected_freq_hz
                )
                freq_error_hz = abs(
                    meas_freq - f_nom_corr
                ) if not math.isnan(meas_freq) else float("nan")
                limit_hz = _freq_error_limit(span_hz, rbw_hz)
                if not math.isnan(freq_error_hz) and freq_error_hz > limit_hz:
                    print(
                        f"  WARNING (cluster): peak at {meas_freq / 1e9:.6f} GHz is far from "
                        f"expected {f_nom_corr / 1e9:.6f} GHz (Δ = {freq_error_hz / 1e3:.1f} kHz, "
                        f"limit ≈ {limit_hz / 1e3:.1f} kHz)."
                    )
                if used_fallback:
                    print(
                        "  NOTE: Cluster search window did not overlap trace; "
                        "global maximum in span was used as fallback for this spur."
                    )

                print(
                    f"  {kind} {spur_def.expression}: {meas_freq / 1e9:.6f} GHz, "
                    f"{p_dbm:.2f} dBm ({rel_dbc:.2f} dBc)"
                    + (" [noise-limited]" if noise_limited else "")
                )

            results.append(
                SpurResult(
                    kind=kind,
                    m=spur_def.m,
                    n=spur_def.n,
                    sign=spur_def.sign,
                    expression=spur_def.expression,
                    expected_freq_hz=spur_def.expected_freq_hz,
                    expected_corr_freq_hz=spur_def.expected_corr_freq_hz,
                    measured_freq_hz=meas_freq,
                    power_dbm=p_dbm,
                    rel_dbc=rel_dbc,
                    error=error_msg,
                    noise_limited=noise_limited,
                    cluster_id=spur_def.cluster_id,
                    cluster_size=spur_def.cluster_size,
                    analytic_coincidence_id=spur_def.analytic_coincidence_id,
                    analytic_coincidence_size=spur_def.analytic_coincidence_size,
                    measurement_scope="cluster",
                    cluster_window_half_width_hz=window_half_width,
                    cluster_window_fallback=used_fallback,
                )
            )

    # Measured coincidence topology (per point)
    assign_measured_coincidence_groups(
        results,
        rbw_hz=rbw_hz,
        measured_coincidence_factor=measured_coincidence_factor,
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
    eps = abs(step) * 1e-6
    while x <= stop + eps:
        vals.append(x)
        x += step
    return vals


def build_csv_row(
    r: SpurResult,
    lo_set_hz: float,
    if_set_hz: float,
    f_lo_model_hz: float,
    f_if_model_hz: float,
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
        freq_error_corr_hz = float("nan")
    else:
        freq_error_hz = r.measured_freq_hz - r.expected_freq_hz
        freq_error_corr_hz = r.measured_freq_hz - r.expected_corr_freq_hz

    return {
        "lo_index": lo_index,
        "if_index": if_index,
        "point_index": point_index,
        "lo_set_Hz": lo_set_hz,
        "if_set_Hz": if_set_hz,
        "lo_model_Hz": f_lo_model_hz,
        "if_model_Hz": f_if_model_hz,
        "mode": mode,
        "kind": r.kind,
        "m": r.m,
        "n": r.n,
        "sign": r.sign,
        "expression": r.expression,
        "expected_freq_Hz": r.expected_freq_hz,
        "expected_corr_freq_Hz": r.expected_corr_freq_hz,
        "measured_freq_Hz": r.measured_freq_hz,
        "freq_error_Hz": freq_error_hz,
        "freq_error_corr_Hz": freq_error_corr_hz,
        "power_dBm": r.power_dbm,
        "rel_dBc_vs_desired": "" if r.rel_dbc is None else r.rel_dbc,
        "noise_limited": 1 if r.noise_limited else 0,
        "cluster_id": r.cluster_id,
        "cluster_size": r.cluster_size,
        "analytic_coincidence_id": r.analytic_coincidence_id,
        "analytic_coincidence_size": r.analytic_coincidence_size,
        "measured_coincidence_id": r.measured_coincidence_id,
        "measured_coincidence_size": r.measured_coincidence_size,
        "measurement_scope": r.measurement_scope,
        "cluster_window_half_width_Hz": r.cluster_window_half_width_hz,
        "cluster_window_fallback": 1 if r.cluster_window_fallback else 0,
        "span_Hz": span_hz,
        "rbw_Hz": rbw_hz,
        "vbw_Hz": "" if vbw_hz is None else vbw_hz,
        "avg_sweeps": avg_sweeps,
        "marker_mode_global": marker_mode_global,
        "marker_mode_effective": _resolve_marker_mode(marker_mode_global, r.kind),
        "error": "" if r.error is None else r.error,
    }


def get_csv_fieldnames() -> List[str]:
    """Canonical CSV header used for both per-point and master CSVs."""
    return [
        "lo_index",
        "if_index",
        "point_index",
        "lo_set_Hz",
        "if_set_Hz",
        "lo_model_Hz",
        "if_model_Hz",
        "mode",
        "kind",
        "m",
        "n",
        "sign",
        "expression",
        "expected_freq_Hz",
        "expected_corr_freq_Hz",
        "measured_freq_Hz",
        "freq_error_Hz",
        "freq_error_corr_Hz",
        "power_dBm",
        "rel_dBc_vs_desired",
        "noise_limited",
        "cluster_id",
        "cluster_size",
        "analytic_coincidence_id",
        "analytic_coincidence_size",
        "measured_coincidence_id",
        "measured_coincidence_size",
        "measurement_scope",
        "cluster_window_half_width_Hz",
        "cluster_window_fallback",
        "span_Hz",
        "rbw_Hz",
        "vbw_Hz",
        "avg_sweeps",
        "marker_mode_global",
        "marker_mode_effective",
        "error",
    ]


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def load_calibration_file(path: Optional[str]) -> Dict[str, Dict[str, float]]:
    """
    Load simple linear calibration parameters from JSON.

    Expected structure (all entries optional):

        {
          "lo": {"a": 1.0, "b": 0.0},
          "if": {"a": 1.0, "b": 0.0}
        }

    Returns empty dict if path is None or file is not found / invalid.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        print(f"WARNING: Calibration file '{p}' not found, ignoring.")
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Calibration JSON is not a dict.")
        return data
    except Exception as ex:
        print(f"WARNING: Failed to load calibration file '{p}': {ex}")
        return {}


def apply_linear_cal(
    f_raw_hz: float,
    cal_entry: Optional[Dict[str, Any]],
) -> float:
    """Apply simple linear calibration f_cal = a * f_raw + b."""
    if not cal_entry:
        return f_raw_hz
    a = float(cal_entry.get("a", 1.0))
    b = float(cal_entry.get("b", 0.0))
    return a * f_raw_hz + b


def build_freq_model(
    set_freq_hz: float,
    readback_func,
    cal_entry: Optional[Dict[str, Any]],
    use_readback: bool,
    instrument_name: str,
) -> float:
    """
    Build modelled frequency:

        - Base = readback (if use_readback) or setpoint
        - Then apply linear calibration if provided
    """
    f_raw = set_freq_hz
    if use_readback and readback_func is not None:
        try:
            f_raw = float(readback_func())
        except Exception as ex:
            print(
                f"WARNING: Failed to read back frequency from {instrument_name}: {ex}. "
                "Using setpoint instead."
            )
    f_model = apply_linear_cal(f_raw, cal_entry)
    return f_model


# ---------------------------------------------------------------------------
# 2D LO/IF sweep main routine
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=(
            "Extended 2D LO/IF sweep for mixer spur scan "
            "(FSW + SMF100A LO source + HP 83711B IF source) "
            "with clusters / coincidences / Δf correction."
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
        "--smf",
        "--smbv",
        dest="smf_resource",
        default="TCPIP::192.168.1.102::HISLIP",
        help="SMF100A VISA resource string (LO generator).",
    )
    p.add_argument(
        "--sma",
        dest="sma_resource",
        default="GPIB0::19::INSTR",
        help="HP 83711B VISA resource string (IF generator), "
             "e.g. 'GPIB0::19::INSTR'.",
    )
    p.add_argument(
        "--fsw-reset",
        action="store_true",
        help="Preset (*RST) the FSW on connect.",
    )
    p.add_argument(
        "--smf-reset",
        "--smbv-reset",
        dest="smf_reset",
        action="store_true",
        help="Preset (*RST) the SMF100A (LO) on connect.",
    )
    p.add_argument(
        "--sma-reset",
        action="store_true",
        help="Preset (*RST) the HP 83711B (IF) on connect.",
    )

    # LO sweep
    p.add_argument(
        "--lo-start",
        type=float,
        required=True,
        help="Start LO frequency in Hz (setpoint).",
    )
    p.add_argument(
        "--lo-stop",
        type=float,
        required=True,
        help="Stop LO frequency in Hz (setpoint).",
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
        help="SMF100A LO output level in dBm.",
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
        help="Start IF frequency in Hz (setpoint).",
    )
    p.add_argument(
        "--if-stop",
        type=float,
        required=True,
        help="Stop IF frequency in Hz (setpoint).",
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
        help="IF output level in dBm.",
    )
    p.add_argument(
        "--if-settle-s",
        type=float,
        default=0.02,
        help="Settling time after changing IF frequency (seconds).",
    )

    # Mixer spur configuration
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
        help="Sweeps to average per product / cluster.",
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

    # Topology / coincidences / clusters
    p.add_argument(
        "--coincidence-factor",
        type=float,
        default=1.0,
        help="RBW multiplier for analytic coincidence tolerance.",
    )
    p.add_argument(
        "--measured-coincidence-factor",
        type=float,
        default=1.5,
        help="RBW multiplier for measured coincidence tolerance.",
    )
    p.add_argument(
        "--min-coincidence-hz",
        type=float,
        default=0.0,
        help="Minimum absolute analytic coincidence tolerance in Hz.",
    )
    p.add_argument(
        "--marker-confusion-factor",
        type=float,
        default=0.25,
        help="Fraction of span used in cluster contamination limit.",
    )
    p.add_argument(
        "--rbw-guard-factor",
        type=float,
        default=10.0,
        help="RBW multiplier in cluster contamination limit.",
    )
    p.add_argument(
        "--max-cluster-distance-hz",
        type=float,
        default=0.0,
        help=(
            "Hard upper bound on cluster contamination distance in Hz. "
            "If > 0, cluster contamination_limit is capped at this value."
        ),
    )

    # Δf / per-point correction
    p.add_argument(
        "--deltaf-low-order-sum",
        type=int,
        default=3,
        help="Max (m+n) for full Δf application (weight=1.0).",
    )
    p.add_argument(
        "--deltaf-mid-order-sum",
        type=int,
        default=5,
        help="Max (m+n) for partial Δf application.",
    )
    p.add_argument(
        "--deltaf-mid-weight",
        type=float,
        default=0.5,
        help="Weight for Δf when low_order_sum < m+n <= mid_order_sum.",
    )
    p.add_argument(
        "--cluster-max-window-rbw",
        type=float,
        default=100.0,
        help=(
            "Maximum half-window for cluster peak search, in units of RBW. "
            "If <= 0, no cap is applied."
        ),
    )

    # Calibration
    p.add_argument(
        "--calibrate-freq",
        action="store_true",
        help=(
            "Use generator frequency readback as base for frequency model "
            "instead of setpoints."
        ),
    )
    p.add_argument(
        "--calibration-file",
        type=str,
        default=None,
        help=(
            "JSON file with linear calibration parameters for LO/IF "
            "(see script header for format)."
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

    lo_values = frange(args.lo_start, args.lo_stop, args.lo_step)
    if_values = frange(args.if_start, args.if_stop, args.if_step)

    print(f"LO sweep points: {len(lo_values)}")
    print(f"IF sweep points: {len(if_values)}")
    total_points = len(lo_values) * len(if_values)
    print(f"Total (LO, IF) operating points: {total_points}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    RsFsw.assert_minimum_version("5.0.0")

    master_rows: List[Dict[str, Any]] = []

    # Load simple calibration
    cal_data = load_calibration_file(args.calibration_file)
    cal_lo = cal_data.get("lo")
    cal_if = cal_data.get("if")

    rm: Optional[pyvisa.ResourceManager] = None
    if_source: Optional[Any] = None     # HP 83711B (IF, pyvisa)
    smf: Optional[RsInstrument] = None  # SMF100A (LO)
    fsw: Optional[RsFsw] = None

    try:
        # LO source (SMF100A)
        print(f"Connecting to SMF100A at '{args.smf_resource}' ...")
        smf = RsInstrument(args.smf_resource, id_query=True, reset=args.smf_reset)
        print(f"SMF100A IDN: {smf.idn_string}")

        smf.write_str("OUTP:STAT ON")
        smf.write_str("SOUR:FREQ:MODE CW")
        smf.write_str(f"SOUR:POW:LEV:IMM:AMPL {args.lo_level_db} dBm")

        # IF source (HP 83711B via GPIB)
        print(f"Connecting to HP 83711B at '{args.sma_resource}' ...")
        rm = pyvisa.ResourceManager()
        if_source = rm.open_resource(args.sma_resource)

        # Optional reset
        if args.sma_reset:
            try:
                if_source.write("*RST")
                time.sleep(1.0)
            except Exception as ex:
                print(f"WARNING: Failed to reset 83711B cleanly: {ex}")

        # Try to identify the instrument
        try:
            idn = if_source.query("*IDN?")
            print(f"HP 83711B IDN: {idn.strip()}")
        except Exception:
            print("HP 83711B did not respond to *IDN? (legacy firmware or non-SCPI mode).")

        # Set IF output level and enable RF
        if_source.write(f"POW:AMPL {args.if_level_db} dBm")
        if_source.write("OUTP:STAT ON")

        # Spectrum analyzer (FSW)
        print(f"Connecting to FSW at '{args.fsw_resource}' ...")
        fsw = RsFsw(args.fsw_resource, reset=args.fsw_reset, id_query=True)
        print(f"FSW IDN: {fsw.utilities.idn_string}")

        fsw.instrument.select.set(fsw_enums.ChannelType.SpectrumAnalyzer)
        fsw.system.display.update.set(True)
        try:
            fsw.initiate.continuous.set(True)
        except Exception:
            pass

        point_index = 0

        for lo_idx, lo_set_hz in enumerate(lo_values):
            print("\n============================================================")
            print(f"LO set = {lo_set_hz / 1e9:.6f} GHz ({lo_idx + 1}/{len(lo_values)})")

            # LO = SMF100A (generic instrument)
            smf.write_str(f"SOUR:FREQ {lo_set_hz}")

            if args.lo_settle_s and args.lo_settle_s > 0:
                time.sleep(args.lo_settle_s)

            # Build LO model (SMF100A)
            f_lo_model_hz = build_freq_model(
                set_freq_hz=lo_set_hz,
                readback_func=(
                    lambda: smf.query_float("SOUR:FREQ?")
                ) if args.calibrate_freq else None,
                cal_entry=cal_lo,
                use_readback=args.calibrate_freq,
                instrument_name="SMF100A",
            )

            for if_idx, if_set_hz in enumerate(if_values):
                point_index += 1
                print("\n------------------------------------------------------------")
                print(
                    f"[{point_index}/{total_points}] "
                    f"LO_set = {lo_set_hz / 1e9:.6f} GHz, IF_set = {if_set_hz / 1e6:.3f} MHz"
                )

                # IF = HP 83711B (pyvisa over GPIB)
                if_source.write(f"FREQ:CW {if_set_hz} Hz")

                if args.if_settle_s and args.if_settle_s > 0:
                    time.sleep(args.if_settle_s)

                # IF model (HP 83711B)
                f_if_model_hz = build_freq_model(
                    set_freq_hz=if_set_hz,
                    readback_func=(
                        lambda: float(if_source.query("FREQ:CW?"))
                    ) if args.calibrate_freq else None,
                    cal_entry=cal_if,
                    use_readback=args.calibrate_freq,
                    instrument_name="HP 83711B",
                )

                try:
                    results = measure_spurs_for_point(
                        fsw=fsw,
                        lo_set_hz=lo_set_hz,
                        if_set_hz=if_set_hz,
                        f_lo_model_hz=f_lo_model_hz,
                        f_if_model_hz=f_if_model_hz,
                        mode=args.mode,
                        m_max=args.m_max,
                        n_max=args.n_max,
                        f_min_hz=args.f_min,
                        f_max_hz=args.f_max,
                        span_hz=args.span,
                        rbw_hz=args.rbw,
                        vbw_hz=args.vbw,
                        avg_sweeps=args.avg,
                        timeout_ms=args.timeout_ms,
                        marker_mode=args.marker_mode,
                        min_power_db=args.min_power_db,
                        min_desired_db=args.min_desired_db,
                        coincidence_factor=args.coincidence_factor,
                        measured_coincidence_factor=args.measured_coincidence_factor,
                        min_coincidence_hz=args.min_coincidence_hz,
                        marker_confusion_factor=args.marker_confusion_factor,
                        rbw_guard_factor=args.rbw_guard_factor,
                        max_cluster_distance_hz=args.max_cluster_distance_hz,
                        deltaf_low_order_sum=args.deltaf_low_order_sum,
                        deltaf_mid_order_sum=args.deltaf_mid_order_sum,
                        deltaf_mid_weight=args.deltaf_mid_weight,
                        cluster_max_window_rbw=args.cluster_max_window_rbw,
                    )
                except Exception as ex:
                    print(
                        "ERROR during spur scan at "
                        f"LO_set={lo_set_hz / 1e9:.6f} GHz, IF_set={if_set_hz / 1e6:.3f} MHz: {ex}"
                    )
                    print("Skipping this point and continuing with sweep.")
                    continue

                rows_for_point: List[Dict[str, Any]] = []
                for r in results:
                    row = build_csv_row(
                        r=r,
                        lo_set_hz=lo_set_hz,
                        if_set_hz=if_set_hz,
                        f_lo_model_hz=f_lo_model_hz,
                        f_if_model_hz=f_if_model_hz,
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

                if not args.no_per_point_csv:
                    lo_ghz = lo_set_hz / 1e9
                    if_mhz = if_set_hz / 1e6
                    csv_name = (
                        f"spur_results_lo_{lo_ghz:.6f}GHz_if_{if_mhz:.3f}MHz.csv"
                    )
                    csv_path = out_dir / csv_name

                    print(f"Writing per-point CSV to '{csv_path}' ...")
                    with csv_path.open("w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=get_csv_fieldnames())
                        writer.writeheader()
                        for row in rows_for_point:
                            writer.writerow(row)

    finally:
        if smf is not None:
            print("\nTurning SMF100A (LO) RF output OFF and closing session.")
            try:
                smf.write_str("OUTP:STAT OFF")
            except Exception as ex:
                print(f"WARNING: Failed to switch SMF LO RF OFF cleanly: {ex}")
            smf.close()

        if if_source is not None:
            print("\nTurning 83711B (IF) RF output OFF and closing session.")
            try:
                if_source.write("OUTP:STAT OFF")
            except Exception as ex:
                print(f"WARNING: Failed to switch 83711B IF RF OFF cleanly: {ex}")
            try:
                if_source.close()
            except Exception as ex:
                print(f"WARNING: Failed to close 83711B VISA session cleanly: {ex}")

        if rm is not None:
            try:
                rm.close()
            except Exception as ex:
                print(f"WARNING: Failed to close VISA ResourceManager: {ex}")

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
