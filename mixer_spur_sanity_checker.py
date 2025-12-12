#!/usr/bin/env python3
"""
Post-pass sanity checker for mixer spur sweeps.

Takes the master CSV from the extended spur tool and writes an
"enhanced" CSV with additional flags:

    - misidentified_candidate (0/1)
    - best_match_m, best_match_n, best_match_sign
    - continuity_outlier (0/1)
    - ambiguous_multi_assign (0/1)
    - cluster_window_suspect (0/1)

Sanity checker logic (summary):

1. For each (LO, IF) point (lo_index, if_index):
   - Rebuild analytic spur list using lo_model_Hz / if_model_Hz,
     m_max/n_max, f_min/f_max, and mode.
   - Optionally apply Δf-based correction from the measured desired spur
     (if --sanity-use-corr is enabled) with the same low/mid order
     weighting logic as the measurement script.
   - For each measured spur at that point:
       * Identify the "self" spur in the analytic list with same (m, n, sign).
       * Find the "best" spur in the analytic list closest in frequency
         to the measured frequency.
       * Flag misidentified_candidate according to RBW-scaled thresholds,
         BUT avoid hard mis-ID if the alternative spur is in the same
         measured coincidence group at that point (ambiguity at analyzer
         resolution).

2. Continuity checks:
   - For each spur family (kind, m, n, sign):
       * Along LO: for each fixed if_index, sort by lo_index and compare
         the change in measured frequency vs the change in expected
         (or expected_corr). Large residuals are flagged.
       * Along IF: for each fixed lo_index, sort by if_index and do the same.

3. Ambiguous multi-assignment:
   - At each (lo_index, if_index), group spurs whose measured_freq_Hz
     are within k_multi_rbw * RBW of each other but have different
     (m, n, sign). Flag those rows as ambiguous_multi_assign.

4. Cluster window fallback highlighting:
   - Any row with cluster_window_fallback == 1 in the master CSV is
     additionally flagged with cluster_window_suspect = 1.

Usage example:

    python3 mixer_spur_sanity_checker.py \\
        --input spur_sweep_results/spur_sweep_master.csv \\
        --output spur_sweep_results/spur_sweep_master_sanity.csv \\
        --m-max 5 --n-max 5 \\
        --f-min 0 --f-max 43e9 \\
        --sanity-use-corr

NOTE:
    - m_max, n_max, f_min, f_max should match the settings used in the
      original sweep.
    - Δf parameters (--deltaf-*) should match the original sweep if you
      want identical corrected expectations.
    - The script expects the master CSV created by the extended spur
      tool (with columns like measured_coincidence_id and
      cluster_window_fallback present, but will still run if they are
      missing, in which case those features simply don't trigger).
"""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def parse_float(val: Any, default: float = float("nan")) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def parse_int(val: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Analytic spur model (local copy, matching the measurement script)
# ---------------------------------------------------------------------------

@dataclass
class AnalyticSpur:
    kind: str
    m: int
    n: int
    sign: int     # +1 or -1
    expression: str
    expected_freq_hz: float
    expected_corr_freq_hz: float = 0.0


def build_analytic_spur_list(
    f_lo_hz: float,
    f_if_hz: float,
    m_max: int,
    n_max: int,
    desired_mode: str,
    f_min_hz: float,
    f_max_hz: float,
) -> List[AnalyticSpur]:
    """
    Build analytic list of spurs for a given (LO, IF) model point,
    mirroring the logic of build_spur_list() in the measurement script.

    NOTE:
        - No de-duplication here; all (m, n, sign, kind) combinations
          are kept if their frequency is within [f_min_hz, f_max_hz].
        - expected_corr_freq_hz is initially equal to expected_freq_hz
          and may be modified later by Δf weighting.
    """
    spur_list: List[AnalyticSpur] = []

    mode_l = desired_mode.lower()
    if mode_l == "lo+if":
        desired_sign = +1
    elif mode_l == "lo-if":
        desired_sign = -1
    else:
        raise ValueError("desired_mode must be 'lo+if' or 'lo-if'")

    # Desired product: 1*LO ± 1*IF
    f_desired = abs(1 * f_lo_hz + desired_sign * 1 * f_if_hz)
    if f_min_hz <= f_desired <= f_max_hz:
        spur_list.append(
            AnalyticSpur(
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
            if m == 0 and n == 0:
                continue

            if m == 0 or n == 0:
                sign_values = (+1,)
            else:
                sign_values = (+1, -1)

            for sign in sign_values:
                # Skip desired already added
                if m == 1 and n == 1 and sign == desired_sign:
                    continue

                freq = abs(m * f_lo_hz + sign * n * f_if_hz)
                if freq < f_min_hz or freq > f_max_hz:
                    continue

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
                    AnalyticSpur(
                        kind=kind,
                        m=m,
                        n=n,
                        sign=sign,
                        expression=expr,
                        expected_freq_hz=freq,
                        expected_corr_freq_hz=freq,
                    )
                )

    spur_list.sort(key=lambda s: s.expected_freq_hz)

    # Ensure desired first if present
    for i, s in enumerate(spur_list):
        if s.kind == "desired":
            spur_list.insert(0, spur_list.pop(i))
            break

    return spur_list


# ---------------------------------------------------------------------------
# Δf weighting (same logic as in the main script)
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
    Order-aware weight for Δf application:

        m+n <= low_order_sum           -> 1.0
        low_order_sum < m+n <= mid     -> mid_weight
        m+n > mid_order_sum           -> 0.0
    """
    order_sum = m + n
    if order_sum <= low_order_sum:
        return 1.0
    if order_sum <= mid_order_sum:
        return float(mid_weight)
    return 0.0


def apply_deltaf_correction(
    spurs: List[AnalyticSpur],
    delta_desired_hz: float,
    low_order_sum: int,
    mid_order_sum: int,
    mid_weight: float,
) -> None:
    """
    Apply Δf from the desired product to all analytic spurs using
    the same weighting strategy as the measurement script.
    """
    for s in spurs:
        w = spur_weight(
            s.m, s.n, s.sign,
            low_order_sum=low_order_sum,
            mid_order_sum=mid_order_sum,
            mid_weight=mid_weight,
        )
        s.expected_corr_freq_hz = s.expected_freq_hz + w * delta_desired_hz


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def misidentification_pass_for_point(
    rows: List[Dict[str, Any]],
    analytic_spurs: List[AnalyticSpur],
    use_corr: bool,
    k1_rbw: float,
    k2_rbw: float,
    alpha: float,
) -> None:
    """
    For a single (LO, IF) point (all rows share same lo_index, if_index),
    perform nearest-spur rematching and set:

        - misidentified_candidate
        - best_match_m / best_match_n / best_match_sign

    IMPORTANT:
        If the analytically best-matching spur lives in the same
        measured coincidence group (same measured_coincidence_id)
        as the current row, the tone is intrinsically ambiguous at
        analyzer resolution, so we DO NOT raise a hard
        misidentified_candidate flag in that case.
    """
    if not rows or not analytic_spurs:
        return

    # Build lookup by (m, n, sign)
    analytic_by_idx: Dict[Tuple[int, int, int], AnalyticSpur] = {}
    for s in analytic_spurs:
        analytic_by_idx[(s.m, s.n, s.sign)] = s

    for row in rows:
        row.setdefault("misidentified_candidate", 0)
        row.setdefault("best_match_m", "")
        row.setdefault("best_match_n", "")
        row.setdefault("best_match_sign", "")

        f_meas = parse_float(row.get("measured_freq_Hz"))
        if math.isnan(f_meas):
            continue

        rbw_hz = parse_float(row.get("rbw_Hz"))
        if rbw_hz <= 0.0 or math.isnan(rbw_hz):
            continue

        m = parse_int(row.get("m"))
        n = parse_int(row.get("n"))
        sign = parse_int(row.get("sign"))
        if m is None or n is None or sign is None:
            continue

        key = (m, n, sign)
        self_spur = analytic_by_idx.get(key)
        if self_spur is None:
            # No analytic match; can't judge mis-ID robustly
            continue

        if use_corr:
            f_self = self_spur.expected_corr_freq_hz
        else:
            f_self = self_spur.expected_freq_hz

        d_self = abs(f_meas - f_self)

        # Find nearest analytic spur to measured frequency
        best_spur: Optional[AnalyticSpur] = None
        d_best = float("inf")
        for s in analytic_spurs:
            f_exp = s.expected_corr_freq_hz if use_corr else s.expected_freq_hz
            d = abs(f_meas - f_exp)
            if d < d_best:
                d_best = d
                best_spur = s

        if best_spur is None:
            continue

        # If best spur is actually the same as self, no mis-ID
        if (best_spur.m, best_spur.n, best_spur.sign) == key:
            continue

        # Check thresholds for potential misidentification
        candidate = (
            d_self > k1_rbw * rbw_hz and
            d_best < k2_rbw * rbw_hz and
            d_best < d_self / max(alpha, 1e-9)
        )

        if not candidate:
            continue

        # Additional check: if the best alternative spur lives in the same
        # measured coincidence group as this row, we do NOT raise a hard
        # mis-ID, since FSW cannot resolve them as separate tones.
        mcid_self = parse_int(row.get("measured_coincidence_id"), default=None)
        if mcid_self is not None and mcid_self >= 0:
            # Look for a row in the same coincidence group with (m,n,sign)
            # equal to best_spur
            for other in rows:
                if other is row:
                    continue
                mcid_other = parse_int(other.get("measured_coincidence_id"), default=None)
                if mcid_other is None or mcid_other != mcid_self:
                    continue
                om = parse_int(other.get("m"))
                on = parse_int(other.get("n"))
                osign = parse_int(other.get("sign"))
                if (
                    om == best_spur.m and
                    on == best_spur.n and
                    osign == best_spur.sign
                ):
                    # Ambiguous coincident case -> skip hard mis-ID
                    candidate = False
                    break

        if not candidate:
            continue

        # Strong mis-ID candidate: thresholds satisfied and not explained
        # as a coincidence group.
        row["misidentified_candidate"] = 1
        row["best_match_m"] = best_spur.m
        row["best_match_n"] = best_spur.n
        row["best_match_sign"] = best_spur.sign


def continuity_pass(
    all_rows: List[Dict[str, Any]],
    use_corr: bool,
    k_continuity_rbw: float,
) -> None:
    """
    Continuity check along LO and IF directions.

    For each spur family (kind, m, n, sign):

        - For each fixed if_index, sort by lo_index and compare
          delta(measured) vs delta(expected).
        - For each fixed lo_index, sort by if_index and do the same.

    Flags rows with 'continuity_outlier' = 1 if residual exceeds
    k_continuity_rbw * RBW.
    """
    for row in all_rows:
        row.setdefault("continuity_outlier", 0)

    # Group rows by spur family
    family_map: Dict[Tuple[str, int, int, int], List[Dict[str, Any]]] = {}
    for row in all_rows:
        kind = row.get("kind", "")
        m = parse_int(row.get("m"))
        n = parse_int(row.get("n"))
        sign = parse_int(row.get("sign"))
        if m is None or n is None or sign is None:
            continue
        key = (kind, m, n, sign)
        family_map.setdefault(key, []).append(row)

    def get_expected(row: Dict[str, Any]) -> float:
        if use_corr:
            return parse_float(row.get("expected_corr_freq_Hz"))
        return parse_float(row.get("expected_freq_Hz"))

    for family_key, family_rows in family_map.items():
        # LO-direction continuity: group by if_index
        by_if: Dict[int, List[Dict[str, Any]]] = {}
        for row in family_rows:
            if_idx = parse_int(row.get("if_index"), default=-1)
            if if_idx is None:
                continue
            by_if.setdefault(if_idx, []).append(row)

        for if_idx, rows_if in by_if.items():
            # Sort by lo_index
            rows_if.sort(key=lambda r: (parse_int(r.get("lo_index"), default=-1) or -1))
            for i in range(len(rows_if) - 1):
                r1 = rows_if[i]
                r2 = rows_if[i + 1]

                f1_meas = parse_float(r1.get("measured_freq_Hz"))
                f2_meas = parse_float(r2.get("measured_freq_Hz"))
                if math.isnan(f1_meas) or math.isnan(f2_meas):
                    continue

                f1_exp = get_expected(r1)
                f2_exp = get_expected(r2)
                if math.isnan(f1_exp) or math.isnan(f2_exp):
                    continue

                rbw1 = parse_float(r1.get("rbw_Hz"))
                rbw2 = parse_float(r2.get("rbw_Hz"))
                rbw = rbw1 if rbw1 > 0 else rbw2
                if rbw <= 0 or math.isnan(rbw):
                    continue

                delta_meas = f2_meas - f1_meas
                delta_exp = f2_exp - f1_exp
                residual = abs(delta_meas - delta_exp)

                if residual > k_continuity_rbw * rbw:
                    r1["continuity_outlier"] = 1
                    r2["continuity_outlier"] = 1

        # IF-direction continuity: group by lo_index
        by_lo: Dict[int, List[Dict[str, Any]]] = {}
        for row in family_rows:
            lo_idx = parse_int(row.get("lo_index"), default=-1)
            if lo_idx is None:
                continue
            by_lo.setdefault(lo_idx, []).append(row)

        for lo_idx, rows_lo in by_lo.items():
            # Sort by if_index
            rows_lo.sort(key=lambda r: (parse_int(r.get("if_index"), default=-1) or -1))
            for i in range(len(rows_lo) - 1):
                r1 = rows_lo[i]
                r2 = rows_lo[i + 1]

                f1_meas = parse_float(r1.get("measured_freq_Hz"))
                f2_meas = parse_float(r2.get("measured_freq_Hz"))
                if math.isnan(f1_meas) or math.isnan(f2_meas):
                    continue

                f1_exp = get_expected(r1)
                f2_exp = get_expected(r2)
                if math.isnan(f1_exp) or math.isnan(f2_exp):
                    continue

                rbw1 = parse_float(r1.get("rbw_Hz"))
                rbw2 = parse_float(r2.get("rbw_Hz"))
                rbw = rbw1 if rbw1 > 0 else rbw2
                if rbw <= 0 or math.isnan(rbw):
                    continue

                delta_meas = f2_meas - f1_meas
                delta_exp = f2_exp - f1_exp
                residual = abs(delta_meas - delta_exp)

                if residual > k_continuity_rbw * rbw:
                    r1["continuity_outlier"] = 1
                    r2["continuity_outlier"] = 1


def multi_assignment_pass(
    point_groups: Dict[Tuple[int, int], List[Dict[str, Any]]],
    k_multi_rbw: float,
) -> None:
    """
    For each (lo_index, if_index) group, find sets of rows that appear to
    share the same measured tone:

        - measured frequencies within k_multi_rbw * RBW
        - but different (m, n, sign)

    Flags such rows with 'ambiguous_multi_assign' = 1.
    """
    for rows in point_groups.values():
        for row in rows:
            row.setdefault("ambiguous_multi_assign", 0)

        # Filter valid measured frequencies
        valid_rows = [
            r for r in rows
            if not math.isnan(parse_float(r.get("measured_freq_Hz")))
        ]
        if len(valid_rows) < 2:
            continue

        valid_rows.sort(key=lambda r: parse_float(r.get("measured_freq_Hz")))

        current_group: List[Dict[str, Any]] = []

        def finalize_group(group: List[Dict[str, Any]]) -> None:
            if len(group) <= 1:
                return
            # Check if at least two rows in group have different (m,n,sign)
            idx_set = set(
                (parse_int(r.get("m")), parse_int(r.get("n")), parse_int(r.get("sign")))
                for r in group
            )
            if len(idx_set) > 1:
                for r in group:
                    r["ambiguous_multi_assign"] = 1

        for r in valid_rows:
            if not current_group:
                current_group = [r]
                continue

            # Use RBW from current row (assumed same across group)
            rbw = parse_float(r.get("rbw_Hz"))
            if rbw <= 0 or math.isnan(rbw):
                rbw = parse_float(current_group[0].get("rbw_Hz"))
            if rbw <= 0 or math.isnan(rbw):
                # Cannot determine RBW; flush group and continue
                finalize_group(current_group)
                current_group = [r]
                continue

            f_ref = parse_float(current_group[0].get("measured_freq_Hz"))
            f = parse_float(r.get("measured_freq_Hz"))
            if math.isnan(f_ref) or math.isnan(f):
                finalize_group(current_group)
                current_group = [r]
                continue

            if abs(f - f_ref) <= k_multi_rbw * rbw:
                current_group.append(r)
            else:
                finalize_group(current_group)
                current_group = [r]

        if current_group:
            finalize_group(current_group)


# ---------------------------------------------------------------------------
# Main CLI logic
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Sanity checker for mixer spur sweep master CSV."
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Input master CSV from the extended spur tool.",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output enhanced master CSV with sanity flags.",
    )

    # Analytic spur generation
    ap.add_argument(
        "--m-max",
        type=int,
        default=5,
        help="Maximum LO order m to consider (0..m_max) for analytic spurs.",
    )
    ap.add_argument(
        "--n-max",
        type=int,
        default=5,
        help="Maximum IF order n to consider (0..n_max) for analytic spurs.",
    )
    ap.add_argument(
        "--f-min",
        type=float,
        default=0.0,
        help="Minimum RF frequency to consider for analytic spurs (Hz).",
    )
    ap.add_argument(
        "--f-max",
        type=float,
        default=43e9,
        help="Maximum RF frequency to consider for analytic spurs (Hz).",
    )

    # Δf behaviour within sanity checker
    ap.add_argument(
        "--sanity-use-corr",
        action="store_true",
        help=(
            "Use Δf-corrected expected frequencies (via local recomputation) "
            "for matching instead of pure analytic expectations."
        ),
    )
    ap.add_argument(
        "--deltaf-low-order-sum",
        type=int,
        default=3,
        help="Max (m+n) for full Δf application (weight=1.0) in sanity checker.",
    )
    ap.add_argument(
        "--deltaf-mid-order-sum",
        type=int,
        default=5,
        help="Max (m+n) for partial Δf application in sanity checker.",
    )
    ap.add_argument(
        "--deltaf-mid-weight",
        type=float,
        default=0.5,
        help="Weight for Δf when low_order_sum < m+n <= mid_order_sum.",
    )

    # Misidentification thresholds
    ap.add_argument(
        "--k1-rbw",
        type=float,
        default=10.0,
        help="RBW multiplier for 'd_self > k1 * RBW' condition.",
    )
    ap.add_argument(
        "--k2-rbw",
        type=float,
        default=3.0,
        help="RBW multiplier for 'd_best < k2 * RBW' condition.",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Factor for 'd_best < d_self / alpha' condition.",
    )

    # Continuity thresholds
    ap.add_argument(
        "--continuity-k-rbw",
        type=float,
        default=10.0,
        help="RBW multiplier for continuity residual threshold.",
    )

    # Multi-assignment threshold
    ap.add_argument(
        "--multi-k-rbw",
        type=float,
        default=1.5,
        help="RBW multiplier for ambiguous multi-assignment threshold.",
    )

    args = ap.parse_args()

    if args.deltaf_mid_order_sum < args.deltaf_low_order_sum:
        raise RuntimeError(
            "--deltaf-mid-order-sum must be >= --deltaf-low-order-sum"
        )

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV '{input_path}' not found")

    print(
        "Sanity checker configuration:\n"
        f"  Analytic spur settings: m_max={args.m_max}, n_max={args.n_max}, "
        f"f_min={args.f_min:.3g} Hz, f_max={args.f_max:.3g} Hz\n"
        f"  Δf params: low_order_sum={args.deltaf_low_order_sum}, "
        f"mid_order_sum={args.deltaf_mid_order_sum}, "
        f"mid_weight={args.deltaf_mid_weight}\n"
        "NOTE: These should match the sweep script settings as closely as possible."
    )

    # ----------------------------------------------------------------------
    # Read input CSV
    # ----------------------------------------------------------------------
    with input_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames_in = reader.fieldnames or []
        rows: List[Dict[str, Any]] = [dict(r) for r in reader]

    if not rows:
        print("Input CSV has no data rows; nothing to do.")
        return

    # ----------------------------------------------------------------------
    # Group rows by (lo_index, if_index) to rebuild per-point spurlists
    # ----------------------------------------------------------------------
    point_groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for row in rows:
        lo_idx = parse_int(row.get("lo_index"), default=-1)
        if_idx = parse_int(row.get("if_index"), default=-1)
        if lo_idx is None or if_idx is None:
            # Skip rows without valid indices
            continue
        key = (lo_idx, if_idx)
        point_groups.setdefault(key, []).append(row)

    # ----------------------------------------------------------------------
    # Per-point misidentification pass (with analytic rebuilding)
    # ----------------------------------------------------------------------
    for key, point_rows in point_groups.items():
        if not point_rows:
            continue

        # Assume all rows in this group share mode, lo_model_Hz, if_model_Hz
        ref = point_rows[0]
        mode = (ref.get("mode") or "").lower()
        f_lo_model = parse_float(ref.get("lo_model_Hz"))
        f_if_model = parse_float(ref.get("if_model_Hz"))

        if math.isnan(f_lo_model) or math.isnan(f_if_model):
            # Can't build analytic spurs reliably, skip this point
            continue

        analytic_spurs = build_analytic_spur_list(
            f_lo_hz=f_lo_model,
            f_if_hz=f_if_model,
            m_max=args.m_max,
            n_max=args.n_max,
            desired_mode=mode,
            f_min_hz=args.f_min,
            f_max_hz=args.f_max,
        )
        if not analytic_spurs:
            continue

        # Optional Δf application, using measured desired from CSV
        if args.sanity_use_corr:
            desired_row = None
            for r in point_rows:
                if (r.get("kind") or "").lower() == "desired":
                    desired_row = r
                    break

            if desired_row is not None:
                f_meas_desired = parse_float(desired_row.get("measured_freq_Hz"))
                # Find analytic desired spur
                desired_analytic = None
                for s in analytic_spurs:
                    if s.kind == "desired":
                        desired_analytic = s
                        break
                if (
                    desired_analytic is not None and
                    not math.isnan(f_meas_desired)
                ):
                    delta_desired = (
                        f_meas_desired - desired_analytic.expected_freq_hz
                    )
                    apply_deltaf_correction(
                        analytic_spurs,
                        delta_desired_hz=delta_desired,
                        low_order_sum=args.deltaf_low_order_sum,
                        mid_order_sum=args.deltaf_mid_order_sum,
                        mid_weight=args.deltaf_mid_weight,
                    )

        # Run misidentification logic for this point
        misidentification_pass_for_point(
            rows=point_rows,
            analytic_spurs=analytic_spurs,
            use_corr=args.sanity_use_corr,
            k1_rbw=args.k1_rbw,
            k2_rbw=args.k2_rbw,
            alpha=args.alpha,
        )

    # ----------------------------------------------------------------------
    # Continuity checks across all points and spur families
    # ----------------------------------------------------------------------
    continuity_pass(
        all_rows=rows,
        use_corr=args.sanity_use_corr,
        k_continuity_rbw=args.continuity_k_rbw,
    )

    # ----------------------------------------------------------------------
    # Multi-assignment check per (LO, IF) point
    # ----------------------------------------------------------------------
    multi_assignment_pass(
        point_groups=point_groups,
        k_multi_rbw=args.multi_k_rbw,
    )

    # ----------------------------------------------------------------------
    # Prepare output fieldnames (input + new ones)
    # ----------------------------------------------------------------------
    new_fields = [
        "misidentified_candidate",
        "best_match_m",
        "best_match_n",
        "best_match_sign",
        "continuity_outlier",
        "ambiguous_multi_assign",
        "cluster_window_suspect",
    ]

    fieldnames_out = list(fieldnames_in)
    for nf in new_fields:
        if nf not in fieldnames_out:
            fieldnames_out.append(nf)

    # Ensure every row has all new fields (with defaults if missing)
    for row in rows:
        row.setdefault("misidentified_candidate", 0)
        row.setdefault("best_match_m", "")
        row.setdefault("best_match_n", "")
        row.setdefault("best_match_sign", "")
        row.setdefault("continuity_outlier", 0)
        row.setdefault("ambiguous_multi_assign", 0)
        # Elevate cluster_window_fallback as a suspect flag
        cwf = parse_int(row.get("cluster_window_fallback"), default=0) or 0
        row.setdefault("cluster_window_suspect", 1 if cwf != 0 else 0)

    # ----------------------------------------------------------------------
    # Write enhanced CSV
    # ----------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_out)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Sanity-enhanced CSV written to '{output_path}'")
    print(f"  Rows: {len(rows)}")
    print("Done.")


if __name__ == "__main__":
    main()
