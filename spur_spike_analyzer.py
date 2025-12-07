#!/usr/bin/env python3
"""
spur_spike_analyzer.py

Post-processing tool for mixer spur IF sweeps.

Problem:
    In small-span spur scans, the marker can sometimes lock onto a
    stronger nearby spur instead of the intended one. This shows up
    in the IF sweep as a sudden large spike (e.g. +30 dB) in a spur's
    curve, where that point's frequency/level matches another spur's
    curve instead.

What this tool does:
    - Reads the master CSV produced by mixer_spur_sweep.py
    - Groups data by spur (kind, m, n, sign)
    - For each spur:
        * Looks for "global" outliers (much higher level than the
          spur's median over IF)
        * Also requires a "local" jump vs neighboring IF points
    - For each candidate spike:
        * Looks at all other spurs at the same IF
        * Searches for another spur whose measured frequency and level
          are nearly identical to the spike
        * Optionally checks that the "source" spur's level is normal
          for that spur
    - Writes a human-readable report listing suspected mismeasurements:
        * IF
        * Mis-measured spur (branch)
        * Its level and deviation from its own curve
        * The likely "true" spur that was measured instead

Usage example:
    python spur_spike_analyzer.py \
        --master ltc5553_5p83g_if_sweep_master.csv \
        --report ltc5553_5p83g_spike_report.txt

You can tweak detection thresholds via command-line options.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpurRow:
    """Parsed representation of a single CSV row for analysis."""
    index: int                    # Position in the overall list
    original: dict                # Original CSV row (string values)

    lo_Hz: float
    if_Hz: float
    if_int: int                   # Rounded IF (Hz) for grouping
    kind: str
    m: int
    n: int
    sign: int
    expression: str
    expected_freq_Hz: float
    measured_freq_Hz: float
    rel_dBc: float                # rel_dBc_vs_desired


@dataclass
class BranchStats:
    branch_key: Tuple[str, int, int, int]
    median_rel: float
    levels: List[float]           # All rel_dBc values (for reference)


@dataclass
class SuspectedSpike:
    """Represents one suspected mismeasurement."""
    mis_row: SpurRow
    mis_dev_global: float         # mis_row.rel_dBc - median for that branch
    mis_local_jump: float         # vs neighbors (max neighbor level)
    src_row: SpurRow              # spur that spike likely belongs to
    src_dev_global: float         # src_row.rel_dBc - median for source branch


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def median(values: List[float]) -> float:
    """Simple median for a list of floats."""
    if not values:
        raise ValueError("median() requires at least one value")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    else:
        return 0.5 * (s[mid - 1] + s[mid])


def parse_float(value: str) -> Optional[float]:
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: str) -> Optional[int]:
    value = value.strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def load_spur_rows(master_csv: Path) -> List[SpurRow]:
    rows: List[SpurRow] = []
    with master_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for idx, r in enumerate(reader):
            rel = parse_float(r.get("rel_dBc_vs_desired", ""))
            if_Hz = parse_float(r.get("if_Hz", ""))
            lo_Hz = parse_float(r.get("lo_Hz", ""))
            meas_f = parse_float(r.get("measured_freq_Hz", ""))
            exp_f = parse_float(r.get("expected_freq_Hz", ""))

            # Skip rows without essential numeric fields
            if (
                rel is None
                or if_Hz is None
                or lo_Hz is None
                or meas_f is None
                or exp_f is None
            ):
                continue

            kind = (r.get("kind") or "").strip()
            m = parse_int(r.get("m", ""))
            n = parse_int(r.get("n", ""))
            sign = parse_int(r.get("sign", ""))
            expr = (r.get("expression") or "").strip()

            if m is None or n is None or sign is None:
                continue

            if_int = int(round(if_Hz))

            rows.append(
                SpurRow(
                    index=idx,
                    original=r,
                    lo_Hz=lo_Hz,
                    if_Hz=if_Hz,
                    if_int=if_int,
                    kind=kind,
                    m=m,
                    n=n,
                    sign=sign,
                    expression=expr,
                    expected_freq_Hz=exp_f,
                    measured_freq_Hz=meas_f,
                    rel_dBc=rel,
                )
            )
    return rows


def build_branch_stats(
    rows: List[SpurRow],
    min_points_per_branch: int = 3,
) -> Dict[Tuple[str, int, int, int], BranchStats]:
    """
    Group rows by spur branch (kind, m, n, sign) and compute median rel_dBc
    for each branch.
    """
    by_branch: Dict[Tuple[str, int, int, int], List[SpurRow]] = {}
    for r in rows:
        key = (r.kind, r.m, r.n, r.sign)
        by_branch.setdefault(key, []).append(r)

    stats: Dict[Tuple[str, int, int, int], BranchStats] = {}
    for key, br_rows in by_branch.items():
        if len(br_rows) < min_points_per_branch:
            # Not enough points to assess outliers
            continue
        levels = [r.rel_dBc for r in br_rows]
        med = median(levels)
        stats[key] = BranchStats(branch_key=key, median_rel=med, levels=levels)
    return stats


def index_by_if(rows: List[SpurRow]) -> Dict[int, List[SpurRow]]:
    """
    Build mapping from integer IF (Hz) to list of rows at that IF.
    """
    by_if: Dict[int, List[SpurRow]] = {}
    for r in rows:
        by_if.setdefault(r.if_int, []).append(r)
    return by_if


def find_suspected_spikes(
    rows: List[SpurRow],
    branch_stats: Dict[Tuple[str, int, int, int], BranchStats],
    global_spike_db: float = 30.0,
    local_jump_db: float = 15.0,
    freq_match_hz: float = 1e5,
    rel_match_db: float = 3.0,
    max_source_dev_db: float = 10.0,
) -> List[SuspectedSpike]:
    """
    Main detection routine.

    global_spike_db:
        Minimum deviation above the branch median (in dB) to consider a global spike.

    local_jump_db:
        Minimum jump vs neighboring IF points (max neighbor level) to treat as local spike.

    freq_match_hz:
        Maximum difference in measured frequency between mismeasured spur and its
        candidate source spur at the same IF.

    rel_match_db:
        Maximum difference in rel_dBc between mismeasured spur and candidate source spur.

    max_source_dev_db:
        Maximum allowed deviation from source spur's median for it to be considered
        a plausible "true" source.
    """
    # Index rows by branch and IF
    branch_to_rows: Dict[Tuple[str, int, int, int], List[SpurRow]] = {}
    for r in rows:
        key = (r.kind, r.m, r.n, r.sign)
        branch_to_rows.setdefault(key, []).append(r)

    for key in branch_to_rows.keys():
        branch_to_rows[key].sort(key=lambda r: r.if_Hz)

    by_if = index_by_if(rows)

    suspects: List[SuspectedSpike] = []

    for branch_key, br_rows in branch_to_rows.items():
        # Skip branch if we don't have stats (too few points)
        stats = branch_stats.get(branch_key)
        if stats is None:
            continue

        # Skip desired product: usually not the issue here
        kind, m, n, sign = branch_key
        if kind == "desired":
            continue

        med = stats.median_rel

        for idx, r in enumerate(br_rows):
            # Global spike check: much higher than median
            global_dev = r.rel_dBc - med
            if global_dev < global_spike_db:
                continue

            # Local neighbors: previous / next IF points for this branch
            neighbor_levels: List[float] = []
            if idx > 0:
                neighbor_levels.append(br_rows[idx - 1].rel_dBc)
            if idx + 1 < len(br_rows):
                neighbor_levels.append(br_rows[idx + 1].rel_dBc)

            if not neighbor_levels:
                # Only one point for this branch; can't assess local jump
                continue

            local_jump = r.rel_dBc - max(neighbor_levels)
            if local_jump < local_jump_db:
                continue

            # At this point, r is a strong candidate spike for this branch.
            # Now look for another spur at the same IF that matches in
            # frequency and level.
            same_if_rows = by_if.get(r.if_int, [])
            best_match: Optional[Tuple[SpurRow, float, float]] = None

            for other in same_if_rows:
                if other.index == r.index:
                    continue

                # Require close measured frequency
                df = abs(other.measured_freq_Hz - r.measured_freq_Hz)
                if df > freq_match_hz:
                    continue

                # Require close rel_dBc
                drel = abs(other.rel_dBc - r.rel_dBc)
                if drel > rel_match_db:
                    continue

                # Evaluate how "normal" this candidate is for its own branch
                other_key = (other.kind, other.m, other.n, other.sign)
                other_stats = branch_stats.get(other_key)
                if other_stats is None:
                    continue  # Not enough data to evaluate this branch

                src_dev = other.rel_dBc - other_stats.median_rel
                if abs(src_dev) > max_source_dev_db:
                    # This candidate is also abnormal for its own spur curve;
                    # treat as less likely to be the true source.
                    continue

                # Keep the best match (smallest |src_dev|)
                score = abs(src_dev)
                if best_match is None or score < best_match[1]:
                    best_match = (other, src_dev, drel)

            if best_match is None:
                # No convincing source spur at the same IF
                continue

            other, src_dev, _ = best_match
            suspects.append(
                SuspectedSpike(
                    mis_row=r,
                    mis_dev_global=global_dev,
                    mis_local_jump=local_jump,
                    src_row=other,
                    src_dev_global=src_dev,
                )
            )

    return suspects


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_suspect(s: SuspectedSpike, case_index: int) -> str:
    r = s.mis_row
    o = s.src_row

    lines = []
    lines.append(f"Case {case_index}:")
    lines.append(f"  IF: {r.if_Hz/1e6:.3f} MHz")

    lines.append("  Suspected mismeasured spur:")
    lines.append(f"    branch: kind={r.kind}, m={r.m}, n={r.n}, sign={r.sign}")
    if r.expression:
        lines.append(f'    expression: "{r.expression}"')
    lines.append(f"    measured_freq: {r.measured_freq_Hz/1e9:.6f} GHz")
    lines.append(f"    expected_freq: {r.expected_freq_Hz/1e9:.6f} GHz")
    lines.append(
        f"    rel_dBc: {r.rel_dBc:.2f} dBc "
        f"(deviation from this spur's median: +{s.mis_dev_global:.2f} dB)"
    )
    lines.append("    local jump vs neighbors: "
                 f"+{s.mis_local_jump:.2f} dB")

    lines.append("  Likely true spur measured instead:")
    lines.append(f"    branch: kind={o.kind}, m={o.m}, n={o.n}, sign={o.sign}")
    if o.expression:
        lines.append(f'    expression: "{o.expression}"')
    lines.append(f"    measured_freq: {o.measured_freq_Hz/1e9:.6f} GHz")
    lines.append(f"    expected_freq: {o.expected_freq_Hz/1e9:.6f} GHz")
    lines.append(
        f"    rel_dBc: {o.rel_dBc:.2f} dBc "
        f"(deviation from this spur's median: {s.src_dev_global:+.2f} dB)"
    )

    lines.append("")  # blank line after case
    return "\n".join(lines)


def write_report(
    master_csv: Path,
    report_path: Path,
    suspects: List[SuspectedSpike],
) -> None:
    with report_path.open("w", newline="\n") as f:
        f.write("Spur mismeasurement analysis report\n")
        f.write("===================================\n\n")
        f.write(f"Master CSV: {master_csv}\n")
        f.write(f"Total suspected mismeasurements: {len(suspects)}\n\n")

        if not suspects:
            f.write("No suspected mismeasurements found with current thresholds.\n")
            return

        for i, s in enumerate(suspects, start=1):
            f.write(format_suspect(s, i))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze mixer spur sweep master CSV for mismeasurements "
                    "caused by marker locking onto the wrong spur."
    )

    p.add_argument(
        "--master",
        required=True,
        help="Path to master CSV from mixer_spur_sweep.py.",
    )
    p.add_argument(
        "--report",
        default="spur_spike_report.txt",
        help="Output text report filename.",
    )

    # Detection thresholds
    p.add_argument(
        "--global-spike-db",
        type=float,
        default=30.0,
        help=(
            "Minimum deviation above spur's median (dB) to treat a point "
            "as a global spike (default: 30 dB)."
        ),
    )
    p.add_argument(
        "--local-jump-db",
        type=float,
        default=15.0,
        help=(
            "Minimum jump vs neighboring IF points (dB) required to treat as "
            "a local spike (default: 15 dB)."
        ),
    )
    p.add_argument(
        "--freq-match-hz",
        type=float,
        default=1e5,
        help=(
            "Max frequency difference (Hz) between mismeasured spur and source spur "
            "at same IF (default: 100 kHz)."
        ),
    )
    p.add_argument(
        "--rel-match-db",
        type=float,
        default=3.0,
        help=(
            "Max difference in rel_dBc between mismeasured spur and source spur "
            "(default: 3 dB)."
        ),
    )
    p.add_argument(
        "--max-source-dev-db",
        type=float,
        default=10.0,
        help=(
            "Max deviation from source spur's median (dB) to treat it as a plausible "
            "true spur (default: 10 dB)."
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()

    master_csv = Path(args.master)
    report_path = Path(args.report)

    if not master_csv.exists():
        raise FileNotFoundError(f"Master CSV not found: {master_csv}")

    print(f"Loading master CSV: {master_csv}")
    rows = load_spur_rows(master_csv)
    print(f"Loaded {len(rows)} valid rows.")

    if not rows:
        print("No valid rows found. Exiting.")
        return

    branch_stats = build_branch_stats(rows)
    print(f"Branches with sufficient data: {len(branch_stats)}")

    suspects = find_suspected_spikes(
        rows=rows,
        branch_stats=branch_stats,
        global_spike_db=args.global_spike_db,
        local_jump_db=args.local_jump_db,
        freq_match_hz=args.freq_match_hz,
        rel_match_db=args.rel_match_db,
        max_source_dev_db=args.max_source_dev_db,
    )

    print(f"Suspected mismeasurements found: {len(suspects)}")
    write_report(master_csv, report_path, suspects)
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()