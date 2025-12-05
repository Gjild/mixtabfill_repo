#!/usr/bin/env python3
"""
IF sweep wrapper for mixer spur measurement

- Controls R&S SMBV100A CW generator as the IF source
- Sweeps IF from start to stop with given step
- For each IF point:
    * Sets SMBV frequency + power
    * Calls the existing mixer spur scanner (FSW script)
    * Writes per-IF spur CSV via the existing code
- After the sweep:
    * Aggregates all per-IF CSVs into one master CSV
    * Generates a plot of RF spur frequency vs dBc for all spurs

Requirements:
    pip install RsSmbv matplotlib
    The existing script must be available as 'mixer_spur_scan.py'
    and must export run_spur_scan(args).
"""

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import matplotlib.pyplot as plt

from RsSmbv import RsSmbv, enums

# Import your existing scanner
#   Make sure your original file is named 'mixer_spur_scan.py'
from mixer_spur_scan import run_spur_scan


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_scan_args(
    fsw_resource: str,
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
    csv_path: Path,
    reset_fsw: bool,
) -> SimpleNamespace:
    """
    Build an argparse-like namespace for run_spur_scan() so we can call it
    programmatically without touching the original code.
    """
    return SimpleNamespace(
        # FSW connection
        resource=fsw_resource,
        reset=reset_fsw,
        # Mixer frequencies
        lo=lo_hz,
        if_freq=if_hz,
        mode=mode,
        # Spur orders
        m_max=m_max,
        n_max=n_max,
        # RF window
        f_min=f_min_hz,
        f_max=f_max_hz,
        # De-duplication
        dedupe_freq=dedupe_freq_hz,
        # Measurement settings
        span=span_hz,
        rbw=rbw_hz,
        vbw=vbw_hz,
        avg=avg_sweeps,
        timeout_ms=timeout_ms,
        marker_mode=marker_mode,
        # Noise / validation
        min_power_db=min_power_db,
        min_desired_db=min_desired_db,
        # Output CSV
        csv=str(csv_path),
    )


def frange(start: float, stop: float, step: float) -> List[float]:
    """Simple inclusive float range for IF sweep."""
    if step <= 0:
        raise ValueError("IF step must be > 0")
    vals: List[float] = []
    x = start
    # Avoid floating rounding problems: use a small epsilon
    eps = abs(step) * 1e-6
    while x <= stop + eps:
        vals.append(x)
        x += step
    return vals


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="IF sweep wrapper for mixer spur scan (FSW + SMBV100A)"
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
        help="SMBV VISA resource string.",
    )
    p.add_argument(
        "--fsw-reset",
        action="store_true",
        help="Preset (*RST) the FSW on first connect for the sweep.",
    )
    p.add_argument(
        "--smbv-reset",
        action="store_true",
        help="Preset (*RST) the SMBV on connect.",
    )

    # Mixer LO / IF sweep
    p.add_argument(
        "--lo",
        type=float,
        required=True,
        help="Mixer LO frequency in Hz (e.g. 10e9).",
    )
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

    # Mixer spur configuration (mirrors your original script)
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

    # Output & plotting
    p.add_argument(
        "--out-dir",
        default="spur_sweep_results",
        help="Directory for per-IF CSVs and outputs.",
    )
    p.add_argument(
        "--master-csv",
        default="spur_sweep_master.csv",
        help="Name of aggregated CSV inside out-dir.",
    )
    p.add_argument(
        "--plot-png",
        default="spur_sweep_plot.png",
        help="Name of spur plot PNG inside out-dir.",
    )

    args = p.parse_args()

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if_start = args.if_start
    if_stop = args.if_stop
    if_step = args.if_step

    if if_stop < if_start:
        raise RuntimeError("if_stop must be >= if_start")

    if_values = frange(if_start, if_stop, if_step)
    print(f"IF sweep points: {len(if_values)}")

    per_if_csv_paths: List[Path] = []

    # Connect to SMBV (IF source)
    print(f"Connecting to SMBV at '{args.smbv_resource}' ...")
    smbv = RsSmbv(args.smbv_resource, reset=args.smbv_reset, id_query=True)
    print(f"SMBV IDN: {smbv.utilities.idn_string}")

    try:
        # Basic CW setup
        smbv.output.state.set_value(True)
        smbv.source.frequency.set_mode(enums.FreqMode.CW)
        smbv.source.power.level.immediate.set_amplitude(args.if_level_db)

        for idx, if_hz in enumerate(if_values):
            print("\n------------------------------------------------------------")
            print(f"[{idx+1}/{len(if_values)}] IF = {if_hz/1e6:.3f} MHz")

            # Set SMBV IF frequency
            smbv.source.frequency.fixed.set_value(if_hz)

            # Optional settling delay
            if args.if_settle_s and args.if_settle_s > 0:
                import time
                time.sleep(args.if_settle_s)

            # Per-IF CSV path (only add to list if scan succeeds)
            if_mhz = if_hz / 1e6
            csv_name = f"spur_results_if_{if_mhz:.3f}MHz.csv"
            csv_path = out_dir / csv_name

            # Reset FSW only on first IF point if requested
            reset_this_fsw = args.fsw_reset and (idx == 0)

            # Build argument namespace for this IF point
            scan_args = build_scan_args(
                fsw_resource=args.fsw_resource,
                lo_hz=args.lo,
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
                csv_path=csv_path,
                reset_fsw=reset_this_fsw,
            )

            # Run the original spur scan for this IF
            print(f"Running spur scan, results -> '{csv_path}'")
            try:
                run_spur_scan(scan_args)
            except Exception as ex:
                print(
                    f"ERROR during spur scan at IF={if_hz/1e6:.3f} MHz: {ex}"
                )
                print("Skipping this IF point and continuing with sweep.")
                continue

            per_if_csv_paths.append(csv_path)

    finally:
        print("\nTurning SMBV RF output OFF and closing session.")
        try:
            smbv.output.state.set_value(False)
        except Exception as ex:
            print(f"WARNING: Failed to switch SMBV RF OFF cleanly: {ex}")
        smbv.close()

    # ------------------------------------------------------------------
    # Aggregate all spur CSVs into one master CSV
    # ------------------------------------------------------------------
    master_csv_path = out_dir / args.master_csv
    print(f"\nAggregating per-IF results into '{master_csv_path}' ...")

    master_rows = []
    master_fieldnames = None

    for csv_path in per_if_csv_paths:
        if not csv_path.exists():
            print(f"WARNING: Missing CSV '{csv_path}', skipping.")
            continue

        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if master_fieldnames is None:
                master_fieldnames = reader.fieldnames
            for row in reader:
                master_rows.append(row)

    if not master_rows:
        print("No data found to aggregate - exiting.")
        return

    # Ensure we have fieldnames
    if master_fieldnames is None:
        master_fieldnames = list(master_rows[0].keys())

    # Write master CSV
    with master_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=master_fieldnames)
        writer.writeheader()
        for row in master_rows:
            writer.writerow(row)

    print(f"Master CSV written with {len(master_rows)} rows.")

    # ------------------------------------------------------------------
    # Build spur frequency vs dBc plot across the IF sweep
    # ------------------------------------------------------------------
    plot_path = out_dir / args.plot_png
    print(f"Creating spur plot: '{plot_path}'")

    freq_ghz: List[float] = []
    rel_dbc: List[float] = []
    kinds: List[str] = []

    for row in master_rows:
        kind = row.get("kind", "")
        # Skip desired mixer product itself if you only want spurs
        # Comment this out if you want desired at 0 dBc in the plot.
        if kind == "desired":
            continue

        rel = row.get("rel_dBc_vs_desired", "")
        f_meas = row.get("measured_freq_Hz", "")

        if rel == "" or f_meas == "":
            continue

        try:
            rel_val = float(rel)
            f_val = float(f_meas)
        except ValueError:
            continue

        freq_ghz.append(f_val / 1e9)
        rel_dbc.append(rel_val)
        kinds.append(kind)

    if not freq_ghz:
        print("No valid spur points for plotting (check CSV contents).")
        return

    # Simple scatter plot: frequency vs dBc
    plt.figure(figsize=(10, 6))
    plt.scatter(freq_ghz, rel_dbc, s=10, alpha=0.7)
    plt.xlabel("RF frequency [GHz]")
    plt.ylabel("Level [dBc vs desired]")
    plt.title("Mixer spur sweep: RF frequency vs dBc across IF sweep")
    plt.grid(True)
    plt.gca().invert_yaxis()  # Spur more negative dBc is better → downwards
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("All done.")
    print(f"  Per-IF CSVs : {out_dir}")
    print(f"  Master CSV  : {master_csv_path}")
    print(f"  Plot PNG    : {plot_path}")


if __name__ == "__main__":
    main()
