#!/usr/bin/env python3
"""
Mixer spur / LO & IF leakage scanner for R&S FSW

- Connects to the FSW over LAN
- Measures desired mixer product first (reference power)
- Sweeps through m·LO ± n·IF combinations and measures each product
- Classifies products as desired, image, LO/IF leakage, harmonics, or spurs
- For each product:
    * Zooms in around the expected frequency (small span)
    * Sets a narrow RBW (and optional VBW)
    * Runs N sweeps and averages the marker power in linear units
    * Reports absolute power and relative rejection in dBc vs desired product
- Results are written to a CSV file, including flags for noise-limited
  measurements and errors for failed measurements.

Requires:
    pip install RsFsw
"""

import argparse
import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from RsFsw import RsFsw, enums, repcap


# ---------------------------------------------------------------------------
# Data classes
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
# Utility functions
# ---------------------------------------------------------------------------

def dbm_list_to_avg_dbm(powers_dbm: List[float]) -> float:
    """Average in linear (mW), return in dBm."""
    if not powers_dbm:
        return float('nan')
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


# ---------------------------------------------------------------------------
# Main measurement routine
# ---------------------------------------------------------------------------

def run_spur_scan(args) -> None:
    # Ensure the RsFsw driver version is recent enough
    RsFsw.assert_minimum_version("5.0.0")

    if args.mode == "lo-if" and args.lo <= args.if_freq:
        print(
            "WARNING: LO <= IF while using lo-if (down-conversion). "
            "Check that your LO and IF frequencies are correct."
        )

    print(f"Connecting to FSW at '{args.resource}' ...")
    fsw: Optional[RsFsw] = None
    try:
        fsw = RsFsw(args.resource, reset=args.reset, id_query=True)
        print(f"FSW IDN: {fsw.utilities.idn_string}")

        # Select spectrum analyzer base application
        fsw.instrument.select.set(enums.ChannelType.SpectrumAnalyzer)

        # Show remote display updates
        fsw.system.display.update.set(True)

        # Build list of mixer products
        spur_defs = build_spur_list(
            f_lo_hz=args.lo,
            f_if_hz=args.if_freq,
            m_max=args.m_max,
            n_max=args.n_max,
            desired_mode=args.mode,
            f_min_hz=args.f_min,
            f_max_hz=args.f_max,
            dedupe_hz=args.dedupe_freq,
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
            d_marker_mode = _resolve_marker_mode(args.marker_mode, desired_def.kind)
            d_meas_freq, d_power_dbm = measure_single_tone(
                fsw=fsw,
                center_hz=desired_def.expected_freq_hz,
                span_hz=args.span,
                rbw_hz=args.rbw,
                vbw_hz=args.vbw,
                avg_sweeps=args.avg,
                sweep_timeout_ms=args.timeout_ms,
                marker_mode=d_marker_mode,
            )
        except Exception as ex:
            print(f"ERROR: Failed to measure desired product: {ex}")
            raise

        freq_error_desired_hz = abs(d_meas_freq - desired_def.expected_freq_hz)
        limit_desired_hz = _freq_error_limit(args.span, args.rbw)
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
        if args.min_desired_db is not None and d_power_dbm < args.min_desired_db:
            raise RuntimeError(
                f"Desired product level {d_power_dbm:.2f} dBm is below "
                f"--min-desired-db threshold of {args.min_desired_db:.2f} dBm. "
                "Aborting spur scan."
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

            meas_freq = float('nan')
            p_dbm = float('nan')
            rel_dbc: Optional[float] = None
            error_msg: Optional[str] = None
            noise_limited = False

            try:
                marker_mode = _resolve_marker_mode(args.marker_mode, kind)
                meas_freq, p_dbm = measure_single_tone(
                    fsw=fsw,
                    center_hz=f_nom,
                    span_hz=args.span,
                    rbw_hz=args.rbw,
                    vbw_hz=args.vbw,
                    avg_sweeps=args.avg,
                    sweep_timeout_ms=args.timeout_ms,
                    marker_mode=marker_mode,
                )

                # Frequency sanity check
                freq_error_hz = abs(meas_freq - f_nom)
                limit_hz = _freq_error_limit(args.span, args.rbw)
                if freq_error_hz > limit_hz:
                    print(
                        f"  WARNING: Marker at {meas_freq/1e9:.6f} GHz is far from "
                        f"expected {f_nom/1e9:.6f} GHz (Δ = {freq_error_hz/1e3:.1f} kHz, "
                        f"limit ≈ {limit_hz/1e3:.1f} kHz). "
                        "Possible wrong peak captured."
                    )

                rel_dbc = p_dbm - d_power_dbm

                # Noise-limited flag (optional)
                if args.min_power_db is not None and p_dbm < args.min_power_db:
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

        # Write CSV
        csv_path = args.csv
        print(f"\nWriting results to '{csv_path}' ...")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            # Include LO/IF/mode and measurement settings on each row
            w.writerow(
                [
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
                ]
            )
            for r in results:
                freq_error_hz = (
                    r.measured_freq_hz - r.expected_freq_hz
                    if not math.isnan(r.measured_freq_hz)
                    else float('nan')
                )

                w.writerow(
                    [
                        args.lo,
                        args.if_freq,
                        args.mode,
                        r.kind,
                        r.m,
                        r.n,
                        r.sign,
                        r.expression,
                        r.expected_freq_hz,
                        r.measured_freq_hz,
                        freq_error_hz,
                        r.power_dbm,
                        "" if r.rel_dbc is None else r.rel_dbc,
                        1 if r.noise_limited else 0,
                        args.span,
                        args.rbw,
                        "" if args.vbw is None else args.vbw,
                        args.avg,
                        args.marker_mode,
                        _resolve_marker_mode(args.marker_mode, r.kind),
                        "" if r.error is None else r.error,
                    ]
                )

        print("Done.")

    finally:
        if fsw is not None:
            print("Closing the FSW session.")
            # Try to restore continuous sweep mode for convenience
            try:
                fsw.initiate.continuous.set(True)
            except Exception:
                pass
            fsw.close()


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Spur / LO & IF leakage measurement for mixer using R&S FSW"
    )

    p.add_argument(
        "--fsw",
        dest="resource",
        default="TCPIP::192.168.1.101::HISLIP",
        help=(
            "FSW VISA resource string "
            "(e.g. TCPIP::192.168.1.101::HISLIP or TCPIP::192.168.1.101::5025::SOCKET)"
        ),
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Send instrument preset (*RST) on connect.",
    )

    # Frequencies
    p.add_argument(
        "--lo",
        type=float,
        required=True,
        help="LO frequency in Hz (e.g. 10e9).",
    )
    p.add_argument(
        "--if",
        dest="if_freq",
        type=float,
        required=True,
        help="IF frequency in Hz (e.g. 140e6).",
    )
    p.add_argument(
        "--mode",
        choices=["lo+if", "lo-if"],
        default="lo-if",
        help="Desired mixing product: LO+IF or LO-IF.",
    )

    # Spur orders
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

    # Frequency window to keep
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
        help="Maximum RF frequency to consider (Hz, default 43 GHz for FSW43).",
    )

    # Optional de-duplication of coincident frequencies
    p.add_argument(
        "--dedupe-freq",
        type=float,
        default=0.0,
        help=(
            "Optional frequency bin (Hz) for merging products with nearly "
            "identical expected frequencies. 0 disables de-duplication."
        ),
    )

    # Zoom / RBW / averaging
    p.add_argument(
        "--span",
        type=float,
        default=1e6,
        help="Zoom span around each product (Hz), e.g. 1e6 = 1 MHz.",
    )
    p.add_argument(
        "--rbw",
        type=float,
        default=1e3,
        help="Resolution bandwidth in Hz (e.g. 1e3 = 1 kHz).",
    )
    p.add_argument(
        "--vbw",
        type=float,
        default=None,
        help=(
            "Video bandwidth in Hz (optional). If omitted, VBW is left unchanged "
            "from the current instrument setting."
        ),
    )
    p.add_argument(
        "--avg",
        type=int,
        default=10,
        help="Number of sweeps to average per product.",
    )
    p.add_argument(
        "--timeout-ms",
        type=int,
        default=None,
        help=(
            "Timeout per sweep (ms) for INIT:IMM *OPC. "
            "If omitted, the script tries to infer a timeout from the sweep time."
        ),
    )

    # Marker behavior
    p.add_argument(
        "--marker-mode",
        choices=["auto", "peak", "at-freq"],
        default="auto",
        help=(
            "Marker positioning mode:\n"
            "  auto    = choose per product kind (recommended)\n"
            "  peak    = marker to maximum in span\n"
            "  at-freq = marker at expected center frequency."
        ),
    )

    # Noise-limited flagging
    p.add_argument(
        "--min-power-db",
        type=float,
        default=None,
        help=(
            "Optional absolute threshold in dBm. If a measured tone is below this "
            "level, it will be flagged as noise-limited in the CSV (noise_limited=1)."
        ),
    )

    # Desired product validation
    p.add_argument(
        "--min-desired-db",
        type=float,
        default=None,
        help=(
            "Optional minimum level in dBm that the desired mixer product must exceed. "
            "If the measured desired product is below this level, the script aborts."
        ),
    )

    # Output
    p.add_argument(
        "--csv",
        default="spur_results.csv",
        help="Output CSV filename.",
    )

    return p.parse_args()


if __name__ == "__main__":
    run_spur_scan(parse_args())