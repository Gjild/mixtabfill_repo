#!/usr/bin/env python3
"""
Run the mixer spur sweep with:

  LO = 5.55 GHz (fixed)
  IF = 950 MHz → 2450 MHz in 60 MHz steps

Instruments:
  SMF100A (LO):  192.168.0.14
  SMA100B (IF):  192.168.0.65
  FSW (RF):      192.168.0.77
  
"""

import subprocess
from pathlib import Path
import sys


def main() -> None:
    # Path to your main spur tool script
    tool_path = "mixer_spur_sweeper_M2.py"

    cmd = [
        "uv", "run", str(tool_path),

        # Instrument VISA resources
        "--fsw", "TCPIP::192.168.0.77::HISLIP",
        "--smf", "TCPIP::192.168.0.14::HISLIP",
        "--sma", "GPIB0::19::INSTR",

        # LO sweep: single point at 5.55 GHz
        "--lo-start", str(21.00e9),
        "--lo-stop",  str(21.00e9),
        "--lo-step",  str(1e6),   # arbitrary non-zero step; only one point used

        # IF sweep: 950 MHz → 2450 MHz in 60 MHz steps
        "--if-start", str(6.5e9),
        "--if-stop",  str(8.0e9),
        "--if-step",  str(15.0e6),

        # Up-conversion mode: LO + IF
        "--mode", "lo+if",
        
        # Spur order and frequency settings
        "--f-min", str(25.0e9),
        "--f-max", str(35.0e9),
        "--m-max", "5",
        "--n-max", "5",

        # Source levels (adjust as needed)
        "--lo-level-db", "16.0",
        "--if-level-db", "-11.0",

        # Measurement settings
        "--span", str(1e3),
        "--rbw",  str(10),
        "--avg",  "10",

        "--marker-mode", "auto",
        "--min-power-db", "-120",

        # Coincidence / cluster settings
        "--coincidence-factor", "1.0",
        "--measured-coincidence-factor", "1.5",
        "--min-coincidence-hz", "0",
        "--marker-confusion-factor", "0.25",
        "--rbw-guard-factor", "10.0",
        "--max-cluster-distance-hz", "0",
        "--deltaf-low-order-sum", "3",
        "--deltaf-mid-order-sum", "5",
        "--deltaf-mid-weight", "0.5",
        "--cluster-max-window-rbw", "100.0",

        # Use generator readback for frequency model
        "--calibrate-freq",

        # Output directory and master CSV name
        "--out-dir", "spur_sweep_lo21p0g_if6p5g_8p0g_15m",
        "--master-csv", "spur_sweep_master.csv",
        
        #timeout 
        "--timeout-ms", "2000",
    ]

    print("Run this command:\n  " + " ".join(cmd) + "\n")


if __name__ == "__main__":
    main()
