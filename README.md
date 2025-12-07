# mixer_spur_scan.py run example: 

uv run mixer_spur_scan.py --fsw "TCPIP::192.168.0.77::HISLIP" --lo 5.83e9 --if 2.45e9 --mode lo+if --m-max 5 --n-max 5 --f-min 0 --f-max 20e9 --span 1e6 --rbw 1e3 --avg 100 --min-power-db -120 --min-desired-db -40 --csv ltc5553_5p83g_2p45g_upconv_spurs.csv --marker-mode peak

# sweep_if_spurs.py run example:

uv run mixer_spur_sweep.py --fsw "TCPIP::192.168.0.77::HISLIP" --smbv "TCPIP::192.168.0.62::HISLIP" --lo 5.55e9 --if-start 950e6 --if-stop 2.45e9 --if-step 15e6 --mode lo+if --m-max 5 --n-max 5 --f-min 0 --f-max 20e9 --span 1e6 --rbw 1e3 --avg 100 --min-power-db -120 --min-desired-db -40 --marker-mode peak --out-dir ltc5553_5p83g_if_sweep_950m_to_2p45g --master-csv ltc5553_5p83g_if_sweep_master.csv --plot-png ltc5553_5p83g_if_sweep_spurs.png

# Replot run example

uv run mixer_spur_replot.py --in-dir ltc5553_5p83g_if_sweep_950m_to_2p45g --master-csv ltc5553_5p83g_if_sweep_master.csv --plot-png ltc5553_5p83g_if_sweep_spurs_improved.png

# Spur Mismeasurement Post Check run example

uv run spur_spike_analyzer.py --master ltc5553_5p83g_if_sweep_950m_to_2p45g/ltc5553_5p83g_if_sweep_master.csv --report ltc5553_5p83g_if_sweep_950m_to_2p45g/ltc5553_5p83g_spike_report.txt


