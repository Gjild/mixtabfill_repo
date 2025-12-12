# mixer_spur_scan.py run example: 

uv run mixer_spur_sweeper.py --fsw TCPIP::192.168.0.77::HISLIP --smbv TCPIP::192.168.0.62::HISLIP --sma TCPIP::192.168.0.64::HISLIP --lo-start 5.55e9 --lo-stop 5.55e9 --lo-step 1e6 --lo-level-db 0.0 --lo-settle-s 0.05 --if-start 9.5e8 --if-stop 2.45e9 --if-step 1.5e7 --if-level-db -10.0 --if-settle-s 0.02 --mode lo+if --m-max 5 --n-max 5 --f-min 1e6 --f-max 4.3e10 --dedupe-freq 0.0 --span 50e3 --rbw 100 --vbw 100 --avg 10 --min-desired-db -23 --timeout-ms 5000 --marker-mode peak --min-power-db -140 --overlap-detect --overlap-local-thr-db 5.0 --overlap-global-thr-db 10.0 --overlap-equal-thr-db 1.0 --overlap-sep-rbw 5.0 --max-remeasure-lo-shift-hz 1e5 --max-remeasure-if-shift-hz 1e5 --out-dir spur_sweep_1p55G_950M_2450M --master-csv spur_sweep_master.csv

# Replot run example

uv run mixer_spur_replot.py --in-dir ltc5553_5p83g_if_sweep_950m_to_2p45g --master-csv ltc5553_5p83g_if_sweep_master.csv --plot-png ltc5553_5p83g_if_sweep_spurs_improved.png

# Run BUC System Predictor

uv run buc_spur_predictor.py sweep --lo2 21e9 --if1-start 950e6 --if1-stop 2450e6 --step 10e6 --if2-filter-csv IF2_PATRON.csv --rf-filter-csv RF_S21_28to31GHz.csv --mixer1-csv LTC5553_spur_sweep_lo5p55g_if950_2450_60m_fine\spur_sweep_master.csv --mixer2-csv-21 HMC329_spur_sweep_lo21p0g_if6p5g_8p0g_15m\spur_sweep_master.csv --mixer2-csv-22 HMC329_spur_sweep_lo22p0g_if6p5g_8p0g_15m\spur_sweep_master.csv --mixer2-csv-23 HMC329_spur_sweep_lo23p0g_if6p5g_8p0g_15m\spur_sweep_master.csv --out ./out/sweep_lo2_21.json


