# Progress — v4 Phase A Session Log

## Session 1 (current)
- Context recovery: read train_snn.py / web\evaluate.js / web\make_web_data.py / web\smoke_test.js / web\snn_runtime.js /
  metrics.json / REPORT.md; verified graph scale 1871/45524, CPU 16 cores.
- Created v4\ and task_plan.md / findings.md / progress.md.
- Wrote v4\train_snn_v4.py (fixed 1 SyntaxError: global declaration position).
- **Speed test completed** (v4\speed_test.json): training batch 0.119s/batch (B=64) → 50K samples single epoch ≈94s, 25 epochs ≈0.65h/seed
  → **no reduction, keep 50K training samples**; B=1 eval forward 6ms/sample.
- 3 seeds launched in background (Start-Process, PID 50288/49120/52544; threads 8/4/4), logs v4\train_seedXXXX.log:
  training 50000 (threats ~24k) / validation 1200 (threats 568, same batch as v3 ✓) / OOD 2000 (threats 978, distribution shifted ✓).
- Wrote v4\TRAIN_STATE.md, v4\make_web_data_v4.py, v4\evaluate_v4.js.
- Baseline hash verification: web\results.json SHA256 = 0E1339A088EF3B5820DCA342B989249403BE41D5A3F7EDB9DDB1D1A91C217E9E (matches requirement).
- **Performance troubleshooting** (see TRAIN_STATE.md scheduling history + batch_probe.py for details):
  - 3-process parallel (8/4/4, OpenMP spin-wait) 500–575s/epoch; exclusive but spin-wait still 629s/epoch → ruled out "inter-process contention" hypothesis;
  - Probe measurement: `OMP_WAIT_POLICY=PASSIVE, KMP_BLOCKTIME=0` makes same-batch computation 3–5× faster → root cause = OpenMP spin-wait grabbing cores;
  - Settings baked into train_snn_v4.py; final config: **one seed at a time on 12 threads, sustained measured 252s/epoch (0.29s/batch)**,
    50K samples × 25 epochs ≈ 1.75h/seed, within the expected "possibly hours/seed" range → **keep 50K samples, no reduction** (speed results written to report §2).
  - Short-burst speed test 0.119s/batch (94s/epoch extrapolation) cannot extrapolate to sustained load; lesson noted in report.
- **Evaluation metrics self-check passed**: v4\evaluate_v4.js reproduces all values in web\results.json bit-for-bit using v3 data (web\snn_data.js read-only)
  (0.801±0.020 / 0.926±0.030 / 10.6 / 7.65; night 0.634, high-speed FAR 0.545, storm 30.7°…) → metrics fully consistent with v3.
  Self-check artifact v4\selftest_v3pipeline.json (full matrix runs in 32s).
- v4\ contains a `full-fly-design.md` deposited by another party (v4 closed-loop body-brain research design) — not in Phase A scope, noted but not modified.
- Scheduling decision: 20240521 exclusive priority (in progress) → after completion, 20240522/20240523 parallel (8+8 threads); estimated total wall clock ~4.5h.
- Next steps: wait for training to finish → make_web_data_v4 + evaluate_v4 (per seed) → report_tables.py → REPORT_v4.md → wrap-up self-check.