# Findings — v4 Phase A

## Environment / Baseline Facts (verified)
- Graph: 1871 nodes / 45524 edges; visual input 210 (LPLC2), wind input 325 (JO-B/JO-C), GF 2, output 169.
- CPU: 16 logical cores; RAM 16GB; torch 2.14.0+cpu (pylibs).
- v3 baseline (REPORT.md / metrics.json, validation set 1200, training seed 20240521):
  - Fusion 0.799 trigger / 0.903 success / 10.7° / 7.65ms; vision only 0.801/0.787/18.4°/7.60ms; wind only 0.527/0.000/68.0°/zero fires.
  - Per-scenario (web\results.json): night recall 0.932→0.634, false alarm 0.316→0.036, latency 8.52ms; storm direction 30.7°, success 0.531;
    high-speed false alarm FAR 0.210→0.305→0.545 (low/medium/high speed).
- v3 reproduction metric key: make_pref(210,325,SEED) pd_pref[:,1] is "reserved parameter not connected to forward pass" — must not be deleted (preserves RNG ordering).
- v3 validation set = build_dataset(1200, SEED+2=20240523); run_eval uses independent rng(777) per mode, B=1 per-sample encoding → v4 keeps consistent.
- make_web_data.py uses SEED=20240521 to rebuild pd_pref/wind_pref: **changing the training seed changes pref**; v4 exports must carry pref_seed.

## Decision Log
- Validation set / OOD set **fixed** across 3 training seeds (val=rng(20240523), OOD=independent fixed seed); std only reflects training randomness;
  seed 20240521 val is identical to v3's sample set → directly comparable with v3.
- Checkpoint selection and early stopping both monitor **validation loss** (v3 selected by score every 5 epochs); this difference will be stated honestly in the report.
- Input augmentation only applies to training; validation evaluation keeps clean encoding (consistent with v3 metrics); OOD encoding uses 2× noise.
- OOD distribution shift (all documented in report): speed 60%~U(7,12)+40%~U(3,12) (capped at 12 m/s, more fast threats),
  d~U(3,40)cm, az~U(-2,2)rad, el~U(-0.9,0.9) (different angle distributions), miss~U(0,10)cm,
  wind pressure ×2 plus environmental wind U(0.01,0.03)cm/ms (stronger wind field), encoding noise σ~U(0,2·NOISE_STD) (sensor noise ×2).
  Label rules unchanged (ṙ<0 AND ttc<50ms AND r<25cm; esc=−r̂).

## Safety Boundaries
- Read-only outside v4\; never write to web\results.json, web\snn_data.js, snn_trained.json;
- Do not run web\evaluate.js / web\make_web_data.py (would overwrite v3 artifacts).

## Seed 20240521 Quick Results (full numbers in v4\metrics_seed20240521.json / v4\results_v4.json)
- Training ran full 25 epochs (early stopping not triggered, val best at ep25=0.6861, still improving → 25 epoch cap is the constraint); train 0.97→0.83;
  gap (train−val) +0.10~+0.20 (augmentation makes training distribution harder; not overfitting — val keeps decreasing).
- Python validation 1200 (clean encoding): fusion 0.762/0.773/18.9°/6.99ms (v3: 0.799/0.903/10.7°/7.65ms)
  → trigger / success rate / direction **regressed**, latency **faster**; vision only 0.783/0.681/20.2°/7.48ms (v3 0.801/0.787/18.4°/7.60);
  wind only 0.570/0.092/46.4°/7.65ms (v3 0.527/0.000/68.0°/zero fires) → **qualitative change: first time discriminative** (recall 0.141/FAR 0.044).
- JS matrix (5 eval seeds × 200): storm success 0.531→0.771, direction 30.7°→20.8° (**significant improvement**); night recall 0.634→0.624 (≈no improvement);
  high-speed FAR 0.545→0.510 (slightly better); clear fusion success 0.926→0.790, direction 10.6°→18.9° (**regressed**); latency universally ~0.5–0.8ms faster.
- Priming: weak vision 0.017 / weak wind 0.167 / weak stacked 0.283 (v3: 0/0/0.017) → cross-modal pre-activation **restored**.
- OOD 2000: fusion 0.592/0.520/30.0°/6.09ms, recall 0.970 / FAR 0.770 (high recall but high false alarm; discriminative ability under distribution shift).
- Mechanistic interpretation (for report): modality dropout + noise pushes the network from "vision-dominant" toward "dual-modality redundancy," trading clean-set direction accuracy (10.7°→18.9°) and success rate for storm / wind-only / priming / latency improvements — a classic robustness-accuracy tradeoff; Phase B suggestions:
  ① direction loss conditioned on modality or grouped learning rates; ② dropout probability annealing (0.12 in first 10 epochs → 0.05 later); ③ extend epochs (val still decreasing at ep25);
  ④ night requires visGain domain randomization (not done this phase because "data-only changes" list did not include environmental augmentation).