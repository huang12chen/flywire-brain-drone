# v4 Phase A Task Plan (Data Upgrade + Overfitting Prevention)

> Iron rule: **only read files outside v4\ is allowed; never modify any file outside v4\** (v3 is frozen).
> web\results.json SHA256 must remain `0E1339A088EF3B5820DCA342B989249403BE41D5A3F7EDB9DDB1D1A91C217E9E`.

## Goal Statement
Fix v3's three major weaknesses (night recall 0.932→0.634, high-speed false alarm FAR 0.210→0.545, storm direction 30.7°) + overfitting checkup;
only modify data per phase (architecture / labels / decision window / loss / activation boundary `>0` identical to train_snn.py bit-for-bit).

## Phases
| # | Content | Status |
|---|---------|--------|
| 1 | Create v4\ and planning files | in_progress |
| 2 | v4\train_snn_v4.py (50K sample target, Gaussian noise 0–0.05, modality dropout 10–15%, OOD 2000, 3 seeds, wd=1e-4, early stopping, train/val curves, checkpoint resume) | pending |
| 3 | Speed test to determine scale (can reduce to 20–30K if too slow; write speed test results to report) | pending |
| 4 | Start-Process to run 3 seeds in background (seed 20240521 priority); v4\TRAIN_STATE.md records progress / resume commands | pending |
| 5 | v4\make_web_data_v4.py, v4\evaluate_v4.js (output v4\results_v4.json etc., never overwrite web\ artifacts) | pending |
| 6 | Evaluation: fusion / vision only / wind only + per-scenario + OOD standalone + overfitting checkup | pending |
| 7 | v4\REPORT_v4.md (item-by-item comparison with v3, mean±std ddof=1, regressions reported honestly) | pending |
| 8 | Wrap-up self-check: node web\smoke_test.js all pass + SHA256 verification written into report | pending |

## Metrics That Must Not Change (consistent with v3)
- Trigger = GF first fire step < 12 (T=12 steps, dt=1ms, β=0.85, Vth=1.0, soft reset, activation boundary `>0`)
- Labels = ṙ<0 AND ttc<50ms AND r<25cm; escape direction = −r̂
- Loss = 0.5·BCE(head) + 0.5·direction cosine (with yTrig mask, untouched this phase) + 2e-3·rate·T + 0.6·GF firing BCE (pos_weight=2.5)
- Direction metric and physical integration metric reported separately, never mixed

## v4 Allowed Changes (data/regularization only)
1. Training samples 5000 → 50000 (can reduce to 20–30K if too slow)
2. Input augmentation: Gaussian noise σ∈[0,0.05] (tunable) + modality dropout p=0.12 (zero out entire channel, randomly drop vision or wind)
3. OOD test set 2000 samples (significantly shifted parameters; forbidden in training/validation)
4. Training seeds 20240521/20240522/20240523, report mean±std (ddof=1)
5. Weight decay 1e-4 + early stopping (monitor validation loss) + train/val loss curves (overfitting checkup)

## Errors Encountered
| Error | Attempt Count | Resolution |
|-------|---------------|------------|
| (To be recorded) | | |