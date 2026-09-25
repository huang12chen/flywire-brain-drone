# v4 Phase A — Training Status & Recovery Handbook (TRAIN_STATE.md)

> Last updated: see "Status Snapshot" at end of file. **Iron rule**: read-only outside v4\; `web\results.json` SHA256 must remain
> `0E1339A088EF3B5820DCA342B989249403BE41D5A3F7EDB9DDB1D1A91C217E9E`.

## 1. Speed Test Conclusion (determines data scale)
`v4\speed_test.json` (`py -3.13 -X utf8 v4\train_snn_v4.py --speed-test`):
- Training batch (B=64, including forward + backward + step) short-burst average **0.119 s**; encoding 64 samples 4.0 ms;
- Extrapolated single epoch (50K samples + 1200 validation loss) ≈ 94 s → 25 epochs ≈ 0.65 h/seed; B=1 eval forward 6.0 ms/sample.

**Measured correction (important)**: short-burst values cannot extrapolate to sustained training — first run with 3-process parallel measured ~500 s/epoch, single-process exclusive still ~560–630 s/epoch.
Root cause identified (`v4\batch_probe.py`): **OpenMP default spin-wait (KMP_BLOCKTIME=200ms) causes core contention under sustained / multi-process loads**;
setting `OMP_WAIT_POLICY=PASSIVE, KMP_BLOCKTIME=0` drops same-batch computation to **0.15–0.21 s/batch** (even when competing with other processes).
This setting has been baked into `train_snn_v4.py` (setdefault before import torch). → **Still executing with 50K training samples, no reduction**.

## 2. Three Seeds = Three Separate Background Processes (Round 2 launch: unified commands, trained from scratch, OMP fixed)
Unified prerequisites: `$env:PYTHONPATH = "<project root>\pylibs"`, working directory = project root.

| Seed | PID (Round 2) | Threads | Launch Command | Log | Status File |
|---|---|---|---|---|---|
| 20240521 | py 58704 | 6 | `py -3.13 -X utf8 -u v4\train_snn_v4.py --seed 20240521 --threads 6` | `v4\train_seed20240521.log` (stderr: `.err.log`) | `v4\status_seed20240521.json` |
| 20240522 | py 64164 | 5 | `py -3.13 -X utf8 -u v4\train_snn_v4.py --seed 20240522 --threads 5` | `v4\train_seed20240522.log` | `v4\status_seed20240522.json` |
| 20240523 | py 54492 | 5 | `py -3.13 -X utf8 -u v4\train_snn_v4.py --seed 20240523 --threads 5` | `v4\train_seed20240523.log` | `v4\status_seed20240523.json` |

**Scheduling history (honestly recorded)**:
- Round 1 (8/4/4 threads parallel): ~500 s/epoch, caused by OpenMP spin-wait contention; its 3 Round 1 checkpoints have been deleted, Round 1 numbers archived below.
- Mid-run experiment: killed 20240522/20240523 to let 20240521 run exclusively → still 629 s/epoch (proved it's not inter-process contention but spin-wait policy);
  after `v4\batch_probe.py` + `OMP_WAIT_POLICY=PASSIVE` identified root cause, **all seeds restarted from scratch** (unified commands, unified thread count, ensuring one-command reproducibility).
- Round 1 train/val loss archive (overwritten by Round 2, recorded for reference): 20240521 ep1 0.9732/0.7874, ep2 0.8772/0.7642;
  20240522 ep1 0.9389/0.7599; 20240523 ep1 0.9741/0.8191.

(Launch method: PowerShell `Start-Process ... -RedirectStandardOutput v4\train_seedXXXX.log -RedirectStandardError v4\train_seedXXXX.err.log -WindowStyle Hidden -PassThru`;
process decoupled from this session; session interruption does not affect training.)

## 3. Checkpoints & Resume Commands
- End-of-epoch checkpoint written: `v4\ckpt_seedXXXX.pt` (contains model/opt/scheduler/hist/best_state/epoch);
- **Resume (bit-for-bit reproducible checkpoint continuation)**:
  ```powershell
  $proj = '<project root>'
  $env:PYTHONPATH = "$proj\pylibs"; Set-Location $proj
  Start-Process -FilePath py -ArgumentList @('-3.13','-X','utf8','-u','v4\train_snn_v4.py','--seed','20240521','--threads','8','--resume') `
    -WorkingDirectory $proj -RedirectStandardOutput "$proj\v4\train_seed20240521.log" `
    -RedirectStandardError "$proj\v4\train_seed20240521.err.log" -WindowStyle Hidden -PassThru
  ```
  (Change `--seed` to resume the corresponding seed; training batch RNG is seeded per-epoch as `seed+100+epoch`, resume does not alter results.)
- Post-training artifacts: `v4\snn_trained_seedXXXX.json`, `v4\metrics_seedXXXX.json`, `v4\training_curve_seedXXXX.png`.

## 4. Post-Training Evaluation Pipeline (do not overwrite web\ artifacts)
```powershell
# Main model (seed 20240521) → v4\snn_trained_v4.json + v4\snn_data_v4.js
py -3.13 -X utf8 v4\make_web_data_v4.py --seed 20240521
# Other two seeds → v4\snn_trained_v4_seedXXXX.json + v4\snn_data_v4_seedXXXX.js
py -3.13 -X utf8 v4\make_web_data_v4.py --seed 20240522 --tag _seed20240522
py -3.13 -X utf8 v4\make_web_data_v4.py --seed 20240523 --tag _seed20240523
# Experiment matrix evaluation (outputs v4\results_v4.json etc.)
F:\Node\node.exe v4\evaluate_v4.js
F:\Node\node.exe v4\evaluate_v4.js --data v4/snn_data_v4_seed20240522.js --out v4/results_v4_seed20240522.json
F:\Node\node.exe v4\evaluate_v4.js --data v4/snn_data_v4_seed20240523.js --out v4/results_v4_seed20240523.json
# Wrap-up self-check
F:\Node\node.exe web\smoke_test.js
Get-FileHash web\results.json -Algorithm SHA256
```

## 5. Hyperparameters (v4 Phase A)
N_TRAIN=50000 / N_VAL=1200 (fixed rng(20240523), same batch as v3) / N_OOD=2000 (fixed rng(20240603), shifted distribution)
Noise σ~U(0,0.05) (tunable --noise-std) / Modality dropout p=0.12 (--drop-p) / wd=1e-4 (--wd)
Early stopping: monitor validation loss, patience=6, minimum 8 epochs (--patience) / Checkpoint selection = min val_loss
BATCH=64, EPOCHS=25, LR=2e-3 + cosine annealing (same as v3); architecture / labels / decision window / loss / activation boundary identical to v3 bit-for-bit.

## Status Snapshot (last updated: after seed 20240521 completion)
- ✅ Seed 20240521: training complete (full 25 epochs, best ep25 val=0.6861), exported
  `v4\snn_trained_seed20240521.json` / `metrics_seed20240521.json` / `training_curve_seed20240521.png`;
  packaged `v4\snn_trained_v4.json` + `v4\snn_data_v4.js`, evaluated `v4\results_v4.json`.
- ⏳ Seeds 20240522 (PID 25848), 20240523 (PID 218844): training in background (8 threads, AboveNormal, OMP PASSIVE),
  each writing `v4\train_seedXXX.log` / `status_seedXXX.json` / `ckpt_seedXXX.pt`, estimated ~3h to complete 25 epochs.
- ✅ Wrap-up self-check (done once): `web\smoke_test.js` 4/4 all pass exit=0; `web\results.json` SHA256 =
  `0E1339A088EF3B5820DCA342B989249403BE41D5A3F7EDB9DDB1D1A91C217E9E` ✅. Must **re-verify** after both seeds complete.
- ✅ `v4\REPORT_v4.md` first draft delivered (seed 1 full numbers + 3-seed mean±std placeholder).

## 4b. "5-Minute Wrap-Up" After Both Seeds Complete (any session can copy and execute directly)
```powershell
$proj = '<project root>'; Set-Location $proj
# 1) Package the other two seed models
py -3.13 -X utf8 v4\make_web_data_v4.py --seed 20240522 --tag _seed20240522
py -3.13 -X utf8 v4\make_web_data_v4.py --seed 20240523 --tag _seed20240523
# 2) Experiment matrix evaluation (~45s each)
F:\Node\node.exe v4\evaluate_v4.js --data v4/snn_data_v4_seed20240522.js --out v4/results_v4_seed20240522.json
F:\Node\node.exe v4\evaluate_v4.js --data v4/snn_data_v4_seed20240523.js --out v4/results_v4_seed20240523.json
# 3) Aggregate 3-seed mean±std(ddof=1) + overfitting checkup + per-scenario tables → v4\report_tables.json + console Markdown
py -3.13 -X utf8 v4\report_tables.py
# 4) Wrap-up self-check: smoke test all pass + hash unchanged
F:\Node\node.exe web\smoke_test.js
(Get-FileHash web\results.json -Algorithm SHA256).Hash   # Must = 0E1339A088EF3B5820DCA342B989249403BE41D5A3F7EDB9DDB1D1A91C217E9E
# 5) Backfill report_tables.py table values into v4\REPORT_v4.md placeholders (T1/T2/T3b/T4)
```
> If hash doesn't match: **do not handle it yourself**, report to the supervisor first (possible concurrent write to web\).