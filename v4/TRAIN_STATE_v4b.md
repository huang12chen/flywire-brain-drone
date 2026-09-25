# v4 Phase B Training Status

**Status**: Completed ✅  
**Seed**: 20240521 (single seed)  
**Duration**: 220 minutes (25 epochs, 50K samples)  
**Completion time**: 2026-09-25

## New Additions (compared to v4 Phase A)
1. **Threat authenticity binary classification head** (`head_real`): 10% fake threat injection during training (long-range type + off-axis type), loss +0.3·BCE
2. **TTC regression head** (`head_ttc`): target = min(ttc,100ms)/50, loss +0.2·MSE

## Validation Set Results (fusion)
| Metric | Value |
|--------|-------|
| Trigger accuracy (GF) | 76.2% |
| Obstacle avoidance success rate | 77.3% |
| Direction error | 18.9° |
| Threat recall | 85.6% |
| False alarm rate | 32.3% |

## OOD Set Results (fusion)
| Metric | Value |
|--------|-------|
| Trigger accuracy (GF) | 59.2% |
| Obstacle avoidance success rate | 52.0% |
| Direction error | 30.0° |
| Threat recall | 97.0% |
| False alarm rate | 77.0% |

## Resume Command
```bash
py -3.13 v4\train_snn_v4b.py --seed 20240521 --resume
```

## Output Files
- `v4/snn_trained_seed20240521.json` — Model weights (includes head_real + head_ttc)
- `v4/metrics_seed20240521.json` — Evaluation metrics
- `v4/training_curve_seed20240521.png` — Training curve
- `v4/status_seed20240521.json` — Status file