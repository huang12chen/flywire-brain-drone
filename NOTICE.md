# Data & Third-Party Notices

## Neural Connectome Data

Data artifacts in this repository (`web\snn_data.js`, `snn_trained.json`, etc.) are **derived from the FlyWire FAFB whole-brain connectome** (a single adult female *Drosophila melanogaster*):

- **Source**: FlyWire Codex, `https://codex.flywire.ai` (dataset=fafb, v783)
- **Citation**: Winding, M., et al. "The neuronal wiring diagram of an adult brain." *Nature* 634, 124–138 (2024). Dorkenwald, S., et al. *Nature* 634, 103–111 (2024).
- **Note**: "Google open source" often seen alongside this data refers to **Neuroglancer** (the viewer software by Google); the data itself is produced by the FlyWire Consortium (Princeton University, HHMI Janelia, MRC LMB, Cambridge).
- This repository contains only a **derived subcircuit summary** of escape-related subcircuits (LPLC2/JO/GF/downstream: 1871 neurons, 45524 connections as a weighted summary), not the raw whole-brain data. When using derived data, please also comply with FlyWire data-use norms and cite the papers above.
- The files here are a **derived subcircuit summary** (not the raw whole-brain data). If you use it, please follow FlyWire data-use norms and cite the papers above.

## Third-Party Code

| Component | License | Description |
|---|---|---|
| three.js r147 (`web\vendor\three.min.js`) | MIT | © 2010-2022 three.js authors |
| Reference project snedea/flybrain | MIT | Engineering reference for whole-brain browser simulation (no code copied) |
| Reference project heyseth/worm-sim | MIT | Predecessor C. elegans simulation (no code copied) |

## Disclaimer

- Flight dynamics parameters are **demo-grade**, not calibrated for real hardware; do not use directly on actual aircraft.
- Control decisions output by this project **do not constitute** flight control advice for any real drone; real-hardware applications must retain manual takeover and safety guardrails.
- Training metrics are from fixed-seed synthetic data evaluation (reproduction commands and hash anchors in README).