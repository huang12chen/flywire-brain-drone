# FlyWire Connectome Brain → Drone Control

A fruit-fly escape circuit extracted from the public FlyWire connectome (1,871 neurons / 45,524 synaptic edges) trained as a LIF spiking neural network and used to pilot a drone in a 3D simulation. When a threat looms, the fly brain decides *when* and *which way* to escape in milliseconds; an adapter layer translates that direction vector into four motor commands.

**Zero build, zero fetch — double-click `web/world.html` and it runs.**

## Quick Start

1. Double-click `web/world.html`
2. Select **Drone** as the agent
3. Press **Auto Threat** — watch the fly brain dodge incoming attacks

## Architecture

```
FlyWire FAFB (1,871 neurons)
  │  LPLC2 visual looming + JO-B/JO-C wind sensing
  ▼
LIF Spiking Neural Network (topology = connectome, weights trainable)
  │  Giant Fiber trigger + escape direction vector
  ▼
DroneAdapter (PID attitude loop + X-quad motor mixing)
  │  State machine: cruise / escape / pilot / land
  ▼
4 Motors → DronePhysics (mass / lift / drag / attitude) → 3D World
```

## Metrics (v3 — matches `web/results.json`)

| Metric | Value |
|---|---|
| Trigger accuracy | 0.799 |
| Escape success rate | 0.903 |
| Direction error | 10.7° |
| GF first-spike latency | 7.65 ms |
| Jitter stability (200 trials × 3 runs, median) | 92.0% all-pass |

Physical integration group (swept-sphere): fly body **0.698** · drone body **0.694** · stationary baseline **0.685**.

## Honest Limits

1. Cruise / obstacle-avoidance steering uses a program baseline mixed with neural escape — **not fully neural**. Full neural cruise = complete fly-brain pipeline (Phase 2, see [`v4/完整果蝇设计方案.md`](v4/完整果蝇设计方案.md)).
2. Flight dynamics are **demo-grade** (small-quadrotor typical values), not calibrated on real hardware.
3. The brain outputs an **escape direction vector**, not direct motor commands. The adapter maps it to motor thrust (biological analog: brain → thoracic ganglion → flight muscle).

## Data & Citations

- **Connectome data**: FlyWire / Princeton — Dorkenwald et al., *Nature* **634**:124–138 (2024).
- **Clarification**: "Google open-sourced the fly brain" actually refers to the **Neuroglancer viewer software**; the connectome data comes from the **FlyWire Consortium** (Princeton, HHMI Janelia, MRC LMB).
- **Engineering reference**: [snedea/flybrain](https://github.com/snedea/flybrain) — browser-based FlyWire whole-brain LIF simulation (139K neurons), MIT.
- See [`NOTICE.md`](NOTICE.md) for full data attribution and third-party notices.

## Reproduction

```bash
export PYTHONPATH=pylibs
python extract_circuit.py        # 1. Extract subcircuit
python train_snn.py              # 2. Train + evaluate + export
python web/make_web_data.py      # 3. Generate frontend data
node web/evaluate.js             # 4. Regression check (writes web/results.json)
```

Hash anchors (any SNN numerical change triggers rollback):

- `web/results.json` SHA256 = `0E1339A088EF3B5820DCA342B989249403BE41D5A3F7EDB9DDB1D1A91C217E9E`
- `web/jitter_sim_log.md` SHA256 = `5C66B3D05A92F17961EC9EFC7F8406BFA6117CB04F5F47260F544AB52B8072AE`

## Demo

![demo](docs/demo.gif)

> Placeholder — replaced with auto-generated screenshots after Phase 6.

## GitHub Pages

Go to **Settings → Pages → Source: `main` branch, folder `/web`** → Save. Your demo will be live at `https://huang12chen.github.io/flywire-brain-drone/`.

## License

MIT — see [`LICENSE`](LICENSE). Connectome-derived data distributed under FlyWire attribution terms (see [`NOTICE.md`](NOTICE.md)).
