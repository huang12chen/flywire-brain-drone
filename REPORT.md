# Fruit Fly Visual-Wind Dual-Modal Threat Escape SNN — Project Experiment Report

> Spiking Neural Network constrained by FlyWire whole-brain connectome · 3D micro-world verification platform · Target application: drone biomimetic obstacle avoidance / anti-interference

---

## 0. Abstract

This project uses a **spiking neural network (SNN) constrained by real connectome topology** to achieve "visual-wind dual-modal threat escape control": using the visual expansion pathway (LPLC2), wind-sense pathway (Johnston's organ JO-B/JO-C), and giant fiber (GF) escape circuit extracted from the fruit fly whole-brain connectome (FlyWire FAFB) as hard topology constraints, training only connection weights on this substrate to implement the complete reflex arc of "threat approach → GF firing trigger → escape away from threat." The 3D open-world webpage (`web\world.html`, "Fruit Fly · Micro World") serves as the system's **verification platform + demo showcase** (not a game product): within the same scene, you can swap bodies, swap environments, apply different sneak attack threats, and use statistical tests with the same metrics as the training pipeline to verify control effectiveness. The target application domain is **drone biomimetic obstacle avoidance / anti-interference**.

**Scientific qualifications (applies to all conclusions herein)**: Training data consists of cue-label mappings **synthesized** from physics formulas (**not electrophysiology / behavioral measurements**); connectome topology is from a **single female adult fly** (FlyWire FAFB), and synapse counts are merely proxies for anatomical connection strength, **not physiological weights**; all conclusions **hold only on this synthetic control task**.

---

## 1. Methods

### 1.1 Data

FlyWire FAFB whole-brain connectome downloaded locally as 4 files:

| File | Content |
|---|---|
| `connections_princeton.csv.gz` | **5.34 million rows** of connection records (edge-level `nt_type` neurotransmitter annotations) |
| `consolidated_cell_types.csv.gz` | **138,327 neurons** (consolidated cell type table) |
| `classification.csv.gz` | Classification hierarchy (super_class, etc.) |
| `neurons.csv.gz` | Neuron attributes (neurotransmitter fallback annotations) |

> Data location: the 4 files should be placed in the directory specified by `DATA_DIR` at the top of `extract_circuit.py` (default: `<FlyWire data directory>`); modify this constant when switching machines.

### 1.2 Key Neurons

| Role | Selection basis | Count |
|---|---|---|
| Visual expansion input | LPLC2 (visual expansion / wide-field motion sensitive projection neurons) | ×210 |
| Wind-sense input | Johnston's organ JO-B / JO-C | ×325 |
| Escape center (giant fiber) | GF: `primary_type=DNp01`, `additional_type` contains `Giant_Fiber` | ×2 |
| Output layer | Downstream / motor neurons within 2 hops of GF | ×169 |

### 1.3 Subgraph Extraction

Only neurons on short paths of **≤3 synaptic edges** from "input → GF" are retained (GF escape latency is only a few milliseconds, indicating an oligosynaptic pathway), and the 210 input nodes, GF, and 169 output nodes are forced to be included, yielding:

> **1871 nodes / 45524 edges** (`syn_count ≥ 3`)

### 1.4 Weight Initialization and Neurotransmitter Signs

- Weight initialization: **W = sign(NT)·log1p(syn_count)**, normalized by postsynaptic total inbound strength (×1.2 scaling); input channels have an additional **learnable gain in_gain**;
- Neurotransmitter signs (fruit fly): **ACH +1 / GABA −1 / GLUT −1 (insect glutamate is inhibitory) / DA, SER, OCT are neuromodulators +1** (and marked `modulatory` in the data).

### 1.5 Model

- LIF spiking neurons: **β = 0.85, Vth = 1.0, dt = 1 ms, simulation window 12 ms**;
- **Topology mask fixed, only weights are trained** (readout head and learnable input gain in_gain are also trained; FlyWire connectome structure serves as an immutable hard constraint);
- Surrogate gradient: snntorch `surrogate.fast_sigmoid()`.

### 1.6 Training

- Data: **5000 synthetic samples × 25 epochs**, cues and labels generated programmatically from physics formulas:
  - Visual expansion rate `dθ/dt = −2s·ṙ/(r²+s²)`;
  - Wind pressure `u ∝ s²·v/r²` (sphere potential flow approximation);
  - Labels: **trigger when approaching (ṙ < 0) AND time-to-collision < 50 ms AND distance < 25 cm**; **escape direction = away from threat** (unit vector).
- Optimizer: **Adam + cosine decay**;
- Loss: **0.5·BCE (readout head) + 0.5·direction cosine loss + energy term (2e-3·spike count·T) + 0.6·GF firing BCE (pos_weight = 2.5)**.

### 1.7 Trigger Criterion (Mechanistic)

**GF firing = escape trigger** — independent of readout head threshold; the criterion itself is a circuit mechanism (giant fiber fires → escape maneuver).

---

## 2. Main Results (Training Evaluation)

Validation set: **1200 samples**, **fixed seed**:

| Group | Trigger Accuracy (GF) | Obstacle Avoidance Success Rate | Direction Error | GF Latency |
|---|---|---|---|---|
| Fusion | 0.799 | 0.903 | 10.7° | 7.65ms |
| Vision only | 0.801 | 0.787 | 18.4° | 7.60ms |
| Wind only | 0.527 | 0.000 | 68.0° | — |

**Metric definitions**: success rate = proportion of effective threats (GF fires AND direction error < 30°); direction error = mean angle between readout direction and "away from threat" ground truth across all effective threat samples; GF latency = mean first-fire time across effective threat samples where GF fired (trigger criterion: first fire step <12 ms).

> Note 1: Wind-only group latency "—" = GF had zero fires within the 12 ms window (trigger criterion is first fire step <12 ms), so latency is undefined (`metrics.json` records null, semantics = "never fired, latency undefined"); wind-only trigger accuracy 0.527 = "always judge no threat" baseline (negative sample proportion 632/1200 = 52.7%), not indicative of discriminative ability.
> Note 2: This main table is a single evaluation from one training seed (SEED=20240521) on 1200 fixed validation samples, without std; 5-seed evaluations with ±std are in §3.1.

---

## 3. Extended Experiment Matrix

> Data in this section is generated by `web\evaluate.js` (deterministic Monte Carlo evaluator); details are stored in `web\results.json` (same seed produces identical SHA256 across runs, bit-for-bit reproducible).
> Each cell is **mean ± sample standard deviation (5 seeds × 200 samples = 1000 samples; std is sample std of the 5 seed-level metric values with ddof=1)**. Groups A/B/C "obstacle avoidance success rate" uses the **direction metric** (GF fires AND direction error < 30° among effective threats); Group D uses the **physical integration metric** (see 3.4); **the two metrics must not be mixed**.

### 3.1 A — Modality Control (Environment = clear meadow)

| Group | Trigger Accuracy | Obstacle Avoidance Success Rate | Direction Error ° | GF Latency ms |
|---|---|---|---|---|
| Fusion | 0.801±0.020 | 0.926±0.030 | 10.6±0.5 | 7.65±0.20 |
| Vision only | 0.796±0.023 | 0.797±0.031 | 18.2±0.5 | 7.62±0.20 |
| Wind only | 0.529±0.014 | 0.000±0.000 | 65.4±1.8 | — (GF zero fires throughout) |

**Interpretation**: Triggering depends almost entirely on vision (fusion 0.801 vs vision only 0.796); wind sense's value lies in **direction** (10.6° vs 18.2°) — the fusion gain primarily manifests in "which direction to dodge" rather than "when to dodge." Wind alone cannot drive GF within the 12 ms window (trigger criterion: first fire step <12 ms; all 5 seeds show zero fires), and its 0.529 is merely the "always judge no threat" baseline (negative samples 52.9%). Cross-validation with training `metrics.json` shows consistent magnitudes (success rate difference 2.3pp, less than 1 seed std).

### 3.2 B — Environment (Clear · Strong Wind · Pitch Black, Fusion Mode)

| Group | Trigger Accuracy | Obstacle Avoidance Success Rate | Direction Error ° | GF Latency ms |
|---|---|---|---|---|
| Clear | 0.801±0.020 | 0.926±0.030 | 10.6±0.5 | 7.65±0.20 |
| Strong wind | 0.782±0.014 | 0.531±0.017 | 30.7±1.1 | 7.75±0.22 |
| Pitch black | 0.809±0.024 | 0.634±0.040 | 11.8±1.0 | 8.52±0.17 |

**Interpretation**: The **(unverified by ablation) hypothesized main cause** of strong wind degradation is the fixed environmental wind direction superimposed on drag-biased wind direction estimates (→ direction error 30.7°, success rate nearly halved; note storm also has visGain=0.85 and no isolated ablation arm, so causal ordering is not confirmed by ablation). Pitch black's "slightly higher trigger accuracy (0.809)" is an **illusion**: recall drops 0.932→0.634, false alarm rate drops 0.316→0.036, GF fire count approximately halves, latency extends to 8.52ms — it trades missed detections for false alarms; looking only at trigger accuracy is misleading (`results.json` contains recall/false-alarm breakdown).

### 3.3 C — Sneak Attack Speed (Low · Medium · High, Fusion × Clear)

| Group | Trigger Accuracy | Obstacle Avoidance Success Rate | Direction Error ° | GF Latency ms |
|---|---|---|---|---|
| Low speed (0.3–3 m/s) | 0.808±0.015 | 0.978±0.032 | 10.2±0.2 | 7.69±0.44 |
| Medium speed (3–7 m/s) | 0.799±0.029 | 0.886±0.035 | 11.3±0.7 | 7.89±0.17 |
| High speed (7–12 m/s) | 0.771±0.016 | 0.935±0.027 | 10.4±0.4 | 7.53±0.21 |

**Interpretation**: Direction error is largely independent of sneak attack speed (10–11°). However, **false alarm rate surges at high speed** (P(GF fires | no threat) = 0.210→0.305→0.545): even large-miss samples at high speed have strong looming + wind pressure stimuli. A practical system needs an additional "threat authenticity / miss magnitude" discriminator after GF trigger, otherwise high-speed scenarios will be overwhelmed by false alarms. Additionally: the low-speed group has only 94 total effective threats (across 5 seeds), so the estimated degrees of freedom for success rate / direction error in this cell are low — interpret with caution.

### 3.4 D — Cross-Platform (Fruit Fly Body vs Drone Body, Fusion × Clear × Medium Speed)

| Group | Trigger Accuracy | Obstacle Avoidance Success Rate* | Direction Error ° | GF Latency ms |
|---|---|---|---|---|
| Fruit fly body | 0.799±0.029 | 0.698±0.049 | 11.3±0.7 | 7.89±0.17 |
| Drone body | 0.799±0.029 | 0.694±0.050 | 11.3±0.7 | 7.89±0.17 |

\* **Physical integration metric**: both parties' motion integrated on a 2D encounter plane with dt=1ms (capped at 3000ms); safe = passed the closest point AND closest distance > sum of radii (threat radius + 0.55cm). Direction-metric reference values: fruit fly / drone both 0.886±0.035 (consistent with Group C medium speed, since they share the same neural data; cross-platform swap does not affect SNN forward pass, only execution dynamics).

**Key finding that must be interpreted honestly**: **The stationary baseline already reaches 0.685** (~68% of effective threats naturally miss); the fraction truly "saved" by maneuvering is only **fruit fly +1.5pp / drone +1.0pp** (`action_saved` metric; net gain physical−stationary is +1.3pp / +1.0pp — fruit fly side has one save offset by one caused collision; save events are only 8/529 and 5/529, ±std 0.006/0.010, **low counts, interpret with caution**). Reason: trigger criterion ttc<50ms, the platform only begins accelerating after ~7.9ms GF latency, so displacement within the 50ms window is only about 1–2cm (drone <1cm). **Conclusion**: this model demonstrates "making the correct response at the correct moment in the correct direction" (neural decision quality), **not "maneuver-envelope hazard rescue"**; and the direction metric (0.886) differs substantially from the integration metric (~0.69–0.70). Implications for drone applications: biomimetic reflex solves "when to dodge and which direction to dodge"; actually avoiding the threat additionally requires **earlier warning distance + stronger execution dynamics** (see Section 4, limitation 7).

---

## 4. Known Issues and Objective Limitations

1. **Training data is synthetic cues** (generated from physics formulas), not electrophysiology / behavioral measurements; conclusions hold only on this control task.
2. **Connectome is from a single female adult fly**; synapse counts are proxies for anatomical connection strength, not physiological weights.
3. **Wind-only modality is weak**: wind direction aligns with threat velocity direction, so when miss magnitude is large, correlation with threat bearing is weak, and the JO→output pathway is sparse — wind sense's value lies in **fusion gain**, not standalone use.
4. **Weak-cue cross-modal pre-activation effects** appeared in early versions (weak vision 15% → stacked 45% GF firing), but weakened in the final v3 model (v3 measured: weak vision 0 / weak wind 0 / weak stacked 1.67% GF firing, see `metrics.json` `priming`; the cost of more decisive triggering), recorded honestly.
5. **The fruit fly is "pure instinct"**: no memory, no online learning (intentionally no online plasticity to ensure reproducibility).
6. **Early development contained a bug where "escape direction labels were inverted"** (numerical metrics were normal but semantics were wrong), caught and fixed by manual review — demonstrating that **manual semantic review is indispensable beyond automatic metrics**.
7. **Maneuver gain is small (revealed by Group D physical integration)**: relative to the stationary baseline of 0.685, true saves are only fruit fly +1.5pp / drone +1.0pp (`action_saved` metric, event counts 8/529 and 5/529, low counts) — within the 50ms escape window, dynamic displacement is limited, and ~68% of effective threats naturally miss. The direction-metric success rate (0.886) differs substantially from the integration metric (~0.69–0.70); the two metrics must not be mixed. Biomimetic reflex solves "when to dodge and which direction to dodge"; actually avoiding the threat additionally requires earlier warning and stronger execution dynamics.
8. **High-speed false alarms**: false alarm rate increases with sneak attack speed (0.210→0.545); practical systems need additional threat authenticity / miss magnitude discrimination.
9. **Early versions of `metrics.json` contained literal `NaN`** (e.g., wind_only GF latency; Python json permits this, strict JSON does not, Node cannot require it). This has been fully remedied: the export side writes NaN as `null` (semantics = "never fired, latency undefined"); the current `metrics.json` is strictly valid JSON.

---

## 5. Future Work (**Not Implemented**)

> All of the following are planned directions, **not implemented** in the current version:

- Dopamine-style online reinforcement learning;
- Memory / habituation;
- PX4 software-in-the-loop simulation → real hardware verification;
- Pilot three-mode control authority arbitration (currently only manual / biomimetic two-state toggle + pilot input immediate takeover implemented).

---

## 6. Reproduction Method

```powershell
$proj = '<project root>'
Set-Location $proj                       # Commands below use relative paths; cd to project directory first
$env:PYTHONPATH = "$proj\pylibs"        # numpy/torch/snntorch etc. dependencies are in pylibs; must set first
py -3.13 -X utf8 extract_circuit.py     # Extract connectome subgraph
py -3.13 -X utf8 train_snn.py           # Train + evaluate + export
py -3.13 -X utf8 web\make_web_data.py   # After retraining, repackage snn_trained.json → web\snn_data.js
node web\smoke_test.js                  # Frontend inference engine smoke test
node web\evaluate.js                    # Extended experiment matrix (generates web\results.json, deterministic reproducible)
```

- **Fixed random seeds** (fully reproducible pipeline);
- Environment dependencies in `pylibs` (`PYTHONPATH` points to it);
- Raw data: place the 4 FlyWire `csv.gz` files in the directory specified by `DATA_DIR` at the top of `extract_circuit.py` (default: `<FlyWire data directory>`); modify this constant when switching machines;
- User-facing operating instructions and A/B/C/D acceptance checklists are in `web\manual.html`; training pipeline and local environment details are in `README.md`.

---

## 7. Conclusions

1. An LIF SNN trained on a connectome-constrained topology can complete the "visual-wind dual-modal threat escape" synthetic control task: fusion mode trigger accuracy 0.799, obstacle avoidance success rate (direction metric) 0.903, direction error 10.7°, GF latency 7.65ms (extended matrix 5-seed evaluation: 0.801±0.020 / 0.926±0.030 / 10.6°±0.5 / 7.65±0.20ms; ±std covers only evaluation sampling seed uncertainty, not training seed repetition) — all B-type numerical thresholds met (thresholds: trigger ≥75% / success ≥85% / direction ≤20° / latency ≤10ms, see `web\manual.html` "numerical acceptance" section).
2. Fusion's value lies in **direction** (10.6° vs vision only 18.2°); triggering depends almost entirely on vision; wind alone is unusable (GF zero fires).
3. Cross-platform experiments show: this model solves the neural decision of "when to dodge and which direction to dodge"; "actually dodging" additionally requires earlier warning + stronger execution dynamics (save rate +1.5pp / +1.0pp, net gain ~+1.3pp / +1.0pp; save events only 8/529 and 5/529, low counts).
4. **Scientific qualification (restated)**: training data is synthetically generated from physics formulas (not electrophysiology / behavioral measurements); connectome is from a single female adult fly, synapse counts are merely proxies for anatomical connection strength, not physiological weights; **all conclusions hold only on this synthetic control task** and do not constitute biological or engineering finality.