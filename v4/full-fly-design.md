# Complete Drosophila Design Plan (v4) — Making the Fruit Fly a Complete Organism: Body-Brain Closed-Loop Sensing the Physical World, and Answering "Can Drosophila Neurons Fly a Drone"

> Research + Design Document (design proposals only, no implementation code).
> Research date: 2026-09-23. Research method: Web search API unpaid (HTTP 402) unavailable, all online information verified by fetching public pages (PyPI / GitHub raw / GitHub API / arXiv / official sites) via `web_fetch`; **anything that could not be verified online is marked "could not be verified online" — no links or conclusions are fabricated**.
> Project status: FlyWire FAFB whole-brain connectome-constrained escape sub-circuit SNN (1871 neurons / 45,524 edges: LPLC2 visual 210 + JO-B/JO-C wind 325 + GF giant fiber 2 + downstream output 169), topology-fixed, weight-only-trained LIF SNN, outputting "should I escape (GF fire) + which direction to flee (direction readout)". v3 metrics: fusion trigger 0.799 / obstacle avoidance success rate 0.903 / direction error 10.7° / GF latency 7.65 ms (see `REPORT.md`).
> Current gap: In the 3D world the fruit fly/drone **only swaps shells** — wing flapping/rotor spinning is hand-written animation, the brain only makes escape decisions, **there is no body-brain closed loop** (visual/wind changes caused by the organism's own actions are not fed back to the brain).

---

## 0. Executive Summary

1. **Whole-body simulation benchmarks — three picks each with a distinct role**: **FlyGym** (fruit fly whole-body biomechanics + compound eyes + force feedback, Apache-2.0, very active) — copy its body and sensory interfaces; **flybrain** (homologous FlyWire whole brain 139,255 neuron LIF browser real-time, MIT) — proves **whole-brain-scale compute is not a bottleneck** and provides whole-brain packaging/visualization references; **OpenWorm** (C. elegans neural + body closed loop, MIT) — the organizational precedent for "connectome → body closed loop".
2. **Key architectural judgment**: The answer to "fruit fly brain flies a drone" is NOT to have the SNN directly stabilize a quadrotor, but rather **use a hierarchical approach**: the fruit fly brain only outputs pseudo-muscular commands of "trigger + direction + turning asymmetry", **stability control is left to the flight controller** (PX4, biologically analogous to the ventral nerve cord + haltere reflexes), and the pilot-takeover arbiter always has the highest priority. Neuro-LIFT (arXiv:2501.19259) has already validated "event camera + SNN + drone" feasibility on a real aircraft.
3. **Closed-loop modification core**: Current cues are passively synthesized and must be changed to **generated in real time from body state** — self-motion enters the optical flow (expansion rate) and wind (relative airflow) formulas, plus an **efference copy** for self-motion compensation.
4. **Honest boundaries**: Wing aerodynamics, flight muscle / CPG motor neurons are **not in the FAFB whole-brain table** (they are in the ventral nerve cord VNC), requiring synthetic models as substitutes or external VNC connectome data; closed-loop stability and sim-to-real are the biggest uncertainties.
5. Recommended roadmap: **Phase 1 (6–10 person-weeks) minimal closed loop, get it flying → Phase 2 (12–20 person-weeks) body-brain integration → Phase 3 (8–14 person-weeks) drone adaptation layer (SITL/HIL) → Phase 4 (12–20 person-weeks) netted-cage real aircraft**.

---

## 1. Open-Source Project Survey (item by item: link / one-sentence value / License / activity level)

### 1.1 Whole-Body / Whole-Brain Simulation Benchmarks

| Project | Link | One-Sentence Value | License | Activity Level (verified this time) |
|---|---|---|---|---|
| **FlyGym (NeuroMechFly v2)** | https://pypi.org/project/flygym/ , https://github.com/NeLy-EPFL/flygym , docs https://neuromechfly.org/ | Adult fruit fly **physical whole-body digital twin**: micro-CT biomechanical model + compound eye vision (hexagonal ommatidial array) + olfaction + brain/VNC hierarchical control interface (descending/ascending representations) + foot adhesion + force feedback (joint angle, active force, contact force) | **Apache-2.0** (PyPI license field verified) | **Very active**: PyPI latest version 2.1.0 released 2026-06-24; 2.x fully rewritten in 2026-03 (CPU ~10x, GPU (Warp/MJWarp) ~300x speedup); maintained by EPFL Ramdya lab; Python ≥3.12 |
| **flybrain** | https://github.com/snedea/flybrain , online version https://flybrain.app | **In-browser FlyWire FAFB v783 whole-brain LIF real-time simulation**: 139,255 neurons / 2.7M connections, Web Worker integration + WebGL spike visualization (grouped by Sensory/Central/Drives/Motor), behavior "emerges" from connection propagation (foraging, touch startle, phototaxis, thermotaxis) | **MIT** (GitHub API spdx=MIT; README points to license.md) | **Active**: created 2026-03-27, last push 2026-08-13, 159 star / 32 fork, JavaScript; data citing Dorkenwald et al., Nature 634:124–138 (2024) |
| **worm-sim** | https://github.com/heyseth/worm-sim , online version https://sethm.me/worm-sim/ | In-browser *C. elegans* 302-neuron connectome simulation + interactive locomotion, **flybrain's predecessor** (note: spelling is `heyseth`; `heyeseth/worm-sim` returns 404 via API verification, does not exist) | **MIT** (GitHub API verified) | Continuously available: created 2018-02, last push 2026-06-07, 425 star / 37 fork |
| **OpenWorm** | https://github.com/openworm/openworm , official site https://openworm.org | *C. elegans* **first whole-body computational model**: c302 nervous system model + Sibernetic 3D soft-body co-simulation producing behavior, the engineering organizational precedent for "connectome → body closed loop" | **MIT** (LICENSE file verified) | Long-term project (since 2012), main repo focused on Docker stack integration + project-level issues; specific star/commit data was truncated during page fetch this time, **could not be verified online** |

**Relationship between flybrain and this project (key borrowing analysis)**:
- **Homologous data**: flybrain uses FlyWire FAFB v783, homologous to our local `connections_princeton.csv.gz` and 3 other tables (Dorkenwald et al. 2024). It does "whole-brain emergent behavior", we do "sub-circuit constrained trainable + closed-loop body" — **complementary, not competitive**.
- **Three things to directly copy**: ① **Binary packaging and region-based grouping** of the whole-brain connectome (`data/neuron_meta.bin.gz` approach), orders of magnitude faster than row-by-row CSV; ② **Web Worker + WebGL spike panel** (139K neurons rendered in real time), portable to our `web\world.html` sandbox as a "whole-brain perspective" debug panel; ③ **Sensory-interactive interface design** (feed / touch / air / light / temp — five stimulus types fed directly into sensory groups) — exactly the interface shape our "closed-loop sensory encoding" needs.
- **Reality check that must be stated**: flybrain's "behavioral emergence" is **demo-grade evidence**; its weight rules and sensory encoding details could not be verified online this time (README does not specify them); its emergent behaviors cannot be taken as scientific conclusions, and its weight generation method cannot be assumed to match our `W=sign(NT)·log1p(syn_count)`.
- **Scale insight**: 139K neurons / 2.7M connections LIF **runs in real time in a browser** → the "whole-brain-scale compute" we worry about is actually not a showstopper (see §5).

### 1.2 Physics Engines

| Project | Link | One-Sentence Value | License | Activity Level (verified this time) |
|---|---|---|---|---|
| **MuJoCo** | https://mujoco.org/ , https://github.com/google-deepmind/mujoco | High-speed, accurate physics engine optimized for contact, supporting **muscle/tendon/slider-crank actuators** (exactly the primitives the fruit fly "pseudo-muscular command" needs), the underlying engine of FlyGym | **Apache-2.0** (LICENSE file verified) | Maintained by Google DeepMind, continuously releasing (FlyGym 2.x already uses its Warp/MJWarp GPU backend) |
| **Brax / MJX** | https://github.com/google/brax | JAX fully-differentiable physics + RL training stack (PPO/SAC/ARS/ES/APG), suitable for large-scale parallel policy training | **Apache-2.0** (LICENSE file verified) | ⚠️ Official README warning: from 0.13.0 onward **only `brax/training` is maintained**; for physics simulation the official recommendation is to switch to **MJX** (`mujoco_mjx`) or **MuJoCo Warp**, with scenes from MuJoCo Playground |
| **Rapier** | https://github.com/dimforge/rapier , https://rapier.rs | Rust 2D/3D physics engine with SIMD batched constraint solving, parallel pipeline, JS/Python bindings, suitable for embedding in a Web-based 3D world | **Apache-2.0** (README badge verified) | Active (maintained by dimforge, Python bindings under development — per README) |

### 1.3 Neural Simulation

| Project | Link | One-Sentence Value | License | Activity Level (verified this time) |
|---|---|---|---|---|
| **Brian2** | https://pypi.org/project/brian2/ , https://brian2.readthedocs.io | Clock-driven SNN simulator, equations-as-code, best suited for **constrained-topology LIF + custom synaptic dynamics** research iteration | **CeCILL-2.1** (PyPI verified) | Active: 2.10.1 released 2025-12-05, multi-platform wheels |
| **NEST** | https://github.com/nest/nest-simulator , https://nest-simulator.org | Large-scale spiking network simulation (laptop to supercomputer), rich model library | **GPLv2+** (README verified) | Active: v3.7 series, conda/pip/Docker multi-channel, README shows continuous CI |
| **GeNN** | https://github.com/genn-team/genn | GPU code generation (CUDA/HIP) SNN simulation, includes **insect mushroom body MNIST classification** example, suitable for GPU-scale runs | **LGPL-2.1** (LICENSE file verified) | Active: 5.4.0 release + master pip installable, Docker images |
| **SpikingJelly** | https://pypi.org/project/spikingjelly/ | PyTorch SNN framework: LIF, surrogate gradients, STDP, ANN→SNN, event dataset family (DVS series) | PyPI labeled **Other/Proprietary License** (no standard SPDX given; check before commercial use) | Stable version 0.0.0.0.14 (2023-03), but 2.0.0 pre-release line continuously updated (2.0.0rc1 2026-08-29) |

> The existing training pipeline uses **snntorch** (`train_snn.py`, `surrogate.fast_sigmoid()`), which was not included in the online verification checklist this time; migration recommendations see §2.3.

### 1.4 Neuromorphic Drones / Flight Controllers / Quadrotor Simulation

| Project | Link | One-Sentence Value | License | Activity Level (verified this time) |
|---|---|---|---|---|
| **Neuro-LIFT** | https://arxiv.org/abs/2501.19259 | **Real-aircraft evidence**: real-time neuromorphic navigation framework on Parrot Bebop2 quadrotor — event camera + SNN + LLM voice commands, dynamic environment obstacle avoidance | Paper CC BY-NC-ND 4.0 (arXiv verified); **code repository could not be verified online** (`github.com/AmoghJoshi/Neuro-LIFT` returned 404) | Accepted at IJCNN 2025; v2 revised 2025-04-26 |
| **gym-pybullet-drones** | https://github.com/utiasDSL/gym-pybullet-drones | Minimal quadrotor Gym environment (PyBullet physics + PID/MRAC control + SB3 RL examples + **Betaflight SITL** integration), suitable for adaptation layer rapid prototyping | Specific License **could not be verified online** (README not labeled; fetched version is the learnsyslab restructured fork) | Active: selected for GitHub Maintainer Spotlight 2026; supports gymnasium / SB3 2.0 |
| **PX4-Autopilot** | https://github.com/PX4/PX4-Autopilot | Industrial-grade open-source flight controller: **SITL / HIL / real-aircraft same stack**, offboard control interface, comprehensive failsafe (RC loss → Land/RTL) | **BSD 3-Clause** (LICENSE file verified) | Very active (2025 copyright line still updating; de facto drone standard) |
| **ArduPilot** | https://github.com/ArduPilot/ardupilot | Another major open-source flight controller, full Copter functionality (alternative; pick one from PX4 or ArduPilot) | **GPLv3** (COPYING.txt verified) | Active (long-term community project; specific version not fetched this time) |

### 1.5 Comparison and Lessons-Learned Summary (FlyGym vs flybrain vs OpenWorm)

| Dimension | FlyGym | flybrain | OpenWorm |
|---|---|---|---|
| Body | ✅ Whole-body biomechanics (micro-CT, hexapod + joints + adhesion + contact force) | ❌ No physical body (2D interactive demo) | ✅ Soft-body whole body (Sibernetic) |
| Brain | ❌ No connectome network (provides brain/VNC interface) | ✅ Whole-brain 139K LIF (same FlyWire data) | ✅ c302 (302 neurons) |
| Sensory | ✅ Compound eye vision + olfaction + force feedback | ✅ Five stimulus interfaces (light/temperature/wind/touch/food) | Simple |
| Closed Loop | ✅ Body-brain interface spec (descending/ascending) | ✅ Stimulus → behavioral emergence | ✅ Neural + soft-body co-simulation |
| What we copy | **Body, compound eye encoding, force feedback, brain/VNC hierarchical interface** | **Whole-brain-scale packaging, spike visualization, stimulus interface shape, confidence that "whole-brain real-time is feasible"** | **Engineering organization of neural-body dual-process co-simulation** |
| What we don't copy | It has no flight/diptera module (README component list doesn't list one; **could not verify online** if a flight extension exists) | No physical body, weight rules not published, "emergence" cannot serve as scientific conclusion | C. elegans soft-body dynamics are irrelevant to fruit flies |

---

## 2. Complete Fruit Fly Blueprint

### 2.0 Current Gaps (Three Shortfalls)

1. **Brain gap**: Only the escape sub-circuit (1871 nodes). No visual motion layer, no turning/heading system, no rhythm generator, no flight muscle motor neurons.
2. **Body gap**: Wings/rotors are hand-written animations, no mechanics; hexapod legs, wing-root joints, and muscles don't exist.
3. **Closed-loop gap**: During training and runtime, visual expansion and wind pressure cues are **passively synthesized from threat motion formulas** (`dθ/dt = −2s·ṙ/(r²+s²)`, `u ∝ s²v/r²`, containing only threat motion terms); once the fruit fly itself moves, optical flow / wind **does not change**, and the brain receives no sensory consequences of self-motion.

### 2.1 Which Neuron Groups Are Still Missing (Including: Can They Be Extracted from the FlyWire Local Tables?)

**Data preamble (honest statement)**: The local 4 tables come from **FAFB = Full Adult Female Brain — brain only**. Flight muscle / leg muscle motor neurons (TTMn, DLMn, etc.), haltere afferents, and rhythm generators (CPG) are all in the **ventral nerve cord (VNC/thoracic segments)** — `README.md` already states "TTMn/DLMn from the Drosophila thorax are not in this whole-brain dataset." Therefore the table below is split into two categories: **Category A = extractable from local tables (within the brain)**; **Category B = not in local tables, use synthetic models as substitutes or supplement with external VNC connectome data** (VNC dataset **could not be verified online** for specific download addresses this time; acquisition paths require separate research).

| Neuron Group | Function | Category | Extraction/Construction Method (following `extract_circuit.py` approach) |
|---|---|---|---|
| Visual motion layer: LC family (LC4/LC6/LC11/LC12 etc. lobula columnar), T4/T5 (local motion detection), Mi1/Tm1/Tm2/Tm3/Tm9/Tm20 (medulla intermediate layers), HS/VS (lobula plate wide-field), VPN (optic bridge) | **Early-stage encoding** of optical flow / expansion / object motion, transforming compound eye pixel streams into motion features beyond LPLC2 (including rotational optical flow needed for turning) | **A** | `primary_type ∈ TYPE_SETS['visual_motion']` as seeds → `hop()` forward/backward 1–2 hops → intersect/union with existing 1871 nodes; seed type names based on actual `distinct primary_type` in local tables (lesson learned: GF is actually called `DNp01`) |
| Auditory / wind extension: JO-A…JO-E all types, antennal mechanosensory accessories | Multi-axis wind / sound / gravity encoding | **A** (JO-B/JO-C already included) | `primary_type LIKE 'JO-%'` collect all, replace existing `WIND_TYPES` |
| Visual / head state: ocelli pathway, compound eye edge neurons | Light level / horizon → attitude reference | **A** (based on actual types in local tables) | Filter by `class/sub_class` keywords like 'ocell' etc. in `classification.csv` (run `Counter` statistics on actual field values first during implementation) |
| Central complex CX: E-PG (heading compass), P-EN/P-EG, PFL3 (turning control), FC2, Δ7, PB/EB/FB neurons | **Turning and heading** — "which direction to flee" upgraded from reflex to a stateful turning controller | **A** | `primary_type ∈ {'E-PG','P-EN','P-EG','P-FL3','FC2','Δ7',...}` (names based on local tables) + `class` keyword 'central complex' |
| Full descending neuron set: DNp/DNa/DNm series (including DNp01=GF) | Brain → body **sole output channel**, motor command carrier | **A** | `classification.csv` where `super_class == 'descending'` **full set** (currently only 169 within 2 hops downstream of GF) |
| In-brain motor neurons | super_class='motor' full set | **A** | Same as above, `super_class == 'motor'` full set |
| Neuromodulatory groups: DA/5-HT/OCT neurons (DOPaminergic, Serotonergic, octopaminergic) | Arousal / gain modulation (biological basis of GF priming) | **A** | `neurons.csv.gz` where `nt_type ∈ {DA, SER, OCT}` + `class` keywords |
| **Wing/leg CPG (half-center oscillators)** | Wing-beat rhythm, gait rhythm | **B** | **Synthetic model**: 2–6 half-center oscillators + parameter-driven ("takeoff/hover/escape" three presets), pseudocode-level design only, no connectome basis — honestly noted |
| **Flight muscle / steering muscle motor neurons** (TTMn jump, DLMn indirect flight muscles, 8 steering muscle MNs) | Motor commands → muscles | **B** | **Synthetic model**: connect Category A DN outputs directly to "pseudo-muscular command space" (§3.1); replace with external VNC connectome when available |
| Haltere afferents + reflexes | Angular velocity sensing (= biological gyroscope) | **B** | **Synthetic model**: IMU angular velocity → sensory encoding (simple to implement); VNC reflex arc supplemented from external data |
| Leg / wing proprioception (chordotonal, campaniform) | Joint angle / wing load feedback | **B** | Initially use FlyGym force feedback interface (joint angle / contact force) as substitute; organ-level encoding added later |

**Extended extraction script design (using all existing mechanisms from `extract_circuit.py`, adding only, not modifying)**:
- Reuse: `load_tables()` (4-table reading, neuropil row merging via `aggregate_edges()`), `hop()` graph expansion, `syn_count ≥ 3` weak-edge pruning, neurotransmitter sign `NT_SIGN = {ACH:+1, GABA:−1, GLUT:−1(insect), DA/SER/OCT:+1 modulatory}`, weight `W = sign(NT)·log1p(syn_count)` + postsynaptic total input strength normalization, `role_of()` role labeling, sparse JSON + CSV dual-format export.
- New additions: ① `TYPE_SETS` multi-seed sets (Category A from table above); ② `super_class` full recruitment (descending / motor / modulatory); ③ multi-hub graph (GF + CX heading group), short-path pruning parameter relaxed from "≤3 edges" to configurable per group (visual motion layer uses ≤2-edge tight layer, CX uses fully connected subgraph); ④ output `escape_network_v2.json`, node roles expanded to `input_vision / input_wind / visual_motion / heading / modulatory / hub_gf / dn / motor / cpg_port / interneuron`.
- Expected scale: given a full-brain 138,327-cell table and 5.34M connection rows, the above Category A union + short-path layer is estimated at **10K–30K nodes / hundreds of thousands of edges** (estimate, actual results may vary); following flybrain's precedent of running 139K neurons in real time, simulation compute is **feasible**.
- First implementation step (must do before writing code): run `distinct primary_type / additional_type(s)` frequency statistics on `consolidated_cell_types.csv.gz`, confirm real names for each group before defining `TYPE_SETS` (the GF=DNp01 lesson must be institutionalized).

### 2.2 Body Model (Hexapod + Diptera Biomechanics/Joints)

**Recommended: flight-first, walking-later two-stage body.**

**Option A (Phase 1, initial): 6-DoF rigid body + flapping aerodynamic average (wingbeat-averaged wrench)**
- Fruit fly thorax as rigid body (mass ~1 mg, wingspan ~2 mm order of magnitude), state = pose + linear/angular velocity;
- Each wing 3 degrees of freedom: **wing-root pitch + stroke + deviation**, driven by CPG outputs for left/right **beat frequency (~200 Hz order), amplitude, angle of attack, stroke asymmetry**;
- Aerodynamics: quasi-steady blade-element / aerodynamic coefficient model (lift, drag, pitch moment as algebraic functions of beat frequency², amplitude, angle of attack), **averaged over the wingbeat cycle** to produce net external wrench (treated as constant wrench within 1 kHz control steps) — deliberately **not** doing flexible-wing CFD (cost-benefit mismatch, see §5);
- Actuator primitives use MuJoCo's muscle/tendon semantics or equivalent "pseudo-muscle → wrench" mapping, ensuring interface compatibility with the Phase 2 FlyGym body.

**Option B (Phase 2, complete): FlyGym biomechanical body + custom dual-wing module**
- Copy FlyGym's hexapod skeleton/joints/foot adhesion/contact force (Apache-2.0, directly modifiable); joints: 3 segments per leg (coxa-trochanter-femur-tibia-tarsus simplified chain), walking CPG-driven;
- Dual-wing aerodynamic module (from Option A) mounted on FlyGym body thorax; ⚠️ FlyGym README component list **does not include** dual-wing/flight module — whether community flight extensions exist **could not be verified online** — plan as if "none exist";
- Force feedback interface copied from FlyGym: joint angle, active force, contact force, custom anatomical points — this is a ready-made source of "proprioception".

### 2.3 Closed-Loop Design: Self-Motion → Optical Flow / Wind Changes → Sensory Encoding → Brain → Motor Commands

**Core modification: cues change from "passive synthesis" to "real-time generation from body state".**

Each simulation step (1 kHz physics / sensory update 100–500 Hz / brain dt=1 ms):

1. **World + body state** (threat pose, self pose/velocity/angular velocity, previous motor command) →
2. **Sensory encoder (new module SensorGen)**:
   - **Optical flow / expansion**: within the compound eye field of view (FlyGym hexagonal ommatidial array or simplified visual cone), each threat's angular size `θ = 2·atan(s/r)`, whose rate of change **contains both threat motion and self-motion**: `ṙ` is relative distance change (flying toward an obstacle yourself also causes `θ` to surge); then superimpose the **global optical flow field** produced by self-motion (rotational component → turning feedback);
   - **Wind**: `u_rel = v_air − v_body − v_downwash(CPG_cmd)` — **relative airflow**, self forward flight / wing-beat downwash both enter JO encoding;
   - **Haltere / gyroscope**: `ω_body` → angular velocity encoding (synthetic);
   - **Force feedback**: FlyGym joint angle / contact force → proprioceptive encoding (Phase 2).
3. **Brain (SNN)**: existing 1871-node network (Phase 1) → + §2.1 groups (Phase 2). Input = above-encoded spikes; output = GF firing (trigger) + direction readout + turning-group left/right asymmetry + DN group firing.
4. **Motor commands (new module MotorGen)**: outputs **pseudo-muscular commands** `M = {lift level, pitch, roll, yaw, escape burst pulse, wing-beat CPG parameters}`; drives CPG/body execution.
5. **Efference copy (Phase 2)**: a copy of `M` fed back to the sensory encoder —
   - cancels out **self-generated wind** in JO encoding (wing-beat downwash + self-flight relative wind);
   - cancels out **self-generated rotational components** in optomotor optical flow (otherwise turning the head would flood the visual field with optical flow, contaminating LPLC2-like channels with self-motion and causing false escape triggers).
   - This is a critical link from "being able to move" to "moving stably"; fruit fly biology has corroborating evidence (corollary discharge), but **there is no data support for the specific pathway within our connectome subgraph — treated as a synthetic model**.

**Closed-loop acceptance criteria (applicable from Phase 1)**:
- Self-rotation at 30°/s: optical flow encoding channel firing rate varies monotonically with `ω` (self-motion visible);
- Forward flight at 0.5 m/s: JO encoding varies with `u_rel` (self-motion visible);
- **Within 2 brain windows (≥24 ms)** after GF-triggered escape, the encoder can detect new optical flow / wind changes caused by the escape maneuver (direct evidence of closed-loop validity);
- Toggle the "self-motion term" of the sensory generator for ablation comparison: disabling it produces self-motion false triggers (expected), enabling it reduces false trigger rate (quantifies closed-loop value).

**Training strategy recommendations**: Keep the existing "topology-fixed, weight-only-training" constraint unchanged; new groups first **freeze anatomical weights** and only train readout/interface gains, then unfreeze for fine-tuning; training data upgraded from "synthetic threat trajectories" to "closed-loop rollout sampling" (optical flow generated by the organism's own flight is the true distribution — this is the first layer of distribution mismatch fix beyond sim-to-real). Simulator selection: existing snntorch works for small networks; once scaling to §2.1's 10K–30K node range, migrate to **Brian2** (research iteration) or **GeNN** (GPU batch rollouts), NEST as backup (GPLv2+ is the most virally licensed — use with caution for commercial drone products).

---

## 3. The Answer Path for "Fruit Fly Brain Flies a Drone"

### 3.1 Motor Commands → Adaptation Layer (Control Allocation: Pseudo-Muscular Commands → Quadrotor RPM/Attitude)

**Architectural judgment (the most important conclusion of this proposal)**: Fruit fly neurons **do not need** to learn to stabilize a quadrotor. In biological fruit flies, stable flight is also not achieved by the GF/brain — it is accomplished by thoracic CPG + haltere reflex + flight muscle mechanical negative feedback (all in the VNC). The drone analog:

```
[Fruit fly SNN brain] --pseudo-muscular command M--> [Adapter] --desired wrench/attitude--> [Flight controller stability loop PX4 (≈VNC+haltere)] --> [ESCs/motors] --> [airframe]
      ^                                  |
      |------ Sensory encoder(SensorGen) <-----+-- IMU/airflow meter(≈compound eyes/JO/haltere) <--+
                                         |
                               [Arbiter Arbiter] <-- Pilot RC (highest priority)
```

- **Pseudo-muscular command space → quadrotor mapping** (biological action semantics → control semantics):
  - `Escape burst pulse (GF fire)` → short-duration thrust step + saccade (angular velocity pulse) in the escape direction — corresponding to fruit fly "jump takeoff + rapid body rotation";
  - `Left/right turning asymmetry (CX/turning group)` → yaw / roll angular velocity commands;
  - `Lift level (CPG beat frequency/amplitude)` → throttle / collective pitch;
  - `Hover / cruise preset` → altitude hold / speed hold (implemented by flight controller; fruit fly brain only provides the "target").
- **Control allocation**: desired wrench (net force + 3-axis moments) → quadrotor mixing (control allocation matrix) → 4-channel motor RPM; connect to **PX4 offboard** (attitude / angular velocity setpoints or actuator direct drive), SITL / real-aircraft same interface.
- **Supporting evidence**: Neuro-LIFT (arXiv:2501.19259) proves event camera + SNN navigation can run on a real Bebop2 aircraft; latency-sensitive tasks can be completed by a neuromorphic stack; its "high-level planning + low-level execution" hierarchy is isomorphic to this proposal (its LLM layer is one we don't need).

### 3.2 Sim-to-Real Layered Roadmap

| Layer | Content | Exit Criteria |
|---|---|---|
| **L0 Simulation Closed Loop** | Fruit fly body + SNN + SensorGen from §2 running closed loop in the 3D world (existing `web\world.html` sandbox + MuJoCo/custom dynamics); same pseudo-muscular interface | All 4 closed-loop acceptance criteria pass (§2.3); escape demo reproducible |
| **L1 SITL** | Adaptation layer connected to **PX4 SITL** (or gym-pybullet-drones for rapid prototyping): fruit fly brain piloting a quadrotor through surprise-threat scenarios | 100 Monte Carlo surprise runs: trigger-escape success rate, no-pilot-intervention no-collision ≥ target (suggested ≥80% as starting point) |
| **L2 Hardware-in-the-Loop HIL** | Real flight controller board + simulator + real sensor loop (event camera DVS or grayscale optical flow, pitot tube / hot-wire anemometer); validate latency budget (perception → command ≤ 20 ms) | End-to-end latency and packet-loss tests pass; takeover logic timing verified (takeover latency ≤ 1 RC frame) |
| **L3 Real Aircraft** | Small quadrotor (250 mm frame size or smaller) + netted cage + low speed; real surprise threats (soft pendulum / airflow disturbance) | Real-aircraft trigger-escape success rate reported; **takeover drill: pilot intervention success rate 100% at any time** |

**Sim-to-real gap mitigation (required at every layer)**: Sensor domain randomization (optical gain, wind noise, DVS noise); latency/jitter injection; actuator saturation / motor first-order delay modeling (gym-pybullet-drones TODO list also acknowledges motor delay as a common pitfall); closed-loop rollout data retraining (not purely synthetic data).

### 3.3 Pilot Takeover Safety Baseline (Non-Negotiable Items)

1. **Arbiter priority is fixed**: `Pilot RC > Safety envelope (geofence/speed limit/angle limit/no-fly zone) > Fruit fly brain recommendation`. The fruit fly brain output is always merely a "recommendation" (trigger + direction), effective only after passing through the arbiter.
2. **Takeover channel is independent**: RC receiver directly connected to flight controller (not through the SNN compute stack), PX4/ArduPilot native RC override; takeover switching is **unconditional, zero-latency, one-way** (pilot→auto can revert, auto cannot seize control from pilot).
3. **Failsafe enabled by default**: RC loss → Land/RTL; compute stack crash → flight controller independent hover/landing (SNN stack decoupled from flight controller, watchdog heartbeat); hardware kill switch.
4. **Flight envelope envelope**: Speed / tilt angle / altitude / radius limits enforced at the flight controller layer (safety envelope has higher priority than the fruit fly brain, preventing escape maneuvers from crashing into the cage).
5. **Flight test discipline**: L3 full-process netted cage + low speed + two-person (pilot + safety officer); start in "shadow mode" (fruit fly brain records recommendations only, does not execute) for calibration, then grant authority.

---

## 4. Phased Roadmap

### Phase 1: Minimal Closed Loop (Fruit Fly Body + CPG Wing Beat + Self-Motion Optical Flow Feedback, First Flight in 3D World)
- **Content**: 6-DoF fruit fly flight body (Option A) + half-center CPG wing beat + SensorGen (optical flow / wind / gyroscope) + existing 1871-node SNN in the loop + pseudo-muscular command space; transform the `web\world.html` sandbox from "only swapping shells" to "actually flying by itself".
- **Deliverables**: ① Closed-loop demo recording (threat → GF → escape → self-motion optical flow change → secondary decision); ② Quantified logs of the 4 closed-loop criteria from §2.3 (including self-motion term on/off ablation); ③ Pseudo-muscular interface specification document (the contract for Phase 3).
- **Effort**: **6–10 person-weeks** (dynamics + CPG 3 weeks, SensorGen 3 weeks, integration debugging 2–4 weeks).
- **Primary risks**: Flapping-average wrench parameters inaccurate causing flight posture to "not look like a fruit fly" (acceptable as long as it is physically plausible); self-motion optical flow computation overhead (can be mitigated with direct geometric formula computation); closed-loop latency budget exceeding 20 ms (requires suppressing sensory update frequency).

### Phase 2: Body-Brain Integration (More Neuron Groups Enter the Loop)
- **Content**: Extended extraction script runs on Category A new groups (visual motion layer, CX turning, full DN set, neuromodulatory groups) → `escape_network_v2`; efference copy self-motion compensation; (optional) introduce FlyGym hexapod body + force feedback; training upgraded to closed-loop rollout sampling; simulator migrated to Brian2/GeNN based on scale.
- **Deliverables**: ① `escape_network_v2.json` + extraction report (group counts, connection statistics); ② flight demo with turning/heading behavior (no longer only "flee-away"); ③ efference copy on/off ablation experiment (quantified reduction in self-motion false triggers); ④ cross-validated spike panel screenshots with flybrain under same-parameter whole-brain simulation (borrowing its visualization).
- **Effort**: **12–20 person-weeks** (extraction 3–4 weeks, integration + training 6–10 weeks, ablation experiments 2–6 weeks).
- **Primary risks**: New group weights based only on anatomical proxies (log1p synapse count) causing network instability (mitigation: freeze + readout fine-tuning, energy regularization); CX group type naming in local tables inconsistent with literature (mitigation: run distinct statistics first); training convergence slowing at tens-of-thousands-of-nodes scale.

### Phase 3: Drone Body Adaptation Layer (Simulation Closed Loop → SITL/HIL)
- **Content**: Adapter (pseudo-muscular → control allocation → PX4 offboard) + arbiter (pilot priority) + L1 SITL scenario library + L2 HIL bench (real flight controller board + sensor loop); Neuro-LIFT-style event camera integration assessment.
- **Deliverables**: ① SITL Monte Carlo report (100 surprise run success rate / takeover records); ② end-to-end latency budget report (perception → motor); ③ takeover logic formalization document + drill records; ④ demonstrable demo ("fruit fly brain flies a quadrotor" in the web sandbox).
- **Effort**: **8–14 person-weeks** (adaptation layer 3–5 weeks, SITL scenarios 3 weeks, HIL 2–6 weeks).
- **Primary risks**: Pseudo-muscular → quadrotor mapping semantics mismatch (escape maneuver causes flip) — mitigated by safety envelope angle limiting; PX4 offboard timing pitfalls (lockstep simulation); HIL equipment/interface preparation timeline unpredictable (off-the-shelf component risk).

### Phase 4: Real Aircraft (Netted Cage → Restricted Field)
- **Content**: Small quadrotor modification (flight controller + Jetson-class edge device running SNN + DVS/optical flow/anemometer); shadow mode → grant authority; real surprise threat tests; full-process pilot takeover assurance.
- **Deliverables**: ① Real-aircraft trigger-escape success rate report (including false trigger rate); ② takeover drill 100% success record; ③ application conclusion white paper: the **quantitative answer** to "can fruit fly neurons fly a drone" (success/failure conditions, gap attribution).
- **Effort**: **12–20 person-weeks** + hardware and test flight windows (calendar time constrained by weather / venue / safety approvals).
- **Primary risks**: Sim-to-real gap (wind noise, vibration, DVS noise) overwhelming sensory encoding (mitigation: inter-layer domain randomization + real-aircraft data backfill fine-tuning); insufficient edge compute (mitigation: GeNN/Loihi-class deployment or network pruning); safety approvals and venue (non-technical risk but often the real bottleneck).

**Total: approximately 38–64 person-weeks (9–16 person-months), roughly 1–1.5 years for one full-time person.**

---

## 5. Honest Assessment

### 5.1 Mature and Copyable (Direct Copy / Minor Modification)
- **Physics and body**: MuJoCo (including muscle/tendon actuators), FlyGym biomechanical body / compound eye encoding / force feedback / brain-VNC interface spec (Apache-2.0);
- **SNN simulation**: Brian2/GeNN/NEST, snntorch/SpikingJelly surrogate gradients and training patterns (current pipeline already uses them);
- **Whole-brain-scale engineering**: flybrain proves 139K neurons / 2.7M connections LIF browser real-time is feasible — **whole-brain compute is a mature item** (its packaging, WebGL panel, stimulus interface can be directly borrowed); OpenWorm's neural-body dual-process co-simulation organizational approach;
- **Drone side**: PX4 (BSD-3) SITL/HIL/real-aircraft same stack + RC override/failsafe native safety mechanisms, ArduPilot (GPLv3) as alternative, gym-pybullet-drones for rapid prototyping;
- **Feasibility evidence**: Neuro-LIFT has already flown "event camera + SNN" on a real aircraft (Bebop2).

### 5.2 Frontier — No Confidence (No Mature Precedent, Treat as Experimental)
- **Drosophila diptera whole-body flight simulation**: FlyGym component list does not include flight/diptera module (**could not verify online** if flight extensions exist); flapping aerodynamic average model parameters require self-calibration, "does it look like fruit fly flight" has no ready benchmark;
- **CPG / flight muscle motor neurons**: Outside FAFB tables (in VNC), not available locally; synthetic CPG is a stopgap, VNC connectome acquisition path **could not be verified online** this time;
- **Efference copy pathway**: Biology has corroborating evidence, but no data support within the subgraph, purely synthetic — poorly tuned compensation amounts will introduce new biases;
- **Closed-loop stability training**: Under the "topology-fixed, weight-only-training" constraint, whether the system can stably fly after adding 10K–30K nodes **has no precedent to cite**; flybrain's "emergence" only holds in demos without a physical body;
- **Biological SNN sim-to-real**: Neuro-LIFT proved "it can fly", but the specific combination of "fruit fly connectome decision + quadrotor" has an unknown gap curve;
- **flybrain weight/encoding details** are not published (README does not specify them), cannot be assumed to match our weight rules.

### 5.3 Top 3 Predicted Difficulties
1. **Closed-loop stability training**: In the action → sensation → brain → action loop, small encoding biases are amplified by self-motion (false trigger → erratic flight → even more chaotic optical flow). Mitigation: efference copy first, energy regularization, freeze topology and fine-tune only, gradual authority release (self-motion toggle ablation first, then full closed loop).
2. **Sim-to-real gap**: Synthetic cues → simulated SensorGen → real-aircraft sensors, three layers of distribution mismatch. Mitigation: domain randomization at each layer + real-aircraft data backfill; L2 HIL to surface latency / packet-loss issues early.
3. **Neuron group "data-function" misalignment**: FAFB is brain-only; functionally necessary CPG/MN are in the VNC; connectome topology ≠ physiological weights (single female fly, synapse count is only a proxy). Mitigation: synthetic modules with explicit boundary statements (conclusions always carry "synthetic data / proxy weight" qualifiers, following the scientific qualification system in `REPORT.md`).

---

## 6. Appendix: Online Verification Details This Time (URL → Verified Content)

| URL | Verification Result |
|---|---|
| https://pypi.org/project/flygym/ | flygym 2.1.0 (released 2026-06-24), Apache-2.0, full NeuroMechFly v2 component descriptions (see §1.1 for body text) |
| https://raw.githubusercontent.com/NeLy-EPFL/flygym/main/README.md | Same content as PyPI; 2.x rewrite, 10x/300x speedup, 1.x migration to flygym-gymnasium |
| https://arxiv.org/search/?query=FlyGym | Search page returned no indexed results (page showed only Tips only) — FlyGym paper follows the Nature Methods link given on the PyPI page |
| https://api.github.com/repos/snedea/flybrain + https://raw.githubusercontent.com/snedea/flybrain/main/readme.md | MIT; 139,255 neurons / 2.7M connections (FlyWire FAFB v783); Web Worker LIF + WebGL panel; 159 star/32 fork; last push 2026-08-13; forked from heyseth/worm-sim |
| https://api.github.com/repos/heyseth/worm-sim | MIT; 425 star/37 fork; created 2018, pushed 2026-06; C. elegans 302-neuron browser simulation |
| https://api.github.com/repos/heyeseth/worm-sim | **404** (this spelling does not exist; correct is `heyseth`) |
| https://raw.githubusercontent.com/openworm/OpenWorm/master/README.md + /LICENSE | c302+Sibernetic co-simulation; MIT |
| https://mujoco.org/ + https://raw.githubusercontent.com/google-deepmind/mujoco/main/LICENSE | Apache-2.0; muscle/tendon actuator catalog |
| https://raw.githubusercontent.com/google/brax/main/README.md + /LICENSE | Apache-2.0; official statement shifting physics to MJX/MuJoCo Warp |
| https://raw.githubusercontent.com/dimforge/rapier/master/README.md | Apache-2.0; Rust 2D/3D; Python bindings under development |
| https://pypi.org/project/brian2/ | CeCILL-2.1; 2.10.1 (2025-12-05) |
| https://raw.githubusercontent.com/nest/nest-simulator/master/README.md | GPLv2+; v3.7 |
| https://raw.githubusercontent.com/genn-team/genn/master/README.md + /LICENSE | LGPL-2.1; 5.4.0; CUDA/HIP code generation |
| https://pypi.org/project/spikingjelly/ | PyPI labeled Other/Proprietary; stable version 2023-03, 2.0.0rc1 2026-08-29 |
| https://arxiv.org/abs/2501.19259 | Neuro-LIFT: Bebop2 real aircraft, event camera + SNN + LLM; IJCNN 2025; CC BY-NC-ND 4.0 |
| https://github.com/AmoghJoshi/Neuro-LIFT | **404, code repository could not be verified online** |
| https://raw.githubusercontent.com/utiasDSL/gym-pybullet-drones/main/README.md | gymnasium/SB3/Betaflight SITL; License **could not be verified online** |
| https://raw.githubusercontent.com/PX4/PX4-Autopilot/main/LICENSE | BSD 3-Clause |
| https://raw.githubusercontent.com/ArduPilot/ardupilot/master/COPYING.txt | GPLv3 |

**Glossary**: GF = giant fiber (DNp01, escape trigger); LPLC2 = visual expansion-sensitive projection neuron; JO-B/JO-C = Johnston's organ mechanoreceptors (wind/sound/gravity); CX = central complex (heading/turning); CPG = central pattern generator (rhythm); DN = descending neurons; VNC = ventral nerve cord (this table's data **does not include** it); efference copy = outgoing copy (self-motion compensation); HIL = hardware-in-the-loop; SITL = software-in-the-loop; DVS = event camera.