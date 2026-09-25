# Drone Simulation Version — Detailed Execution Specification (agents follow this, do not improvise)

> Project root: <project root>
> Iron rules: ①SNN values unchanged (web\results.json SHA256=0E1339A088EF3B5820DCA342B989249403BE41D5A3F7EDB9DDB1D1A91C217E9E, jitter_sim_log.md=5C66B3D05A92F17961EC9EFC7F8406BFA6117CB04F5F47260F544AB52B8072AE) ②Pilot non-modifier-key takeover has highest priority (Shift/Ctrl/Alt/Meta/AltGraph exempt) ③file:// double-click usable: zero fetch / zero ES module / zero build; three.js uses web\vendor\three.min.js ④Coordinate mapping (x,z,y) unchanged ⑤Direction error formula = jump moment −r̂ unchanged.

## Phase 1: drone_physics.js (Flight Dynamics)

Create `web\drone_physics.js` (plain `<script>`, attached to window.DronePhysics):

```js
// class DronePhysics
// State: pos(Vector3), vel(Vector3), quat(quaternion), angVel(Vector3)
// Input update(dtSec, motors[4]) — 4 motor speeds (normalized 0..1.2)
// Parameters (demo-grade, world units, noted as "not calibrated for real hardware"):
//   mass=1.0, gravity=9.8, kf=0.12 (thrust coefficient), km=0.018 (anti-torque coefficient),
//   dragLin=0.35, dragQuad=0.02, maxTilt=40°, inertia=(0.02,0.02,0.04)
// Formulas:
//   Total thrust T = kf * Σ(motors[i]^2), along body -y axis (world frame rotated by quat)
//   Torque: roll=(m0-m2), pitch=(m1-m3), yaw=km*(m0-m2+m1-m3) (noted as simplified quadcopter X-type mixing)
//   Translation: a = (T_body + F_drag)/m + g; F_drag = -dragLin*v - dragQuad*|v|*v
//   Attitude: quaternion integration q += 0.5*q⊗ω*dt, normalize each step; euler angles clamped to ±maxTilt
//   Integration: semi-implicit Euler (update vel first, then pos), dt clamped ≤0.033s
// Hover calibration: hoverOmega = sqrt(mass*gravity/(4*kf)), exported as DronePhysics.HOVER
// Self-test function DronePhysics.selfTest(): hover 5s position drift <0.3, uniform acceleration, attitude response — three PASS/FAIL string results
```

Integration: In world.html's driveAgent, switch the "drone agent" path to use DronePhysics (fruit fly agent keeps the existing kinematics — fruit flies fly, they're not helicopters); add a "⚙️ Flight Dynamics" toggle to the console (on = new physics, off = old kinematics, default on); HUD adds 4 motor speed bars + attitude angles (roll/pitch).

## Phase 2: drone_adapter.js (Brain → Motor Adaptation Layer)

Create `web\drone_adapter.js` (attached to window.DroneAdapter):

```js
// Biomimetic layering: fruit fly brain (decision) → this adapter layer (thoracic ganglion) → motors (flight muscle); PID attitude loop = haltere reflex
// class DroneAdapter { update(brainOut, droneState, dtMs, pilotActive) -> motors[4] }
// State machine: CRUISE / ESCAPE / PILOT / LAND
// - PILOT highest priority: pilotActive (world.html existing non-modifier-key logic) → direct old kinematics takeover, this layer bypassed
// - ESCAPE: brainOut.triggered==true triggers, lasts 300ms:
//     Throttle = HOVER*1.5 (step, simulating GF burst)
//     Target tilt = clamp(escapeDir projected to horizontal plane, ±35°) → desired roll/pitch
// - CRUISE: throttle=HOVER*1.02, desired attitude = slight random walk (new target heading every 2-4s, angular velocity ≤15°/s, use injected rng for determinism)
// - LAND: gradually reduce throttle to HOVER*0.8 until landing (pos.y<threshold) → throttle 0
// PID attitude loop: kp=0.045, ki=0.001, kd=0.028 (error=desired attitude - current attitude, output added to motors)
// Mixing matrix: [m0,m1,m2,m3] = HOVER*[throttle, +roll, -roll, ...] (X-type, document each term)
// Always export lastState (HUD displays state machine: cruise/escape/pilot/land)
```

Pilot takeover verification: On the frame any non-modifier key is pressed, the dynamics toggle steps aside and old kinematics takes over immediately (existing logic preserved as-is).

## Phase 3: Continuous Flight Behavior

world.html logic (brain stays online, runs SNN every decision cycle):
- Cruise (CRUISE): slow wandering + obstacle avoidance steering — sample nearest obstacle bearing from the loom field via LPLC2 input; when 0.35<loom<threshold, yaw away from obstacle (**does not trigger escape**, only steers); above threshold, proceed with escape as before.
- Escape (ESCAPE): existing SNN escape unchanged, only executed via the adapter layer (throttle step + tilt angle).
- Land (LAND): console button + battery decoration bar (100→0 over 10 minutes, auto-LAND at 0; pilot can take off again anytime).
- HUD additions: state machine text (cruise/escape/pilot/land) + 4 motor bars + attitude angles.

## Phase 4: Regression Self-Tests (run immediately after each phase change)

1. `F:\Node\node.exe web\smoke_test.js` 4/4 must remain passing
2. New `web\smoke_drone.js`: hover stability (5s drift<0.3), escape response (after trigger <300ms velocity component toward escape direction>0), pilot takeover (takes over within one frame), DronePhysics.selfTest() all PASS
3. After changing any web\*.js, run `F:\Node\node.exe web\evaluate.js`; results.json hash must equal the anchor value; any drift triggers rollback of that step
4. world.html inline script syntax check (node new Function)

## Phase 5: GitHub Packaging

1. `LICENSE` (MIT, Copyright (c) 2026 + username placeholder)
2. `.gitignore`: pylibs/, *.pt, v4/ckpt*, __pycache__, %TEMP% artifacts
3. `README.md` rewrite (bilingual: Chinese first, English summary after):
   - Title: FlyWire Connectome Brain → Drone Control
   - 30-second quick start: double-click web\world.html → select "Drone" → press "Auto Threat" to watch the fruit fly brain fly a plane dodging attacks
   - Architecture diagram (ASCII): FlyWire FAFB (1871 neurons) → LIF SNN → escape direction → DroneAdapter (PID + mixing) → 4 motors → 3D world
   - Metrics table: v3 numbers (trigger 0.799 / success rate 0.903 / direction 10.7° / latency 7.65ms / jitter 92% all items pass) + Group D physics metrics
   - Three honest boundaries (see task_plan.md)
   - Citations: Dorkenwald et al. Nature 634:124–138 (2024); correction "Google open source = Neuroglancer viewer, data = FlyWire"; snedea/flybrain link
   - Reproduction: extract_circuit.py → train_snn.py → make_web_data.py → evaluate.js commands + two hash anchors
   - Demo GIF placeholder: `docs/demo.gif` (to be replaced after Phase 6 output)
4. `docs/` index: full-fly-design.md, this spec, v4\REPORT_v4.md excerpt
5. GitHub Pages deployment section (Settings→Pages→main→/web)

## Phase 6: Demo GIF (optional)

Fixed-seed automated script captures 6 key frames (cruise → threat enters → brain fires ⚡ → escape → safe → brain panel close-up) stitched into GIF; if not possible, provide the user with a manual recording script (30-second storyboard).

## Reporting Discipline (all agents)
Each phase ≤10 lines: which files changed, self-test results, one-line hash verification, next step. No lengthy tables allowed.