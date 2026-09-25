/* evaluate.js — Deterministic Monte Carlo Experiment Matrix Evaluator (Groups A/B/C/D)
 * ============================================================================
 * Usage: node web/evaluate.js
 * Output: web/results.json (mean/std details and sample sizes for all cells) + console Markdown table
 *
 * Design notes (strictly following task specifications):
 *   1) Math.random is completely disabled — self-implemented mulberry32 seeded PRNG;
 *      scene sampling and spike noise each use an independent random stream (both derived
 *      from the same seed), so all cells share the same seed set forming "common random
 *      numbers" paired comparisons, and each cell is reproducible.
 *   2) snn_runtime.encode() originally called Math.random internally; this file has
 *      patched the "only instance" to: (this.rng || Math.random)(). Default behavior
 *      is completely unchanged; this evaluator injects a seeded noise source via
 *      rt.rng = fn to achieve determinism.
 *   3) Sensor physics, label criteria, and metric definitions are identical to
 *      train_snn.py (see function annotations below).
 *
 * Experiment matrix (each cell = 5 different seeds × 200 samples, reporting
 *   5-seed mean ± sample standard deviation):
 *   Group A (modality control × meadow)        : fusion / vision_only / wind_only
 *   Group B (environment × fusion)             : meadow / storm / night
 *   Group C (attack speed × fusion × meadow)   : slow[0.3,3] / mid[3,7] / high[7,12] m/s
 *   Group D (carrier swap × fusion × meadow × mid) : fly / drone (2D planar integration for physical collision avoidance)
 */
'use strict';

const fs = require('fs');
const path = require('path');

// -------------------- Load exported model data and inference engine --------------------
global.window = {};
require('./snn_data.js');
const SNNRuntime = require('./snn_runtime.js');
const DATA = global.window.SNN_DATA;
if (DATA.meta && DATA.meta.dir_flipped) throw new Error('Model direction readout has flip flag (--flip-dir export), direction criteria for this evaluation requires manual confirmation');

// -------------------- Constants and presets (task-specified, not invented) --------------------
const SEEDS = [20240521, 20240522, 20240523, 20240524, 20240525]; // 5 different seeds per cell
const SAMPLES_PER_SEED = 200;          // 200 samples per seed
const TAU_TTC_MS = 50.0;               // Label: time-to-collision < 50 ms
const R_TRIGGER_CM = 25.0;             // Label: and distance < 25 cm
const SUCC_ANGLE_DEG = 30.0;           // Escape success: direction error < 30°
const FLY_RADIUS_CM = 0.55;            // Fly radius (used in carrier-swap collision detection)
const DT_MS = 1.0;                     // Carrier-swap integration step
const T_MAX_MS = 3000;                 // Carrier-swap integration time limit (sufficient for one encounter)

// Environment presets: visGain applies to visual expansion rate; ambientWind added to wind vector (cm/ms)
const ENVS = {
  meadow: { visGain: 1.00, ambientWind: [0.000, 0, 0.000] },
  storm:  { visGain: 0.85, ambientWind: [0.020, 0, 0.012] },
  night:  { visGain: 0.30, ambientWind: [0.000, 0, 0.000] },
};

// Attack speed sub-intervals (m/s)
const SPEED_RANGES = {
  full: [0.3, 12.0],
  slow: [0.3, 3.0],
  mid:  [3.0, 7.0],
  high: [7.0, 12.0],
};

// Carrier dynamics (carrier-swap experiment D): acceleration cm/ms², max speed cm/ms
const BODIES = {
  fly:  { accel: 0.0022, vmax: 0.16 },
  drone:{ accel: 0.0009, vmax: 0.10 },
};

// -------------------- Deterministic random number: mulberry32 --------------------
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Derive sub-stream seeds from master seed (scene stream uses seed directly, spike noise stream uses derived value to avoid correlation between the two streams)
function deriveSeed(seed, tag) {
  let h = (seed ^ 0x9e3779b9) >>> 0;
  for (let i = 0; i < tag.length; i++) h = (Math.imul(h ^ tag.charCodeAt(i), 0x85ebca6b) >>> 0);
  return h >>> 0;
}

function uniform(rng, lo, hi) { return lo + (hi - lo) * rng(); }

// -------------------- Small vector utilities --------------------
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function norm3(a) { return Math.hypot(a[0], a[1], a[2]); }

// Direction error (degrees) = acos(dot(escapeDir, esc)), with defensive normalization and clamping on input
function angleDeg(dir, esc) {
  const n = norm3(dir);
  if (!(n > 1e-8)) return NaN;
  const c = Math.max(-1, Math.min(1, (dir[0] * esc[0] + dir[1] * esc[1] + dir[2] * esc[2]) / n));
  return (Math.acos(c) * 180) / Math.PI;
}

// -------------------- Scene sampling (same range and sampling order as training-side sample_threat) --------------------
function sampleThreat(rng, speedRange) {
  const d0 = uniform(rng, 3.0, 35.0);            // Current distance (cm)
  const az = uniform(rng, -Math.PI, Math.PI);    // Azimuth angle (threat relative to fly heading +x)
  const el = uniform(rng, -0.6, 0.6);            // Elevation angle
  const speed = uniform(rng, speedRange[0], speedRange[1]); // m/s (Group C/D use sub-intervals)
  const miss = uniform(rng, 0.0, 8.0);           // Miss distance (cm)
  const sSize = uniform(rng, 0.3, 3.0);          // Threat object radius (cm)

  // Line-of-sight direction (unit vector from threat relative to fly position)
  const rhat = [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];
  // Horizontal tangent (consistent with training side)
  let tangent = [-rhat[1], rhat[0], 0.0];
  const tn = norm3(tangent);
  tangent = tn > 1e-9 ? [tangent[0] / tn, tangent[1] / tn, 0.0] : [0, 1, 0];

  // Velocity direction = approaching direction mixed with tangent by miss/d, then normalized
  const k = miss / (d0 + 1e-9);
  const mix = Math.sqrt(Math.max(0.0, 1 - k * k));
  let vdir = [-rhat[0] * mix + tangent[0] * k,
              -rhat[1] * mix + tangent[1] * k,
              -rhat[2] * mix + tangent[2] * k];
  const vn = norm3(vdir);
  vdir = vn > 1e-9 ? [vdir[0] / vn, vdir[1] / vn, vdir[2] / vn] : [-rhat[0], -rhat[1], -rhat[2]];

  const v = [vdir[0] * speed * 0.1, vdir[1] * speed * 0.1, vdir[2] * speed * 0.1]; // m/s -> cm/ms (÷10)

  return { d: d0, az, el, speed, miss, s: sSize, rhat, tangent, v, rvec: [rhat[0] * d0, rhat[1] * d0, rhat[2] * d0] };
}

// -------------------- Physical cues and labels (same criteria as training-side cues_and_labels) --------------------
function cuesAndLabels(st, env) {
  const r = st.d, s = st.s;
  const rdot = dot3(st.v, st.rhat);                    // cm/ms, negative during approach
  // Visual expansion rate: dθ/dt = -2 s r' / (r² + s²) (rad/s), multiplied by environment visual gain
  const looming = Math.max(0.0, (-2.0 * s * rdot) / (r * r + s * s)) * 1000.0 * env.visGain;
  // Wind pressure (sphere potential flow approximation): u = s²|v|/r², direction along threat velocity; plus ambient wind
  const vm = norm3(st.v);
  const u = (s * s) * vm / (r * r + 1e-9);
  const wind = [u * (st.v[0] / (vm + 1e-9)) + env.ambientWind[0],
                u * (st.v[1] / (vm + 1e-9)) + env.ambientWind[1],
                u * (st.v[2] / (vm + 1e-9)) + env.ambientWind[2]];
  // Label: effective threat ⇔ rdot<0 and ttc=r/(-rdot)<50ms and r<25cm
  const ttcMs = rdot >= 0 ? Infinity : r / -rdot;
  const yTrig = rdot < 0 && ttcMs < TAU_TTC_MS && r < R_TRIGGER_CM ? 1 : 0;
  // Escape direction ground truth = away from threat
  const esc = [-st.rhat[0], -st.rhat[1], -st.rhat[2]];
  return { looming, wind, az: st.az, yTrig, ttcMs, esc };
}

// -------------------- Carrier-swap experiment: 2D planar integration (Group D) --------------------
/* The plane is spanned by rhat and its horizontal tangent (threat position, velocity, and escape
 * direction all lie within it). The fly is fixed at the origin. The threat flies at constant
 * velocity along its original direction; the carrier accelerates along escapeDir after GF
 * latency L(ms) (limited by accel / vmax), integrated with dt=1ms, with full quadratic
 * closest-point sweep at each step:
 * "Any step sweeps closest point < sum of radii (collision)" or "Full sweep completes
 * with no collision (safe)". latencyMs == null means GF did not fire (carrier stays
 * still, used as baseline). */
function planarEscapeOutcome(st, latencyMs, escapeDir3, body) {
  const e1 = st.rhat, e2 = st.tangent;               // Plane basis (3D unit vectors, mutually orthogonal)
  const pT = [st.d, 0.0];                            // Threat initial position (e1, e2)
  const vT = [dot3(st.v, e1), dot3(st.v, e2)];       // Threat constant velocity cm/ms
  // Escape direction projected onto plane and normalized (carrier can only maneuver within plane)
  let eD = [dot3(escapeDir3, e1), dot3(escapeDir3, e2)];
  const eDn = Math.hypot(eD[0], eD[1]);
  eD = eDn < 1e-6 ? [-1.0, 0.0] : [eD[0] / eDn, eD[1] / eDn];

  const rSum = st.s + FLY_RADIUS_CM;                 // Sum of radii (threat radius + fly radius 0.55cm)
  let pE = [0.0, 0.0], vE = [0.0, 0.0];              // Carrier position / velocity
  let minDist = st.d;
  if (minDist < rSum) return { safe: false, minDist, reason: 'start_overlap' };

  for (let t = 1; t <= T_MAX_MS; t++) {
    if (latencyMs !== null && t >= latencyMs) {      // After latency period, accelerate along escapeDir
      vE[0] += body.accel * eD[0] * DT_MS;
      vE[1] += body.accel * eD[1] * DT_MS;
      const sp = Math.hypot(vE[0], vE[1]);
      if (sp > body.vmax) { vE[0] *= body.vmax / sp; vE[1] *= body.vmax / sp; }
    }
    const r0x = pT[0] - pE[0], r0y = pT[1] - pE[1];  // Relative position at step start
    pE[0] += vE[0] * DT_MS; pE[1] += vE[1] * DT_MS;
    pT[0] += vT[0] * DT_MS; pT[1] += vT[1] * DT_MS;
    // In-step sweep distance check: quadratic closest point from current relative segment to origin (including endpoints; prevents 1ms-step valley penetration)
    const r1x = pT[0] - pE[0], r1y = pT[1] - pE[1];
    const sx = r1x - r0x, sy = r1y - r0y, s2 = sx * sx + sy * sy;
    let tc = s2 > 0 ? -(r0x * sx + r0y * sy) / s2 : 0; tc = tc < 0 ? 0 : (tc > 1 ? 1 : tc);
    const sweptMin = Math.hypot(r0x + sx * tc, r0y + sy * tc);
    if (sweptMin < rSum) return { safe: false, minDist: Math.min(minDist, sweptMin), reason: 'collision' };
    // [v5 P5①] Complete quadratic closest-point evaluation: take the minimum closest distance across all steps, do not use "local distance increase" to declare safety early
    //   (relative trajectory under maneuver acceleration is piecewise quadratic; distance² can form a double valley; returning when the first valley rises would miss the second approach → false safety report)
    if (sweptMin < minDist) minDist = sweptMin;
  }
  return { safe: minDist > rSum, minDist, reason: 'escaped' };   // Full sweep completes with no collision = truly safe
}

// -------------------- Statistics: 5 seeds mean ± sample standard deviation (ddof=1) --------------------
function aggregate(values) {
  const xs = values.filter((x) => Number.isFinite(x));
  if (xs.length === 0) return { mean: NaN, std: NaN, n_defined: 0 };
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  let ss = 0;
  for (const x of xs) ss += (x - mean) * (x - mean);
  const std = xs.length > 1 ? Math.sqrt(ss / (xs.length - 1)) : 0.0;
  return { mean, std, n_defined: xs.length };
}

// -------------------- Single cell evaluation: 5 seeds × 200 samples --------------------
function evalCell(cell) {
  const rt = new SNNRuntime(DATA);
  rt.mode = cell.mode;                               // 'fusion' | 'vision_only' | 'wind_only'
  const env = ENVS[cell.env];
  const speedRange = SPEED_RANGES[cell.speed];
  const body = cell.body ? BODIES[cell.body] : null;

  const perSeed = [];
  for (const seed of SEEDS) {
    const scenRng = mulberry32(seed);                // Scene sampling stream
    rt.rng = mulberry32(deriveSeed(seed, 'spike'));  // Spike noise stream (injected via snn_runtime patch)

    let n = 0, nPos = 0, nTrig = 0, correct = 0, hits = 0;
    let tp = 0, fp = 0, fn = 0, tn = 0;             // Diagnostic confusion matrix counts (does not change criteria)
    const dirErrs = [], lats = [];
    let physSafe = 0, baseSafe = 0, actionSaved = 0; // Group D: carrier-swap integration statistics

    for (let i = 0; i < SAMPLES_PER_SEED; i++) {
      const st = sampleThreat(scenRng, speedRange);
      const cue = cuesAndLabels(st, env);
      const out = rt.simulate(cue.looming, cue.wind, cue.az);
      n++;
      const pred = out.triggered;                    // GF fires within 12ms window
      const isPos = cue.yTrig === 1;
      if (pred === isPos) correct++;
      if (pred) nTrig++;
      if (pred && isPos) tp++; else if (pred) fp++; else if (isPos) fn++; else tn++;

      if (cue.yTrig === 1) {
        nPos++;
        const dirErr = angleDeg(out.escapeDir, cue.esc);
        if (Number.isFinite(dirErr)) dirErrs.push(dirErr);
        if (pred) {
          if (out.firstSpikeMs !== null) lats.push(out.firstSpikeMs);
          if (Number.isFinite(dirErr) && dirErr < SUCC_ANGLE_DEG) hits++;
        }
        if (body) {
          // Carrier-swap integration: if GF fires, start escape maneuver after latency L; if not, stay still
          const L = pred ? out.firstSpikeMs : null;
          const withAction = planarEscapeOutcome(st, L, out.escapeDir, body);
          const noAction = planarEscapeOutcome(st, null, out.escapeDir, body); // Stationary baseline (natural evasion)
          if (withAction.safe) physSafe++;
          if (noAction.safe) baseSafe++;
          if (withAction.safe && !noAction.safe) actionSaved++;
        }
      }
    }

    perSeed.push({
      seed,
      n_samples: n,
      n_threat: nPos,
      n_triggered: nTrig,
      trigger_acc: correct / Math.max(1, n),
      threat_recall: nPos > 0 ? tp / nPos : NaN,          // Diagnostic: P(GF fires | yTrig=1)
      false_alarm_rate: n - nPos > 0 ? fp / (n - nPos) : NaN, // Diagnostic: P(GF fires | yTrig=0)
      n_tp: tp, n_fp: fp, n_fn: fn, n_tn: tn,
      escape_success_rate: nPos > 0 ? hits / nPos : NaN,
      dir_mae_deg: dirErrs.length ? dirErrs.reduce((a, b) => a + b, 0) / dirErrs.length : NaN,
      gf_first_spike_ms: lats.length ? lats.reduce((a, b) => a + b, 0) / lats.length : NaN,
      ...(body ? {
        physical_success_rate: nPos > 0 ? physSafe / nPos : NaN,
        stationary_safe_rate: nPos > 0 ? baseSafe / nPos : NaN,
        action_saved_rate: nPos > 0 ? actionSaved / nPos : NaN,
      } : {}),
    });
  }

  // Cross-seed aggregation
  const metrics = {
    trigger_acc: aggregate(perSeed.map((r) => r.trigger_acc)),
    threat_recall: aggregate(perSeed.map((r) => r.threat_recall)),
    false_alarm_rate: aggregate(perSeed.map((r) => r.false_alarm_rate)),
    escape_success_rate: aggregate(perSeed.map((r) => r.escape_success_rate)),
    dir_mae_deg: aggregate(perSeed.map((r) => r.dir_mae_deg)),
    gf_first_spike_ms: aggregate(perSeed.map((r) => r.gf_first_spike_ms)),
    ...(body ? {
      physical_success_rate: aggregate(perSeed.map((r) => r.physical_success_rate)),
      stationary_safe_rate: aggregate(perSeed.map((r) => r.stationary_safe_rate)),
      action_saved_rate: aggregate(perSeed.map((r) => r.action_saved_rate)),
    } : {}),
  };

  return {
    id: cell.id,
    label: cell.label,
    group: cell.group,
    config: { mode: cell.mode, env: cell.env, speed_range: cell.speed, speed_range_ms: SPEED_RANGES[cell.speed], body: cell.body },
    n_seeds: SEEDS.length,
    samples_per_seed: SAMPLES_PER_SEED,
    n_samples: SEEDS.length * SAMPLES_PER_SEED,
    n_threat_total: perSeed.reduce((a, r) => a + r.n_threat, 0),
    metrics,
    per_seed: perSeed,
  };
}

// -------------------- Experiment matrix --------------------
const CELLS = [
  // Group A: modality control × meadow
  { group: 'A', id: 'A1', label: 'fusion',       mode: 'fusion',      env: 'meadow', speed: 'full', body: null },
  { group: 'A', id: 'A2', label: 'vision_only',  mode: 'vision_only', env: 'meadow', speed: 'full', body: null },
  { group: 'A', id: 'A3', label: 'wind_only',    mode: 'wind_only',   env: 'meadow', speed: 'full', body: null },
  // Group B: environment × fusion
  { group: 'B', id: 'B1', label: 'meadow',       mode: 'fusion',      env: 'meadow', speed: 'full', body: null },
  { group: 'B', id: 'B2', label: 'storm',        mode: 'fusion',      env: 'storm',  speed: 'full', body: null },
  { group: 'B', id: 'B3', label: 'night',        mode: 'fusion',      env: 'night',  speed: 'full', body: null },
  // Group C: attack speed × fusion × meadow
  { group: 'C', id: 'C1', label: 'slow',         mode: 'fusion',      env: 'meadow', speed: 'slow', body: null },
  { group: 'C', id: 'C2', label: 'mid',          mode: 'fusion',      env: 'meadow', speed: 'mid',  body: null },
  { group: 'C', id: 'C3', label: 'high',         mode: 'fusion',      env: 'meadow', speed: 'high', body: null },
  // Group D: carrier swap × fusion × meadow × mid speed
  { group: 'D', id: 'D1', label: 'fly',          mode: 'fusion',      env: 'meadow', speed: 'mid',  body: 'fly' },
  { group: 'D', id: 'D2', label: 'drone',        mode: 'fusion',      env: 'meadow', speed: 'mid',  body: 'drone' },
];

const GROUP_TITLES = {
  A: 'Group A: Modality Control × Meadow (full speed 0.3–12 m/s)',
  B: 'Group B: Environment × Fusion (full speed 0.3–12 m/s)',
  C: 'Group C: Attack Speed × Fusion × Meadow',
  D: 'Group D: Carrier Swap × Fusion × Meadow × Mid (3–7 m/s)',
};

// -------------------- Output helpers --------------------
function fmtMS(v, digits) {
  if (!Number.isFinite(v)) return 'n/a';
  return v.toFixed(digits);
}
function fmtPair(m, digits) {
  return `${fmtMS(m.mean, digits)}±${fmtMS(m.std, digits)}`;
}
// JSON sanitization: NaN/Infinity -> null
function sanitize(x) {
  if (typeof x === 'number') return Number.isFinite(x) ? x : null;
  if (Array.isArray(x)) return x.map(sanitize);
  if (x && typeof x === 'object') {
    const o = {};
    for (const k of Object.keys(x)) o[k] = sanitize(x[k]);
    return o;
  }
  return x;
}

function printGroupTable(group, results) {
  const cells = results.filter((r) => r.group === group);
  console.log(`\n### ${GROUP_TITLES[group]}`);
  console.log('');
  console.log('| Group | Trigger Accuracy | Escape Success Rate | Direction Error(°) | GF Latency(ms) |');
  console.log('| --- | --- | --- | --- | --- |');
  for (const c of cells) {
    // Group D "escape success rate" is determined by carrier-swap integration (safe/sum of radii); other groups use "triggered and direction error<30°"
    const succKey = group === 'D' ? 'physical_success_rate' : 'escape_success_rate';
    console.log(`| ${c.label} | ${fmtPair(c.metrics.trigger_acc, 3)} | ${fmtPair(c.metrics[succKey], 3)} | ` +
                `${fmtPair(c.metrics.dir_mae_deg, 1)} | ${fmtPair(c.metrics.gf_first_spike_ms, 2)} |`);
  }
}

// -------------------- Main flow --------------------
function main() {
  const t0 = Date.now();
  console.log('Deterministic Monte Carlo Experiment Matrix Evaluator — Drosophila Escape SNN');
  console.log(`Model: ${DATA.num_nodes} nodes / ${DATA.edges.length} edges / T=${DATA.meta.t_steps}ms | ` +
              `Per cell: ${SEEDS.length} seeds × ${SAMPLES_PER_SEED} samples = ${SEEDS.length * SAMPLES_PER_SEED}`);
  console.log(`Seeds: ${SEEDS.join(', ')} (shared across cells, forming common random number paired comparisons) | PRNG: mulberry32 (injected into snn_runtime.encode)`);

  const byId = {};
  const results = [];
  for (const cell of CELLS) {
    const r = evalCell(cell);
    byId[cell.id] = r;
    results.push(r);
    console.log(`  [${cell.id}] ${cell.label} completed (valid threats ${r.n_threat_total}/${r.n_samples})... ${(Date.now() - t0) / 1000 | 0}s`);
  }

  // -------------------- Consistency / determinism checks --------------------
  const neuralKeys = ['trigger_acc', 'escape_success_rate', 'dir_mae_deg', 'gf_first_spike_ms'];
  const neuralSig = (r) => JSON.stringify(neuralKeys.map((k) => r.per_seed.map((s) => s[k])));
  const checks = {
    // A1(fusion×meadow×full speed) and B1(meadow×fusion×full speed) share config + shared seeds → should be bitwise identical
    A1_equals_B1: neuralSig(byId.A1) === neuralSig(byId.B1),
    // C2(mid) and D1(fly) have same neural config and seeds → should be bitwise identical (D only adds carrier-swap integration)
    C2_equals_D1_neural: neuralSig(byId.C2) === neuralSig(byId.D1),
    D1_equals_D2_neural: neuralSig(byId.D1) === neuralSig(byId.D2), // Carrier swap does not change SNN forward pass
  };

  // -------------------- Console Markdown table --------------------
  console.log('\n================ Experiment Matrix Results (5-seed mean ± sample std) ================');
  for (const g of ['A', 'B', 'C', 'D']) printGroupTable(g, results);
  console.log('\nNotes:');
  console.log('  * Trigger accuracy = proportion where pred(GF fires) == yTrig; GF latency is only computed for samples with effective threats (yTrig=1) where GF fired.');
  console.log('  * Groups A/B/C escape success rate = proportion of yTrig=1 samples that are "triggered and direction error<30°"; direction error is averaged over yTrig=1 samples.');
  console.log('  * Group D escape success rate = physical collision avoidance success rate determined by 2D planar integration (accelerates along escapeDir after latency, safe = closest distance > sum of radii);');
  console.log('    As a reference, Group D success rate by direction criteria (triggered and direction error<30°): ' +
              `fly ${fmtPair(byId.D1.metrics.escape_success_rate, 3)} / drone ${fmtPair(byId.D2.metrics.escape_success_rate, 3)};`);
  console.log('    Stationary baseline (no maneuver, pure natural evasion) safety rate: ' +
              `fly ${fmtPair(byId.D1.metrics.stationary_safe_rate, 3)} / drone ${fmtPair(byId.D2.metrics.stationary_safe_rate, 3)};` +
              'Proportion truly saved by maneuver: ' +
              `fly ${fmtPair(byId.D1.metrics.action_saved_rate, 3)} / drone ${fmtPair(byId.D2.metrics.action_saved_rate, 3)}.`);
  console.log('  * Determinism checks: A1==B1 bitwise identical=' + checks.A1_equals_B1 +
              ', C2==D1(neural metrics)=' + checks.C2_equals_D1_neural +
              ', D1==D2(neural metrics)=' + checks.D1_equals_D2_neural);
  console.log('  * Trigger behavior decomposition (diagnostic, not acceptance metric) — threat recall P(GF fires|yTrig=1) / no-threat false alarm rate P(GF fires|yTrig=0):');
  for (const c of results) {
    console.log(`      [${c.id}] ${c.label}: ${fmtPair(c.metrics.threat_recall, 3)} / ${fmtPair(c.metrics.false_alarm_rate, 3)}` +
                ` | per-seed GF firing counts ${c.per_seed.map((s) => s.n_triggered).join('/')}`);
  }

  // -------------------- Write results.json --------------------
  const payload = {
    meta: {
      generated_by: 'web/evaluate.js',
      deterministic: true,
      prng: 'mulberry32 (seeded, Math.random disabled)',
      rng_injection: "web/snn_runtime.js encode() Math.random -> (this.rng || Math.random)(), default behavior unchanged, injected via rt.rng",
      seeds: SEEDS,
      samples_per_seed: SAMPLES_PER_SEED,
      std_definition: 'Sample standard deviation (ddof=1), computed across the 5 seeds per cell',
      model: { num_nodes: DATA.num_nodes, n_edges: DATA.edges.length, t_steps: DATA.meta.t_steps, dt_ms: DATA.meta.dt_ms },
      label_rule: 'yTrig=1 ⇔ rdot<0 and ttc=r/(-rdot)<50ms and r<25cm; esc = -rhat',
      metrics_def: {
        trigger_acc: 'Proportion where pred(triggered) == yTrig (triggered = GF fires within 12ms window)',
        threat_recall: '(diagnostic) P(GF fires | yTrig=1)',
        false_alarm_rate: '(diagnostic) P(GF fires | yTrig=0)',
        escape_success_rate: 'Proportion of yTrig=1 samples where "triggered and direction error<30°" (Groups A/B/C escape success rate)',
        dir_mae_deg: 'Mean of acos(dot(escapeDir, esc)) angle (averaged over yTrig=1 samples)',
        gf_first_spike_ms: 'Mean of firstSpikeMs (only computed for triggered samples with effective threats yTrig=1)',
        physical_success_rate: '(Group D escape success rate) Proportion determined safe by 2D planar integration (yTrig=1 samples)',
        stationary_safe_rate: '(Group D baseline) Proportion safe (natural evasion) when carrier stays still',
        action_saved_rate: '(Group D) Proportion safe with maneuver but collided in stationary baseline (truly saved by maneuver)',
      },
      envs: ENVS,
      speed_ranges_ms: SPEED_RANGES,
      bodies: { fly: BODIES.fly, drone: BODIES.drone, fly_radius_cm: FLY_RADIUS_CM },
      scene_sampling: 'r∈[3,35]cm, az∈[-π,π], el∈[-0.6,0.6], speed∈speed sub-interval(m/s), miss∈[0,8]cm, s∈[0.3,3]cm; same range as train_snn.sample_threat',
      physics: 'looming=max(0,-2*s*rdot/(r²+s²))*1000*visGain (rad/s); u=s²|v|/r², wind=u*v/|v|+ambientWind (cm/ms); speed m/s÷10→cm/ms',
      carrier_dynamics: 'Carrier accelerates along escapeDir after GF latency L (limited by accel/vmax), threat at constant speed, dt=1ms 2D planar integration until closest distance > sum of radii (safe) or distance < sum of radii (collision)',
    },
    checks,
    groups: {},
    group_titles: GROUP_TITLES,
  };
  for (const g of ['A', 'B', 'C', 'D']) {
    payload.groups[g] = results.filter((r) => r.group === g);
  }
  const outPath = path.join(__dirname, 'results.json');
  fs.writeFileSync(outPath, JSON.stringify(sanitize(payload), null, 2), 'utf8');
  console.log(`\nWritten: ${outPath} (elapsed ${((Date.now() - t0) / 1000).toFixed(1)}s)`);
}

main();