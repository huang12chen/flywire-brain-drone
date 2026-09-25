/* evaluate_v4.js — （A/B/C/D ）· v4  A 
 * ============================================================================
 *  web\evaluate.js（v3 ，）；：
 *   1)  -> v4\snn_data_v4.js（ --data ， v4/snn_data_v4_seed20240522.js）；
 *   2)  web\snn_runtime.js（，）；
 *   3)  -> v4\results_v4.json（ --out ）。 web\results.json。
 * ： /  /  /  /  /  v3 ，。
 *
 * ：node v4/evaluate_v4.js [--data <snn_data_v4*.js>] [--out <results_v4*.json>]（）
 *
 * （）：
 *   1)  Math.random ——  mulberry32  PRNG；
 *      （ seed ），
 *      「」，。
 *   2) snn_runtime.encode() 「」：(this.rng || Math.random)()。
 *      ； rt.rng = fn ，。
 *   3) 、、 train_snn_v4.py （= v3 ）。
 *
 * （ = 5  × 200 ， 5 ±）：
 *   A （ × meadow）        ：fusion / vision_only / wind_only
 *   B （ × fusion）            ：meadow / storm / night
 *   C （ × fusion × meadow）：slow[0.3,3] / mid[3,7] / high[7,12] m/s
 *   D （ × fusion × meadow × mid）：fly / drone（）
 */
'use strict';

const fs = require('fs');
const path = require('path');

// -------------------- （v4 ） --------------------
function argVal(name, dflt) {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : dflt;
}
const DATA_PATH = path.resolve(process.cwd(), argVal('--data', path.join(__dirname, 'snn_data_v4.js')));
const OUT_PATH = path.resolve(process.cwd(), argVal('--out', path.join(__dirname, 'results_v4.json')));
if (fs.existsSync(path.join(__dirname, '..', 'web', 'results.json')) &&
    OUT_PATH === path.resolve(path.join(__dirname, '..', 'web', 'results.json'))) {
  throw new Error(' web/results.json（v3 ）');
}

// --------------------  --------------------
global.window = {};
require(DATA_PATH);
const SNNRuntime = require(path.join(__dirname, '..', 'web', 'snn_runtime.js'));
const DATA = global.window.SNN_DATA;
if (DATA.meta && DATA.meta.dir_flipped) throw new Error('（--flip-dir ），');

// -------------------- （，； v3 evaluate.js ） --------------------
const SEEDS = [20240521, 20240522, 20240523, 20240524, 20240525]; //  5 
const SAMPLES_PER_SEED = 200;          //  200 
const TAU_TTC_MS = 50.0;               // ： < 50 ms
const R_TRIGGER_CM = 25.0;             // ： < 25 cm
const SUCC_ANGLE_DEG = 30.0;           // ： < 30°
const FLY_RADIUS_CM = 0.55;            // （）
const DT_MS = 1.0;                     // 
const T_MAX_MS = 3000;                 // （）

// ：visGain ；ambientWind （cm/ms）
const ENVS = {
  meadow: { visGain: 1.00, ambientWind: [0.000, 0, 0.000] },
  storm:  { visGain: 0.85, ambientWind: [0.020, 0, 0.012] },
  night:  { visGain: 0.30, ambientWind: [0.000, 0, 0.000] },
};

// （m/s）
const SPEED_RANGES = {
  full: [0.3, 12.0],
  slow: [0.3, 3.0],
  mid:  [3.0, 7.0],
  high: [7.0, 12.0],
};

// （ D）： cm/ms²、 cm/ms
const BODIES = {
  fly:  { accel: 0.0022, vmax: 0.16 },
  drone:{ accel: 0.0009, vmax: 0.10 },
};

// -------------------- ：mulberry32 --------------------
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

// （ seed，，）
function deriveSeed(seed, tag) {
  let h = (seed ^ 0x9e3779b9) >>> 0;
  for (let i = 0; i < tag.length; i++) h = (Math.imul(h ^ tag.charCodeAt(i), 0x85ebca6b) >>> 0);
  return h >>> 0;
}

function uniform(rng, lo, hi) { return lo + (hi - lo) * rng(); }

// --------------------  --------------------
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function norm3(a) { return Math.hypot(a[0], a[1], a[2]); }

// （）= acos(dot(escapeDir, esc))，
function angleDeg(dir, esc) {
  const n = norm3(dir);
  if (!(n > 1e-8)) return NaN;
  const c = Math.max(-1, Math.min(1, (dir[0] * esc[0] + dir[1] * esc[1] + dir[2] * esc[2]) / n));
  return (Math.acos(c) * 180) / Math.PI;
}

// -------------------- （ sample_threat 、） --------------------
function sampleThreat(rng, speedRange) {
  const d0 = uniform(rng, 3.0, 35.0);            //  cm
  const az = uniform(rng, -Math.PI, Math.PI);    // （ +x）
  const el = uniform(rng, -0.6, 0.6);            // 
  const speed = uniform(rng, speedRange[0], speedRange[1]); // m/s（C/D ）
  const miss = uniform(rng, 0.0, 8.0);           //  cm
  const sSize = uniform(rng, 0.3, 3.0);          //  cm

  // （）
  const rhat = [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];
  // （）
  let tangent = [-rhat[1], rhat[0], 0.0];
  const tn = norm3(tangent);
  tangent = tn > 1e-9 ? [tangent[0] / tn, tangent[1] / tn, 0.0] : [0, 1, 0];

  //  =  miss/d 
  const k = miss / (d0 + 1e-9);
  const mix = Math.sqrt(Math.max(0.0, 1 - k * k));
  let vdir = [-rhat[0] * mix + tangent[0] * k,
              -rhat[1] * mix + tangent[1] * k,
              -rhat[2] * mix + tangent[2] * k];
  const vn = norm3(vdir);
  vdir = vn > 1e-9 ? [vdir[0] / vn, vdir[1] / vn, vdir[2] / vn] : [-rhat[0], -rhat[1], -rhat[2]];

  const v = [vdir[0] * speed * 0.1, vdir[1] * speed * 0.1, vdir[2] * speed * 0.1]; // m/s -> cm/ms（÷10）

  return { d: d0, az, el, speed, miss, s: sSize, rhat, tangent, v, rvec: [rhat[0] * d0, rhat[1] * d0, rhat[2] * d0] };
}

// -------------------- （ cues_and_labels ） --------------------
function cuesAndLabels(st, env) {
  const r = st.d, s = st.s;
  const rdot = dot3(st.v, st.rhat);                    // cm/ms，
  // ：dθ/dt = -2 s r' / (r² + s²)（rad/s），
  const looming = Math.max(0.0, (-2.0 * s * rdot) / (r * r + s * s)) * 1000.0 * env.visGain;
  // （）：u = s²|v|/r²，；
  const vm = norm3(st.v);
  const u = (s * s) * vm / (r * r + 1e-9);
  const wind = [u * (st.v[0] / (vm + 1e-9)) + env.ambientWind[0],
                u * (st.v[1] / (vm + 1e-9)) + env.ambientWind[1],
                u * (st.v[2] / (vm + 1e-9)) + env.ambientWind[2]];
  // ： ⇔ rdot<0  ttc=r/(-rdot)<50ms  r<25cm
  const ttcMs = rdot >= 0 ? Infinity : r / -rdot;
  const yTrig = rdot < 0 && ttcMs < TAU_TTC_MS && r < R_TRIGGER_CM ? 1 : 0;
  //  = 
  const esc = [-st.rhat[0], -st.rhat[1], -st.rhat[2]];
  return { looming, wind, az: st.az, yTrig, ttcMs, esc };
}

// -------------------- ：（D ） --------------------
/*  rhat （、、），
 * 。； GF  L(ms)  escapeDir 
 * （ accel / vmax ），dt=1ms ，「 > （）」
 * 「 < （）」。latencyMs == null  GF （，）。 */
function planarEscapeOutcome(st, latencyMs, escapeDir3, body) {
  const e1 = st.rhat, e2 = st.tangent;               // （3D ，）
  const pT = [st.d, 0.0];                            //  (e1, e2)
  const vT = [dot3(st.v, e1), dot3(st.v, e2)];       //  cm/ms
  // （）
  let eD = [dot3(escapeDir3, e1), dot3(escapeDir3, e2)];
  const eDn = Math.hypot(eD[0], eD[1]);
  eD = eDn < 1e-6 ? [-1.0, 0.0] : [eD[0] / eDn, eD[1] / eDn];

  const rSum = st.s + FLY_RADIUS_CM;                 // （ +  0.55cm）
  let pE = [0.0, 0.0], vE = [0.0, 0.0];              //  / 
  let minDist = st.d, prevDist = st.d;
  if (minDist < rSum) return { safe: false, minDist, reason: 'start_overlap' };

  for (let t = 1; t <= T_MAX_MS; t++) {
    if (latencyMs !== null && t >= latencyMs) {      //  escapeDir 
      vE[0] += body.accel * eD[0] * DT_MS;
      vE[1] += body.accel * eD[1] * DT_MS;
      const sp = Math.hypot(vE[0], vE[1]);
      if (sp > body.vmax) { vE[0] *= body.vmax / sp; vE[1] *= body.vmax / sp; }
    }
    const r0x = pT[0] - pE[0], r0y = pT[1] - pE[1];  // 
    pE[0] += vE[0] * DT_MS; pE[1] += vE[1] * DT_MS;
    pT[0] += vT[0] * DT_MS; pT[1] += vT[1] * DT_MS;
    // ：（ 1ms ）
    const r1x = pT[0] - pE[0], r1y = pT[1] - pE[1];
    const sx = r1x - r0x, sy = r1y - r0y, s2 = sx * sx + sy * sy;
    let tc = s2 > 0 ? -(r0x * sx + r0y * sy) / s2 : 0; tc = tc < 0 ? 0 : (tc > 1 ? 1 : tc);
    const sweptMin = Math.hypot(r0x + sx * tc, r0y + sy * tc);
    const dist = Math.hypot(r1x, r1y);
    if (sweptMin < rSum) return { safe: false, minDist: Math.min(minDist, sweptMin), reason: 'collision' };
    const newMin = Math.min(sweptMin, dist);
    if (newMin < minDist) minDist = newMin;
    if (dist > prevDist && minDist > rSum) return { safe: true, minDist, reason: 'escaped' }; // 
    prevDist = dist;
  }
  return { safe: minDist > rSum, minDist, reason: 'timeout' };
}

// -------------------- ：5  ± (ddof=1) --------------------
function aggregate(values) {
  const xs = values.filter((x) => Number.isFinite(x));
  if (xs.length === 0) return { mean: NaN, std: NaN, n_defined: 0 };
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  let ss = 0;
  for (const x of xs) ss += (x - mean) * (x - mean);
  const std = xs.length > 1 ? Math.sqrt(ss / (xs.length - 1)) : 0.0;
  return { mean, std, n_defined: xs.length };
}

// -------------------- ：5  × 200  --------------------
function evalCell(cell) {
  const rt = new SNNRuntime(DATA);
  rt.mode = cell.mode;                               // 'fusion' | 'vision_only' | 'wind_only'
  const env = ENVS[cell.env];
  const speedRange = SPEED_RANGES[cell.speed];
  const body = cell.body ? BODIES[cell.body] : null;

  const perSeed = [];
  for (const seed of SEEDS) {
    const scenRng = mulberry32(seed);                // 
    rt.rng = mulberry32(deriveSeed(seed, 'spike'));  // （ snn_runtime ）

    let n = 0, nPos = 0, nTrig = 0, correct = 0, hits = 0;
    let tp = 0, fp = 0, fn = 0, tn = 0;             // （）
    const dirErrs = [], lats = [];
    let physSafe = 0, baseSafe = 0, actionSaved = 0; // D ：

    for (let i = 0; i < SAMPLES_PER_SEED; i++) {
      const st = sampleThreat(scenRng, speedRange);
      const cue = cuesAndLabels(st, env);
      const out = rt.simulate(cue.looming, cue.wind, cue.az);
      n++;
      const pred = out.triggered;                    // GF  12ms 
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
          // ：GF  L ；
          const L = pred ? out.firstSpikeMs : null;
          const withAction = planarEscapeOutcome(st, L, out.escapeDir, body);
          const noAction = planarEscapeOutcome(st, null, out.escapeDir, body); // （）
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
      threat_recall: nPos > 0 ? tp / nPos : NaN,          // ：P(GF | yTrig=1)
      false_alarm_rate: n - nPos > 0 ? fp / (n - nPos) : NaN, // ：P(GF | yTrig=0)
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

  // 
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

// -------------------- （ v3 ） --------------------
const CELLS = [
  // A ： × meadow
  { group: 'A', id: 'A1', label: 'fusion',       mode: 'fusion',      env: 'meadow', speed: 'full', body: null },
  { group: 'A', id: 'A2', label: 'vision_only',  mode: 'vision_only', env: 'meadow', speed: 'full', body: null },
  { group: 'A', id: 'A3', label: 'wind_only',    mode: 'wind_only',   env: 'meadow', speed: 'full', body: null },
  // B ： × fusion
  { group: 'B', id: 'B1', label: 'meadow',       mode: 'fusion',      env: 'meadow', speed: 'full', body: null },
  { group: 'B', id: 'B2', label: 'storm',        mode: 'fusion',      env: 'storm',  speed: 'full', body: null },
  { group: 'B', id: 'B3', label: 'night',        mode: 'fusion',      env: 'night',  speed: 'full', body: null },
  // C ： × fusion × meadow
  { group: 'C', id: 'C1', label: 'slow',         mode: 'fusion',      env: 'meadow', speed: 'slow', body: null },
  { group: 'C', id: 'C2', label: 'mid',          mode: 'fusion',      env: 'meadow', speed: 'mid',  body: null },
  { group: 'C', id: 'C3', label: 'high',         mode: 'fusion',      env: 'meadow', speed: 'high', body: null },
  // D ： × fusion × meadow × mid 
  { group: 'D', id: 'D1', label: 'fly',          mode: 'fusion',      env: 'meadow', speed: 'mid',  body: 'fly' },
  { group: 'D', id: 'D2', label: 'drone',        mode: 'fusion',      env: 'meadow', speed: 'mid',  body: 'drone' },
];

const GROUP_TITLES = {
  A: 'A ： × meadow（ 0.3–12 m/s）',
  B: 'B ： × fusion（ 0.3–12 m/s）',
  C: 'C ： × fusion × meadow',
  D: 'D ： × fusion × meadow × mid（3–7 m/s）',
};

// --------------------  --------------------
function fmtMS(v, digits) {
  if (!Number.isFinite(v)) return 'n/a';
  return v.toFixed(digits);
}
function fmtPair(m, digits) {
  return `${fmtMS(m.mean, digits)}±${fmtMS(m.std, digits)}`;
}
// JSON ：NaN/Infinity -> null
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
  console.log('|  |  |  | (°) | GF(ms) |');
  console.log('| --- | --- | --- | --- | --- |');
  for (const c of cells) {
    // D 「」（/），「<30°」
    const succKey = group === 'D' ? 'physical_success_rate' : 'escape_success_rate';
    console.log(`| ${c.label} | ${fmtPair(c.metrics.trigger_acc, 3)} | ${fmtPair(c.metrics[succKey], 3)} | ` +
                `${fmtPair(c.metrics.dir_mae_deg, 1)} | ${fmtPair(c.metrics.gf_first_spike_ms, 2)} |`);
  }
}

// --------------------  --------------------
function main() {
  const t0 = Date.now();
  console.log(' ——  SNN（v4  A）');
  console.log(`：${DATA_PATH}`);
  console.log(`：${DATA.num_nodes}  / ${DATA.edges.length}  / T=${DATA.meta.t_steps}ms / ` +
              `train_seed=${DATA.meta.train_seed ?? 'n/a'} | ` +
              ` ${SEEDS.length}  × ${SAMPLES_PER_SEED}  = ${SEEDS.length * SAMPLES_PER_SEED}`);
  console.log(`：${SEEDS.join(', ')}（，）| PRNG: mulberry32（ snn_runtime.encode）`);

  const byId = {};
  const results = [];
  for (const cell of CELLS) {
    const r = evalCell(cell);
    byId[cell.id] = r;
    results.push(r);
    console.log(`  [${cell.id}] ${cell.label} （ ${r.n_threat_total}/${r.n_samples}）... ${(Date.now() - t0) / 1000 | 0}s`);
  }

  // --------------------  /  --------------------
  const neuralKeys = ['trigger_acc', 'escape_success_rate', 'dir_mae_deg', 'gf_first_spike_ms'];
  const neuralSig = (r) => JSON.stringify(neuralKeys.map((k) => r.per_seed.map((s) => s[k])));
  const checks = {
    // A1(fusion×meadow×)  B1(meadow×fusion×)  +  => 
    A1_equals_B1: neuralSig(byId.A1) === neuralSig(byId.B1),
    // C2(mid)  D1(fly)  => （D ）
    C2_equals_D1_neural: neuralSig(byId.C2) === neuralSig(byId.D1),
    D1_equals_D2_neural: neuralSig(byId.D1) === neuralSig(byId.D2), //  SNN 
  };

  // --------------------  Markdown  --------------------
  console.log('\n================ （5 ±） ================');
  for (const g of ['A', 'B', 'C', 'D']) printGroupTable(g, results);
  console.log('\n：');
  console.log('  *  = pred(GF) == yTrig ；GF (yTrig=1)  GF 。');
  console.log('  * A/B/C  = yTrig=1 「  <30°」； yTrig=1 。');
  console.log('  * D  = （ escapeDir ，=>）；');
  console.log('    ，D （<30°）：' +
              `fly ${fmtPair(byId.D1.metrics.escape_success_rate, 3)} / drone ${fmtPair(byId.D2.metrics.escape_success_rate, 3)}；`);
  console.log('    （、）：' +
              `fly ${fmtPair(byId.D1.metrics.stationary_safe_rate, 3)} / drone ${fmtPair(byId.D2.metrics.stationary_safe_rate, 3)}；` +
              '：' +
              `fly ${fmtPair(byId.D1.metrics.action_saved_rate, 3)} / drone ${fmtPair(byId.D2.metrics.action_saved_rate, 3)}。`);
  console.log('  * ：A1==B1 =' + checks.A1_equals_B1 +
              '，C2==D1()=' + checks.C2_equals_D1_neural +
              '，D1==D2()=' + checks.D1_equals_D2_neural);
  console.log('  * （，）——  P(GF|yTrig=1) /  P(GF|yTrig=0)：');
  for (const c of results) {
    console.log(`      [${c.id}] ${c.label}: ${fmtPair(c.metrics.threat_recall, 3)} / ${fmtPair(c.metrics.false_alarm_rate, 3)}` +
                ` |  GF  ${c.per_seed.map((s) => s.n_triggered).join('/')}`);
  }

  // --------------------  results_v4.json --------------------
  const payload = {
    meta: {
      generated_by: 'v4/evaluate_v4.js（ web/evaluate.js；，）',
      model_data: path.relative(process.cwd(), DATA_PATH).replace(/\\/g, '/'),
      model_train_seed: DATA.meta.train_seed ?? null,
      model_pref_seed: DATA.meta.pref_seed ?? null,
      deterministic: true,
      prng: 'mulberry32（， Math.random）',
      rng_injection: "web/snn_runtime.js encode()  Math.random -> (this.rng || Math.random)()，， rt.rng ",
      seeds: SEEDS,
      samples_per_seed: SAMPLES_PER_SEED,
      std_definition: ' (ddof=1)， 5 ',
      model: { num_nodes: DATA.num_nodes, n_edges: DATA.edges.length, t_steps: DATA.meta.t_steps, dt_ms: DATA.meta.dt_ms },
      label_rule: 'yTrig=1 ⇔ rdot<0  ttc=r/(-rdot)<50ms  r<25cm；esc = -rhat',
      metrics_def: {
        trigger_acc: 'pred(triggered) == yTrig （triggered = GF  12ms ）',
        threat_recall: '（）P(GF | yTrig=1)',
        false_alarm_rate: '（）P(GF | yTrig=0)',
        escape_success_rate: 'yTrig=1 「triggered  <30°」（A/B/C ）',
        dir_mae_deg: 'acos(dot(escapeDir, esc)) （ yTrig=1 ）',
        gf_first_spike_ms: 'firstSpikeMs （ yTrig=1  triggered ）',
        physical_success_rate: '（D ）（yTrig=1 ）',
        stationary_safe_rate: '（D ）（）',
        action_saved_rate: '（D ）（）',
      },
      envs: ENVS,
      speed_ranges_ms: SPEED_RANGES,
      bodies: { fly: BODIES.fly, drone: BODIES.drone, fly_radius_cm: FLY_RADIUS_CM },
      scene_sampling: 'r∈[3,35]cm, az∈[-π,π], el∈[-0.6,0.6], speed∈(m/s), miss∈[0,8]cm, s∈[0.3,3]cm； train_snn.sample_threat ',
      physics: 'looming=max(0,-2*s*rdot/(r²+s²))*1000*visGain (rad/s)；u=s²|v|/r²，wind=u*v/|v|+ambientWind (cm/ms)； m/s÷10→cm/ms',
      carrier_dynamics: 'GF  L  escapeDir （ accel/vmax），，dt=1ms >（）<（）',
    },
    checks,
    groups: {},
    group_titles: GROUP_TITLES,
  };
  for (const g of ['A', 'B', 'C', 'D']) {
    payload.groups[g] = results.filter((r) => r.group === g);
  }
  fs.writeFileSync(OUT_PATH, JSON.stringify(sanitize(payload), null, 2), 'utf8');
  console.log(`\n：${OUT_PATH}（ ${((Date.now() - t0) / 1000).toFixed(1)}s）`);
}

main();
