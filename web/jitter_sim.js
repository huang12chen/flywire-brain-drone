#!/usr/bin/env node
/* jitter_sim.js — "One-click 100-run statistics" (bBatch) reproducible archive simulation for sampling jitter
 *
 * Purpose: Provides reviewable data (fixed seed) for "jitter band" and "3-run median decision" in User Manual Type B acceptance tests.
 * Replication target: world.html's bBatch sampling and metrics (trigger = label agreement rate; success = triggered and direction error <30°,
 *   statistics for all valid threat samples yTrig=1, including non-firing rounds; direction = mean angle to −r̂ for yTrig=1;
 *   latency = mean first-spike time among firing rounds in yTrig=1).
 * Usage: node web\jitter_sim.js   → outputs web\jitter_sim_log.md (deterministic, bit-reproducible with fixed seed)
 */
const fs = require('fs');
const path = require('path');
global.window = {};
require('./snn_data.js');
const SNNRuntime = require('./snn_runtime.js');
const DATA = global.window.SNN_DATA;
if (DATA.meta && DATA.meta.dir_flipped) { console.error('❌ Direction readout headset flip flag detected'); process.exit(1); }

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SEED = 20240521;          // Consistent with other parts of the project
const N = 100;                  // Number of scenarios per run (= bBatch's N)
const RUNS = 40;                // Number of jitter band runs
const TRIALS = 200;             // Number of "3-run median" simulation groups
const VIS_GAIN = 1.0;           // Grassland (sunny): consistent with manual Type B test conditions

const v3 = (x, y, z) => ({ x, y, z });
const vlen = (a) => Math.hypot(a.x, a.y, a.z);
const vdot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;

function run100(rt, rnd) {
  let trigOK = 0, hits = 0, nPos = 0, angSum = 0, latSum = 0, latN = 0;
  for (let i = 0; i < N; i++) {
    const az = (rnd() * 2 - 1) * Math.PI, el = (rnd() - .5) * 1.2;
    const d = 3 + rnd() * 32, s = .3 + rnd() * 2.7, speed = .3 + rnd() * 11.7;
    const rhat = v3(Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el));
    const miss = rnd() * 8;
    // tan = normalize(-rhat.y, rhat.x, 0) (training frame tangent, consistent with bBatch)
    let tan = v3(-rhat.y, rhat.x, 0); const tl = vlen(tan) || 1;
    tan = v3(tan.x / tl, tan.y / tl, tan.z / tl);
    const k = Math.sqrt(Math.max(0, 1 - (miss / (d + 1e-9)) ** 2));   // P5③: miss-mixing coefficient unified as miss/(d+1e-9) (aligned with train/evaluate)
    let v = v3(rhat.x * -k + tan.x * (miss / (d + 1e-9)), rhat.y * -k + tan.y * (miss / (d + 1e-9)), rhat.z * -k + tan.z * (miss / (d + 1e-9)));
    const vl = vlen(v) || 1, sp = speed / 10;
    v = v3(v.x / vl * sp, v.y / vl * sp, v.z / vl * sp);
    const rdot = vdot(v, rhat);
    const loom = Math.max(0, -2 * s * rdot / (d * d + s * s)) * 1000 * VIS_GAIN;
    const vn = vlen(v);
    const u = s * s * vn / (d * d + 1e-9);
    const wind = vn > 1e-9 ? v3(v.x / vn * u, v.y / vn * u, v.z / vn * u) : v3(0, 0, 0);
    const ttc = rdot < 0 ? d / (-rdot) : 1e9;
    const yTrig = (rdot < 0 && ttc < 50 && d < 25) ? 1 : 0;
    const res = rt.simulate(loom, [wind.x, wind.y, wind.z], Math.atan2(rhat.y, rhat.x));
    const pred = res.triggered ? 1 : 0;
    if (pred === yTrig) trigOK++;
    if (yTrig === 1) {
      nPos++;
      const esc = v3(-rhat.x, -rhat.y, -rhat.z);
      let dv = v3(res.escapeDir[0], res.escapeDir[1], res.escapeDir[2]);
      const dl = vlen(dv) || 1; dv = v3(dv.x / dl, dv.y / dl, dv.z / dl);
      const ang = Math.acos(Math.max(-1, Math.min(1, vdot(dv, esc)))) * 180 / Math.PI;
      angSum += ang;
      if (res.firstSpikeMs != null && isFinite(res.firstSpikeMs)) { latSum += res.firstSpikeMs; latN++; }
      if (pred === 1 && ang < 30) hits++;
    }
  }
  return {
    trig: trigOK / N * 100,
    succ: hits / Math.max(1, nPos) * 100,
    dir: nPos ? angSum / nPos : 999,
    lat: latN ? latSum / latN : 99,
    nPos,
  };
}

const median = (xs) => xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const fmt = (x, n = 1) => x.toFixed(n);

function main() {
  const rt = new SNNRuntime(DATA);
  rt.rng = mulberry32(SEED ^ 0x5F5F5F5F);   // Inject deterministic random source (Poisson spikes) — without this it falls back to Math.random, breaking reproducibility
  // ---- Part A: 40 runs × 100 scenarios jitter band ----
  const rnd = mulberry32(SEED);
  const runs = [];
  for (let r = 0; r < RUNS; r++) runs.push(run100(rt, rnd));
  // ---- Part B: 200 groups "3-run median" decision simulation ----
  const trnd = mulberry32(SEED ^ 0x9E3779B9);
  let passAll = 0, passT = 0, passS = 0, passD = 0, passL = 0;
  for (let t = 0; t < TRIALS; t++) {
    const rs = [run100(rt, trnd), run100(rt, trnd), run100(rt, trnd)];
    const trig = median(rs.map(r => r.trig)), succ = median(rs.map(r => r.succ));
    const dir = median(rs.map(r => r.dir)), lat = median(rs.map(r => r.lat));
    const okT = trig >= 75, okS = succ >= 85, okD = dir <= 20, okL = lat <= 10;
    if (okT) passT++; if (okS) passS++; if (okD) passD++; if (okL) passL++;
    if (okT && okS && okD && okL) passAll++;
  }

  const trigs = runs.map(r => r.trig), succs = runs.map(r => r.succ);
  const dirs = runs.map(r => r.dir), lats = runs.map(r => r.lat);
  const lo = (xs) => Math.min(...xs), hi = (xs) => Math.max(...xs);

  const md = [];
  md.push('# "One-click 100-run" Sampling Jitter — Reproducible Archive Simulation');
  md.push('');
  md.push(`- Script: \`web\\jitter_sim.js\` (fixed seed, deterministic) | Seed ${SEED} | N=${N} scenarios per run | Grassland (sunny) visGain=${VIS_GAIN}`);
  md.push(`- Metrics: identical to \`world.html\` bBatch (trigger = label agreement rate; success = triggered and direction error <30°, including non-firing rounds; direction = mean angle to −r̂ for all valid threats; latency = mean first-spike time among firing rounds in valid threats)`);
  md.push(`- Environment: node ${process.version} | Model meta.t_steps=${DATA.meta.t_steps}ms`);
  md.push('');
  md.push('## A. Jitter Band (40 runs × 100 scenarios)');
  md.push('');
  md.push(`| Metric | Min | Median | Max |`);
  md.push(`|---|---|---|---|`);
  md.push(`| Trigger Accuracy | ${fmt(lo(trigs))}% | ${fmt(median(trigs))}% | ${fmt(hi(trigs))}% |`);
  md.push(`| Escape Success Rate | ${fmt(lo(succs))}% | ${fmt(median(succs))}% | ${fmt(hi(succs))}% |`);
  md.push(`| Direction Error | ${fmt(lo(dirs))}° | ${fmt(median(dirs))}° | ${fmt(hi(dirs))}° |`);
  md.push(`| GF Latency | ${fmt(lo(lats))} ms | ${fmt(median(lats))} ms | ${fmt(hi(lats))} ms |`);
  md.push('');
  md.push('## B. "Run 3 Times Take Median" Decision Simulation (200 groups)');
  md.push('');
  md.push(`- All four criteria met (trigger ≥75%, success ≥85%, direction ≤20°, latency ≤10ms): **${fmt(passAll / TRIALS * 100, 1)}%**`);
  md.push(`- Individual pass rates: Trigger ${fmt(passT / TRIALS * 100, 1)}% | Success ${fmt(passS / TRIALS * 100, 1)}% | Direction ${fmt(passD / TRIALS * 100, 1)}% | Latency ${fmt(passL / TRIALS * 100, 1)}%`);
  md.push('');
  md.push('## Per-run Details (Part A)');
  md.push('');
  md.push('| Run | Trigger | Success | Direction | Latency |');
  md.push('|---|---|---|---|---|');
  runs.forEach((r, i) => md.push(`| ${i + 1} | ${fmt(r.trig)}% | ${fmt(r.succ)}% | ${fmt(r.dir)}° | ${fmt(r.lat)} ms |`));
  md.push('');
  md.push('> This file was generated by \`node web\\jitter_sim.js\`; bit-reproducible under a fixed seed.');

  const out = path.join(__dirname, 'jitter_sim_log.md');
  fs.writeFileSync(out, md.join('\n'), 'utf8');
  console.log(`Jitter band: Trigger ${fmt(lo(trigs))}–${fmt(hi(trigs))}% | Success ${fmt(lo(succs))}–${fmt(hi(succs))}% | Direction ${fmt(lo(dirs))}–${fmt(hi(dirs))}° | Latency ${fmt(lo(lats))}–${fmt(hi(lats))}ms`);
  console.log(`3-run median all-criteria pass rate: ${fmt(passAll / TRIALS * 100, 1)}%  (individual T${fmt(passT / TRIALS * 100, 1)}/S${fmt(passS / TRIALS * 100, 1)}/D${fmt(passD / TRIALS * 100, 1)}/L${fmt(passL / TRIALS * 100, 1)})`);
  console.log(`Written to ${out}`);
}
main();
