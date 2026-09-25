/* evaluate_v4.js — 确定性蒙特卡洛实验矩阵评估器（A/B/C/D 四组）· v4 阶段 A 版
 * ============================================================================
 * 复制自 web\evaluate.js（v3 冻结品，未修改）；仅改三处：
 *   1) 模型数据加载 -> v4\snn_data_v4.js（可用 --data 覆盖，如 v4/snn_data_v4_seed20240522.js）；
 *   2) 推理引擎复用 web\snn_runtime.js（只读使用，不改动）；
 *   3) 结果输出 -> v4\results_v4.json（可用 --out 覆盖）。绝不写 web\results.json。
 * 其余：场景采样 / 物理 / 标签 / 指标定义 / 实验矩阵 / 确定性检查与 v3 逐字一致，保证可比。
 *
 * 用法：node v4/evaluate_v4.js [--data <snn_data_v4*.js>] [--out <results_v4*.json>]（相对路径按当前工作目录解析）
 *
 * 设计要点（严格遵循任务口径）：
 *   1) 全程禁用 Math.random —— 自实现 mulberry32 种子化 PRNG；场景采样与脉冲噪声
 *      各用一条独立随机流（都由同一 seed 派生），因此所有格子共享同一组种子时构成
 *      「共同随机数」配对比较，且逐格可复现。
 *   2) snn_runtime.encode() 有「唯一一处」随机源可注入补丁：(this.rng || Math.random)()。
 *      默认行为完全不变；本评估器通过 rt.rng = fn 注入种子化噪声源，实现确定性。
 *   3) 传感器物理、标签口径、指标定义与 train_snn_v4.py 完全一致（= v3 口径）。
 *
 * 实验矩阵（每格 = 5 个不同种子 × 200 样本，报 5 种子均值±样本标准差）：
 *   A 组（模态对照 × meadow）        ：fusion / vision_only / wind_only
 *   B 组（环境 × fusion）            ：meadow / storm / night
 *   C 组（突袭速度 × fusion × meadow）：slow[0.3,3] / mid[3,7] / high[7,12] m/s
 *   D 组（换体 × fusion × meadow × mid）：fly / drone（二维平面积分判定物理避障）
 */
'use strict';

const fs = require('fs');
const path = require('path');

// -------------------- 命令行参数（v4 新增） --------------------
function argVal(name, dflt) {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : dflt;
}
const DATA_PATH = path.resolve(process.cwd(), argVal('--data', path.join(__dirname, 'snn_data_v4.js')));
const OUT_PATH = path.resolve(process.cwd(), argVal('--out', path.join(__dirname, 'results_v4.json')));
if (fs.existsSync(path.join(__dirname, '..', 'web', 'results.json')) &&
    OUT_PATH === path.resolve(path.join(__dirname, '..', 'web', 'results.json'))) {
  throw new Error('拒绝输出到 web/results.json（v3 冻结品）');
}

// -------------------- 加载导出的模型数据与推理引擎 --------------------
global.window = {};
require(DATA_PATH);
const SNNRuntime = require(path.join(__dirname, '..', 'web', 'snn_runtime.js'));
const DATA = global.window.SNN_DATA;
if (DATA.meta && DATA.meta.dir_flipped) throw new Error('模型方向读出头带翻转标记（--flip-dir 导出），本评估的方向口径需人工确认');

// -------------------- 常量与预设（任务给定，不自行发明；与 v3 evaluate.js 一致） --------------------
const SEEDS = [20240521, 20240522, 20240523, 20240524, 20240525]; // 每格 5 个不同种子
const SAMPLES_PER_SEED = 200;          // 每种子 200 样本
const TAU_TTC_MS = 50.0;               // 标签：碰撞时间 < 50 ms
const R_TRIGGER_CM = 25.0;             // 标签：且距离 < 25 cm
const SUCC_ANGLE_DEG = 30.0;           // 避障成功：方向误差 < 30°
const FLY_RADIUS_CM = 0.55;            // 果蝇半径（换体实验碰撞判定用）
const DT_MS = 1.0;                     // 换体积分步长
const T_MAX_MS = 3000;                 // 换体积分时间上限（足够完成一次交会）

// 环境预设：visGain 作用于视觉膨胀率；ambientWind 叠加到风矢量（cm/ms）
const ENVS = {
  meadow: { visGain: 1.00, ambientWind: [0.000, 0, 0.000] },
  storm:  { visGain: 0.85, ambientWind: [0.020, 0, 0.012] },
  night:  { visGain: 0.30, ambientWind: [0.000, 0, 0.000] },
};

// 突袭速度子区间（m/s）
const SPEED_RANGES = {
  full: [0.3, 12.0],
  slow: [0.3, 3.0],
  mid:  [3.0, 7.0],
  high: [7.0, 12.0],
};

// 载体动力学（换体实验 D）：加速度 cm/ms²、最大速度 cm/ms
const BODIES = {
  fly:  { accel: 0.0022, vmax: 0.16 },
  drone:{ accel: 0.0009, vmax: 0.10 },
};

// -------------------- 确定性随机数：mulberry32 --------------------
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

// 由主种子派生子流种子（场景流直接用 seed，脉冲噪声流用派生值，避免两流相关）
function deriveSeed(seed, tag) {
  let h = (seed ^ 0x9e3779b9) >>> 0;
  for (let i = 0; i < tag.length; i++) h = (Math.imul(h ^ tag.charCodeAt(i), 0x85ebca6b) >>> 0);
  return h >>> 0;
}

function uniform(rng, lo, hi) { return lo + (hi - lo) * rng(); }

// -------------------- 小型向量工具 --------------------
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function norm3(a) { return Math.hypot(a[0], a[1], a[2]); }

// 方向误差（度）= acos(dot(escapeDir, esc))，对输入做防御性归一与裁剪
function angleDeg(dir, esc) {
  const n = norm3(dir);
  if (!(n > 1e-8)) return NaN;
  const c = Math.max(-1, Math.min(1, (dir[0] * esc[0] + dir[1] * esc[1] + dir[2] * esc[2]) / n));
  return (Math.acos(c) * 180) / Math.PI;
}

// -------------------- 场景采样（与训练端 sample_threat 同量程、同抽样顺序） --------------------
function sampleThreat(rng, speedRange) {
  const d0 = uniform(rng, 3.0, 35.0);            // 当前距离 cm
  const az = uniform(rng, -Math.PI, Math.PI);    // 方位角（威胁相对果蝇朝向 +x）
  const el = uniform(rng, -0.6, 0.6);            // 仰角
  const speed = uniform(rng, speedRange[0], speedRange[1]); // m/s（C/D 组按子区间）
  const miss = uniform(rng, 0.0, 8.0);           // 脱靶量 cm
  const sSize = uniform(rng, 0.3, 3.0);          // 威胁物半径 cm

  // 视线方向（威胁相对果蝇的位置单位向量）
  const rhat = [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];
  // 水平切向（与训练端一致）
  let tangent = [-rhat[1], rhat[0], 0.0];
  const tn = norm3(tangent);
  tangent = tn > 1e-9 ? [tangent[0] / tn, tangent[1] / tn, 0.0] : [0, 1, 0];

  // 速度方向 = 指向果蝇方向与切向按 miss/d 混合后归一
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

// -------------------- 物理线索与标签（与训练端 cues_and_labels 同口径） --------------------
function cuesAndLabels(st, env) {
  const r = st.d, s = st.s;
  const rdot = dot3(st.v, st.rhat);                    // cm/ms，接近为负
  // 视觉膨胀率：dθ/dt = -2 s r' / (r² + s²)（rad/s），再乘环境视觉增益
  const looming = Math.max(0.0, (-2.0 * s * rdot) / (r * r + s * s)) * 1000.0 * env.visGain;
  // 风压（球体势流近似）：u = s²|v|/r²，方向沿威胁速度方向；再叠加环境风
  const vm = norm3(st.v);
  const u = (s * s) * vm / (r * r + 1e-9);
  const wind = [u * (st.v[0] / (vm + 1e-9)) + env.ambientWind[0],
                u * (st.v[1] / (vm + 1e-9)) + env.ambientWind[1],
                u * (st.v[2] / (vm + 1e-9)) + env.ambientWind[2]];
  // 标签：有效威胁 ⇔ rdot<0 且 ttc=r/(-rdot)<50ms 且 r<25cm
  const ttcMs = rdot >= 0 ? Infinity : r / -rdot;
  const yTrig = rdot < 0 && ttcMs < TAU_TTC_MS && r < R_TRIGGER_CM ? 1 : 0;
  // 逃逸方向真值 = 背离威胁
  const esc = [-st.rhat[0], -st.rhat[1], -st.rhat[2]];
  return { looming, wind, az: st.az, yTrig, ttcMs, esc };
}

// -------------------- 换体实验：二维平面积分（D 组） --------------------
/* 平面取由 rhat 与其水平切向张成的交会平面（威胁位置、速度、逃逸方向均落在其中），
 * 果蝇固定在原点。威胁沿原方向恒速飞行；载体在 GF 潜伏期 L(ms) 后沿 escapeDir 加速
 * （受 accel / vmax 限制），dt=1ms 积分，直到「已过最近点且最近距离 > 半径和（安全）」
 * 或「距离 < 半径和（撞）」。latencyMs == null 表示 GF 未放电（载体不动，作基线用）。 */
function planarEscapeOutcome(st, latencyMs, escapeDir3, body) {
  const e1 = st.rhat, e2 = st.tangent;               // 平面基（3D 单位向量，互相正交）
  const pT = [st.d, 0.0];                            // 威胁初始位置 (e1, e2)
  const vT = [dot3(st.v, e1), dot3(st.v, e2)];       // 威胁恒定速度 cm/ms
  // 逃逸方向投影到平面并归一（载体只能在平面内机动）
  let eD = [dot3(escapeDir3, e1), dot3(escapeDir3, e2)];
  const eDn = Math.hypot(eD[0], eD[1]);
  eD = eDn < 1e-6 ? [-1.0, 0.0] : [eD[0] / eDn, eD[1] / eDn];

  const rSum = st.s + FLY_RADIUS_CM;                 // 半径和（威胁半径 + 果蝇半径 0.55cm）
  let pE = [0.0, 0.0], vE = [0.0, 0.0];              // 载体位置 / 速度
  let minDist = st.d, prevDist = st.d;
  if (minDist < rSum) return { safe: false, minDist, reason: 'start_overlap' };

  for (let t = 1; t <= T_MAX_MS; t++) {
    if (latencyMs !== null && t >= latencyMs) {      // 潜伏期后沿 escapeDir 加速
      vE[0] += body.accel * eD[0] * DT_MS;
      vE[1] += body.accel * eD[1] * DT_MS;
      const sp = Math.hypot(vE[0], vE[1]);
      if (sp > body.vmax) { vE[0] *= body.vmax / sp; vE[1] *= body.vmax / sp; }
    }
    const r0x = pT[0] - pE[0], r0y = pT[1] - pE[1];  // 步首相对位置
    pE[0] += vE[0] * DT_MS; pE[1] += vE[1] * DT_MS;
    pT[0] += vT[0] * DT_MS; pT[1] += vT[1] * DT_MS;
    // 步内扫掠判距：本步相对线段到原点的最近距离（防 1ms 步内凹谷穿透）
    const r1x = pT[0] - pE[0], r1y = pT[1] - pE[1];
    const sx = r1x - r0x, sy = r1y - r0y, s2 = sx * sx + sy * sy;
    let tc = s2 > 0 ? -(r0x * sx + r0y * sy) / s2 : 0; tc = tc < 0 ? 0 : (tc > 1 ? 1 : tc);
    const sweptMin = Math.hypot(r0x + sx * tc, r0y + sy * tc);
    const dist = Math.hypot(r1x, r1y);
    if (sweptMin < rSum) return { safe: false, minDist: Math.min(minDist, sweptMin), reason: 'collision' };
    const newMin = Math.min(sweptMin, dist);
    if (newMin < minDist) minDist = newMin;
    if (dist > prevDist && minDist > rSum) return { safe: true, minDist, reason: 'escaped' }; // 已过最近点且安全
    prevDist = dist;
  }
  return { safe: minDist > rSum, minDist, reason: 'timeout' };
}

// -------------------- 统计：5 个种子的均值 ± 样本标准差(ddof=1) --------------------
function aggregate(values) {
  const xs = values.filter((x) => Number.isFinite(x));
  if (xs.length === 0) return { mean: NaN, std: NaN, n_defined: 0 };
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  let ss = 0;
  for (const x of xs) ss += (x - mean) * (x - mean);
  const std = xs.length > 1 ? Math.sqrt(ss / (xs.length - 1)) : 0.0;
  return { mean, std, n_defined: xs.length };
}

// -------------------- 单格评估：5 种子 × 200 样本 --------------------
function evalCell(cell) {
  const rt = new SNNRuntime(DATA);
  rt.mode = cell.mode;                               // 'fusion' | 'vision_only' | 'wind_only'
  const env = ENVS[cell.env];
  const speedRange = SPEED_RANGES[cell.speed];
  const body = cell.body ? BODIES[cell.body] : null;

  const perSeed = [];
  for (const seed of SEEDS) {
    const scenRng = mulberry32(seed);                // 场景采样流
    rt.rng = mulberry32(deriveSeed(seed, 'spike'));  // 脉冲噪声流（经 snn_runtime 补丁注入）

    let n = 0, nPos = 0, nTrig = 0, correct = 0, hits = 0;
    let tp = 0, fp = 0, fn = 0, tn = 0;             // 诊断用混淆计数（不改变口径）
    const dirErrs = [], lats = [];
    let physSafe = 0, baseSafe = 0, actionSaved = 0; // D 组：换体积分统计

    for (let i = 0; i < SAMPLES_PER_SEED; i++) {
      const st = sampleThreat(scenRng, speedRange);
      const cue = cuesAndLabels(st, env);
      const out = rt.simulate(cue.looming, cue.wind, cue.az);
      n++;
      const pred = out.triggered;                    // GF 在 12ms 窗口内放电
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
          // 换体积分：GF 放电则在潜伏期 L 后开始逃逸机动；未放电则不动
          const L = pred ? out.firstSpikeMs : null;
          const withAction = planarEscapeOutcome(st, L, out.escapeDir, body);
          const noAction = planarEscapeOutcome(st, null, out.escapeDir, body); // 静止基线（自然脱靶）
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
      threat_recall: nPos > 0 ? tp / nPos : NaN,          // 诊断：P(GF放电 | yTrig=1)
      false_alarm_rate: n - nPos > 0 ? fp / (n - nPos) : NaN, // 诊断：P(GF放电 | yTrig=0)
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

  // 跨种子聚合
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

// -------------------- 实验矩阵（与 v3 逐字一致） --------------------
const CELLS = [
  // A 组：模态对照 × meadow
  { group: 'A', id: 'A1', label: 'fusion',       mode: 'fusion',      env: 'meadow', speed: 'full', body: null },
  { group: 'A', id: 'A2', label: 'vision_only',  mode: 'vision_only', env: 'meadow', speed: 'full', body: null },
  { group: 'A', id: 'A3', label: 'wind_only',    mode: 'wind_only',   env: 'meadow', speed: 'full', body: null },
  // B 组：环境 × fusion
  { group: 'B', id: 'B1', label: 'meadow',       mode: 'fusion',      env: 'meadow', speed: 'full', body: null },
  { group: 'B', id: 'B2', label: 'storm',        mode: 'fusion',      env: 'storm',  speed: 'full', body: null },
  { group: 'B', id: 'B3', label: 'night',        mode: 'fusion',      env: 'night',  speed: 'full', body: null },
  // C 组：突袭速度 × fusion × meadow
  { group: 'C', id: 'C1', label: 'slow',         mode: 'fusion',      env: 'meadow', speed: 'slow', body: null },
  { group: 'C', id: 'C2', label: 'mid',          mode: 'fusion',      env: 'meadow', speed: 'mid',  body: null },
  { group: 'C', id: 'C3', label: 'high',         mode: 'fusion',      env: 'meadow', speed: 'high', body: null },
  // D 组：换体 × fusion × meadow × mid 速度
  { group: 'D', id: 'D1', label: 'fly',          mode: 'fusion',      env: 'meadow', speed: 'mid',  body: 'fly' },
  { group: 'D', id: 'D2', label: 'drone',        mode: 'fusion',      env: 'meadow', speed: 'mid',  body: 'drone' },
];

const GROUP_TITLES = {
  A: 'A 组：模态对照 × meadow（全速域 0.3–12 m/s）',
  B: 'B 组：环境 × fusion（全速域 0.3–12 m/s）',
  C: 'C 组：突袭速度 × fusion × meadow',
  D: 'D 组：换体 × fusion × meadow × mid（3–7 m/s）',
};

// -------------------- 输出辅助 --------------------
function fmtMS(v, digits) {
  if (!Number.isFinite(v)) return 'n/a';
  return v.toFixed(digits);
}
function fmtPair(m, digits) {
  return `${fmtMS(m.mean, digits)}±${fmtMS(m.std, digits)}`;
}
// JSON 友好化：NaN/Infinity -> null
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
  console.log('| 组别 | 触发准确率 | 避障成功率 | 方向误差(°) | GF潜伏期(ms) |');
  console.log('| --- | --- | --- | --- | --- |');
  for (const c of cells) {
    // D 组的「避障成功率」按换体积分判定（安全/半径和），其余组为「触发且方向误差<30°」
    const succKey = group === 'D' ? 'physical_success_rate' : 'escape_success_rate';
    console.log(`| ${c.label} | ${fmtPair(c.metrics.trigger_acc, 3)} | ${fmtPair(c.metrics[succKey], 3)} | ` +
                `${fmtPair(c.metrics.dir_mae_deg, 1)} | ${fmtPair(c.metrics.gf_first_spike_ms, 2)} |`);
  }
}

// -------------------- 主流程 --------------------
function main() {
  const t0 = Date.now();
  console.log('确定性蒙特卡洛实验矩阵评估器 —— 果蝇逃逸 SNN（v4 阶段 A）');
  console.log(`模型数据：${DATA_PATH}`);
  console.log(`模型：${DATA.num_nodes} 节点 / ${DATA.edges.length} 边 / T=${DATA.meta.t_steps}ms / ` +
              `train_seed=${DATA.meta.train_seed ?? 'n/a'} | ` +
              `每格 ${SEEDS.length} 种子 × ${SAMPLES_PER_SEED} 样本 = ${SEEDS.length * SAMPLES_PER_SEED}`);
  console.log(`种子：${SEEDS.join(', ')}（各格共用，构成共同随机数配对比较）| PRNG: mulberry32（已注入 snn_runtime.encode）`);

  const byId = {};
  const results = [];
  for (const cell of CELLS) {
    const r = evalCell(cell);
    byId[cell.id] = r;
    results.push(r);
    console.log(`  [${cell.id}] ${cell.label} 完成（有效威胁 ${r.n_threat_total}/${r.n_samples}）... ${(Date.now() - t0) / 1000 | 0}s`);
  }

  // -------------------- 一致性 / 确定性检查 --------------------
  const neuralKeys = ['trigger_acc', 'escape_success_rate', 'dir_mae_deg', 'gf_first_spike_ms'];
  const neuralSig = (r) => JSON.stringify(neuralKeys.map((k) => r.per_seed.map((s) => s[k])));
  const checks = {
    // A1(fusion×meadow×全速) 与 B1(meadow×fusion×全速) 配置相同 + 共用种子 => 应逐位一致
    A1_equals_B1: neuralSig(byId.A1) === neuralSig(byId.B1),
    // C2(mid) 与 D1(fly) 的神经指标同配置同种子 => 应逐位一致（D 仅多出换体积分）
    C2_equals_D1_neural: neuralSig(byId.C2) === neuralSig(byId.D1),
    D1_equals_D2_neural: neuralSig(byId.D1) === neuralSig(byId.D2), // 换体不改变 SNN 前向
  };

  // -------------------- 控制台 Markdown 表格 --------------------
  console.log('\n================ 实验矩阵结果（5 种子均值±样本标准差） ================');
  for (const g of ['A', 'B', 'C', 'D']) printGroupTable(g, results);
  console.log('\n注：');
  console.log('  * 触发准确率 = pred(GF放电) == yTrig 的比例；GF 潜伏期仅统计有效威胁(yTrig=1) 中 GF 放电的样本。');
  console.log('  * A/B/C 组避障成功率 = yTrig=1 样本中「触发 且 方向误差<30°」的比例；方向误差对 yTrig=1 样本取平均。');
  console.log('  * D 组避障成功率 = 换体二维平面积分判定的物理避障成功率（潜伏期后沿 escapeDir 加速，安全=最近距离>半径和）；');
  console.log('    作为对照，D 组按方向口径（触发且方向误差<30°）的成功率：' +
              `fly ${fmtPair(byId.D1.metrics.escape_success_rate, 3)} / drone ${fmtPair(byId.D2.metrics.escape_success_rate, 3)}；`);
  console.log('    静止基线（不作机动、纯自然脱靶）安全率：' +
              `fly ${fmtPair(byId.D1.metrics.stationary_safe_rate, 3)} / drone ${fmtPair(byId.D2.metrics.stationary_safe_rate, 3)}；` +
              '机动真正救回的比例：' +
              `fly ${fmtPair(byId.D1.metrics.action_saved_rate, 3)} / drone ${fmtPair(byId.D2.metrics.action_saved_rate, 3)}。`);
  console.log('  * 确定性检查：A1==B1 逐位一致=' + checks.A1_equals_B1 +
              '，C2==D1(神经指标)=' + checks.C2_equals_D1_neural +
              '，D1==D2(神经指标)=' + checks.D1_equals_D2_neural);
  console.log('  * 触发行为分解（诊断，非验收指标）—— 威胁召回率 P(GF放电|yTrig=1) / 无威胁误报率 P(GF放电|yTrig=0)：');
  for (const c of results) {
    console.log(`      [${c.id}] ${c.label}: ${fmtPair(c.metrics.threat_recall, 3)} / ${fmtPair(c.metrics.false_alarm_rate, 3)}` +
                ` | 种子内 GF 放电次数 ${c.per_seed.map((s) => s.n_triggered).join('/')}`);
  }

  // -------------------- 写出 results_v4.json --------------------
  const payload = {
    meta: {
      generated_by: 'v4/evaluate_v4.js（复制自 web/evaluate.js；口径逐字一致，仅改数据加载与输出路径）',
      model_data: path.relative(process.cwd(), DATA_PATH).replace(/\\/g, '/'),
      model_train_seed: DATA.meta.train_seed ?? null,
      model_pref_seed: DATA.meta.pref_seed ?? null,
      deterministic: true,
      prng: 'mulberry32（种子化，禁用 Math.random）',
      rng_injection: "web/snn_runtime.js encode() 内 Math.random -> (this.rng || Math.random)()，默认行为不变，经 rt.rng 注入",
      seeds: SEEDS,
      samples_per_seed: SAMPLES_PER_SEED,
      std_definition: '样本标准差 (ddof=1)，对每格 5 个种子的指标值',
      model: { num_nodes: DATA.num_nodes, n_edges: DATA.edges.length, t_steps: DATA.meta.t_steps, dt_ms: DATA.meta.dt_ms },
      label_rule: 'yTrig=1 ⇔ rdot<0 且 ttc=r/(-rdot)<50ms 且 r<25cm；esc = -rhat',
      metrics_def: {
        trigger_acc: 'pred(triggered) == yTrig 的比例（triggered = GF 在 12ms 窗口内放电）',
        threat_recall: '（诊断）P(GF放电 | yTrig=1)',
        false_alarm_rate: '（诊断）P(GF放电 | yTrig=0)',
        escape_success_rate: 'yTrig=1 样本中「triggered 且 方向误差<30°」的比例（A/B/C 组避障成功率）',
        dir_mae_deg: 'acos(dot(escapeDir, esc)) 的角度均值（对 yTrig=1 样本）',
        gf_first_spike_ms: 'firstSpikeMs 的均值（仅统计有效威胁 yTrig=1 中 triggered 的样本）',
        physical_success_rate: '（D 组避障成功率）换体二维平面积分判定为安全的比例（yTrig=1 样本）',
        stationary_safe_rate: '（D 组基线）载体不作机动时安全（自然脱靶）的比例',
        action_saved_rate: '（D 组）有机动时安全而静止基线碰撞的比例（机动真正救回）',
      },
      envs: ENVS,
      speed_ranges_ms: SPEED_RANGES,
      bodies: { fly: BODIES.fly, drone: BODIES.drone, fly_radius_cm: FLY_RADIUS_CM },
      scene_sampling: 'r∈[3,35]cm, az∈[-π,π], el∈[-0.6,0.6], speed∈速度子区间(m/s), miss∈[0,8]cm, s∈[0.3,3]cm；与 train_snn.sample_threat 同量程',
      physics: 'looming=max(0,-2*s*rdot/(r²+s²))*1000*visGain (rad/s)；u=s²|v|/r²，wind=u*v/|v|+ambientWind (cm/ms)；速度 m/s÷10→cm/ms',
      carrier_dynamics: 'GF 潜伏期 L 后沿 escapeDir 加速（限 accel/vmax），威胁恒速，dt=1ms 二维平面积分至最近距离>半径和（安全）或距离<半径和（撞）',
    },
    checks,
    groups: {},
    group_titles: GROUP_TITLES,
  };
  for (const g of ['A', 'B', 'C', 'D']) {
    payload.groups[g] = results.filter((r) => r.group === g);
  }
  fs.writeFileSync(OUT_PATH, JSON.stringify(sanitize(payload), null, 2), 'utf8');
  console.log(`\n已写出：${OUT_PATH}（耗时 ${((Date.now() - t0) / 1000).toFixed(1)}s）`);
}

main();
