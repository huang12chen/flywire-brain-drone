#!/usr/bin/env node
/* jitter_sim.js — 「一键跑 100 次统计」（bBatch）抽样抖动的可复现存档模拟
 *
 * 目的：给使用手册 B 类验收的"抖动带"与"3 次中位数判定"提供可复核数据（固定种子）。
 * 复刻对象：world.html 的 bBatch 采样与口径（触发=标签一致率；成功=触发且方向误差<30°，
 *   对全部有效威胁样本 yTrig=1 统计、含未放电回合；方向=对 yTrig=1 的 −r̂ 夹角均值；
 *   潜伏=yTrig=1 中放电回合的首次放电均值）。
 * 运行：node web\jitter_sim.js   → 输出 web\jitter_sim_log.md（确定性，固定种子可逐位复现）
 */
const fs = require('fs');
const path = require('path');
global.window = {};
require('./snn_data.js');
const SNNRuntime = require('./snn_runtime.js');
const DATA = global.window.SNN_DATA;
if (DATA.meta && DATA.meta.dir_flipped) { console.error('❌ 方向读出头带翻转标记'); process.exit(1); }

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SEED = 20240521;          // 与项目其它部分一致
const N = 100;                  // 每轮场景数（= bBatch 的 N）
const RUNS = 40;                // 抖动带轮数
const TRIALS = 200;             // "3 次中位数"模拟组数
const VIS_GAIN = 1.0;           // 草原（晴）：与手册 B 类测试条件一致

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
    // tan = normalize(-rhat.y, rhat.x, 0)（训练帧切向，与 bBatch 一致）
    let tan = v3(-rhat.y, rhat.x, 0); const tl = vlen(tan) || 1;
    tan = v3(tan.x / tl, tan.y / tl, tan.z / tl);
    const k = Math.sqrt(Math.max(0, 1 - (miss / (d + 1e-9)) ** 2));   // P5③：脱靶混合系数统一 miss/(d+1e-9)（对齐 train/evaluate）
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
  rt.rng = mulberry32(SEED ^ 0x5F5F5F5F);   // 注入确定性随机源（泊松脉冲）——缺此会回落 Math.random，"可复现"就不成立
  // ---- A 部分：40 轮 × 100 场景的抖动带 ----
  const rnd = mulberry32(SEED);
  const runs = [];
  for (let r = 0; r < RUNS; r++) runs.push(run100(rt, rnd));
  // ---- B 部分：200 组「3 次取中位数」判定模拟 ----
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
  md.push('# 「一键 100 次」抽样抖动 —— 可复现存档模拟');
  md.push('');
  md.push(`- 脚本：\`web\\jitter_sim.js\`（固定种子，确定性）｜种子 ${SEED}｜每轮 N=${N} 场景｜草原（晴）visGain=${VIS_GAIN}`);
  md.push(`- 口径：与 \`world.html\` bBatch 完全一致（触发=标签一致率；成功=触发且方向误差<30°，含未放电回合；方向=对全部有效威胁的 −r̂ 夹角均值；潜伏=有效威胁中放电回合首次放电均值）`);
  md.push(`- 环境：node ${process.version}｜模型 meta.t_steps=${DATA.meta.t_steps}ms`);
  md.push('');
  md.push('## A. 抖动带（40 轮 × 100 场景）');
  md.push('');
  md.push(`| 指标 | 最小 | 中位 | 最大 |`);
  md.push(`|---|---|---|---|`);
  md.push(`| 触发准确率 | ${fmt(lo(trigs))}% | ${fmt(median(trigs))}% | ${fmt(hi(trigs))}% |`);
  md.push(`| 逃逸成功率 | ${fmt(lo(succs))}% | ${fmt(median(succs))}% | ${fmt(hi(succs))}% |`);
  md.push(`| 方向误差 | ${fmt(lo(dirs))}° | ${fmt(median(dirs))}° | ${fmt(hi(dirs))}° |`);
  md.push(`| GF 潜伏期 | ${fmt(lo(lats))} ms | ${fmt(median(lats))} ms | ${fmt(hi(lats))} ms |`);
  md.push('');
  md.push('## B. 「连跑 3 次取中位数」判定模拟（200 组）');
  md.push('');
  md.push(`- 四项全达标（触发≥75%、成功≥85%、方向≤20°、潜伏≤10ms）：**${fmt(passAll / TRIALS * 100, 1)}%**`);
  md.push(`- 单项达标率：触发 ${fmt(passT / TRIALS * 100, 1)}%｜成功 ${fmt(passS / TRIALS * 100, 1)}%｜方向 ${fmt(passD / TRIALS * 100, 1)}%｜潜伏 ${fmt(passL / TRIALS * 100, 1)}%`);
  md.push('');
  md.push('## 每轮明细（A 部分）');
  md.push('');
  md.push('| 轮 | 触发 | 成功 | 方向 | 潜伏 |');
  md.push('|---|---|---|---|---|');
  runs.forEach((r, i) => md.push(`| ${i + 1} | ${fmt(r.trig)}% | ${fmt(r.succ)}% | ${fmt(r.dir)}° | ${fmt(r.lat)} ms |`));
  md.push('');
  md.push('> 本文件由 `node web\\jitter_sim.js` 生成；固定种子下逐位可复现。');

  const out = path.join(__dirname, 'jitter_sim_log.md');
  fs.writeFileSync(out, md.join('\n'), 'utf8');
  console.log(`抖动带: 触发 ${fmt(lo(trigs))}–${fmt(hi(trigs))}% | 成功 ${fmt(lo(succs))}–${fmt(hi(succs))}% | 方向 ${fmt(lo(dirs))}–${fmt(hi(dirs))}° | 潜伏 ${fmt(lo(lats))}–${fmt(hi(lats))}ms`);
  console.log(`3次中位数 全项达标率: ${fmt(passAll / TRIALS * 100, 1)}%  (单项 T${fmt(passT / TRIALS * 100, 1)}/S${fmt(passS / TRIALS * 100, 1)}/D${fmt(passD / TRIALS * 100, 1)}/L${fmt(passL / TRIALS * 100, 1)})`);
  console.log(`已写入 ${out}`);
}
main();
