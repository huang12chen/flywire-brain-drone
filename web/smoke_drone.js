/* =====================================================================
 * web\smoke_drone.js —— 阶段 4：无人机物理 + 适配层回归自测（Node 运行）
 *   F:\Node\node.exe web\smoke_drone.js
 * 四项：① 悬停稳定（5s 漂移<0.3）② 逃逸响应（触发后 <300ms 速度逃逸分量>0）
 *       ③ 飞手一帧接管 ④ DronePhysics.selfTest() 全 PASS
 * ===================================================================== */
'use strict';
const fs = require('fs');
const path = require('path');
const DronePhysics = require('./drone_physics.js');
const DroneAdapter = require('./drone_adapter.js');

let pass = 0, fail = 0;
function chk(ok, name, extra) {
  console.log(`  ${ok ? '✅' : '❌'} ${name}${extra ? '  ' + extra : ''}`);
  if (ok) pass++; else fail++;
}
const stOf = p => ({ pos: p.pos, vel: p.vel, roll: p.eulerAngles().roll, pitch: p.eulerAngles().pitch, yaw: p.eulerAngles().yaw, groundY: 6 });

console.log('==== 无人机物理+适配层自测（阶段4）====');

/* ① 悬停稳定：4 电机 = HOVER，悬停 5s，位置漂移 <0.3 */
{
  const p = new DronePhysics();
  p.pos.y = 10;
  const h = DronePhysics.HOVER;
  for (let t = 0; t < 5; t += 0.016) p.update(0.016, [h, h, h, h]);
  const drift = Math.hypot(p.pos.x, p.pos.y - 10, p.pos.z);
  chk(drift < 0.3, `悬停稳定：5s 漂移=${drift.toFixed(4)} < 0.3`);
}

/* ② 逃逸响应：脑触发后 <300ms，速度在逃逸方向分量 >0
 * escapeDir 网络帧 (0,1,2)=(x,z,y) 世界分量：[0.6,-0.8,0] → 世界水平 (x=0.6, z=-0.8) */
{
  const p = new DronePhysics();
  p.pos.y = 10;
  const ad = new DroneAdapter({ rng: DroneAdapter.makeRng(20240521) });
  for (let i = 0; i < 30; i++) p.update(0.016, ad.update(null, stOf(p), 16, false));   // CRUISE 先稳一下
  let okAt = -1;
  for (let i = 0; i < 20; i++) {
    const bo = i === 0 ? { triggered: true, escapeDir: [0.6, -0.8, 0] } : null;
    p.update(0.016, ad.update(bo, stOf(p), 16, false));
    const dot = p.vel.x * 0.6 + p.vel.z * (-0.8);
    if (dot > 0 && okAt < 0) okAt = (i + 1) * 16;
  }
  chk(okAt >= 0 && okAt < 300, `逃逸响应：${okAt}ms 时速度逃逸分量 >0（要求 <300ms）`);
}

/* ③ 飞手一帧接管：pilotActive=true 当帧适配层旁路（0 电机、状态=飞手）；
 *    world.html 现有立即接管逻辑原样（keydown 同帧 selAgent.bionic=false → 旧运动学，适配层被旁路） */
{
  const p = new DronePhysics();
  p.pos.y = 10;
  const ad = new DroneAdapter({ rng: DroneAdapter.makeRng(7) });
  for (let i = 0; i < 10; i++) p.update(0.016, ad.update({ triggered: true, escapeDir: [1, 0, 0] }, stOf(p), 16, false));
  const m = ad.update(null, stOf(p), 16, true);      // 飞手接管的那一帧
  const bypassOK = ad.lastStateCN === '飞手' && m.every(v => v === 0);
  const html = fs.readFileSync(path.join(__dirname, 'world.html'), 'utf8');
  const keydownOK = /addEventListener\('keydown'[\s\S]{0,600}?selAgent\.bionic = false;/.test(html);
  const gateOK = /function ensureDronePhys\(a\)[\s\S]{0,240}?a\.kind === 'drone' && DRONE_PHYS_ON && a\.bionic/.test(html);
  chk(bypassOK && keydownOK && gateOK,
    `飞手一帧接管：适配层同帧旁路=${bypassOK}｜keydown 同帧切断=${keydownOK}｜bionic 门控=${gateOK}`);
}

/* ④ DronePhysics.selfTest() 全 PASS */
{
  const s = DronePhysics.selfTest();
  console.log('  ' + s);
  chk(/\[PASS\]/.test(s) && !/\[FAIL\]/.test(s) && /总评 PASS/.test(s), 'DronePhysics.selfTest() 全 PASS');
}

console.log(`\n==== 自检：${pass}/4 通过 ====`);
if (fail > 0) process.exit(1);
