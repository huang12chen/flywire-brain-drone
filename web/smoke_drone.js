/* =====================================================================
 * web\smoke_drone.js —— Phase 4: Drone Physics + Adapter Layer Regression Self-Test (Node Run)
 *   F:\Node\node.exe web\smoke_drone.js
 * Four items: ① Hover Stability (5s drift<0.3) ② Escape Response (triggered <300ms escape velocity component>0)
 *       ③ Pilot One-Frame Takeover ④ DronePhysics.selfTest() All PASS
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

console.log('==== Drone Physics + Adapter Layer Self-Test (Phase 4) ====');

/* ① Hover Stability: 4 motors = HOVER, hover for 5s, position drift <0.3 */
{
  const p = new DronePhysics();
  p.pos.y = 10;
  const h = DronePhysics.HOVER;
  for (let t = 0; t < 5; t += 0.016) p.update(0.016, [h, h, h, h]);
  const drift = Math.hypot(p.pos.x, p.pos.y - 10, p.pos.z);
  chk(drift < 0.3, `Hover Stability: 5s drift=${drift.toFixed(4)} < 0.3`);
}

/* ② Escape Response: After brain trigger, <300ms, velocity in escape direction component >0
 * escapeDir network frame (0,1,2)=(x,z,y) world component: [0.6,-0.8,0] → world horizontal (x=0.6, z=-0.8) */
{
  const p = new DronePhysics();
  p.pos.y = 10;
  const ad = new DroneAdapter({ rng: DroneAdapter.makeRng(20240521) });
  for (let i = 0; i < 30; i++) p.update(0.016, ad.update(null, stOf(p), 16, false));   // CRUISE stabilize first
  let okAt = -1;
  for (let i = 0; i < 20; i++) {
    const bo = i === 0 ? { triggered: true, escapeDir: [0.6, -0.8, 0] } : null;
    p.update(0.016, ad.update(bo, stOf(p), 16, false));
    const dot = p.vel.x * 0.6 + p.vel.z * (-0.8);
    if (dot > 0 && okAt < 0) okAt = (i + 1) * 16;
  }
  chk(okAt >= 0 && okAt < 300, `Escape Response: at ${okAt}ms escape velocity component >0 (requires <300ms)`);
}

/* ③ Pilot One-Frame Takeover: pilotActive=true, adapter layer bypassed that frame (0 motors, state=pilot);
 *    world.html existing immediate takeover logic unchanged (keydown same frame selAgent.bionic=false → old kinematics, adapter layer bypassed) */
{
  const p = new DronePhysics();
  p.pos.y = 10;
  const ad = new DroneAdapter({ rng: DroneAdapter.makeRng(7) });
  for (let i = 0; i < 10; i++) p.update(0.016, ad.update({ triggered: true, escapeDir: [1, 0, 0] }, stOf(p), 16, false));
  const m = ad.update(null, stOf(p), 16, true);      // The frame when pilot takes over
  const bypassOK = ad.lastStateCN === '飞手' && m.every(v => v === 0);
  const html = fs.readFileSync(path.join(__dirname, 'world.html'), 'utf8');
  const keydownOK = /addEventListener\('keydown'[\s\S]{0,600}?selAgent\.bionic = false;/.test(html);
  const gateOK = /function ensureDronePhys\(a\)[\s\S]{0,240}?a\.kind === 'drone' && DRONE_PHYS_ON && a\.bionic/.test(html);
  chk(bypassOK && keydownOK && gateOK,
    `Pilot One-Frame Takeover: adapter same-frame bypass=${bypassOK}｜keydown same-frame cut=${keydownOK}｜bionic gating=${gateOK}`);
}

/* ④ DronePhysics.selfTest() All PASS */
{
  const s = DronePhysics.selfTest();
  console.log('  ' + s);
  chk(/\[PASS\]/.test(s) && !/\[FAIL\]/.test(s) && /总评 PASS/.test(s), 'DronePhysics.selfTest() All PASS');
}

console.log(`\n==== Self-Test: ${pass}/4 Passed ====`);
if (fail > 0) process.exit(1);
