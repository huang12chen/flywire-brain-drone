/* =====================================================================
 * web\drone_adapter.js —— 阶段 2：脑→电机适配层
 * 仿生分层：果蝇脑(决策) → 本适配层(胸神经节) → 电机(飞行肌)；PID 姿态环 = 平衡棒反射
 * 普通 <script>，挂 window.DroneAdapter；Node 端 module.exports（供阶段 4 smoke_drone.js）
 * 零依赖（零 fetch、零 ESM）；随机数 rng 由外部注入保确定性。
 * ===================================================================== */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.DroneAdapter = api;
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this), function () {
  'use strict';
  const DEG = Math.PI / 180;
  const clamp = (v, lo, hi) => v < lo ? lo : (v > hi ? hi : v);

  /* 取 DronePhysics（浏览器按脚本顺序读 window；Node 端 require），只用其 HOVER */
  const DronePhysics = (typeof module === 'object' && module.exports)
    ? require('./drone_physics.js')
    : (typeof window !== 'undefined' ? window.DronePhysics : null);
  const HOVER = DronePhysics ? DronePhysics.HOVER : Math.sqrt(1.0 * 9.8 / (4 * 0.12));  // 悬停转速 ≈4.5185

  /* 状态机：CRUISE(巡航) / ESCAPE(逃逸) / PILOT(飞手) / LAND(降落) */
  const CN = { CRUISE: '巡航', ESCAPE: '逃逸', PILOT: '飞手', LAND: '降落' };

  class DroneAdapter {
    constructor(opts) {
      opts = opts || {};
      /* PID 姿态环：kp=0.045, ki=0.001, kd=0.028（误差=期望姿态-当前姿态，输出加进 motors） */
      this.kp = opts.kp !== undefined ? opts.kp : 0.045;
      this.ki = opts.ki !== undefined ? opts.ki : 0.001;
      this.kd = opts.kd !== undefined ? opts.kd : 0.028;
      this.rng = opts.rng || Math.random;          // 注入 rng 保确定性
      this.escapeMs = 300;                          // ESCAPE 持续 300ms（GF 爆发）
      this.landY = opts.landY !== undefined ? opts.landY : 0;   // 落地阈值基准（world 传 groundY）
      /* 内部状态 */
      this.state = 'CRUISE';
      this.lastState = 'CRUISE';                    // 永远导出（HUD 用）
      this.lastStateCN = CN.CRUISE;
      this.lastMotors = [0, 0, 0, 0];
      this.tMs = 0;
      this.escapeUntil = -1e9; this.lastEscapeEnd = -1e9;
      this.escapeVec = null;                        // 逃逸方向（世界 (x,y,z)，来自脑输出换轴后）
      this.landRequested = false; this.landed = false; this.throttle = 1.02;
      this.yawTarget = 0; this.nextHeadingT = 0;
      this.tiltTarget = { roll: 0, pitch: 0 };
      this.iErr = { roll: 0, pitch: 0, yaw: 0 };
      this.prevErr = { roll: 0, pitch: 0, yaw: 0 };
      this.hoverOmega = HOVER;
    }

    /* ---------- 对外命令（阶段 3 接按钮/电量用） ---------- */
    enterLand() { this.landRequested = true; this.landed = false; if (this.state !== 'ESCAPE') this.state = 'LAND'; }
    enterCruise() { this.landRequested = false; this.landed = false; this.throttle = 1.02; if (this.state !== 'ESCAPE') this.state = 'CRUISE'; }
    /* 阶段3 巡航避障：朝远离障碍方向偏航（只转向、不触发逃逸） */
    setYawTarget(yaw) { this.yawTarget = yaw; this.nextHeadingT = this.tMs + 2000 + this.rng() * 2000; }

    /* 混控矩阵（X 型简化，每项写清）：
     *   base = HOVER * throttle
     *   m0 = base + HOVER*(+rollCmd + yawCmd)    前左（对角 1）
     *   m1 = base + HOVER*(+pitchCmd + yawCmd)   前右（对角 2）
     *   m2 = base + HOVER*(-rollCmd - yawCmd)    后右（对角 1）
     *   m3 = base + HOVER*(-pitchCmd - yawCmd)   后左（对角 2）
     * → 动力学端（规格公式）：roll 力矩=(m0-m2)=2*HOVER*(rollCmd+yawCmd)
     *   pitch 力矩=(m1-m3)=2*HOVER*(pitchCmd+yawCmd)
     *   yaw 力矩=km*(m0-m2+m1-m3)=2*km*HOVER*(rollCmd+pitchCmd+2*yawCmd)
     * 规格简化混控下 yaw 通道与 roll/pitch 有耦合，由 PID 闭环吸收（已注明）。 */
    _mix(throttle, rollCmd, pitchCmd, yawCmd) {
      const base = HOVER * throttle;
      return [
        clamp(base + HOVER * (+rollCmd + yawCmd), 0, 2 * HOVER),
        clamp(base + HOVER * (+pitchCmd + yawCmd), 0, 2 * HOVER),
        clamp(base + HOVER * (-rollCmd - yawCmd), 0, 2 * HOVER),
        clamp(base + HOVER * (-pitchCmd - yawCmd), 0, 2 * HOVER),
      ];
    }

    /* update(brainOut, droneState, dtMs, pilotActive) -> motors[4]
     * brainOut：SNN 输出（triggered/escapeDir）；无威胁或飞手时传 null
     * droneState：{ pos, vel, roll, pitch, yaw, groundY }（roll/pitch/yaw 弧度；vel=世界单位/秒） */
    update(brainOut, s, dtMs, pilotActive) {
      const dt = clamp(dtMs, 0, 50);
      this.tMs += dt;
      s = s || { pos: { x: 0, y: 0, z: 0 }, vel: { x: 0, y: 0, z: 0 }, roll: 0, pitch: 0, yaw: 0, groundY: 0 };

      /* PILOT 优先级最高：飞手接管（world.html 现有非修饰键逻辑）→ 旧运动学立即接管，本层旁路 */
      if (pilotActive) {
        this.state = 'PILOT'; this.iErr = { roll: 0, pitch: 0, yaw: 0 };
        return this._commit([0, 0, 0, 0]);
      }
      if (this.state === 'PILOT') this.state = this.landRequested ? 'LAND' : 'CRUISE';

      /* ESCAPE：brainOut.triggered==true 触发，持续 300ms */
      if (brainOut && brainOut.triggered && this.state !== 'ESCAPE' && !this.landRequested
          && this.tMs - this.lastEscapeEnd > 150) {
        this.state = 'ESCAPE';
        this.escapeUntil = this.tMs + this.escapeMs;
        /* escapeDir 网络帧 (0,1,2)=(x,z,y) 世界分量 → 世界 (x,y,z)（交换 y/z，口径不动） */
        const d = brainOut.escapeDir || [0, 0, 0];
        const ex = d[0], ey = d[2], ez = d[1];
        const hl = Math.hypot(ex, ez);
        this.escapeVec = hl > 1e-6 ? { x: ex / hl, z: ez / hl } : { x: 0, z: 0 };
        this.iErr = { roll: 0, pitch: 0, yaw: 0 };
      }
      if (this.state === 'ESCAPE' && this.tMs > this.escapeUntil) {
        this.state = this.landRequested ? 'LAND' : 'CRUISE';
        this.lastEscapeEnd = this.tMs;
      }
      if (this.landRequested && this.state !== 'ESCAPE') this.state = 'LAND';

      /* —— 各状态给油门 + 期望姿态 —— */
      let throttle = 1.02, desRoll = 0, desPitch = 0, desYaw = s.yaw;
      if (this.state === 'ESCAPE') {
        /* 油门 = HOVER*1.5（阶跃，模拟 GF 爆发）
         * 目标倾角 = escapeDir 投影到水平面，倾角 35°、roll/pitch 分量各钳 ±35° */
        throttle = 1.5;
        const a35 = 35 * DEG, si = Math.sin(a35);
        const h = this.escapeVec || { x: 0, z: 0 };
        const dx = h.x * si, dz = h.z * si;
        const cy = Math.cos(s.yaw), sy = Math.sin(s.yaw);
        let th = sy * dx + cy * dz;    // 期望 pitch（世界逃逸方向 → 机体轴）
        let ph = -cy * dx + sy * dz;   // 期望 roll
        desPitch = clamp(th, -a35, a35);
        desRoll = clamp(ph, -a35, a35);
      } else if (this.state === 'LAND') {
        /* 降落：慢速降油门至 HOVER*0.8 直到落地（pos.y<阈值）→ 油门 0 */
        this.throttle = Math.max(0.8, this.throttle - dt * 0.00012);
        throttle = this.throttle;
        const gy = (s.groundY !== undefined ? s.groundY : this.landY);
        if (s.pos.y <= gy + 0.5) { this.landed = true; throttle = 0; }
        desRoll = this.tiltTarget.roll; desPitch = this.tiltTarget.pitch;
      } else {
        /* CRUISE：油门=HOVER*1.02，期望姿态=轻微随机游走（每 2-4s 换目标航向，角速度 ≤15°/s） */
        throttle = 1.02;
        if (this.tMs >= this.nextHeadingT) {
          this.nextHeadingT = this.tMs + 2000 + this.rng() * 2000;
          this.yawTarget = s.yaw + (this.rng() * 2 - 1) * 60 * DEG;
          this.tiltTarget.roll = (this.rng() * 2 - 1) * 6 * DEG;
          this.tiltTarget.pitch = (this.rng() * 2 - 1) * 6 * DEG;
        }
        let dy = this.yawTarget - s.yaw;
        while (dy > Math.PI) dy -= 2 * Math.PI;
        while (dy < -Math.PI) dy += 2 * Math.PI;
        desYaw = s.yaw + clamp(dy, -15 * DEG * dt / 1000, 15 * DEG * dt / 1000);  // 航向角速度 ≤15°/s
        desRoll = this.tiltTarget.roll; desPitch = this.tiltTarget.pitch;
      }

      /* PID 姿态环：误差=期望姿态-当前姿态，输出加进 motors（dt 用秒，量纲自洽） */
      const dtSec = Math.max(dt, 1e-3) / 1000;
      const err = { roll: desRoll - s.roll, pitch: desPitch - s.pitch, yaw: desYaw - s.yaw };
      err.roll = clamp(err.roll, -0.6, 0.6); err.pitch = clamp(err.pitch, -0.6, 0.6);
      let dyaw = err.yaw; while (dyaw > Math.PI) dyaw -= 2 * Math.PI; while (dyaw < -Math.PI) dyaw += 2 * Math.PI;
      err.yaw = dyaw;
      const out = {};
      for (const k of ['roll', 'pitch', 'yaw']) {
        this.iErr[k] = clamp(this.iErr[k] + err[k] * dtSec, -1, 1);
        const dErr = (err[k] - this.prevErr[k]) / dtSec;
        out[k] = this.kp * err[k] + this.ki * this.iErr[k] + this.kd * dErr;
        this.prevErr[k] = err[k];
      }
      return this._commit(this._mix(throttle, out.roll, out.pitch, out.yaw));
    }

    _commit(m) {
      this.lastMotors = m;
      this.lastState = this.state;
      this.lastStateCN = CN[this.state] || this.state;
      return m;
    }
  }

  /* 确定性随机源（mulberry32），world.html/自测注入用 */
  DroneAdapter.makeRng = function (seed) {
    let a = (seed >>> 0) || 1;
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  };
  DroneAdapter.HOVER = HOVER;

  return DroneAdapter;
});
