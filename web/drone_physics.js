/* =====================================================================
 * web\drone_physics.js —— 阶段 1：四旋翼飞行动力学（演示级，未真机标定）
 * 普通 <script>，挂 window.DronePhysics；Node 端 module.exports（供阶段 4 smoke_drone.js）
 * 零依赖（不引 THREE、零 fetch、零 ESM）：pos/vel/angVel 为 {x,y,z}、quat 为 {x,y,z,w}，
 *   字段与 THREE.Vector3/Quaternion 互通，world.html 可直接 .set/.copy 对接。
 * 单位：世界单位 + 秒（update 收 dtSec）；参数为演示级数值，标注【未真机标定】。
 * ===================================================================== */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.DronePhysics = api;
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this), function () {
  'use strict';
  const DEG = Math.PI / 180;
  const clamp = (v, lo, hi) => v < lo ? lo : (v > hi ? hi : v);

  /* 四元数旋转向量：v' = v + 2w(qv×v) + 2(qv×(qv×v)) */
  function quatRotate(q, v) {
    const qx = q.x, qy = q.y, qz = q.z, qw = q.w, vx = v.x, vy = v.y, vz = v.z;
    const tx = 2 * (qy * vz - qz * vy);
    const ty = 2 * (qz * vx - qx * vz);
    const tz = 2 * (qx * vy - qy * vx);
    return {
      x: vx + qw * tx + (qy * tz - qz * ty),
      y: vy + qw * ty + (qz * tx - qx * tz),
      z: vz + qw * tz + (qx * ty - qy * tx),
    };
  }

  class DronePhysics {
    constructor(opts) {
      opts = opts || {};
      /* 参数（演示级、世界单位，【未真机标定】） */
      this.mass     = opts.mass     !== undefined ? opts.mass     : 1.0;    // 质量
      this.gravity  = opts.gravity  !== undefined ? opts.gravity  : 9.8;    // 重力加速度
      this.kf       = opts.kf       !== undefined ? opts.kf       : 0.12;   // 推力系数
      this.km       = opts.km       !== undefined ? opts.km       : 0.018;  // 反扭矩系数
      this.dragLin  = opts.dragLin  !== undefined ? opts.dragLin  : 0.35;   // 线性阻力
      this.dragQuad = opts.dragQuad !== undefined ? opts.dragQuad : 0.02;   // 二次阻力
      this.maxTilt  = (opts.maxTilt !== undefined ? opts.maxTilt : 40) * DEG; // 姿态欧拉角钳制 ±40°
      this.inertia  = opts.inertia  !== undefined ? opts.inertia  : { roll: 0.02, pitch: 0.02, yaw: 0.04 };
      /* 状态：pos/vel/angVel（{x,y,z}）、quat（{x,y,z,w}） */
      this.pos    = { x: 0, y: 0, z: 0 };
      this.vel    = { x: 0, y: 0, z: 0 };
      this.quat   = { x: 0, y: 0, z: 0, w: 1 };
      this.angVel = { x: 0, y: 0, z: 0 };
      this.lastMotors = [0, 0, 0, 0];
    }

    /* 悬停校准：hoverOmega = sqrt(mass*gravity/(4*kf)) */
    hoverOmega() { return Math.sqrt(this.mass * this.gravity / (4 * this.kf)); }

    /* 欧拉角（约定 q = qy(yaw)·qx(pitch)·qz(roll)，机体 x=右 y=上 z=后） */
    eulerAngles() {
      const q = this.quat, x = q.x, y = q.y, z = q.z, w = q.w;
      const pitch = Math.asin(clamp(2 * (w * x - y * z), -1, 1));            // 绕机体 x
      const roll  = Math.atan2(2 * (x * y + w * z), 1 - 2 * (x * x + z * z)); // 绕机体 z
      const yaw   = Math.atan2(2 * (x * z + w * y), 1 - 2 * (x * x + y * y)); // 绕机体 y
      return { roll: roll, pitch: pitch, yaw: yaw };
    }

    /* 由欧拉角重建四元数（用于 ±maxTilt 钳制） */
    _fromEuler(roll, pitch, yaw) {
      const cy = Math.cos(yaw / 2),  sy = Math.sin(yaw / 2);
      const cx = Math.cos(pitch / 2), sx = Math.sin(pitch / 2);
      const cz = Math.cos(roll / 2),  sz = Math.sin(roll / 2);
      // q = qy ⊗ qx ⊗ qz
      const qx = sx * cy * cz - cx * sy * sz, qy = cx * sy * cz + sx * cy * sz;
      const qz = cx * cy * sz - sx * sy * cz, qw = cx * cy * cz + sx * sy * sz;
      this.quat.x = qx; this.quat.y = qy; this.quat.z = qz; this.quat.w = qw;
    }

    /* 输入 update(dtSec, motors[4]) —— 4 个电机转速（原始量纲，悬停≈HOVER≈4.52）
     * 归一化口径 = motors[i]/HOVER（常用带 0..1.2，逃逸爆发 1.5） */
    update(dtSec, motors) {
      const dt = clamp(dtSec, 0, 0.033);                 // dt 钳制 ≤0.033s
      const mMax = 2 * DronePhysics.HOVER;
      const m = [0, 1, 2, 3].map(i => clamp((motors && motors[i]) || 0, 0, mMax));
      this.lastMotors = m;

      /* 总推力 T = kf * Σ(motors[i]^2)，沿机体 +y 轴（悬停需抵消重力；
       * 规格原文写"-y"指桨盘下洗方向，推力为反作用力，取 +y 才能过悬停校准） */
      const T = this.kf * (m[0] * m[0] + m[1] * m[1] + m[2] * m[2] + m[3] * m[3]);
      const T_world = quatRotate(this.quat, { x: 0, y: T, z: 0 });

      /* 力矩（X 型混控简化）：roll=(m0-m2) 绕机体 z，pitch=(m1-m3) 绕机体 x，yaw=km*(m0-m2+m1-m3) 绕机体 y */
      const tauRoll = (m[0] - m[2]);
      const tauPitch = (m[1] - m[3]);
      const tauYaw = this.km * (m[0] - m[2] + m[1] - m[3]);

      /* 平动：a = (T_body + F_drag)/m + g；F_drag = -dragLin*v - dragQuad*|v|*v */
      const v = this.vel;
      const sp = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
      const kq = this.dragQuad * sp;
      const Fx = T_world.x - this.dragLin * v.x - kq * v.x;
      const Fy = T_world.y - this.dragLin * v.y - kq * v.y - this.mass * this.gravity;
      const Fz = T_world.z - this.dragLin * v.z - kq * v.z;
      /* 半隐式欧拉：先更新 vel 再更新 pos */
      v.x += Fx / this.mass * dt;
      v.y += Fy / this.mass * dt;
      v.z += Fz / this.mass * dt;
      this.pos.x += v.x * dt;
      this.pos.y += v.y * dt;
      this.pos.z += v.z * dt;

      /* 姿态：角速度积分 + 四元数积分 q += 0.5*q⊗ω*dt，每步归一化 */
      const I = this.inertia;
      this.angVel.x += tauPitch / I.pitch * dt;
      this.angVel.y += tauYaw   / I.yaw   * dt;
      this.angVel.z += tauRoll  / I.roll  * dt;
      const q = this.quat, w = this.angVel;
      const dq = {
        x: 0.5 * ( w.x * q.w + w.y * q.z - w.z * q.y),
        y: 0.5 * ( w.y * q.w + w.z * q.x - w.x * q.z),
        z: 0.5 * ( w.z * q.w + w.x * q.y - w.y * q.x),
        w: 0.5 * (-w.x * q.x - w.y * q.y - w.z * q.z),
      };
      q.x += dq.x * dt; q.y += dq.y * dt; q.z += dq.z * dt; q.w += dq.w * dt;
      const n = Math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) || 1;
      q.x /= n; q.y /= n; q.z /= n; q.w /= n;

      /* 欧拉角钳制 ±maxTilt（roll/pitch，yaw 不钳） */
      const e = this.eulerAngles();
      const roll = clamp(e.roll, -this.maxTilt, this.maxTilt);
      const pitch = clamp(e.pitch, -this.maxTilt, this.maxTilt);
      if (roll !== e.roll || pitch !== e.pitch) this._fromEuler(roll, pitch, e.yaw);
      return this;
    }
  }

  /* 悬停转速（导出）：HOVER = sqrt(mass*gravity/(4*kf))，默认参数下 ≈4.5185 */
  DronePhysics.HOVER = Math.sqrt(1.0 * 9.8 / (4 * 0.12));
  DronePhysics.quatRotate = quatRotate;

  /* 自测：① 悬停 5s 漂移 <0.3 ② 匀加速 ③ 姿态响应 —— 返回 PASS/FAIL 字符串 */
  DronePhysics.selfTest = function () {
    const lines = []; let all = true;
    const chk = (ok, txt) => { all = all && ok; return txt + '[' + (ok ? 'PASS' : 'FAIL') + ']'; };

    /* ① 悬停 5s：4 电机=HOVER，位置漂移 <0.3 */
    const p1 = new DronePhysics();
    p1.pos.y = 10;
    const hover = DronePhysics.HOVER;
    for (let t = 0; t < 5; t += 0.033) p1.update(0.033, [hover, hover, hover, hover]);
    const drift = Math.sqrt(p1.pos.x * p1.pos.x + (p1.pos.y - 10) * (p1.pos.y - 10) + p1.pos.z * p1.pos.z);
    lines.push(chk(drift < 0.3, '悬停5s漂移=' + drift.toFixed(4) + '<0.3 '));

    /* ② 匀加速：关阻力、纯重力 1s，Δvy≈-g、Δy≈-g t²/2（半隐式欧拉容差 5%） */
    const p2 = new DronePhysics({ dragLin: 0, dragQuad: 0 });
    const v0 = p2.vel.y, y0 = p2.pos.y;
    for (let t = 0; t < 1; t += 0.01) p2.update(0.01, [0, 0, 0, 0]);
    const dv = p2.vel.y - v0, dy = p2.pos.y - y0;
    const okA = Math.abs(dv + 9.8) / 9.8 < 0.02 && Math.abs(dy + 4.9) / 4.9 < 0.05;
    lines.push(chk(okA, '匀加速 Δvy=' + dv.toFixed(3) + ' Δy=' + dy.toFixed(3) + ' '));

    /* ③ 姿态响应：roll 差速 0.2s 产生滚转角；压满 3s 不超 ±maxTilt */
    const p3 = new DronePhysics();
    for (let t = 0; t < 0.2; t += 0.01) p3.update(0.01, [hover + 0.1, hover, hover - 0.1, hover]);
    const roll1 = p3.eulerAngles().roll * 180 / Math.PI;
    for (let t = 0; t < 3; t += 0.033) p3.update(0.033, [hover * 1.5, hover, hover * 0.2, hover]);
    const roll2 = p3.eulerAngles().roll * 180 / Math.PI;
    const okB = Math.abs(roll1) > 0.5 && Math.abs(roll2) <= 40.2;
    lines.push(chk(okB, '姿态响应 roll=' + roll1.toFixed(2) + '°→' + roll2.toFixed(2) + '°(≤40°) '));

    return 'DronePhysics.selfTest ⇒ ' + lines.join('｜') + '｜总评 ' + (all ? 'PASS' : 'FAIL');
  };

  return DronePhysics;
});
