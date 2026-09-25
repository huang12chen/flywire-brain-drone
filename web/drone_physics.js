/* =====================================================================
 * web\drone_physics.js —— Phase 1: Quadrotor flight dynamics (demo-level, not calibrated to real hardware)
 * Regular <script>, attaches to window.DronePhysics; Node module.exports (for Phase 4 smoke_drone.js)
 * Zero dependencies (no THREE, zero fetch, zero ESM): pos/vel/angVel are {x,y,z}, quat is {x,y,z,w},
 *   fields are compatible with THREE.Vector3/Quaternion, world.html can directly .set/.copy to interface.
 * Units: world units + seconds (update takes dtSec); parameters are demo-level values, marked [NOT CALIBRATED TO REAL HARDWARE].
 * ===================================================================== */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.DronePhysics = api;
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this), function () {
  'use strict';
  const DEG = Math.PI / 180;
  const clamp = (v, lo, hi) => v < lo ? lo : (v > hi ? hi : v);

  /* Quaternion rotate vector: v' = v + 2w(qv×v) + 2(qv×(qv×v)) */
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
      /* Parameters (demo-level, world units, [NOT CALIBRATED TO REAL HARDWARE]) */
      this.mass     = opts.mass     !== undefined ? opts.mass     : 1.0;    // Mass
      this.gravity  = opts.gravity  !== undefined ? opts.gravity  : 9.8;    // Gravitational acceleration
      this.kf       = opts.kf       !== undefined ? opts.kf       : 0.12;   // Thrust coefficient
      this.km       = opts.km       !== undefined ? opts.km       : 0.018;  // Anti-torque coefficient
      this.dragLin  = opts.dragLin  !== undefined ? opts.dragLin  : 0.35;   // Linear drag
      this.dragQuad = opts.dragQuad !== undefined ? opts.dragQuad : 0.02;   // Quadratic drag
      this.maxTilt  = (opts.maxTilt !== undefined ? opts.maxTilt : 40) * DEG; // Attitude Euler angle clamp ±40°
      this.inertia  = opts.inertia  !== undefined ? opts.inertia  : { roll: 0.02, pitch: 0.02, yaw: 0.04 };
      /* State: pos/vel/angVel ({x,y,z}), quat ({x,y,z,w}) */
      this.pos    = { x: 0, y: 0, z: 0 };
      this.vel    = { x: 0, y: 0, z: 0 };
      this.quat   = { x: 0, y: 0, z: 0, w: 1 };
      this.angVel = { x: 0, y: 0, z: 0 };
      this.lastMotors = [0, 0, 0, 0];
    }

    /* Hover calibration: hoverOmega = sqrt(mass*gravity/(4*kf)) */
    hoverOmega() { return Math.sqrt(this.mass * this.gravity / (4 * this.kf)); }

    /* Euler angles (convention q = qy(yaw)·qx(pitch)·qz(roll), body frame x=right y=up z=back) */
    eulerAngles() {
      const q = this.quat, x = q.x, y = q.y, z = q.z, w = q.w;
      const pitch = Math.asin(clamp(2 * (w * x - y * z), -1, 1));            // Around body x
      const roll  = Math.atan2(2 * (x * y + w * z), 1 - 2 * (x * x + z * z)); // Around body z
      const yaw   = Math.atan2(2 * (x * z + w * y), 1 - 2 * (x * x + y * y)); // Around body y
      return { roll: roll, pitch: pitch, yaw: yaw };
    }

    /* Rebuild quaternion from Euler angles (used for ±maxTilt clamping) */
    _fromEuler(roll, pitch, yaw) {
      const cy = Math.cos(yaw / 2),  sy = Math.sin(yaw / 2);
      const cx = Math.cos(pitch / 2), sx = Math.sin(pitch / 2);
      const cz = Math.cos(roll / 2),  sz = Math.sin(roll / 2);
      // q = qy ⊗ qx ⊗ qz
      const qx = sx * cy * cz - cx * sy * sz, qy = cx * sy * cz + sx * cy * sz;
      const qz = cx * cy * sz - sx * sy * cz, qw = cx * cy * cz + sx * sy * sz;
      this.quat.x = qx; this.quat.y = qy; this.quat.z = qz; this.quat.w = qw;
    }

    /* Input update(dtSec, motors[4]) — 4 motor speeds (raw dimensions, hover ≈ HOVER ≈ 4.52)
     * Normalized form = motors[i]/HOVER (commonly ranges 0..1.2, escape burst 1.5) */
    update(dtSec, motors) {
      const dt = clamp(dtSec, 0, 0.033);                 // dt clamp ≤0.033s
      const mMax = 2 * DronePhysics.HOVER;
      const m = [0, 1, 2, 3].map(i => clamp((motors && motors[i]) || 0, 0, mMax));
      this.lastMotors = m;

      /* Total thrust T = kf * Σ(motors[i]^2), along body +y axis (hover must counteract gravity;
       * Spec originally says "-y" referring to rotor downwash direction, thrust is the reaction force, taking +y passes hover calibration) */
      const T = this.kf * (m[0] * m[0] + m[1] * m[1] + m[2] * m[2] + m[3] * m[3]);
      const T_world = quatRotate(this.quat, { x: 0, y: T, z: 0 });

      /* Torque (X-configuration mix simplified): roll=(m0-m2) around body z, pitch=(m1-m3) around body x, yaw=km*(m0-m2+m1-m3) around body y */
      const tauRoll = (m[0] - m[2]);
      const tauPitch = (m[1] - m[3]);
      const tauYaw = this.km * (m[0] - m[2] + m[1] - m[3]);

      /* Translation: a = (T_body + F_drag)/m + g; F_drag = -dragLin*v - dragQuad*|v|*v */
      const v = this.vel;
      const sp = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
      const kq = this.dragQuad * sp;
      const Fx = T_world.x - this.dragLin * v.x - kq * v.x;
      const Fy = T_world.y - this.dragLin * v.y - kq * v.y - this.mass * this.gravity;
      const Fz = T_world.z - this.dragLin * v.z - kq * v.z;
      /* Semi-implicit Euler: update vel first, then update pos */
      v.x += Fx / this.mass * dt;
      v.y += Fy / this.mass * dt;
      v.z += Fz / this.mass * dt;
      this.pos.x += v.x * dt;
      this.pos.y += v.y * dt;
      this.pos.z += v.z * dt;

      /* Attitude: angular velocity integration + quaternion integration q += 0.5*q⊗ω*dt, normalize each step */
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

      /* Euler angle clamp ±maxTilt (roll/pitch, yaw not clamped) */
      const e = this.eulerAngles();
      const roll = clamp(e.roll, -this.maxTilt, this.maxTilt);
      const pitch = clamp(e.pitch, -this.maxTilt, this.maxTilt);
      if (roll !== e.roll || pitch !== e.pitch) this._fromEuler(roll, pitch, e.yaw);
      return this;
    }
  }

  /* Hover speed (exported): HOVER = sqrt(mass*gravity/(4*kf)), defaults to ≈4.5185 */
  DronePhysics.HOVER = Math.sqrt(1.0 * 9.8 / (4 * 0.12));
  DronePhysics.quatRotate = quatRotate;

  /* Self-test: (1) Hover 5s drift <0.3 (2) Uniform acceleration (3) Attitude response — returns PASS/FAIL string */
  DronePhysics.selfTest = function () {
    const lines = []; let all = true;
    const chk = (ok, txt) => { all = all && ok; return txt + '[' + (ok ? 'PASS' : 'FAIL') + ']'; };

    /* (1) Hover 5s: 4 motors=HOVER, position drift <0.3 */
    const p1 = new DronePhysics();
    p1.pos.y = 10;
    const hover = DronePhysics.HOVER;
    for (let t = 0; t < 5; t += 0.033) p1.update(0.033, [hover, hover, hover, hover]);
    const drift = Math.sqrt(p1.pos.x * p1.pos.x + (p1.pos.y - 10) * (p1.pos.y - 10) + p1.pos.z * p1.pos.z);
    lines.push(chk(drift < 0.3, 'Hover 5s drift=' + drift.toFixed(4) + '<0.3 '));

    /* (2) Uniform acceleration: drag off, pure gravity 1s, Δvy≈-g, Δy≈-g t²/2 (semi-implicit Euler tolerance 5%) */
    const p2 = new DronePhysics({ dragLin: 0, dragQuad: 0 });
    const v0 = p2.vel.y, y0 = p2.pos.y;
    for (let t = 0; t < 1; t += 0.01) p2.update(0.01, [0, 0, 0, 0]);
    const dv = p2.vel.y - v0, dy = p2.pos.y - y0;
    const okA = Math.abs(dv + 9.8) / 9.8 < 0.02 && Math.abs(dy + 4.9) / 4.9 < 0.05;
    lines.push(chk(okA, 'Uniform accel Δvy=' + dv.toFixed(3) + ' Δy=' + dy.toFixed(3) + ' '));

    /* (3) Attitude response: roll differential 0.2s produces roll angle; full deflection 3s does not exceed ±maxTilt */
    const p3 = new DronePhysics();
    for (let t = 0; t < 0.2; t += 0.01) p3.update(0.01, [hover + 0.1, hover, hover - 0.1, hover]);
    const roll1 = p3.eulerAngles().roll * 180 / Math.PI;
    for (let t = 0; t < 3; t += 0.033) p3.update(0.033, [hover * 1.5, hover, hover * 0.2, hover]);
    const roll2 = p3.eulerAngles().roll * 180 / Math.PI;
    const okB = Math.abs(roll1) > 0.5 && Math.abs(roll2) <= 40.2;
    lines.push(chk(okB, 'Attitude response roll=' + roll1.toFixed(2) + '°→' + roll2.toFixed(2) + '°(≤40°) '));

    return 'DronePhysics.selfTest ⇒ ' + lines.join('｜') + '｜Overall ' + (all ? 'PASS' : 'FAIL');
  };

  return DronePhysics;
});
