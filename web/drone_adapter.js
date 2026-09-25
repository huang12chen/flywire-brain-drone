/* =====================================================================
 * web\drone_adapter.js —— Phase 2: Brain → Motor adapter layer
 * Bionic hierarchical: Drosophila brain (decision) → this adapter (thoracic ganglia) → motors (flight muscles); PID attitude loop = haltere reflex
 * Plain <script>, attaches window.DroneAdapter; Node module.exports (for phase 4 smoke_drone.js)
 * Zero dependencies (zero fetch, zero ESM); rng injected externally to ensure determinism.
 * ===================================================================== */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.DroneAdapter = api;
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this), function () {
  'use strict';
  const DEG = Math.PI / 180;
  const clamp = (v, lo, hi) => v < lo ? lo : (v > hi ? hi : v);

  /* Get DronePhysics (browsers read window by script order; Node uses require), only uses its HOVER */
  const DronePhysics = (typeof module === 'object' && module.exports)
    ? require('./drone_physics.js')
    : (typeof window !== 'undefined' ? window.DronePhysics : null);
  const HOVER = DronePhysics ? DronePhysics.HOVER : Math.sqrt(1.0 * 9.8 / (4 * 0.12));  // Hover RPM ≈4.5185

  /* State machine: CRUISE / ESCAPE / PILOT / LAND */
  const CN = { CRUISE: 'Cruise', ESCAPE: 'Escape', PILOT: 'Pilot', LAND: 'Land' };

  class DroneAdapter {
    constructor(opts) {
      opts = opts || {};
      /* PID attitude loop: kp=0.045, ki=0.001, kd=0.028 (error = desired attitude - current attitude, output added to motors) */
      this.kp = opts.kp !== undefined ? opts.kp : 0.045;
      this.ki = opts.ki !== undefined ? opts.ki : 0.001;
      this.kd = opts.kd !== undefined ? opts.kd : 0.028;
      this.rng = opts.rng || Math.random;          // Inject rng to ensure determinism
      this.escapeMs = 300;                          // ESCAPE lasts 300ms (GF burst)
      this.landY = opts.landY !== undefined ? opts.landY : 0;   // Landing threshold baseline (world passes groundY)
      /* Internal state */
      this.state = 'CRUISE';
      this.lastState = 'CRUISE';                    // Always exported (for HUD use)
      this.lastStateCN = CN.CRUISE;
      this.lastMotors = [0, 0, 0, 0];
      this.tMs = 0;
      this.escapeUntil = -1e9; this.lastEscapeEnd = -1e9;
      this.escapeVec = null;                        // Escape direction (world (x,y,z), from brain output after axis swap)
      this.landRequested = false; this.landed = false; this.throttle = 1.02;
      this.yawTarget = 0; this.nextHeadingT = 0;
      this.tiltTarget = { roll: 0, pitch: 0 };
      this.iErr = { roll: 0, pitch: 0, yaw: 0 };
      this.prevErr = { roll: 0, pitch: 0, yaw: 0 };
      this.hoverOmega = HOVER;
    }

    /* ---------- External commands (phase 3 for buttons/battery) ---------- */
    enterLand() { this.landRequested = true; this.landed = false; if (this.state !== 'ESCAPE') this.state = 'LAND'; }
    enterCruise() { this.landRequested = false; this.landed = false; this.throttle = 1.02; if (this.state !== 'ESCAPE') this.state = 'CRUISE'; }
    /* Phase 3 cruise obstacle avoidance: yaw away from obstacles (turn only, no escape trigger) */
    setYawTarget(yaw) { this.yawTarget = yaw; this.nextHeadingT = this.tMs + 2000 + this.rng() * 2000; }

    /* Mixing matrix (X-frame simplified, each term explicit):
     *   base = HOVER * throttle
     *   m0 = base + HOVER*(+rollCmd + yawCmd)    Front-left (diagonal 1)
     *   m1 = base + HOVER*(+pitchCmd + yawCmd)   Front-right (diagonal 2)
     *   m2 = base + HOVER*(-rollCmd - yawCmd)    Rear-right (diagonal 1)
     *   m3 = base + HOVER*(-pitchCmd - yawCmd)   Rear-left (diagonal 2)
     * → Dynamics end (specification formula): roll torque = (m0-m2) = 2*HOVER*(rollCmd+yawCmd)
     *   pitch torque = (m1-m3) = 2*HOVER*(pitchCmd+yawCmd)
     *   yaw torque = km*(m0-m2+m1-m3) = 2*km*HOVER*(rollCmd+pitchCmd+2*yawCmd)
     * In the simplified spec mixing, the yaw channel couples with roll/pitch, absorbed by the PID closed loop (noted). */
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
     * brainOut: SNN output (triggered/escapeDir); pass null when no threat or pilot active
     * droneState: { pos, vel, roll, pitch, yaw, groundY } (roll/pitch/yaw in radians; vel = world units/sec) */
    update(brainOut, s, dtMs, pilotActive) {
      const dt = clamp(dtMs, 0, 50);
      this.tMs += dt;
      s = s || { pos: { x: 0, y: 0, z: 0 }, vel: { x: 0, y: 0, z: 0 }, roll: 0, pitch: 0, yaw: 0, groundY: 0 };

      /* PILOT has highest priority: pilot takes over (existing non-modifier key logic in world.html) → legacy kinematics takes over immediately, this layer bypassed */
      if (pilotActive) {
        this.state = 'PILOT'; this.iErr = { roll: 0, pitch: 0, yaw: 0 };
        return this._commit([0, 0, 0, 0]);
      }
      if (this.state === 'PILOT') this.state = this.landRequested ? 'LAND' : 'CRUISE';

      /* ESCAPE: triggered by brainOut.triggered==true, lasts 300ms */
      if (brainOut && brainOut.triggered && this.state !== 'ESCAPE' && !this.landRequested
          && this.tMs - this.lastEscapeEnd > 150) {
        this.state = 'ESCAPE';
        this.escapeUntil = this.tMs + this.escapeMs;
        /* escapeDir network frame (0,1,2)=(x,z,y) world components → world (x,y,z) (swap y/z, convention unchanged) */
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

      /* —— Throttle + desired attitude for each state —— */
      let throttle = 1.02, desRoll = 0, desPitch = 0, desYaw = s.yaw;
      if (this.state === 'ESCAPE') {
        /* Throttle = HOVER*1.5 (step, simulating GF burst)
         * Target tilt angle = escapeDir projected onto horizontal plane, tilt 35°, roll/pitch components each clamped ±35° */
        throttle = 1.5;
        const a35 = 35 * DEG, si = Math.sin(a35);
        const h = this.escapeVec || { x: 0, z: 0 };
        const dx = h.x * si, dz = h.z * si;
        const cy = Math.cos(s.yaw), sy = Math.sin(s.yaw);
        let th = sy * dx + cy * dz;    // Desired pitch (world escape direction → body frame)
        let ph = -cy * dx + sy * dz;   // Desired roll
        desPitch = clamp(th, -a35, a35);
        desRoll = clamp(ph, -a35, a35);
      } else if (this.state === 'LAND') {
        /* Landing: gradually reduce throttle to HOVER*0.8 until touchdown (pos.y < threshold) → throttle 0 */
        this.throttle = Math.max(0.8, this.throttle - dt * 0.00012);
        throttle = this.throttle;
        const gy = (s.groundY !== undefined ? s.groundY : this.landY);
        if (s.pos.y <= gy + 0.5) { this.landed = true; throttle = 0; }
        desRoll = this.tiltTarget.roll; desPitch = this.tiltTarget.pitch;
      } else {
        /* CRUISE: throttle=HOVER*1.02, desired attitude=slight random walk (change target heading every 2-4s, angular velocity ≤15°/s) */
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
        desYaw = s.yaw + clamp(dy, -15 * DEG * dt / 1000, 15 * DEG * dt / 1000);  // Heading angular velocity ≤15°/s
        desRoll = this.tiltTarget.roll; desPitch = this.tiltTarget.pitch;
      }

      /* PID attitude loop: error = desired attitude - current attitude, output added to motors (dt in seconds, units consistent) */
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

  /* Deterministic random source (mulberry32), injected by world.html / self-tests */
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
