/* snn_runtime.js — Browser inference engine for the trained fruit fly escape SNN
 * Line-by-line corresponds to the forward computation in train_snn.py (LIF, dt=1ms, soft reset, hard threshold firing)
 * Input: visual expansion rate looming (rad/s) + wind vector wind (cm/ms) + threat azimuth az (rad)
 * Output: whether GF fires, first spike latency, escape direction (3D unit vector)
 *
 * [v5 - visualization bypass recording, numerical values unchanged bit-for-bit] simulate() 4th optional parameter rec:
 *   When passed a plain object, step-by-step read-only snapshots of each group's spikes and GF membrane potential (rec.steps[t] =
 *   {vis:[],wind:[],hid:[],gf:[],out:[],gfV:[]}, plus rec.vth/rec.T/rec.firstStep).
 *   Recording only reads s/v values into independent buffers, never participates in any floating-point computation path/sequence/type;
 *   When rec is not passed, the execution path is identical to v3 (unchanged bit-for-bit). For world.html "fruit fly brain" panel rendering only.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.SNNRuntime = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var RATE_MAX_HZ = 200.0, BASE_RATE_HZ = 5.0, WIND_HALF_SAT = 0.012;

  function SNNRuntime(data) {
    this.meta = data.meta;
    this.n = data.num_nodes;
    this.beta = data.meta.beta;
    this.vth = data.meta.threshold;
    this.T = data.meta.t_steps;

    var E = data.edges.length;
    this.src = new Int32Array(E);
    this.dst = new Int32Array(E);
    this.w = new Float32Array(E);
    for (var e = 0; e < E; e++) {
      this.src[e] = data.edges[e][0];
      this.dst[e] = data.edges[e][1];
      this.w[e] = data.edges[e][2];
    }

    this.inVision = Int32Array.from(data.input_vision_indices);
    this.inWind = Int32Array.from(data.input_wind_indices);
    this.hub = Int32Array.from(data.hub_gf_indices);
    this.out = Int32Array.from(data.output_indices);
    this.inGain = Float32Array.from(data.in_gain);

    this.trigW = Float32Array.from(data.head_trig.w);
    this.trigB = data.head_trig.b;
    this.dirW = data.head_dir.w.map(function (r) { return Float32Array.from(r); });
    this.dirB = Float32Array.from(data.head_dir.b);

    // Sensor encoding parameters (identical to make_pref on training side)
    this.pdPrefAz = Float32Array.from(data.pd_pref.map(function (r) { return r[0]; }));
    this.windPref = data.wind_pref.map(function (r) { return Float32Array.from(r); });

    this._v = new Float32Array(this.n);
    this._s = new Float32Array(this.n);
    this._cur = new Float32Array(this.n);
  }

  /* Poisson firing rate encoding — corresponds to train_snn.encode_spikes */
  SNNRuntime.prototype.encode = function (looming, windVec, threatAz) {
    var nVis = this.inVision.length, nIn = nVis + this.inWind.length;
    var loomN = Math.min(1, looming / (looming + 2.0));          // min(1,·) clamping, consistent with train:140
    var wm = Math.hypot(windVec[0], windVec[1], windVec[2]);
    var windN = Math.min(1, wm / (wm + WIND_HALF_SAT));          // min(1,·) clamping, consistent with train:142
    var rates = new Float32Array(nIn);
    var i;
    for (i = 0; i < nVis; i++) {
      var cv = Math.cos(this.pdPrefAz[i] - threatAz);
      rates[i] = BASE_RATE_HZ + loomN * Math.max(0, cv) * RATE_MAX_HZ * 0.8;
    }
    if (wm > 1e-9) {
      var wx = windVec[0] / wm, wy = windVec[1] / wm, wz = windVec[2] / wm;
      for (i = 0; i < this.inWind.length; i++) {
        var p = this.windPref[i], proj = p[0] * wx + p[1] * wy + p[2] * wz;
        rates[nVis + i] = BASE_RATE_HZ + windN * Math.max(0, proj) * RATE_MAX_HZ * 0.8;
      }
    } else {
      for (i = 0; i < this.inWind.length; i++) rates[nVis + i] = BASE_RATE_HZ;
    }
    // Ablation mode: consistent with training-side controlled experiments (disable all spikes for a channel)
    if (this.mode === 'vision_only') for (i = nVis; i < nIn; i++) rates[i] = 0;
    if (this.mode === 'wind_only') for (i = 0; i < nVis; i++) rates[i] = 0;

    // Generate [T][N_in] 0/1 spikes
    var sp = [];
    for (var t = 0; t < this.T; t++) {
      var row = new Float32Array(nIn);
      // [Patch - injectable random source] After construction, can set rt.rng = fn (for deterministic experiments); defaults to Math.random if not set, behavior completely unchanged
      for (var k = 0; k < nIn; k++) row[k] = (this.rng || Math.random)() < Math.min(0.9, rates[k] * 0.001) ? 1 : 0;   // p<=0.9 clamping, consistent with train:157
      sp.push(row);
    }
    return sp;
  };

  /* Forward simulation — corresponds to EscapeSNN.forward
   * rec (optional): visualization bypass recorder; only collects read-only state snapshots, does not participate in any numerical computation */
  SNNRuntime.prototype.simulate = function (looming, windVec, threatAz, rec) {
    var spikes = this.encode(looming, windVec, threatAz);
    var n = this.n, T = this.T, E = this.src.length;
    var v = this._v, s = this._s, cur = this._cur;
    v.fill(0); s.fill(0);
    var nHub = this.hub.length, nOut = this.out.length;
    var vHubSum = new Float32Array(nHub), vOutSum = new Float32Array(nOut);
    var firstStep = T, hubSpk = 0, vHubMax = -1e9, spikeCount = 0;
    var nVis = this.inVision.length;
    if (rec) { rec.steps = []; rec.vth = this.vth; rec.T = T; }

    for (var t = 0; t < T; t++) {
      cur.fill(0);
      for (var e = 0; e < E; e++) cur[this.dst[e]] += this.w[e] * s[this.src[e]];
      var row = spikes[t];
      for (var i = 0; i < nVis; i++) cur[this.inVision[i]] += row[i] * this.inGain[i];
      for (var j = 0; j < this.inWind.length; j++) cur[this.inWind[j]] += row[nVis + j] * this.inGain[nVis + j];

      for (var q = 0; q < n; q++) {
        v[q] = this.beta * v[q] + cur[q] - this.vth * s[q];
        s[q] = (v[q] - this.vth) > 0 ? 1 : 0;   // boundary >0 consistent with snntorch FastSigmoid
        spikeCount += s[q];
      }
      var fired = false;
      for (var h = 0; h < nHub; h++) {
        vHubSum[h] += v[this.hub[h]];
        if (v[this.hub[h]] > vHubMax) vHubMax = v[this.hub[h]];
        if (s[this.hub[h]] > 0) { fired = true; hubSpk++; }
      }
      if (fired && firstStep === T) firstStep = t + 1;
      for (var o = 0; o < nOut; o++) vOutSum[o] += v[this.out[o]];
      // —— Visualization bypass recording (only collected when rec is passed; read-only s/v snapshots, does not affect any numerical values above) ——
      if (rec) {
        var st = { vis: [], wind: [], hid: [], gf: [], out: [], gfV: [] }, z;
        for (z = 0; z < nVis; z++) if (s[this.inVision[z]] > 0) st.vis.push(this.inVision[z]);
        for (z = 0; z < this.inWind.length; z++) if (s[this.inWind[z]] > 0) st.wind.push(this.inWind[z]);
        for (z = 0; z < nHub; z++) { st.gf.push(s[this.hub[z]] > 0 ? 1 : 0); st.gfV.push(v[this.hub[z]]); }
        for (z = 0; z < nOut; z++) if (s[this.out[z]] > 0) st.out.push(this.out[z]);
        for (z = 0; z < n; z++) if (s[z] > 0) st.hid.push(z);   // Full firing snapshot (includes all groups, panel groups them on its own)
        rec.steps.push(st);
      }
    }

    var trigLogit = this.trigB;
    for (var a = 0; a < nHub; a++) trigLogit += this.trigW[a] * (vHubSum[a] / T);
    var dir = [this.dirB[0], this.dirB[1], this.dirB[2]];
    for (var b = 0; b < 3; b++) {
      for (var c = 0; c < nOut; c++) dir[b] += this.dirW[b][c] * (vOutSum[c] / T);
    }
    var dl = Math.hypot(dir[0], dir[1], dir[2]) + 1e-8;   // Normalization consistent with train:342 (norm + 1e-8)
    dir[0] /= dl; dir[1] /= dl; dir[2] /= dl;
    if (rec) rec.firstStep = firstStep;

    return {
      triggered: firstStep < T,               // GF firing = escape triggered (mechanistic criterion)
      firstSpikeMs: firstStep < T ? firstStep : null,
      triggerProb: 1 / (1 + Math.exp(-trigLogit)),
      escapeDir: dir,                          // Escape direction (away from threat)
      vHubMax: vHubMax,
      gfSpikes: hubSpk,
      spikeCount: spikeCount / (n * T)
    };
  };

  return SNNRuntime;
});
