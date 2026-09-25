/* brain_panel.js —— 🧠 Drosophila Brain Panel (real connectome circuit map + decision window spike conduction + GF membrane potential curve)
 * Data source: window.SNN_DATA (FlyWire topology-constrained real connectome: 1871 neurons / 45524 edges)
 *   LPLC2 vision 210 → JO wind-sense 325 → (interneurons 1165) → GF giant fiber 2 → downstream 169
 * Rendering: pure canvas 2D, zero fetch / zero ES module / zero networking, works by double-clicking file://
 * Input: bypass recording from snn_runtime.simulate(..., rec) (read-only snapshot, does not affect SNN values).
 * Concept inspired by snedea/flybrain "real-time spike panel" (dot matrix + light pulse conduction + potential curve), independently rewritten.
 */
(function (root) {
  'use strict';
  var D = root.SNN_DATA;
  if (!D) { console.error('brain_panel: SNN_DATA not loaded'); return; }

  /* ---------- Grouping (real index sets, mutually exclusive) ---------- */
  var N = D.num_nodes;
  var grp = new Int8Array(N);                 // 0=hid 1=vis 2=wind 3=gf 4=out
  D.input_vision_indices.forEach(function (i) { grp[i] = 1; });
  D.input_wind_indices.forEach(function (i) { grp[i] = 2; });
  D.hub_gf_indices.forEach(function (i) { grp[i] = 3; });
  D.output_indices.forEach(function (i) { grp[i] = 4; });

  // LPLC2: sorted by preferred azimuth angle (dot matrix = azimuth sectors, making azimuth codes visible)
  var visOrder = D.input_vision_indices.map(function (id, k) { return { id: id, key: D.pd_pref[k][0] }; })
    .sort(function (a, b) { return a.key - b.key; }).map(function (o) { return o.id; });
  // JO: sorted by sensitive axis azimuth angle
  var windOrder = D.input_wind_indices.map(function (id, k) {
    var p = D.wind_pref[k]; return { id: id, key: Math.atan2(p[1], p[0]) };
  }).sort(function (a, b) { return a.key - b.key; }).map(function (o) { return o.id; });
  var outOrder = D.output_indices.slice();
  // Interneurons 1165: uniform sampling of 128 representative points (fits on screen, labels are clear)
  var hidAll = [];
  for (var i = 0; i < N; i++) if (grp[i] === 0) hidAll.push(i);
  var hidShow = [];
  for (var hi = 0; hi < 128; hi++) hidShow.push(hidAll[Math.floor(hi * hidAll.length / 128)]);

  /* ---------- Real connectome: inter-group aggregation (edge count / weight sum) ---------- */
  var GNAME = ['Interneurons', 'LPLC2', 'JO', 'GF', 'Downstream'];
  var flow = {};                              // 'a->b' => {n, w}
  D.edges.forEach(function (e) {
    var k = grp[e[0]] + '>' + grp[e[1]];
    var f = flow[k] || (flow[k] = { n: 0, w: 0 });
    f.n++; f.w += e[2];
  });

  /* ---------- Layout (canvas 640×470) ---------- */
  var W = 640, H = 470;
  function grid(n, cols, x0, y0, gap, r) {
    var pts = [];
    for (var i = 0; i < n; i++) pts.push({ x: x0 + (i % cols) * gap, y: y0 + Math.floor(i / cols) * gap, r: r, id: null });
    return pts;
  }
  var visPts = grid(210, 15, 34, 66, 7.6, 2.3);            // 15×14
  var windPts = grid(325, 25, 34, 246, 7.0, 2.1);          // 25×13
  var hidPts = grid(128, 8, 262, 130, 8.4, 2.2);           // 8×16 vertical
  var gfPts = [{ x: 408, y: 158, r: 9 }, { x: 408, y: 205, r: 9 }];
  var outPts = grid(169, 13, 502, 116, 7.4, 2.2);          // 13×13
  visPts.forEach(function (p, i) { p.id = visOrder[i]; });
  windPts.forEach(function (p, i) { p.id = windOrder[i]; });
  hidPts.forEach(function (p, i) { p.id = hidShow[i]; });
  outPts.forEach(function (p, i) { p.id = outOrder[i]; });

  // Dot matrix fast lookup: id -> (point object)
  var ptOf = {};
  [visPts, windPts, hidPts, gfPts, outPts].forEach(function (arr) {
    arr.forEach(function (p) { if (p.id != null) ptOf[p.id] = p; });
  });

  /* ---------- Flow ribbons (real edge aggregation; a,b are group numbers, cp is control point) ---------- */
  var FLOWS = [
    { a: 1, b: 0, sx: 155, sy: 120, tx: 262, ty: 165, cx: 200, cy: 110, label: 'LPLC2→Inter' },
    { a: 2, b: 0, sx: 155, sy: 285, tx: 262, ty: 205, cx: 200, cy: 275, label: 'JO→Inter' },
    { a: 0, b: 3, sx: 330, sy: 180, tx: 399, ty: 182, cx: 365, cy: 150, label: 'Inter→GF' },
    { a: 1, b: 3, sx: 155, sy: 92, tx: 400, ty: 150, cx: 285, cy: 40, label: 'LPLC2→GF direct', dash: true },
    { a: 2, b: 3, sx: 155, sy: 300, tx: 400, ty: 212, cx: 285, cy: 322, label: 'JO→GF direct', dash: true },
    { a: 3, b: 4, sx: 418, sy: 182, tx: 502, ty: 170, cx: 462, cy: 140, label: 'GF→Downstream' },
    { a: 0, b: 4, sx: 330, sy: 215, tx: 502, ty: 215, cx: 420, cy: 268, label: 'Inter→Downstream' },
  ];

  /* ---------- Replay state ---------- */
  var view = { pend: null, cur: null, t: 0, lastTick: 0, holding: 0, lastRecAt: 0 };
  var STEP_MS = 130, HOLD_MS = 750;
  var glow = new Float64Array(N);             // Afterglow (display only, unrelated to SNN)

  var Panel = {
    open: false,
    beginRec: function () { return this.open ? { steps: [] } : null; },   // Bypass recorder (not passed = zero changes)
    show: function (rec, out, cues) {
      view.pend = { rec: rec, out: out, cues: cues, at: performance.now() };
      view.lastRecAt = view.pend.at;
    },
    toggle: function () {
      this.open = !this.open;
      var el = document.getElementById('brain');
      if (el) el.style.display = this.open ? 'block' : 'none';
      var b = document.getElementById('bBrain');
      if (b) { b.classList.toggle('on', this.open); b.textContent = this.open ? '🧠 Brain Panel: ON' : '🧠 Drosophila Brain Panel'; }
      if (this.open && !this._raf) this._loop();
    },
  };

  function groupFired(st, g) {
    // st.hid is the full spike snapshot; count by group
    var ids = st ? st.hid : [], n = 0;
    for (var i = 0; i < ids.length; i++) if (grp[ids[i]] === g) n++;
    return n;
  }

  /* ---------- Drawing ---------- */
  function draw(now) {
    var cv = document.getElementById('brainCanvas');
    if (!cv) return;
    var g = cv.getContext('2d');
    g.clearRect(0, 0, W, H);

    var cur = view.cur, steps = cur ? cur.rec.steps : [];
    var T = cur ? cur.rec.T : 12, vth = cur ? cur.rec.vth : 1.0;
    var t = Math.min(view.t, steps.length - 1);
    var st = steps[t] || null, stPrev = steps[t - 1] || null;
    if (st) for (var q = 0; q < st.hid.length; q++) glow[st.hid[q]] = 1;

    // —— Title status ——
    g.font = '12px "Microsoft YaHei", sans-serif';
    g.fillStyle = '#8fd2ff';
    var trig = cur && cur.out && cur.out.triggered;
    g.fillText('Decision window replay t = ' + (t + 1) + '/' + T + ' ms' + (trig ? '　⚡ GF fired! Escape' : '　Resting'),
      16, 20);
    if (cur && cur.cues) {
      g.fillStyle = '#8aa0b8';
      g.fillText('loom ' + cur.cues.loom.toFixed(2) + ' rad/s　|u| ' +
        Math.hypot(cur.cues.wind.x, cur.cues.wind.y, cur.cues.wind.z).toFixed(4), 16, 38);
    } else {
      g.fillStyle = '#8aa0b8';
      g.fillText('Waiting for threat decision… (GF decision window will replay in real-time after threat is deployed)', 16, 38);
    }

    // —— Group labels ——
    g.fillStyle = '#ffd479';
    g.fillText('LPLC2 Vision 210', 34, 60);
    g.fillText('JO Wind-sense 325', 34, 240);
    g.fillStyle = '#c9b8ff';
    g.fillText('Interneurons 1165 (sampled 128)', 244, 118);
    g.fillStyle = '#ffd479';
    g.fillText('GF ×2', 386, 138);
    g.fillText('Downstream 169', 502, 110);

    // —— Flow ribbons (real edge count label + activity particles) ——
    var active = [];
    if (st) active = st.hid;
    FLOWS.forEach(function (f) {
      var meta = flow[f.a + '>' + f.b];
      var srcHot = stPrev ? groupFired(stPrev, f.a) : 0;
      var dstHot = st ? groupFired(st, f.b) : 0;
      var hot = (srcHot + dstHot) > 0;
      g.beginPath();
      g.moveTo(f.sx, f.sy);
      g.quadraticCurveTo(f.cx, f.cy, f.tx, f.ty);
      g.setLineDash(f.dash ? [3, 4] : []);
      g.strokeStyle = hot ? 'rgba(110,231,160,.55)' : 'rgba(53,80,110,.45)';
      g.lineWidth = hot ? 2.2 : 1.2;
      g.stroke();
      g.setLineDash([]);
      // Particles along ribbon (visual beat = replay step)
      if (hot) {
        for (var pi = 0; pi < 3; pi++) {
          var u = ((now * 0.0012) + pi * 0.33) % 1;
          var mx = (1 - u) * (1 - u) * f.sx + 2 * (1 - u) * u * f.cx + u * u * f.tx;
          var my = (1 - u) * (1 - u) * f.sy + 2 * (1 - u) * u * f.cy + u * u * f.ty;
          g.beginPath(); g.arc(mx, my, 1.8, 0, 6.2832);
          g.fillStyle = '#6ee7a0'; g.fill();
        }
      }
      if (meta) {
        g.font = '10px "Microsoft YaHei", sans-serif';
        g.fillStyle = hot ? '#6ee7a0' : '#5d7590';
        g.fillText(meta.n + ' edges', f.cx - 12, f.cy - 5);
        g.font = '12px "Microsoft YaHei", sans-serif';
      }
    });

    // —— Dot matrix (real spike per-neuron light-up + afterglow decay) ——
    function drawPts(pts, firedSet, base) {
      pts.forEach(function (p) {
        var v = glow[p.id] || 0;
        var isF = firedSet && firedSet[p.id];
        if (isF) v = 1;
        var r = p.r + v * 1.8;
        g.beginPath(); g.arc(p.x, p.y, r, 0, 6.2832);
        g.fillStyle = v > 0.5 ? '#ffe9a3' : base;
        g.globalAlpha = v > 0.5 ? 1 : 0.32 + v * 0.6;
        g.fill();
        g.globalAlpha = 1;
      });
    }
    var fs = {};
    if (st) st.hid.forEach(function (id) { fs[id] = 1; });
    drawPts(visPts, fs, '#7fd1ff');
    drawPts(windPts, fs, '#8fe8c7');
    drawPts(hidPts, fs, '#9aa8c8');
    drawPts(outPts, fs, '#ff9ad5');
    gfPts.forEach(function (p, gi) {
      var fired = st && st.gf[gi] > 0;
      var v = st ? st.gfV[gi] : 0;
      g.beginPath(); g.arc(p.x, p.y, fired ? 13 : 9, 0, 6.2832);
      g.fillStyle = fired ? '#fff2a8' : (v > vth * 0.7 ? '#ffb86b' : '#e8c9ff');
      g.fill();
      g.strokeStyle = '#ffd479'; g.lineWidth = 1.5; g.stroke();
      g.fillStyle = '#233548'; g.font = 'bold 10px sans-serif';
      g.fillText('GF' + (gi + 1), p.x - 10, p.y + 3.5);
      g.fillStyle = '#ffd479'; g.font = '12px "Microsoft YaHei", sans-serif';
    });
    // Afterglow decay (display-only cache)
    for (var q2 = 0; q2 < N; q2++) if (glow[q2] > 0) glow[q2] *= 0.86;

    // —— GF membrane potential curve (with Vth threshold line; crossing = ⚡ escape) ——
    var cx0 = 34, cy0 = 350, cw = 580, ch = 96;
    g.strokeStyle = '#35506e'; g.lineWidth = 1;
    g.strokeRect(cx0, cy0, cw, ch);
    g.fillStyle = '#8aa0b8'; g.font = '11px "Microsoft YaHei", sans-serif';
    g.fillText('GF membrane potential (decision window 12 steps, Vth=' + vth.toFixed(2) + ' crossing=⚡ escape)', cx0, cy0 - 8);
    if (steps.length) {
      var vmax = vth * 1.35, vmin = -0.25;
      for (var gi2 = 0; gi2 < steps.length; gi2++)
        for (var kk = 0; kk < steps[gi2].gfV.length; kk++)
          vmax = Math.max(vmax, steps[gi2].gfV[kk] + 0.05);
      var Y = function (v) { return cy0 + ch - (v - vmin) / (vmax - vmin) * ch; };
      var X = function (i) { return cx0 + (i + 0.5) / T * cw; };
      // Vth threshold line
      g.setLineDash([5, 4]);
      g.strokeStyle = '#ffd479'; g.beginPath();
      g.moveTo(cx0, Y(vth)); g.lineTo(cx0 + cw, Y(vth)); g.stroke();
      g.setLineDash([]);
      g.fillStyle = '#ffd479';
      g.fillText('Vth', cx0 + cw - 30, Y(vth) - 4);
      // Two GF curves
      var colors = ['#7fd1ff', '#c9a3ff'];
      for (var gq = 0; gq < 2; gq++) {
        g.strokeStyle = colors[gq]; g.lineWidth = 1.8;
        g.beginPath();
        for (var i2 = 0; i2 < steps.length; i2++) {
          var yy = Y(steps[i2].gfV[gq]);
          if (i2 === 0) g.moveTo(X(i2), yy); else g.lineTo(X(i2), yy);
        }
        g.stroke();
        for (var i3 = 0; i3 <= t && i3 < steps.length; i3++) {
          var yv = Y(steps[i3].gfV[gq]);
          g.beginPath(); g.arc(X(i3), yv, steps[i3].gf[gq] > 0 ? 4 : 2, 0, 6.2832);
          g.fillStyle = steps[i3].gf[gq] > 0 ? '#fff2a8' : colors[gq];
          g.fill();
          if (steps[i3].gf[gq] > 0) {          // Crossed threshold spike → ⚡
            g.fillStyle = '#ffd479'; g.font = '13px sans-serif';
            g.fillText('⚡', X(i3) - 4, yv - 8);
            g.font = '11px "Microsoft YaHei", sans-serif';
          }
        }
      }
      // Current step cursor
      g.strokeStyle = 'rgba(255,255,255,.25)';
      g.beginPath(); g.moveTo(X(t), cy0); g.lineTo(X(t), cy0 + ch); g.stroke();
      g.fillStyle = '#8aa0b8';
      g.fillText('t=' + (t + 1) + 'ms', X(t) - 12, cy0 + ch + 12);
    } else {
      g.fillStyle = '#5d7590';
      g.fillText('(No decision window records yet)', cx0 + 8, cy0 + ch / 2);
    }
    // Legend
    g.font = '10px "Microsoft YaHei", sans-serif';
    g.fillStyle = '#7fd1ff'; g.fillText('GF1', cx0 + 8, cy0 + 12);
    g.fillStyle = '#c9a3ff'; g.fillText('GF2', cx0 + 42, cy0 + 12);
    g.fillStyle = '#8aa0b8';
    g.fillText('Real connectome: FlyWire topology 1871 neurons / 45524 edges (weights fine-tuned for synthetic task)', cx0 + 80, cy0 + 12);
  }

  /* ---------- Replay main loop (independent rAF, only runs while panel is open) ---------- */
  Panel._loop = function () {
    if (!Panel.open) { Panel._raf = null; return; }
    Panel._raf = requestAnimationFrame(Panel._loop);
    var now = performance.now();
    if (view.pend && (!view.cur || now - view.lastTick > 200)) {
      // New decision arrived: if current window finished / too long then take over, otherwise queue (prevents 25 Hz decisions from pinning replay to step 1)
      if (!view.cur || view.holding || view.t >= (view.cur.rec.steps.length - 1) || now - view.lastRecAt > 2500) {
        view.cur = view.pend; view.t = 0; view.holding = 0; view.lastTick = now;
      }
    }
    if (view.cur) {
      if (now - view.lastTick > STEP_MS) {
        view.lastTick = now;
        if (view.t < view.cur.rec.steps.length - 1) view.t++;
        else if (now - (view.cur.at || 0) > 0) {
          if (!view.holding) view.holding = now;
          else if (now - view.holding > HOLD_MS) {           // After holding, replay current window (or switch to new window)
            if (view.pend && view.pend !== view.cur) { view.cur = view.pend; }
            view.t = 0; view.holding = 0;
          }
        }
      }
    }
    draw(now);
  };

  root.BrainPanel = Panel;
})(window);
