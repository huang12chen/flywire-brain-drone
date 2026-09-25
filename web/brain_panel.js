/* brain_panel.js —— 🧠 果蝇大脑面板（真实连接组回路图 + 决策窗脉冲传导 + GF 膜电位曲线）
 * 数据来源：window.SNN_DATA（FlyWire 拓扑约束的真实连接组：1871 神经元 / 45524 边）
 *   LPLC2 视觉 210 → JO 风觉 325 →（中间神经元 1165）→ GF 巨纤维 2 → 下游 169
 * 渲染：纯 canvas 2D，零 fetch / 零 ES module / 零联网，file:// 双击即用。
 * 输入：snn_runtime.simulate(..., rec) 的旁路记录（只读快照，不影响 SNN 数值）。
 * 思路参考 snedea/flybrain 的"实时放电面板"（点阵 + 光点传导 + 电位曲线），独立重写。
 */
(function (root) {
  'use strict';
  var D = root.SNN_DATA;
  if (!D) { console.error('brain_panel: SNN_DATA 未加载'); return; }

  /* ---------- 分组（真实索引集合，互斥） ---------- */
  var N = D.num_nodes;
  var grp = new Int8Array(N);                 // 0=hid 1=vis 2=wind 3=gf 4=out
  D.input_vision_indices.forEach(function (i) { grp[i] = 1; });
  D.input_wind_indices.forEach(function (i) { grp[i] = 2; });
  D.hub_gf_indices.forEach(function (i) { grp[i] = 3; });
  D.output_indices.forEach(function (i) { grp[i] = 4; });

  // LPLC2：按偏好方位角排序（点阵列=方位扇区，让方位码看得见）
  var visOrder = D.input_vision_indices.map(function (id, k) { return { id: id, key: D.pd_pref[k][0] }; })
    .sort(function (a, b) { return a.key - b.key; }).map(function (o) { return o.id; });
  // JO：按敏感轴方位角排序
  var windOrder = D.input_wind_indices.map(function (id, k) {
    var p = D.wind_pref[k]; return { id: id, key: Math.atan2(p[1], p[0]) };
  }).sort(function (a, b) { return a.key - b.key; }).map(function (o) { return o.id; });
  var outOrder = D.output_indices.slice();
  // 中间神经元 1165：均匀抽样 128 个代表点（画得下，标注清楚）
  var hidAll = [];
  for (var i = 0; i < N; i++) if (grp[i] === 0) hidAll.push(i);
  var hidShow = [];
  for (var hi = 0; hi < 128; hi++) hidShow.push(hidAll[Math.floor(hi * hidAll.length / 128)]);

  /* ---------- 真实连接组：群间聚合（边数 / 权重和） ---------- */
  var GNAME = ['中间神经元', 'LPLC2', 'JO', 'GF', '下游'];
  var flow = {};                              // 'a->b' => {n, w}
  D.edges.forEach(function (e) {
    var k = grp[e[0]] + '>' + grp[e[1]];
    var f = flow[k] || (flow[k] = { n: 0, w: 0 });
    f.n++; f.w += e[2];
  });

  /* ---------- 布局（canvas 640×470） ---------- */
  var W = 640, H = 470;
  function grid(n, cols, x0, y0, gap, r) {
    var pts = [];
    for (var i = 0; i < n; i++) pts.push({ x: x0 + (i % cols) * gap, y: y0 + Math.floor(i / cols) * gap, r: r, id: null });
    return pts;
  }
  var visPts = grid(210, 15, 34, 66, 7.6, 2.3);            // 15×14
  var windPts = grid(325, 25, 34, 246, 7.0, 2.1);          // 25×13
  var hidPts = grid(128, 8, 262, 130, 8.4, 2.2);           // 8×16 竖排
  var gfPts = [{ x: 408, y: 158, r: 9 }, { x: 408, y: 205, r: 9 }];
  var outPts = grid(169, 13, 502, 116, 7.4, 2.2);          // 13×13
  visPts.forEach(function (p, i) { p.id = visOrder[i]; });
  windPts.forEach(function (p, i) { p.id = windOrder[i]; });
  hidPts.forEach(function (p, i) { p.id = hidShow[i]; });
  outPts.forEach(function (p, i) { p.id = outOrder[i]; });

  // 点阵快速定位：id -> (点亮值)
  var ptOf = {};
  [visPts, windPts, hidPts, gfPts, outPts].forEach(function (arr) {
    arr.forEach(function (p) { if (p.id != null) ptOf[p.id] = p; });
  });

  /* ---------- 流向带（真实边聚合；a,b 为群号，qp 为控制点 ---------- */
  var FLOWS = [
    { a: 1, b: 0, sx: 155, sy: 120, tx: 262, ty: 165, cx: 200, cy: 110, label: 'LPLC2→中间' },
    { a: 2, b: 0, sx: 155, sy: 285, tx: 262, ty: 205, cx: 200, cy: 275, label: 'JO→中间' },
    { a: 0, b: 3, sx: 330, sy: 180, tx: 399, ty: 182, cx: 365, cy: 150, label: '中间→GF' },
    { a: 1, b: 3, sx: 155, sy: 92, tx: 400, ty: 150, cx: 285, cy: 40, label: 'LPLC2→GF 直连', dash: true },
    { a: 2, b: 3, sx: 155, sy: 300, tx: 400, ty: 212, cx: 285, cy: 322, label: 'JO→GF 直连', dash: true },
    { a: 3, b: 4, sx: 418, sy: 182, tx: 502, ty: 170, cx: 462, cy: 140, label: 'GF→下游' },
    { a: 0, b: 4, sx: 330, sy: 215, tx: 502, ty: 215, cx: 420, cy: 268, label: '中间→下游' },
  ];

  /* ---------- 重放状态 ---------- */
  var view = { pend: null, cur: null, t: 0, lastTick: 0, holding: 0, lastRecAt: 0 };
  var STEP_MS = 130, HOLD_MS = 750;
  var glow = new Float64Array(N);             // 余晖（纯显示，与 SNN 无关）

  var Panel = {
    open: false,
    beginRec: function () { return this.open ? { steps: [] } : null; },   // 旁路记录器（不传=零改动）
    show: function (rec, out, cues) {
      view.pend = { rec: rec, out: out, cues: cues, at: performance.now() };
      view.lastRecAt = view.pend.at;
    },
    toggle: function () {
      this.open = !this.open;
      var el = document.getElementById('brain');
      if (el) el.style.display = this.open ? 'block' : 'none';
      var b = document.getElementById('bBrain');
      if (b) { b.classList.toggle('on', this.open); b.textContent = this.open ? '🧠 大脑面板：开' : '🧠 果蝇大脑面板'; }
      if (this.open && !this._raf) this._loop();
    },
  };

  function groupFired(st, g) {
    // st.hid 为全体放电快照；按群计数
    var ids = st ? st.hid : [], n = 0;
    for (var i = 0; i < ids.length; i++) if (grp[ids[i]] === g) n++;
    return n;
  }

  /* ---------- 绘制 ---------- */
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

    // —— 标题状态 ——
    g.font = '12px "Microsoft YaHei", sans-serif';
    g.fillStyle = '#8fd2ff';
    var trig = cur && cur.out && cur.out.triggered;
    g.fillText('决策窗重放 t = ' + (t + 1) + '/' + T + ' ms' + (trig ? '　⚡ GF 放电！逃逸' : '　静息'),
      16, 20);
    if (cur && cur.cues) {
      g.fillStyle = '#8aa0b8';
      g.fillText('loom ' + cur.cues.loom.toFixed(2) + ' rad/s　|u| ' +
        Math.hypot(cur.cues.wind.x, cur.cues.wind.y, cur.cues.wind.z).toFixed(4), 16, 38);
    } else {
      g.fillStyle = '#8aa0b8';
      g.fillText('等待威胁决策…（投放威胁后 GF 决策窗将实时重放）', 16, 38);
    }

    // —— 群标签 ——
    g.fillStyle = '#ffd479';
    g.fillText('LPLC2 视觉 210', 34, 60);
    g.fillText('JO 风觉 325', 34, 240);
    g.fillStyle = '#c9b8ff';
    g.fillText('中间神经元 1165（抽样 128）', 244, 118);
    g.fillStyle = '#ffd479';
    g.fillText('GF ×2', 386, 138);
    g.fillText('下游 169', 502, 110);

    // —— 流向带（真实边数标注 + 活动粒子） ——
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
      // 沿带粒子（视觉节拍 = 重放步）
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
        g.fillText(meta.n + ' 条', f.cx - 12, f.cy - 5);
        g.font = '12px "Microsoft YaHei", sans-serif';
      }
    });

    // —— 点阵（真实放电逐个点亮 + 余晖衰减） ——
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
    // 余晖衰减（纯显示缓存）
    for (var q2 = 0; q2 < N; q2++) if (glow[q2] > 0) glow[q2] *= 0.86;

    // —— GF 膜电位曲线（含 Vth 阈值线；越线=⚡逃逸） ——
    var cx0 = 34, cy0 = 350, cw = 580, ch = 96;
    g.strokeStyle = '#35506e'; g.lineWidth = 1;
    g.strokeRect(cx0, cy0, cw, ch);
    g.fillStyle = '#8aa0b8'; g.font = '11px "Microsoft YaHei", sans-serif';
    g.fillText('GF 膜电位（决策窗 12 步，Vth=' + vth.toFixed(2) + ' 越线=⚡逃逸）', cx0, cy0 - 8);
    if (steps.length) {
      var vmax = vth * 1.35, vmin = -0.25;
      for (var gi2 = 0; gi2 < steps.length; gi2++)
        for (var kk = 0; kk < steps[gi2].gfV.length; kk++)
          vmax = Math.max(vmax, steps[gi2].gfV[kk] + 0.05);
      var Y = function (v) { return cy0 + ch - (v - vmin) / (vmax - vmin) * ch; };
      var X = function (i) { return cx0 + (i + 0.5) / T * cw; };
      // Vth 阈值线
      g.setLineDash([5, 4]);
      g.strokeStyle = '#ffd479'; g.beginPath();
      g.moveTo(cx0, Y(vth)); g.lineTo(cx0 + cw, Y(vth)); g.stroke();
      g.setLineDash([]);
      g.fillStyle = '#ffd479';
      g.fillText('Vth', cx0 + cw - 30, Y(vth) - 4);
      // 两条 GF 曲线
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
          if (steps[i3].gf[gq] > 0) {          // 越线放电 → ⚡
            g.fillStyle = '#ffd479'; g.font = '13px sans-serif';
            g.fillText('⚡', X(i3) - 4, yv - 8);
            g.font = '11px "Microsoft YaHei", sans-serif';
          }
        }
      }
      // 当前步游标
      g.strokeStyle = 'rgba(255,255,255,.25)';
      g.beginPath(); g.moveTo(X(t), cy0); g.lineTo(X(t), cy0 + ch); g.stroke();
      g.fillStyle = '#8aa0b8';
      g.fillText('t=' + (t + 1) + 'ms', X(t) - 12, cy0 + ch + 12);
    } else {
      g.fillStyle = '#5d7590';
      g.fillText('（尚无决策窗记录）', cx0 + 8, cy0 + ch / 2);
    }
    // 图例
    g.font = '10px "Microsoft YaHei", sans-serif';
    g.fillStyle = '#7fd1ff'; g.fillText('GF1', cx0 + 8, cy0 + 12);
    g.fillStyle = '#c9a3ff'; g.fillText('GF2', cx0 + 42, cy0 + 12);
    g.fillStyle = '#8aa0b8';
    g.fillText('真实连接组：FlyWire 拓扑 1871 神经元 / 45524 边（权重为合成任务微调）', cx0 + 80, cy0 + 12);
  }

  /* ---------- 重放主循环（独立 rAF，仅面板开启时运行） ---------- */
  Panel._loop = function () {
    if (!Panel.open) { Panel._raf = null; return; }
    Panel._raf = requestAnimationFrame(Panel._loop);
    var now = performance.now();
    if (view.pend && (!view.cur || now - view.lastTick > 200)) {
      // 新决策到达：若当前窗播完/太久则接管，否则排队（防 25Hz 决策把重放钉死在第 1 步）
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
          else if (now - view.holding > HOLD_MS) {           // 停留后重播当前窗（或切入新窗）
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
