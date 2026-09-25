/* scene_fx.js —— P2 建模精修 + 四场景实质差异（纯 three.js 基元 + 程序纹理，零外部资源）
 * 🪰 果蝇：红黑复眼（程序复眼纹理）+ 六足关节（2 节腿×膝关节）+ 翅脉翅振（程序翅脉纹理）
 *        + 胸腹分节（头/胸/腹节环纹）
 * 🚁 无人机：四旋翼桨叶（双叶桨 + 模糊盘）+ 俯仰云台相机 + 航向/状态指示灯
 * 🌦️ 四场景：白日草浪云影 / 黑夜萤火虫月光 / 暴风斜雨风线 / 大风气流粒子
 * 光影升级：阴影贴图 + 半球环境光，主模型/树/蘑菇投影。
 */
(function (root) {
  'use strict';
  var THREE = root.THREE;
  var FX = {};

  /* ================= 程序纹理 ================= */
  function canvasTex(w, h, draw) {
    var c = document.createElement('canvas'); c.width = w; c.height = h;
    draw(c.getContext('2d'));
    var t = new THREE.CanvasTexture(c);
    return t;
  }
  // 翅脉：半透明膜 + 深色脉络（前缘粗、放射分支）
  var WING_TEX = canvasTex(128, 96, function (g) {
    g.clearRect(0, 0, 128, 96);
    g.strokeStyle = 'rgba(60,45,30,0.85)'; g.lineCap = 'round';
    g.lineWidth = 3.2;                                  // 前缘脉
    g.beginPath(); g.moveTo(4, 22); g.bezierCurveTo(40, 8, 90, 14, 124, 34); g.stroke();
    g.lineWidth = 1.6;
    var veins = [[10, 26, 26, 88], [34, 16, 48, 90], [58, 14, 74, 88], [82, 18, 98, 80], [104, 26, 116, 66]];
    veins.forEach(function (v) {
      g.beginPath(); g.moveTo(v[0], v[1]); g.quadraticCurveTo((v[0] + v[2]) / 2 + 6, (v[1] + v[3]) / 2, v[2], v[3]); g.stroke();
    });
    g.lineWidth = 1.1;                                  // 横脉
    g.beginPath(); g.moveTo(26, 88); g.lineTo(48, 90); g.moveTo(48, 90); g.lineTo(74, 88); g.stroke();
  });
  // 复眼：暗红底 + 黑色小眼面点阵（红黑复眼）
  var EYE_TEX = canvasTex(64, 64, function (g) {
    g.fillStyle = '#7a1018'; g.fillRect(0, 0, 64, 64);
    g.fillStyle = '#1a0507';
    for (var y = 0; y < 8; y++) for (var x = 0; x < 8; x++) {
      g.beginPath(); g.arc(x * 8 + (y % 2 ? 4 : 0) + 2, y * 8 + 2, 2.6, 0, 6.2832); g.fill();
    }
  });
  // 腹节环纹：深浅相间
  var BELLY_TEX = canvasTex(32, 64, function (g) {
    for (var i = 0; i < 8; i++) {
      g.fillStyle = i % 2 ? '#3a2b22' : '#c98a3d';
      g.fillRect(0, i * 8, 32, 8);
    }
  });
  // 云影柔斑
  var SHADOW_TEX = canvasTex(128, 128, function (g) {
    var r = g.createRadialGradient(64, 64, 8, 64, 64, 62);
    r.addColorStop(0, 'rgba(20,40,25,0.42)'); r.addColorStop(1, 'rgba(20,40,25,0)');
    g.fillStyle = r; g.fillRect(0, 0, 128, 128);
  });

  function shade(o) { o.traverse(function (m) { if (m.isMesh) { m.castShadow = true; } }); return o; }
  function mat(color, opt) { return new THREE.MeshLambertMaterial(Object.assign({ color: color }, opt || {})); }

  /* ================= 🪰 果蝇 ================= */
  FX.buildFlyMesh = function (FLY_R) {
    var g = new THREE.Group();
    var s = FLY_R;                                     // 视觉以碰撞半径为比例基准（碰撞语义不变）

    // —— 头 + 红黑复眼 ——
    var head = new THREE.Mesh(new THREE.SphereGeometry(s * 0.52, 14, 10), mat(0x241a16));
    head.position.set(0, s * 0.12, s * 1.05); g.add(head);
    var eyeMat = new THREE.MeshPhongMaterial({ map: EYE_TEX, shininess: 30 });
    for (var ei = 0; ei < 2; ei++) {
      var side = ei ? 1 : -1;
      var eye = new THREE.Mesh(new THREE.SphereGeometry(s * 0.42, 16, 12), eyeMat);
      eye.position.set(side * s * 0.4, s * 0.18, s * 1.02); eye.scale.set(0.8, 1, 1);
      g.add(eye);
    }
    // 触角（2 根，前伸分节感）
    for (var ai = 0; ai < 2; ai++) {
      var asd = ai ? 1 : -1;
      var ant = new THREE.Mesh(new THREE.CylinderGeometry(0.02 * s, 0.03 * s, s * 1.1, 5), mat(0x1a1210));
      ant.position.set(asd * s * 0.22, s * 0.5, s * 1.5);
      ant.rotation.set(Math.PI / 2.6, 0, asd * 0.35); g.add(ant);
    }

    // —— 胸（黑褐，背板 + 侧板分节感）——
    var thorax = new THREE.Mesh(new THREE.SphereGeometry(s * 0.66, 14, 10), mat(0x4a3226));
    thorax.scale.set(0.95, 0.9, 1.15); thorax.position.set(0, s * 0.1, s * 0.1); g.add(thorax);
    var scut = new THREE.Mesh(new THREE.SphereGeometry(s * 0.4, 10, 8), mat(0x241812));
    scut.scale.set(1, 0.5, 1.2); scut.position.set(0, s * 0.5, s * 0.05); g.add(scut);

    // —— 腹：分节（4 节递缩 + 环纹）——
    for (var bi = 0; bi < 4; bi++) {
      var r0 = s * (0.52 - bi * 0.085);
      var seg = new THREE.Mesh(new THREE.SphereGeometry(r0, 12, 9),
        new THREE.MeshLambertMaterial({ map: BELLY_TEX }));
      seg.scale.set(0.95, 0.85, 1.0);
      seg.position.set(0, s * 0.02, -s * (0.55 + bi * 0.62));
      g.add(seg);
    }

    // —— 六足（每侧 3 条：基节-腿节 + 膝 + 胫节，含关节枢轴）——
    g.userData.legs = [];
    for (var li = 0; li < 6; li++) {
      var side2 = li < 3 ? -1 : 1, row = li % 3;
      var hip = new THREE.Group();                     // 髋关节枢轴
      hip.position.set(side2 * s * 0.5, -s * 0.15, s * (0.5 - row * 0.55));
      hip.rotation.z = side2 * (0.75 + row * 0.12);
      var femur = new THREE.Mesh(new THREE.CylinderGeometry(0.035 * s, 0.03 * s, s * 0.9, 5), mat(0x1c1310));
      femur.position.y = -s * 0.45; hip.add(femur);
      var knee = new THREE.Group();                    // 膝关节枢轴
      knee.position.y = -s * 0.9;
      var tibia = new THREE.Mesh(new THREE.CylinderGeometry(0.025 * s, 0.014 * s, s * 1.0, 5), mat(0x120c0a));
      tibia.position.y = -s * 0.5; knee.add(tibia);
      knee.rotation.x = 0.5 + row * 0.15;
      hip.add(knee);
      g.add(hip);
      g.userData.legs.push({ hip: hip, knee: knee, phase: li * 1.1, baseHip: hip.rotation.x, baseKnee: knee.rotation.x });
    }

    // —— 翅（翅脉纹理膜；根部枢轴，翅振=绕根部摆动）——
    g.userData.wings = [];
    for (var wi = 0; wi < 2; wi++) {
      var wsd = wi ? 1 : -1;
      var wing = new THREE.Group();
      wing.position.set(wsd * s * 0.28, s * 0.62, s * 0.15);
      var mem = new THREE.Mesh(new THREE.PlaneGeometry(s * 2.6, s * 1.5),
        new THREE.MeshLambertMaterial({ map: WING_TEX, transparent: true, side: THREE.DoubleSide, depthWrite: false }));
      mem.position.x = wsd * s * 1.3;
      mem.rotation.y = wsd > 0 ? 0 : Math.PI;
      wing.add(mem);
      g.add(wing);
      g.userData.wings.push(wing);
    }
    // 平衡棒（后翅退化为楫翅，一对小球）
    for (var hi = 0; hi < 2; hi++) {
      var hsd = hi ? 1 : -1;
      var hal = new THREE.Mesh(new THREE.SphereGeometry(s * 0.1, 6, 5), mat(0x8a6a3d));
      hal.position.set(hsd * s * 0.35, s * 0.3, -s * 0.45); g.add(hal);
    }
    return shade(g);
  };

  /* ================= 🚁 无人机 ================= */
  FX.buildDroneMesh = function () {
    var g = new THREE.Group();
    var bodyMat = mat(0x37485f), darkMat = mat(0x222b38), metalMat = new THREE.MeshPhongMaterial({ color: 0x93a7bd, shininess: 60 });

    // —— 机身（碳纤维色 + 顶盖分层）——
    var body = new THREE.Mesh(new THREE.BoxGeometry(3.4, 1.1, 2.2), bodyMat); g.add(body);
    var top = new THREE.Mesh(new THREE.BoxGeometry(2.4, 0.5, 1.6), mat(0x46597a));
    top.position.y = 0.75; g.add(top);
    var belly = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.4, 1.4), darkMat);
    belly.position.y = -0.7; g.add(belly);

    // —— 四旋翼：X 形机臂 + 双叶桨 + 模糊盘 ——
    g.userData.rotors = [];                            // 接口保持：world.html 旋翼动画
    g.userData.props = [];
    for (var i = 0; i < 4; i++) {
      var x = [2.4, 2.4, -2.4, -2.4][i], z = [2.4, -2.4, 2.4, -2.4][i];
      var arm = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.18, 3.0, 6), bodyMat);
      arm.rotation.z = Math.PI / 2; arm.rotation.y = Math.atan2(z, x);
      arm.position.set(x / 2, 0, z / 2); g.add(arm);
      var motor = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.42, 0.6, 10), metalMat);
      motor.position.set(x, 0.5, z); g.add(motor);
      var prop = new THREE.Group();                    // 桨毂 + 双叶
      prop.position.set(x, 0.9, z);
      for (var b = 0; b < 2; b++) {
        var blade = new THREE.Mesh(new THREE.BoxGeometry(2.6, 0.045, 0.42), mat(0x1c2530));
        blade.position.x = b ? -1.3 : 1.3;
        blade.rotation.y = b ? 0.25 : -0.25;
        prop.add(blade);
      }
      var hubc = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.18, 8), metalMat);
      prop.add(hubc);
      var disc = new THREE.Mesh(new THREE.CylinderGeometry(1.45, 1.45, 0.02, 20),
        new THREE.MeshLambertMaterial({ color: 0x8fa6bf, transparent: true, opacity: 0.22 }));
      disc.position.y = 0.02; g.add(disc);             // 高速旋转模糊盘
      g.add(prop);
      g.userData.rotors.push(prop);
      g.userData.props.push(prop);
    }

    // —— 云台相机（前下俯仰）——
    var gimbal = new THREE.Group(); gimbal.position.set(0, -0.9, 1.1);
    var yawRing = new THREE.Mesh(new THREE.TorusGeometry(0.5, 0.08, 6, 14), darkMat);
    gimbal.add(yawRing);
    var pitchBox = new THREE.Group(); pitchBox.position.y = -0.1;
    var cam = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.55, 0.6), darkMat);
    var lens = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.22, 0.3, 10),
      new THREE.MeshPhongMaterial({ color: 0x101820, shininess: 90 }));
    lens.rotation.x = Math.PI / 2; lens.position.z = 0.35;
    pitchBox.add(cam, lens); gimbal.add(pitchBox);
    gimbal.userData.pitch = pitchBox;
    g.add(gimbal);
    g.userData.gimbal = gimbal;

    // —— 指示灯：前白×2 / 后红×1 / 尾闪红×1（含夜航闪）——
    g.userData.leds = [];
    function led(color, x, y, z) {
      var l = new THREE.Mesh(new THREE.SphereGeometry(0.16, 8, 6),
        new THREE.MeshLambertMaterial({ color: color, emissive: color, emissiveIntensity: 0.9 }));
      l.position.set(x, y, z); g.add(l); g.userData.leds.push(l); return l;
    }
    led(0xffffff, 1.2, 0.2, 1.15); led(0xffffff, -1.2, 0.2, 1.15);
    led(0xff3344, 1.2, 0.2, -1.15); led(0xff3344, -1.2, 0.2, -1.15);
    led(0xffaa22, 0, 0.3, -1.35);                     // 尾部橙闪

    // —— 起落架 ——
    for (var gi = 0; gi < 4; gi++) {
      var lx = [1.4, -1.4, 1.4, -1.4][gi], lz = [0.8, 0.8, -0.8, -0.8][gi];
      var leg = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.05, 1.2, 6), darkMat);
      leg.position.set(lx, -1.1, lz); g.add(leg);
    }
    return shade(g);
  };

  /* ================= 🌦️ 四场景实质差异 ================= */
  var fx = null;
  FX.init = function (scene) {
    fx = { scene: scene };

    // —— 云影（白日草浪云影）：柔斑大圆盘贴地漂移 ——
    fx.cloudShadows = new THREE.Group();
    for (var i = 0; i < 8; i++) {
      var m = new THREE.Mesh(new THREE.CircleGeometry(26 + Math.random() * 30, 20),
        new THREE.MeshBasicMaterial({ map: SHADOW_TEX, transparent: true, depthWrite: false }));
      m.rotation.x = -Math.PI / 2;
      m.position.set(Math.random() * 500 - 250, 0.06, Math.random() * 500 - 250);
      fx.cloudShadows.add(m);
    }
    scene.add(fx.cloudShadows);

    // —— 月亮 + 星空（黑夜萤火虫月光）——
    fx.nightSky = new THREE.Group();
    var moon = new THREE.Mesh(new THREE.SphereGeometry(22, 18, 14),
      new THREE.MeshBasicMaterial({ color: 0xf5f2df }));
    moon.position.set(-320, 300, -420); fx.nightSky.add(moon);
    var moonGlow = new THREE.Mesh(new THREE.SphereGeometry(34, 18, 14),
      new THREE.MeshBasicMaterial({ color: 0xbfd4ff, transparent: true, opacity: 0.22 }));
    moonGlow.position.copy(moon.position); fx.nightSky.add(moonGlow);
    var starGeo = new THREE.BufferGeometry();
    var sPos = new Float32Array(340 * 3);
    for (var si = 0; si < 340; si++) {
      var a = Math.random() * Math.PI * 2, e = 0.12 + Math.random() * 0.8, d = 1100;
      sPos[si * 3] = Math.cos(a) * Math.cos(e) * d;
      sPos[si * 3 + 1] = Math.sin(e) * d * 0.8 + 60;
      sPos[si * 3 + 2] = Math.sin(a) * Math.cos(e) * d;
    }
    starGeo.setAttribute('position', new THREE.BufferAttribute(sPos, 3));
    fx.nightSky.add(new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0xdfe8ff, size: 2.2, transparent: true, opacity: 0.9 })));
    scene.add(fx.nightSky);

    // —— 萤火虫（暖黄绿闪烁 + 漫游）——
    var ffGeo = new THREE.BufferGeometry();
    var ffPos = new Float32Array(140 * 3), ffSeed = new Float32Array(140);
    for (var fi = 0; fi < 140; fi++) {
      var fa = Math.random() * Math.PI * 2, fd = Math.sqrt(Math.random()) * 160;
      ffPos[fi * 3] = Math.cos(fa) * fd;
      ffPos[fi * 3 + 1] = 1.5 + Math.random() * 9;
      ffPos[fi * 3 + 2] = Math.sin(fa) * fd;
      ffSeed[fi] = Math.random() * 100;
    }
    ffGeo.setAttribute('position', new THREE.BufferAttribute(ffPos, 3));
    fx.fireflies = new THREE.Points(ffGeo, new THREE.PointsMaterial({
      color: 0xd8ff7a, size: 1.5, transparent: true, opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false }));
    fx.fireflies.userData.seed = ffSeed;
    scene.add(fx.fireflies);

    // —— 斜雨（暴风）：线段雨幕，带风向斜率 ——
    var rGeo = new THREE.BufferGeometry();
    var RN = 700, rPos = new Float32Array(RN * 6);
    fx.rain = new THREE.LineSegments(rGeo, new THREE.LineBasicMaterial({ color: 0x9fc4e8, transparent: true, opacity: 0.5 }));
    fx.rain.userData.pos = rPos; fx.rain.userData.n = RN;
    for (var ri = 0; ri < RN; ri++) resetDrop(rPos, ri, true);
    rGeo.setAttribute('position', new THREE.BufferAttribute(rPos, 3));
    scene.add(fx.rain);

    // —— 风线（暴风）：高速水平流线 ——
    var wGeo = new THREE.BufferGeometry();
    var WN = 130, wPos = new Float32Array(WN * 6);
    fx.streaks = new THREE.LineSegments(wGeo, new THREE.LineBasicMaterial({ color: 0xd8ecff, transparent: true, opacity: 0.35 }));
    fx.streaks.userData.pos = wPos; fx.streaks.userData.n = WN;
    for (var wi = 0; wi < WN; wi++) resetStreak(wPos, wi, true);
    wGeo.setAttribute('position', new THREE.BufferAttribute(wPos, 3));
    scene.add(fx.streaks);

    // —— 气流粒子（大风）：流场平流的可视气流 ——
    var aGeo = new THREE.BufferGeometry();
    var AN = 520, aPos = new Float32Array(AN * 3);
    for (var ai = 0; ai < AN; ai++) {
      aPos[ai * 3] = Math.random() * 360 - 180;
      aPos[ai * 3 + 1] = 0.5 + Math.random() * 34;
      aPos[ai * 3 + 2] = Math.random() * 360 - 180;
    }
    aGeo.setAttribute('position', new THREE.BufferAttribute(aPos, 3));
    fx.airflow = new THREE.Points(aGeo, new THREE.PointsMaterial({
      color: 0xbfe8ff, size: 0.55, transparent: true, opacity: 0.55, depthWrite: false }));
    scene.add(fx.airflow);

    return fx;
  };

  function resetDrop(p, i, init) {
    var o = i * 6;
    var x = Math.random() * 400 - 200, y = init ? Math.random() * 120 : 90 + Math.random() * 30, z = Math.random() * 400 - 200;
    p[o] = x; p[o + 1] = y; p[o + 2] = z;
    p[o + 3] = x - 1.6; p[o + 4] = y - 3.4; p[o + 5] = z - 0.9;   // 斜率=风向（斜雨）
  }
  function resetStreak(p, i, init) {
    var o = i * 6;
    var x = init ? Math.random() * 420 - 210 : 200 + Math.random() * 30, y = 1 + Math.random() * 46, z = Math.random() * 420 - 210;
    var len = 8 + Math.random() * 14;
    p[o] = x; p[o + 1] = y; p[o + 2] = z;
    p[o + 3] = x - len; p[o + 4] = y; p[o + 5] = z - len * 0.35;
  }

  // 场景开关（实质差异：各场景挂不同粒子系统）
  FX.setScene = function (name) {
    if (!fx) return;
    fx.cloudShadows.visible = (name === 'meadow');
    fx.nightSky.visible = (name === 'night');
    fx.fireflies.visible = (name === 'night');
    fx.rain.visible = (name === 'storm');
    fx.streaks.visible = (name === 'storm' || name === 'gale');
    fx.airflow.visible = (name === 'gale');
  };

  // 逐帧动画
  FX.update = function (now, dtMs, name) {
    if (!fx) return;
    var t = now * 0.001;
    if (fx.cloudShadows.visible) fx.cloudShadows.children.forEach(function (m, i) {
      m.position.x += 0.006 * dtMs * (1 + (i % 3) * 0.3);
      if (m.position.x > 300) m.position.x = -300;
    });
    if (fx.fireflies.visible) {
      var p = fx.fireflies.geometry.attributes.position, seed = fx.fireflies.userData.seed;
      for (var i = 0; i < seed.length; i++) {
        p.array[i * 3] += Math.sin(t * 0.7 + seed[i]) * 0.012 * dtMs * 0.06;
        p.array[i * 3 + 1] += Math.cos(t * 0.5 + seed[i] * 1.3) * 0.008 * dtMs * 0.06;
        p.array[i * 3 + 2] += Math.cos(t * 0.6 + seed[i] * 0.7) * 0.012 * dtMs * 0.06;
      }
      p.needsUpdate = true;
      fx.fireflies.material.opacity = 0.55 + 0.4 * Math.abs(Math.sin(t * 2.2));   // 群体闪烁
    }
    if (fx.rain.visible) {
      var rp = fx.rain.userData.pos, RN2 = fx.rain.userData.n;
      for (var r = 0; r < RN2; r++) {
        var o = r * 6, dy = 0.75 * dtMs, dx = -0.35 * dtMs, dz = -0.2 * dtMs;
        rp[o] += dx; rp[o + 1] -= dy; rp[o + 2] += dz;
        rp[o + 3] += dx; rp[o + 4] -= dy; rp[o + 5] += dz;
        if (rp[o + 1] < 0) resetDrop(rp, r, false);
      }
      fx.rain.geometry.attributes.position.needsUpdate = true;
    }
    if (fx.streaks.visible) {
      var sp = fx.streaks.userData.pos, SN = fx.streaks.userData.n;
      for (var s = 0; s < SN; s++) {
        var so = s * 6, vx = 0.5 * dtMs, vz = 0.18 * dtMs;
        sp[so] -= vx; sp[so + 2] -= vz; sp[so + 3] -= vx; sp[so + 5] -= vz;
        if (sp[so + 3] < -230) resetStreak(sp, s, false);
      }
      fx.streaks.geometry.attributes.position.needsUpdate = true;
    }
    if (fx.airflow.visible) {
      var ap = fx.airflow.geometry.attributes.position;
      for (var a = 0; a < ap.count; a++) {
        var x = ap.array[a * 3], y = ap.array[a * 3 + 1], z = ap.array[a * 3 + 2];
        // 无散度流场（正弦剪切）平流——可视气流
        var vx2 = 0.055 * dtMs * (1 + 0.5 * Math.sin(y * 0.25 + t * 0.8));
        var vz2 = 0.03 * dtMs * Math.sin(x * 0.05 + t * 0.6);
        x -= vx2; z -= vz2; y += 0.004 * dtMs * Math.sin(x * 0.08 + z * 0.05 + t);
        if (x < -190) x = 190;
        if (z > 190) z = -190; else if (z < -190) z = 190;
        if (y < 0.5) y = 34; else if (y > 34) y = 0.5;
        ap.array[a * 3] = x; ap.array[a * 3 + 1] = y; ap.array[a * 3 + 2] = z;
      }
      ap.needsUpdate = true;
    }
    // 月夜恒定辉光呼吸
    if (fx.nightSky.visible) fx.nightSky.children[1].material.opacity = 0.18 + 0.06 * Math.sin(t * 0.8);
  };

  FX.state = function () { return fx; };
  root.FX = FX;
})(window);
