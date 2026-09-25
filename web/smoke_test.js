/* smoke_test.js — 用 Node 快速验证训练好的 SNN 推理引擎（不需要浏览器）
 * 运行：node web/smoke_test.js
 * 判定按"放电率"做（脉冲编码有随机性，单次判定会偶发假红/假绿）。
 */
global.window = {};
require('./snn_data.js');
const SNNRuntime = require('./snn_runtime.js');

const rt = new SNNRuntime(window.SNN_DATA);
if (window.SNN_DATA.meta && window.SNN_DATA.meta.dir_flipped) { console.error('❌ 方向读出头带翻转标记（--flip-dir 导出）'); process.exit(1); }
console.log(`模型：${rt.n} 节点 / ${rt.src.length} 边 / T=${rt.T}`);

function scenario(name, looming, wind, az) {
  const out = rt.simulate(looming, wind, az);
  console.log(`\n[${name}]`);
  console.log(`  GF放电=${out.triggered}  潜伏期=${out.firstSpikeMs ?? '—'} ms  ` +
              `触发概率=${(out.triggerProb * 100).toFixed(1)}%  v_GF峰值=${out.vHubMax.toFixed(2)}`);
  console.log(`  逃逸方向=(${out.escapeDir.map(x => x.toFixed(2)).join(', ')})`);
  return out;
}
function fireRate(looming, wind, az, n) {
  let f = 0;
  for (let i = 0; i < n; i++) if (rt.simulate(looming, wind, az).triggered) f++;
  return f / n;
}
function dirMeanX(looming, wind, az, n) {
  let s = 0;
  for (let i = 0; i < n; i++) s += rt.simulate(looming, wind, az).escapeDir[0];
  return s / n;
}

// 1) 强烈逼近威胁，位于方位角 0°（+x 方向）→ 期望：GF 放电，逃逸方向≈(-1, 0, 0)
const a = scenario('强威胁 @0°（在 +x 方向）', 4.0, [0.08, 0, 0], 0);
// 2) 无威胁 → 期望：GF 不放电
const b = scenario('无威胁', 0.0, [0, 0, 0], 0);
// 3) 弱线索 → 期望：多数情况不放电（临界状态）
const c = scenario('弱线索 @180°', 0.8, [0.01, 0, 0], Math.PI);

// —— 语义自检（按放电率判定，抗随机性）——
const strongRate = fireRate(4.0, [0.08, 0, 0], 0, 10);          // 强威胁：应高概率放电
const silentRate = fireRate(0.0, [0, 0, 0], 0, 20);             // 无威胁：应绝不放电
const weakRate = fireRate(0.8, [0.01, 0, 0], Math.PI, 20);      // 弱线索：应基本安静
const dirMean = dirMeanX(4.0, [0.08, 0, 0], 0, 10);            // 抗随机：10 次均值判定背离
const dirOK = dirMean < -0.5;                                   // 逃逸方向背离 +x 方向的威胁
const strongOK = strongRate >= 0.6;
const silentOK = silentRate <= 0.05;                            // 20 次抽样严格 0/20 有 ~3% 假红率，放宽到 ≤5%（实测真实率 0/2000）
const weakOK = weakRate <= 0.3;
console.log('\n==== 自检 ====');
console.log(`强威胁触发 GF 放电: ${strongOK ? '✅' : '❌'} (10 次放电率 ${(strongRate * 100).toFixed(0)}%，要求 ≥60%)`);
console.log(`无威胁保持静息:     ${silentOK ? '✅' : '❌'} (20 次放电率 ${(silentRate * 100).toFixed(0)}%，要求 ≤5%)`);
console.log(`逃逸方向背离威胁:   ${dirOK ? '✅' : '❌'} (单次 x=${a.escapeDir[0].toFixed(2)}，10 次均值=${dirMean.toFixed(2)}, 期望 < -0.5)`);
console.log(`弱线索保持安静:     ${weakOK ? '✅' : '❌'} (20 次放电率 ${(weakRate * 100).toFixed(0)}%，要求 ≤30%)`);
process.exit((strongOK && silentOK && dirOK && weakOK) ? 0 : 1);
