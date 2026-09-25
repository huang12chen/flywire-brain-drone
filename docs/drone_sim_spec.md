# 无人机模拟版·详细执行规格书（agent 照此施工，勿自创）

> 项目根：<项目根>
> 铁律：①SNN 数值不动（web\results.json SHA256=0E1339A088EF3B5820DCA342B989249403BE41D5A3F7EDB9DDB1D1A91C217E9E、jitter_sim_log.md=5C66B3D05A92F17961EC9EFC7F8406BFA6117CB04F5F47260F544AB52B8072AE）②飞手非修饰键接管最高优先级（Shift/Ctrl/Alt/Meta/AltGraph 豁免）③file:// 双击可用：零 fetch/零 ES module/零构建，three.js 用 web\vendor\three.min.js ④坐标映射 (x,z,y) 不变 ⑤方向误差口径=起跳时刻 −r̂ 不变。

## 阶段 1：drone_physics.js（飞行动力学）

新建 `web\drone_physics.js`（普通 <script>，挂 window.DronePhysics）：

```js
// class DronePhysics
// 状态：pos(Vector3), vel(Vector3), quat(四元数), angVel(Vector3)
// 输入 update(dtSec, motors[4])——4 个电机转速(归一化 0..1.2)
// 参数（演示级、世界单位，标注"未真机标定"）：
//   mass=1.0, gravity=9.8, kf=0.12(推力系数), km=0.018(反扭矩系数),
//   dragLin=0.35, dragQuad=0.02, maxTilt=40°, inertia=(0.02,0.02,0.04)
// 公式：
//   总推力 T = kf * Σ(motors[i]^2)，沿机体 -y 轴（世界系经 quat 旋转）
//   力矩：roll=(m0-m2), pitch=(m1-m3), yaw=km*(m0-m2+m1-m3)（标注四旋翼 X 型混控简化）
//   平动：a = (T_body + F_drag)/m + g；F_drag = -dragLin*v - dragQuad*|v|*v
//   姿态：四元数积分 q += 0.5*q⊗ω*dt，每步归一化；欧拉角钳制 ±maxTilt
//   积分：半隐式欧拉（先更新 vel 再更新 pos），dt 钳制 ≤0.033s
// 悬停校准：hoverOmega = sqrt(mass*gravity/(4*kf))，导出为 DronePhysics.HOVER
// 自测函数 DronePhysics.selfTest()：悬停 5s 位置漂移 <0.3、匀加速、姿态响应三项 PASS/FAIL 返回字符串
```

集成：world.html 的 driveAgent 中"无人机 agent"改走 DronePhysics（果蝇 agent 保持现运动学——果蝇会飞不是直升机）；控制台加"⚙️ 飞行动力学"开关（开=新物理，关=旧运动学，默认开）；HUD 增 4 条电机转速条+姿态角(roll/pitch)。

## 阶段 2：drone_adapter.js（脑→电机适配层）

新建 `web\drone_adapter.js`（挂 window.DroneAdapter）：

```js
// 仿生分层：果蝇脑(决策) → 本适配层(胸神经节) → 电机(飞行肌)；PID 姿态环=平衡棒反射
// class DroneAdapter { update(brainOut, droneState, dtMs, pilotActive) -> motors[4] }
// 状态机：CRUISE(巡航) / ESCAPE(逃逸) / PILOT(飞手) / LAND(降落)
// - PILOT 优先级最高：pilotActive（world.html 现有非修饰键逻辑）→ 直接旧运动学接管，本层旁路
// - ESCAPE：brainOut.triggered==true 触发，持续 300ms：
//     油门 = HOVER*1.5（阶跃，模拟 GF 爆发）
//     目标倾角 = clamp(escapeDir 投影到水平面, ±35°) → 期望 roll/pitch
// - CRUISE：油门=HOVER*1.02，期望姿态=轻微随机游走（每 2-4s 换目标航向，角速度 ≤15°/s，用注入 rng 保确定性）
// - LAND：慢速降油门至 HOVER*0.8 直到落地（pos.y<阈值）→ 油门 0
// PID 姿态环：kp=0.045, ki=0.001, kd=0.028（误差=期望姿态-当前姿态，输出加进 motors）
// 混控矩阵：[m0,m1,m2,m3] = HOVER*[throttle, +roll, -roll, ...]（X 型，写清每项）
// 永远导出 lastState（HUD 显示状态机：巡航/逃逸/飞手/降落）
```

飞手接管验收：按下任意非修饰键那一帧，动力学开关退居二线、旧运动学立即接管（现有逻辑原样保留）。

## 阶段 3：持续飞行行为

world.html 逻辑（大脑持续在线，每决策周期跑 SNN）：
- 巡航（CRUISE）：慢速游荡 + 避障转向——用 LPLC2 输入的 loom 场采样最近障碍方位，当 0.35<loom<触发阈值时朝远离方向偏航（**不触发逃逸**，只转向）；触发阈值以上照旧走逃逸。
- 逃逸（ESCAPE）：现有 SNN 逃逸不动，只是改由适配层执行（油门阶跃+倾角）。
- 降落（LAND）：控制台按钮 + 电量装饰条（100→0 用时 10min，到 0 自动 LAND；飞手可随时重新起飞）。
- HUD 增：状态机文字（巡航/逃逸/飞手/降落）+ 电机 4 条 + 姿态角。

## 阶段 4：回归自测（每阶段改完立刻跑）

1. `F:\Node\node.exe web\smoke_test.js` 4/4 必须保持
2. 新增 `web\smoke_drone.js`：悬停稳定（5s 漂移<0.3）、逃逸响应（触发后 <300ms 速度指向逃逸方向分量>0）、飞手接管（一帧内接管）、自测 DronePhysics.selfTest() 全 PASS
3. 改动任何 web\*.js 后跑 `F:\Node\node.exe web\evaluate.js`，results.json 哈希必须=锚点值；漂移即回滚当步
4. world.html 内联脚本语法检查（node new Function）

## 阶段 5：GitHub 打包

1. `LICENSE`（MIT，Copyright (c) 2026 + 用户名占位）
2. `.gitignore`：pylibs/、*.pt、v4/ckpt*、__pycache__、%TEMP% 产物
3. `README.md` 重写（双语：中文在前英文摘要在后）：
   - 标题：FlyWire Connectome Brain → Drone Control（果蝇连接组大脑开无人机）
   - 30 秒速览：双击 web\world.html → 选"无人机" → 按"自动威胁"看果蝇脑开飞机躲攻击
   - 架构图（ASCII）：FlyWire FAFB(1871神经元) → LIF SNN → 逃逸方向 → DroneAdapter(PID+混控) → 4电机 → 3D世界
   - 指标表：v3 数字（触发 0.799/成功率 0.903/方向 10.7°/潜伏 7.65ms/抖动 92% 全项达标）+ D组物理口径
   - 诚实边界三条（见 task_plan.md）
   - 引用：Dorkenwald et al. Nature 634:124–138 (2024)；勘误"谷歌开源=Neuroglancer 查看器，数据=FlyWire"；snedea/flybrain 链接
   - 复现：extract_circuit.py → train_snn.py → make_web_data.py → evaluate.js 命令 + 两个哈希锚点
   - 动图占位：`docs/demo.gif`（阶段 6 产出后替换）
4. `docs/` 收纳：full-fly-design.md、本规格书、v4\REPORT_v4.md 摘录
5. GitHub Pages 部署段落（Settings→Pages→main→/web）

## 阶段 6：演示动图（可选）

固定种子自动脚本拍 6 帧关键瞬间（巡航→威胁入画→脑放电⚡→逃逸→脱险→大脑面板特写）拼 GIF；做不了就给用户手录操作剧本（30 秒分镜）。

## 汇报纪律（所有 agent）
每阶段≤10 行中文：改了什么文件、自测结果、哈希验证一行、下一步。禁长篇表格。
