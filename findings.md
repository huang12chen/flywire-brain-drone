# 发现与决策记录（无人机模拟版 + GitHub）

## 已定决策
1. **不用 Rapier 物理引擎**：包仅 ESM/需打包器，违 file:// 零构建约束（已核实官方 README）→ 手写动力学。
2. **分层控制**（引自 v4\full-fly-design.md）：果蝇脑=决策（伪肌肉指令），PID 姿态环=稳定（生物对应 VNC/平衡棒反射），飞手遥控=最高优先级。
3. **SNN 数值锚点**：results.json / jitter_sim_log.md 哈希锁死，任何阶段改动后必须复验。
4. **GitHub 许可选 MIT**（代码）；数据引用 FlyWire（Dorkenwald et al., Nature 634:124–138, 2024）；注明"谷歌开源"=Neuroglancer 查看器、数据=FlyWire/Princeton。

## 可借鉴资源
- snedea/flybrain（MIT）：全脑 13.9 万神经元浏览器实时仿真的工程参考（Web Worker LIF + WebGL 面板）。
- heyseth/worm-sim（MIT）：前身项目，线虫 302 神经元浏览器仿真。
- PX4 offboard / gym-pybullet-drones：未来真机阶段的控制栈参考。
- 训练结论（v4 实验）：数据增强治风觉（0→55%触发）但方向精度从 10.7° 退到 19°；瓶颈在模型架构不在数据——GitHub 项目用 v3 模型（10.7°/90.3%）。

## 风险
- world.html 改动回归风险 → 每阶段跑 smoke_test + 哈希验证。
- 飞行动力学数值稳定性（dt 大时积分发散）→ 半隐式欧拉 + dt 钳制。
