# FlyWire Connectome Brain → Drone Control（果蝇连接组大脑开无人机）

> 用 FlyWire 公开连接组抽出的果蝇逃逸回路（1871 神经元 / 45524 突触边）训练成 LIF 脉冲神经网络，装进 3D 世界里开无人机——威胁来袭时，果蝇脑在毫秒级决定"何时逃、往哪逃"，适配层把它翻译成 4 个电机的指令。
> 纯前端演示：零构建、零 fetch、双击 `web\world.html` 即可运行（file:// 协议直接可用）。

## 30 秒速览

1. 双击 `web\world.html`
2. 控制台里把载体选成"**无人机**"
3. 按下"**自动威胁**"——看果蝇脑开飞机躲攻击

## 架构

```
FlyWire FAFB (1871 神经元)
   │  LPLC2 视觉膨胀 + JO-B/JO-C 风觉
   ▼
LIF SNN（拓扑=连接组，边权可训练）
   │  GF 巨纤维触发 + 逃逸方向向量
   ▼
DroneAdapter（PID 姿态环 + X 型混控）
   │  状态机：巡航 / 逃逸 / 飞手 / 降落
   ▼
4 电机 ──► DronePhysics（质量/升力/阻力/姿态）──► 3D 世界
```

## 指标（v3 口径，与 `web\results.json` 一致）

| 指标 | 数值 |
|---|---|
| 触发准确率 | 0.799 |
| 避障成功率 | 0.903 |
| 方向误差 | 10.7° |
| GF 潜伏期 | 7.65 ms |
| 抖动（200 试，连跑 3 次取中位数判定，四项全达标） | 92.0% |

D 组物理积分口径（单列，不与方向口径混用）：果蝇身体 **0.698** | 无人机身体 **0.694** | 静止基线 **0.685**。

## 诚实边界（三条）

1. 巡航/避障的"转向偏好"是程序基线 + 神经逃逸混合，**不是全神经生成**（全神经巡航 = 完整果蝇路线阶段 2，见 [`v4\完整果蝇设计方案.md`](v4/完整果蝇设计方案.md)）。
2. 飞行动力学参数为**演示级**（小四旋翼典型值），未做真机标定。
3. 脑的输出是"**逃逸方向向量**"，适配层把它翻译成电机指令（生物对应：脑 → 胸神经节 → 飞行肌）。

## 引用与致谢

- 数据：FlyWire / Princeton，见 Dorkenwald et al., *Nature* **634**:124–138 (2024)。
- **勘误**：坊间所谓"谷歌开源了果蝇全脑"，开源的是 **Neuroglancer 查看器**，数据本身来自 **FlyWire**（Princeton 等）。
- 工程参考：[snedea/flybrain](https://github.com/snedea/flybrain)（浏览器内 FlyWire 全脑 LIF 实时仿真，MIT）。
- 代码许可：MIT（见 [`LICENSE`](LICENSE)）。

## 复现

```powershell
$env:PYTHONPATH = "pylibs"
py -3.13 -X utf8 extract_circuit.py        # ① 抽取子网络
py -3.13 -X utf8 train_snn.py              # ② 训练 + 评估 + 导出
py -3.13 -X utf8 web\make_web_data.py      # ③ 生成前端数据
node web\evaluate.js                       # ④ 回归评估（结果写 web\results.json）
```

两个哈希锚点（改动任何 SNN 数值即回滚）：

- `web\results.json` SHA256 = `0E1339A088EF3B5820DCA342B989249403BE41D5A3F7EDB9DDB1D1A91C217E9E`
- `web\jitter_sim_log.md` SHA256 = `5C66B3D05A92F17961EC9EFC7F8406BFA6117CB04F5F47260F544AB52B8072AE`

## 演示动图

![demo](docs/demo.gif)

> 占位：`docs/demo.gif` 由阶段 6 自动截图脚本产出后替换。

## GitHub Pages 部署

仓库 **Settings → Pages → Source 选 `main` 分支 → 目录 `/web`** → Save，稍候即可通过 `https://<用户名>.github.io/<仓库名>/` 访问在线演示（本地仍以 file:// 双击 `web\world.html` 为准）。

---

## English Summary

**FlyWire Connectome Brain → Drone Control**: an escape circuit extracted from the public FlyWire fruit-fly connectome (1,871 neurons / 45,524 synaptic edges) is trained as a LIF spiking network and pilots a drone in a 3D world. On looming threat the fly brain decides *when* and *which way* to escape in milliseconds; an adapter layer (PID attitude loop + X-quad motor mixing) turns that direction vector into four motor commands.

- **Quick start**: double-click `web\world.html`, pick "Drone", press "Auto Threat" — zero build, zero fetch, works straight from `file://`.
- **Metrics (v3)**: trigger 0.799 / success 0.903 / direction 10.7° / latency 7.65 ms / jitter 200-trial median 92.0% all-pass. Physical-integration group: fly body 0.698 | drone body 0.694 | stationary baseline 0.685.
- **Honest limits**: cruise/obstacle-avoidance steering is a program baseline mixed with neural escape (not fully neural); flight dynamics are demo-grade, not calibrated on real hardware; the brain outputs an escape direction vector, not motor commands.
- **Data**: FlyWire (Dorkenwald et al., *Nature* 634:124–138, 2024). Clarification: "Google open-sored the fly brain" actually refers to the **Neuroglancer viewer**; the data comes from **FlyWire**. Engineering reference: [snedea/flybrain](https://github.com/snedea/flybrain). Code under MIT.
