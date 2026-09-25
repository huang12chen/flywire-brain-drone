# 完整果蝇设计方案（v4）——让果蝇成为完整生物：身-脑闭环感受物理世界，并回答"果蝇神经元能不能开无人机"

> 调研+设计文档（只写方案，不含实现代码）。
> 调研日期：2026-09-23。调研方式：网页搜索接口欠费（HTTP 402）不可用，全部在线信息均以 `web_fetch` 抓取公开页面（PyPI / GitHub raw / GitHub API / arXiv / 官网）核实；**核实不到的一律标注"未能在线核实"，不编造链接与结论**。
> 项目现状：FlyWire FAFB 全脑连接组约束的逃逸子回路 SNN（1871 神经元 / 45524 边：LPLC2 视觉 210 + JO-B/JO-C 风觉 325 + GF 巨纤维 2 + 下游输出 169），拓扑固定、只训权重的 LIF SNN，输出"该不该逃（GF 放电）+ 往哪逃（方向读出）"。v3 指标：融合触发 0.799 / 避障成功率 0.903 / 方向误差 10.7° / GF 潜伏期 7.65 ms（见 `REPORT.md`）。
> 现状缺口：3D 世界里果蝇/无人机**只换壳**——挥翅膀/转旋翼是手写动画，脑只做逃逸决策，**没有身-脑闭环**（自己动作引起的视觉/风觉变化不回馈脑）。

---

## 0. 摘要速览

1. **全身仿真标杆三选一各有分工**：**FlyGym**（果蝇全身力学+复眼+力学反馈，Apache-2.0，极活跃）抄身体与感觉接口；**flybrain**（同源 FlyWire 全脑 139,255 神经元 LIF 浏览器实时，MIT）证明**全脑规模算力不是瓶颈**并提供全脑打包/可视化借鉴；**OpenWorm**（线虫神经+身体闭环，MIT）是"连接组→身体闭环"的组织方式先例。
2. **关键架构判断**："果蝇脑开无人机"的答案不是让 SNN 直接稳定四旋翼，而是**分层**：果蝇脑只输出"触发+方向+转向不对称"的伪肌肉级指令，**稳定环留给飞控**（PX4，生物学上相当于腹神经索+平衡棒反射），飞手接管仲裁器永远最高优先级。Neuro-LIFT（arXiv:2501.19259）已在真机上验证"事件相机+SNN+无人机"可行。
3. **闭环改造核心**：现在的 cues 是被动合成的，必须改成**由身体状态实时生成**——自运动进入光流（膨胀率）与风觉（相对气流）公式，再加一份**传出副本（efference copy）**做自运动补偿。
4. **诚实边界**：翅膀气动、翅肌/CPG 运动神经元**不在 FAFB 全脑表内**（在腹神经索 VNC），需合成模型顶替或外源 VNC 连接组补充；闭环稳定性与 sim-to-real 是最大不确定项。
5. 推荐路线：**阶段 1（6–10 人周）最小闭环先飞起来 → 阶段 2（12–20 人周）身脑合一 → 阶段 3（8–14 人周）无人机适配层（SITL/HIL）→ 阶段 4（12–20 人周）网笼真机**。

---

## 1. 开源项目调研（逐项：链接 / 一句话价值 / License / 活跃度）

### 1.1 全身 / 全脑仿真标杆

| 项目 | 链接 | 一句话价值 | License | 活跃度（本次核实） |
|---|---|---|---|---|
| **FlyGym（NeuroMechFly v2）** | https://pypi.org/project/flygym/ 、https://github.com/NeLy-EPFL/flygym 、文档 https://neuromechfly.org/ | 成年果蝇**物理全身数字孪生**：micro-CT 生物力学模型+复眼视觉（六角小眼光栅）+嗅觉+脑/VNC 分层控制接口（下行/上行表征）+足端粘附+力学反馈（关节角、主动力、接触力） | **Apache-2.0**（PyPI license 字段核实） | **极活跃**：PyPI 最新版 2.1.0 发布于 2026-06-24；2.x 于 2026-03 全面重写（CPU ~10x、GPU(Warp/MJWarp) ~300x 加速）；EPFL Ramdya 实验室维护；Python ≥3.12 |
| **flybrain** | https://github.com/snedea/flybrain 、在线版 https://flybrain.app | **浏览器内 FlyWire FAFB v783 全脑 LIF 实时仿真**：139,255 神经元 / 270 万连接，Web Worker 积分 + WebGL 放电可视化（按 Sensory/Central/Drives/Motor 分组），行为从连接传播中"涌现"（觅食、触碰惊跳、趋光、趋温） | **MIT**（GitHub API spdx=MIT；README 指向 license.md） | **活跃**：2026-03-27 创建，2026-08-13 最后推送，159 star / 32 fork，JavaScript；数据引 Dorkenwald et al., Nature 634:124–138 (2024) |
| **worm-sim** | https://github.com/heyseth/worm-sim 、在线版 https://sethm.me/worm-sim/ | 浏览器内 *C. elegans* 302 神经元连接组仿真+可交互运动，**flybrain 的前身**（注意：拼写是 `heyseth`；`heyeseth/worm-sim` 经 API 核实为 404，不存在） | **MIT**（GitHub API 核实） | 持续可用：2018-02 创建，2026-06-07 最后推送，425 star / 37 fork |
| **OpenWorm** | https://github.com/openworm/openworm 、官网 https://openworm.org | 线虫 *C. elegans* **首个全身计算模型**：c302 神经系统模型 + Sibernetic 3D 软体身体并行联跑出行为，"连接组→身体闭环"工程组织先例 | **MIT**（LICENSE 文件核实） | 长期项目（2012 年起），主仓库以 Docker 栈集成+项目级 issue 为主；具体 star/commit 数据本次抓取页面被截断，**未能在线核实** |

**flybrain 与本项目的关系（重点借鉴分析）**：
- **同源数据**：flybrain 用的 FlyWire FAFB v783 与我们本地 `connections_princeton.csv.gz` 等 4 表同源（Dorkenwald et al. 2024）。它做"整脑涌现行为"，我们做"子回路约束可训练+闭环身体"——**互补而非竞争**。
- **直接抄的三件事**：① 全脑连接组的**二进制打包与按区分组**（`data/neuron_meta.bin.gz` 思路），比逐行 CSV 快几个量级；② **Web Worker + WebGL 放电面板**（13.9 万神经元实时画出来），可移植到我们的 `web\world.html` 沙盒做"全脑视角"调试面板；③ **感觉-交互接口设计**（feed / touch / air / light / temp 五类刺激直接打进感觉群）——正是我们"闭环感觉编码"需要的接口形状。
- **必须泼的冷水**：flybrain 的"行为涌现"是**演示级证据**，其权重规则、感觉编码细节本次未能在线核实（README 未写明）；不能拿它的涌现行为当科学结论，也不能假定它的权重生成方式与我们的 `W=sign(NT)·log1p(syn_count)` 一致。
- **规模启示**：13.9 万神经元 / 270 万连接的 LIF **在浏览器里都能实时跑** → 我们担心的"全脑规模算力"其实不是拦路虎（详见 §5）。

### 1.2 物理引擎

| 项目 | 链接 | 一句话价值 | License | 活跃度（本次核实） |
|---|---|---|---|---|
| **MuJoCo** | https://mujoco.org/ 、https://github.com/google-deepmind/mujoco | 面向接触优化的高速精确物理引擎，支持**肌肉/肌腱/滑块曲柄执行器**（正是果蝇"伪肌肉指令"需要的原语），FlyGym 的底层 | **Apache-2.0**（LICENSE 文件核实） | Google DeepMind 维护，持续发布（FlyGym 2.x 已用其 Warp/MJWarp GPU 后端） |
| **Brax / MJX** | https://github.com/google/brax | JAX 全可微物理+RL 训练栈（PPO/SAC/ARS/ES/APG），适合大规模并行训策略 | **Apache-2.0**（LICENSE 文件核实） | ⚠️ README 官方警告：自 0.13.0 起**只有 `brax/training` 在维护**，物理仿真官方建议改用 **MJX**（`mujoco_mjx`）或 **MuJoCo Warp**、场景用 MuJoCo Playground |
| **Rapier** | https://github.com/dimforge/rapier 、https://rapier.rs | Rust 2D/3D 物理引擎，SIMD 批量约束求解、并行管线、JS/Python 绑定，适合嵌进 Web 端 3D 世界 | **Apache-2.0**（README 徽章核实） | 活跃（dimforge 维护，Python 绑定开发中——README 自述） |

### 1.3 神经仿真

| 项目 | 链接 | 一句话价值 | License | 活跃度（本次核实） |
|---|---|---|---|---|
| **Brian2** | https://pypi.org/project/brian2/ 、https://brian2.readthedocs.io | 时钟驱动 SNN 仿真器，方程即代码，最适合**约束拓扑 LIF + 自定义突触动力学**的科研迭代 | **CeCILL-2.1**（PyPI 核实） | 活跃：2.10.1 发布于 2025-12-05，多平台 wheel |
| **NEST** | https://github.com/nest/nest-simulator 、https://nest-simulator.org | 大规模脉冲网络仿真（笔记本到超算），模型库丰富 | **GPLv2+**（README 核实） | 活跃：v3.7 系列，conda/pip/Docker 多渠道，README 显示持续 CI |
| **GeNN** | https://github.com/genn-team/genn | GPU 代码生成（CUDA/HIP）SNN 仿真，含**昆虫蘑菇体 MNIST 分类**示例，适合上 GPU 大规模 | **LGPL-2.1**（LICENSE 文件核实） | 活跃：5.4.0 发布版+master pip 直装，Docker 镜像 |
| **SpikingJelly** | https://pypi.org/project/spikingjelly/ | PyTorch SNN 框架：LIF、代理梯度、STDP、ANN→SNN、事件数据集全家桶（DVS 系列） | PyPI 标注 **Other/Proprietary License**（未给标准 SPDX；商用需自查） | 稳定版 0.0.0.0.14（2023-03），但 2.0.0 预发布线持续更新（2.0.0rc1 于 2026-08-29） |

> 现有训练管线用的是 **snntorch**（`train_snn.py`，`surrogate.fast_sigmoid()`），本次未列入联网核实清单；迁移建议见 §2.3。

### 1.4 神经形态无人机 / 飞控 / 四旋翼仿真

| 项目 | 链接 | 一句话价值 | License | 活跃度（本次核实） |
|---|---|---|---|---|
| **Neuro-LIFT** | https://arxiv.org/abs/2501.19259 | **真机佐证**：Parrot Bebop2 四旋翼上的实时神经形态导航框架——事件相机+SNN+LLM 语音指令，动态环境避障 | 论文 CC BY-NC-ND 4.0（arXiv 核实）；**代码仓库未能在线核实**（尝试 `github.com/AmoghJoshi/Neuro-LIFT` 返回 404） | IJCNN 2025 接收；v2 于 2025-04-26 修订 |
| **gym-pybullet-drones** | https://github.com/utiasDSL/gym-pybullet-drones | 极简四旋翼 Gym 环境（PyBullet 物理+PID/MRAC 控制+SB3 强化学习示例+**Betaflight SITL** 联调），适合做适配层快速验证 | 具体 License **未能在线核实**（README 未标注；抓取的为 learnsyslab 重构版） | 活跃：入选 GitHub Maintainer Spotlight 2026；支持 gymnasium / SB3 2.0 |
| **PX4-Autopilot** | https://github.com/PX4/PX4-Autopilot | 工业级开源飞控：**SITL / HIL / 真机同栈**，offboard 控制接口，安全失效保护（RC 丢失→Land/RTL）完善 | **BSD 3-Clause**（LICENSE 文件核实） | 极活跃（2025 年版权行仍在更新；无人机事实标准之一） |
| **ArduPilot** | https://github.com/ArduPilot/ardupilot | 另一大开源飞控，Copter 全功能（备选，与 PX4 二选一即可） | **GPLv3**（COPYING.txt 核实） | 活跃（长期社区项目；具体版本号本次未抓取） |

### 1.5 对比与借鉴结论（FlyGym vs flybrain vs OpenWorm）

| 维度 | FlyGym | flybrain | OpenWorm |
|---|---|---|---|
| 身体 | ✅ 全身生物力学（micro-CT，六足+关节+粘附+接触力） | ❌ 无物理身体（2D 交互演示） | ✅ 软体全身（Sibernetic） |
| 脑 | ❌ 不含连接组网络（提供脑/VNC 接口） | ✅ 全脑 139K LIF（同 FlyWire 数据） | ✅ c302（302 神经元） |
| 感觉 | ✅ 复眼视觉+嗅觉+力学反馈 | ✅ 五类刺激接口（光/温/风/触/食） | 简单 |
| 闭环 | ✅ 身-脑接口规范（下行/上行） | ✅ 刺激→行为涌现 | ✅ 神经→软体并行联跑 |
| 我们抄什么 | **身体、复眼编码、力学反馈、脑/VNC 分层接口** | **全脑规模打包、放电可视化、刺激接口形状、"全脑可实时"的信心** | **神经-身体双进程联跑的工程组织方式** |
| 我们不抄什么 | 它无飞行/双翅模块（README 组件列表未列；**未能在线核实**有无飞行扩展） | 它无物理身体、权重规则未公开、"涌现"不可作科学结论 | 线虫软体动力学与果蝇无关 |

---

## 2. 完整果蝇蓝图

### 2.0 现状差距（三个缺口）

1. **脑缺口**：只有逃逸子回路（1871 节点）。没有视觉运动层、没有转向/航向系统、没有节律发生器、没有翅肌运动神经元。
2. **身体缺口**：翅膀/旋翼是手写动画，没有力学；六足、翅根关节、肌肉都不存在。
3. **闭环缺口**：训练与运行时的视觉膨胀、风压 cues 是**按威胁运动公式被动合成**的（`dθ/dt = −2s·ṙ/(r²+s²)`、`u ∝ s²v/r²`，只含威胁运动项）；果蝇自己一动，光流/风觉**不会变**，脑收不到自运动的感觉后果。

### 2.1 还缺哪些神经元群（含：能否从 FlyWire 本地表提取）

**数据前提（诚实声明）**：本地 4 表来自 **FAFB = Full Adult Female Brain，只有脑**。翅肌/足肌运动神经元（TTMn、DLMn 等）、平衡棒传入、节律发生器（CPG）都在**腹神经索（VNC/胸节）**——`README.md` 已声明"果蝇胸段的 TTMn/DLMn 不在该全脑数据集中"。因此下表分两类：**A 类=可从本地表提取（脑内）**；**B 类=本地表没有，用合成模型顶替或外源 VNC 连接组补充**（VNC 数据集本次**未能在线核实**具体下载地址，获取途径需另行调研）。

| 神经元群 | 作用 | 类别 | 提取/构造方法（沿用 `extract_circuit.py` 思路） |
|---|---|---|---|
| 视觉运动层：LC 家族（LC4/LC6/LC11/LC12 等 lobula columnar）、T4/T5（局部运动检测）、Mi1/Tm1/Tm2/Tm3/Tm9/Tm20（medulla 中间层）、HS/VS（lobula plate 宽场）、VPN（视脑桥） | 光流/膨胀/物体运动的**前级编码**，把复眼像素流变成 LPLC2 之外的运动特征（含转向所需的旋转光流） | **A** | `primary_type ∈ TYPE_SETS['visual_motion']` 做种子 → `hop()` 前向/后向 1–2 跳 → 与现有 1871 节点求交/并；种子类型名以本地表实际 `distinct primary_type` 为准（教训：GF 实际叫 `DNp01`） |
| 听觉/风觉扩展：JO-A…JO-E 全型、触角机械感觉附属 | 风/声/重力多轴编码 | **A**（JO-B/JO-C 已入网） | `primary_type LIKE 'JO-%'` 全收，替换现有 `WIND_TYPES` |
| 视觉/头部状态：单眼（ocelli）通路、复眼边缘神经元 | 光照水平/地平线→姿态参考 | **A**（以本地表实际类型为准） | `classification.csv` 的 `class/sub_class` 含 'ocell' 等关键字筛（具体字段值落地时先 `Counter` 统计） |
| 中央复合体 CX：E-PG（航向罗盘）、P-EN/P-EG、PFL3（转向控制）、FC2、Δ7、PB/EB/FB 神经元 | **转向与航向**——"往哪逃"从反射升级为有状态的转向控制器 | **A** | `primary_type ∈ {'E-PG','P-EN','P-EG','P-FL3','FC2','Δ7',...}`（名称以本地表为准）+ `class` 关键字 'central complex' |
| 下行神经元全集：DNp/DNa/DNm 系列（含 DNp01=GF） | 脑→身体的**唯一出口**，运动指令载体 | **A** | `classification.csv` 中 `super_class == 'descending'` **全量**（现仅取了 GF 下游 2 跳内 169 个） |
| 脑内运动神经元 | super_class='motor' 全量 | **A** | 同上，`super_class == 'motor'` 全量 |
| 调质群：DA/5-HT/OCT 神经元（DOPaminergic、Serotonergic、octopaminergic） | 唤醒/增益调节（GF priming 的生物学基础） | **A** | `neurons.csv.gz` 的 `nt_type ∈ {DA, SER, OCT}` + `class` 关键字 |
| **翅/足 CPG（半中心振荡器）** | 挥翅节律、步态节律 | **B** | **合成模型**：2–6 个半中心振荡器（half-center oscillator）+ 参数驱动（"起飞/悬停/逃逸"三档），伪代码级设计即可，无连接组依据——诚实标注 |
| **翅肌/转向肌运动神经元**（TTMn 跳跃、DLMn 间接飞行肌、8 个转向肌 MN） | 运动指令→肌肉 | **B** | **合成模型**：把 A 类 DN 输出直接接到"伪肌肉指令空间"（§3.1）；有条件时外源 VNC 连接组替换 |
| 平衡棒（haltere）传入+反射 | 角速度感觉（=生物陀螺仪） | **B** | **合成模型**：IMU 角速度→感觉编码（简单可做）；其 VNC 反射弧外源数据补齐 |
| 足/翅本体感觉（chordotonal、campaniform） | 关节角/翅载荷反馈 | **B** | 先用 FlyGym 力学反馈接口（关节角/接触力）代替；器官级编码后续补 |

**扩展提取脚本设计（沿用 `extract_circuit.py` 全部机制，只加不改）**：
- 复用：`load_tables()`（4 表读取、neuropil 行合并 `aggregate_edges()`）、`hop()` 图扩展、`syn_count ≥ 3` 弱边剔除、递质符号 `NT_SIGN = {ACH:+1, GABA:−1, GLUT:−1(昆虫), DA/SER/OCT:+1 调质}`、权重 `W = sign(NT)·log1p(syn_count)` + 突触后总入强度归一化、`role_of()` 角色标注、sparse JSON+CSV 双格式导出。
- 新增：① `TYPE_SETS` 多种子集（上表 A 类）；② `super_class` 全量收编（descending / motor / 调质）；③ 多 hub 图（GF + CX 航向群），短路径剪枝参数从"≤3 边"放宽为分群可配（视觉运动层用 ≤2 边紧密层，CX 用全连通子图）；④ 输出 `escape_network_v2.json`，节点角色扩展为 `input_vision / input_wind / visual_motion / heading / modulatory / hub_gf / dn / motor / cpg_port / interneuron`。
- 预期规模：全脑 138,327 细胞表、534 万连接行的前提下，以上 A 类并集+短路径层预计 **1–3 万节点 / 数十万边**（估算，以实际跑出为准）；参照 flybrain 139K 神经元实时运行的先例，仿真算力**可行**。
- 落地第一步（写代码前必做）：对 `consolidated_cell_types.csv.gz` 跑一次 `distinct primary_type / additional_type(s)` 频次统计，确认各群真实名称再定 `TYPE_SETS`（GF=DNp01 的教训必须制度化）。

### 2.2 身体模型（六足 + 双翅的力学/关节）

**推荐：飞行优先、走路后补的两段式身体。**

**方案 A（阶段 1，起步）：6-DoF 刚体 + 扑翼气动平均（wingbeat-averaged wrench）**
- 果蝇胸段为刚体（质量 ~1 mg、翅展 ~2 mm 量级），状态 = 位姿 + 线/角速度；
- 双翅各 3 个自由度：**翅根俯仰（pitch）+ 行程（stroke）+ 展翅（deviation）**，由 CPG 输出左右**拍频（~200 Hz 量级）、振幅、攻角、行程不对称度**；
- 气动：准定常叶素/气动力系数模型（升力、阻力、俯仰力矩关于拍频²、振幅、攻角的代数式），**按翅拍周期取平均**得到合外力/力矩（1 kHz 控制步内视为常值 wrench）——刻意**不做**柔性翅 CFD（成本与收益不匹配，见 §5）；
- 执行器原语直接用 MuJoCo 的 muscle/tendon 语义或等价的"伪肌肉→wrench"映射，保证与阶段 2 的 FlyGym 身体同接口。

**方案 B（阶段 2，完整体）：FlyGym 生物力学身体 + 自研双翅模块**
- 抄 FlyGym 的六足骨架/关节/足端粘附/接触力（Apache-2.0，可直接改）；关节：每足 3 节（coxa-trochanter-femur-tibia-tarsus 简化链），走路 CPG 驱动；
- 双翅气动模块（方案 A 的）挂在 FlyGym 身体 thorax 上；⚠️ FlyGym README 组件列表**未含**双翅/飞行模块，是否已有社区飞行扩展**未能在线核实**——按"没有"规划；
- 力学反馈接口照抄 FlyGym：关节角、主动力、接触力、自定义解剖点位——这就是"本体感觉"的现成来源。

### 2.3 闭环设计：自运动 → 光流/风觉变化 → 感觉编码 → 脑 → 运动指令

**核心改造：cues 由"被动合成"改为"身体状态实时生成"。**

每仿真步（1 kHz 物理 / 感觉更新 100–500 Hz / 脑 dt=1 ms）：

1. **世界+身体状态**（威胁位姿、自身位姿/速度/角速度、上一步运动指令）→
2. **感觉编码器（新模块 SensorGen）**：
   - **光流/膨胀**：复眼视场（FlyGym 六角小眼光栅或简化视锥）内每个威胁的视角 `θ = 2·atan(s/r)`，其变化率**同时含威胁运动与自运动**：`ṙ` 是相对距离变化（自己飞向障碍物同样让 `θ` 暴涨）；再叠加自运动产生的**全局光流场**（旋转分量→转向反馈）；
   - **风觉**：`u_rel = v_air − v_body − v_downwash(CPG_cmd)`——**相对气流**，自己前飞/挥翅下洗流都进 JO 编码；
   - **平衡棒/陀螺**：`ω_body` → 角速度编码（合成）；
   - **力学反馈**：FlyGym 关节角/接触力 → 本体编码（阶段 2）。
3. **脑（SNN）**：现有 1871 节点网（阶段 1）→ +§2.1 各群（阶段 2）。输入 = 上述编码后的脉冲；输出 = GF 放电（触发）+ 方向读出 + 转向群左右不对称 + DN 群发放。
4. **运动指令（新模块 MotorGen）**：输出**伪肌肉指令** `M = {升力档位, 俯仰, 滚转, 偏航, 逃逸爆发脉冲, 挥翅 CPG 参数}`；驱动 CPG/身体执行。
5. **传出副本（efference copy，阶段 2）**：`M` 的副本反馈给感觉编码器——
   - 抵消 JO 编码中的**自致风觉**（挥翅下洗+自飞相对风）；
   - 抵消视动光流中的**自致旋转分量**（不然自己一转头满视野光流，LPLC2 类通道会被自运动污染，逃逸回路误触发）。
   - 这是从"能动"到"动得稳"的关键一环；果蝇生物学有旁证（ corollary discharge ），但**在我们连接组子图里的具体通路无数据依据，按合成模型处理**。

**闭环验收判据（阶段 1 即适用）**：
- 自转 30°/s：光流编码通道放电率随 `ω` 单调变化（自运动可见）；
- 前飞 0.5 m/s：JO 编码随 `u_rel` 变化（自运动可见）；
- GF 触发逃逸后 **2 个脑窗（≥24 ms）内**，编码器能测到逃逸动作引起的新光流/风觉（闭环成立的直接证据）；
- 开/关感觉生成器的"自运动项"做对照：关掉时出现自运动误触发（预期），打开时误触发率下降（闭环价值量化）。

**训练策略建议**：保留现有"拓扑固定、只训权重"约束不变；新增群先**冻结解剖权重**只训读出/接口增益，再放开微调；训练素材从"合成威胁轨迹"升级为"闭环 rollout 采样"（自己飞出来的光流才是真实分布——这是 sim-to-real 之外的第一层分布差修复）。仿真器选型：现有 snntorch 小网可继续；上到 §2.1 的 1–3 万节点规模后迁 **Brian2**（科研迭代）或 **GeNN**（GPU 批量 rollout），NEST 备选（GPLv2+ 传染性最强，商用无人机产品化慎用）。

---

## 3. "果蝇脑开无人机"的答案路径

### 3.1 运动指令 → 适配层（控制分配：伪肌肉指令 → 四旋翼转速/姿态）

**架构判断（本方案最重要的结论）**：果蝇神经元**不需要**学会稳定四旋翼。生物果蝇里稳定飞行也不是 GF/脑完成的——是胸节 CPG+平衡棒反射+翅肌机械负反馈（都在 VNC）。对应到无人机：

```
[果蝇 SNN 脑] --伪肌肉指令 M--> [适配层 Adapter] --期望 wrench/姿态--> [飞控稳定环 PX4（≈VNC+平衡棒）] --> [电调/电机] --> [机身]
      ^                                  |
      |------ 感觉编码器(SensorGen) <-----+-- IMU/光流/气流计(≈复眼/JO/平衡棒) <--+
                                        |
                               [仲裁器 Arbiter] <-- 飞手 RC（最高优先级）
```

- **伪肌肉指令空间 → 四旋翼映射**（生物动作语义 → 控制语义）：
  - `逃逸爆发脉冲（GF 放电）` → 短时推力阶跃 + 沿逃逸方向的 saccade（角速度脉冲）——对应果蝇"跳跃起飞+快速转体"；
  - `左右转向不对称（CX/转向群）` → 偏航/滚转角速度指令；
  - `升力档位（CPG 拍频/振幅）` → 油门/总距；
  - `悬停/巡航档` → 定高/定速保持（由飞控实现，果蝇脑只给"目标"）。
- **控制分配**：期望 wrench（合力+三轴力矩）→ 四旋翼混控（控制分配矩阵）→ 4 路电机转速；接 **PX4 offboard**（姿态/角速度设定点或 actuator 直驱）即可，SITL/真机同接口。
- **佐证**：Neuro-LIFT（arXiv:2501.19259）证明事件相机+SNN 导航可跑在真机 Bebop2 上、延迟敏感任务可由神经形态栈完成；其"高层规划+底层执行"分层与本方案同构（其 LLM 层我们不需要）。

### 3.2 sim-to-real 分层路线

| 层 | 内容 | 出口判据 |
|---|---|---|
| **L0 仿真闭环** | §2 的果蝇身体+SNN+SensorGen 在 3D 世界（现 `web\world.html` 沙盒 + MuJoCo/自研动力学）跑通闭环；同一套伪肌肉接口 | 闭环验收 4 条全过（§2.3）；逃逸演示可复现 |
| **L1 SITL** | 适配层接 **PX4 SITL**（或 gym-pybullet-drones 做快速原型）：果蝇脑驾驶四旋翼穿越突袭威胁场景 | 100 次蒙特卡洛突袭：触发-逃逸成功率、无飞手干预无碰撞 ≥目标值（建议 ≥80% 起步） |
| **L2 硬件在环 HIL** | 真实飞控板 + 仿真器 + 真实传感器回路（事件相机 DVS 或灰度光流、皮托管/热线风速计）；验证延迟预算（感知→指令 ≤ 20 ms） | 端到端延迟与丢包测试通过；接管逻辑时序验证（接管延迟 ≤ 1 帧 RC） |
| **L3 真机** | 小型四旋翼（250 mm 轴距以内）+ 网笼 + 低速；真实突袭（软质摆锤/气流扰动） | 真机触发-逃逸成功率报告；**接管演练：任意时刻飞手介入成功率 100%** |

**sim-to-real 差距治理（每一层都要做）**：传感器域随机化（光增益、风噪、DVS 噪声）；时延抖动注入；执行器饱和/电机一阶延迟建模（gym-pybullet-drones TODO 列表也承认 motor delay 是普遍坑）；闭环 rollout 数据再训练（而不是纯合成数据）。

### 3.3 飞手接管安全底线（不可妥协项）

1. **仲裁器优先级固定**：`飞手 RC > 安全罩（geofence/限速/限角/禁飞区）> 果蝇脑建议`。果蝇脑输出永远只是"建议"（触发+方向），经仲裁器才生效。
2. **接管通道独立**：RC 接收机直连飞控（不经过 SNN 计算栈），PX4/ArduPilot 原生 RC override；接管切换**无条件、无延迟、单向**（飞手→自动可回，自动抢不回飞手）。
3. **失效保护默认开**：RC 丢失 → Land/RTL；计算栈死机 → 飞控独立悬停/降落（SNN 栈与飞控解耦，watchdog 心跳）；硬件 kill switch。
4. **飞行包线罩**：速度/倾角/高度/半径限幅在飞控层执行（安全罩比果蝇脑优先级高，防止逃逸机动撞护栏）。
5. **试飞纪律**：L3 全程网笼+低速+双人（飞手+安全员）；先"影子模式"（果蝇脑只记录建议不执行）校准，再放权。

---

## 4. 分阶段路线图

### 阶段 1：最小闭环（果蝇身体 + CPG 挥翅 + 自运动光流回馈，3D 世界先飞起来）
- **内容**：6-DoF 果蝇飞行体（方案 A）+ 半中心 CPG 挥翅 + SensorGen（光流/风觉/陀螺）+ 现有 1871 节点 SNN 入环 + 伪肌肉指令空间；`web\world.html` 沙盒里从"只换壳"变成"真的自己飞"。
- **可验收产物**：① 闭环演示录屏（威胁→GF→逃逸→自运动光流变化→二次决策）；② §2.3 四条闭环判据的量化日志（含自运动项开关对照）；③ 伪肌肉接口规范文档（给阶段 3 的合同）。
- **工作量**：**6–10 人周**（动力学+CPG 3 周、SensorGen 3 周、入环调试 2–4 周）。
- **主要风险**：扑翼平均 wrench 参数不准导致飞姿"不像果蝇"（可接受，物理合理即可）；自运动光流计算开销（用几何公式直算可规避）；闭环延迟预算超 20 ms（需压感觉更新频率）。

### 阶段 2：身脑合一（更多神经元群入环）
- **内容**：扩展提取脚本跑 A 类新群（视觉运动层、CX 转向、DN 全集、调质群）→ `escape_network_v2`；efference copy 自运动补偿；（可选）引入 FlyGym 六足身体+力学反馈；训练升级为闭环 rollout 采样；仿真器视规模迁 Brian2/GeNN。
- **可验收产物**：① `escape_network_v2.json` + 提取报告（各群计数、连接统计）；② 有转向/航向行为的飞行演示（不再只会"背离逃"）；③ efference copy 开/关对照实验（自运动误触发率下降的量化）；④ 与 flybrain 同参数全脑仿真交叉验证的 spike 面板截图（借鉴其可视化）。
- **工作量**：**12–20 人周**（提取 3–4 周、入环+训练 6–10 周、对照实验 2–6 周）。
- **主要风险**：新群权重只有解剖代理（log1p 突触数）导致网络不稳（缓解：冻结+读出微调、能量正则）；CX 群在本地表的类型命名与文献不一致（缓解：先跑 distinct 统计）；规模上到数万节点后训练收敛变慢。

### 阶段 3：无人机身体适配层（仿真闭环 → SITL/HIL）
- **内容**：Adapter（伪肌肉→控制分配→PX4 offboard）+ 仲裁器（飞手优先）+ L1 SITL 场景库 + L2 HIL 台架（真实飞控板+传感器回路）；Neuro-LIFT 式事件相机接入评估。
- **可验收产物**：① SITL 蒙特卡洛报告（100 次突袭成功率/接管记录）；② 端到端延迟预算报告（感知→电机）；③ 接管逻辑形式化文档+演练记录；④ 可演示 demo（网页沙盒里"果蝇脑开四旋翼"）。
- **工作量**：**8–14 人周**（适配层 3–5 周、SITL 场景 3 周、HIL 2–6 周）。
- **主要风险**：伪肌肉→四旋翼映射语义不对（逃逸机动翻机）——用安全罩限角兜底；PX4 offboard 时序坑（锁步仿真）；HIL 设备/接口准备周期不可控（外购件风险）。

### 阶段 4：真机（网笼 → 受限场地）
- **内容**：小型四旋翼改装（飞控+Jetson 级边缘机跑 SNN+DVS/光流/风速计）；影子模式→放权；真实突袭测试；全程飞手接管保障。
- **可验收产物**：① 真机触发-逃逸成功率报告（含误触发率）；② 接管演练 100% 成功记录；③ 应用结论白皮书："果蝇神经元能否开无人机"的**定量答案**（成功/失败条件、差距归因）。
- **工作量**：**12–20 人周** + 硬件与试飞窗口（日历时间受天气/场地/安全审批制约）。
- **主要风险**：sim-to-real 差距（风噪、振动、DVS 噪声）压垮感觉编码（缓解：层间域随机化+真机数据回灌微调）；边缘算力不足（缓解：GeNN/Loihi 类部署或裁剪网络）；安全审批与场地（非技术风险但常是真正的瓶颈）。

**累计：约 38–64 人周（9–16 人月），单人全职约 1–1.5 年。**

---

## 5. 诚实评估

### 5.1 成熟可抄的（照搬/小改即可）
- **物理与身体**：MuJoCo（含 muscle/tendon 执行器）、FlyGym 生物力学身体/复眼编码/力学反馈/脑-VNC 接口规范（Apache-2.0）；
- **SNN 仿真**：Brian2/GeNN/NEST、snntorch/SpikingJelly 的代理梯度与训练套路（现管线已在用）；
- **全脑规模工程**：flybrain 证明 139K 神经元/270 万连接 LIF 浏览器实时可行——**全脑算力是成熟项**（其打包、WebGL 面板、刺激接口可直接借鉴）；OpenWorm 的神经-身体双进程联跑组织方式；
- **无人机侧**：PX4（BSD-3）SITL/HIL/真机同栈 + RC override/failsafe 原生安全机制、ArduPilot（GPLv3）备选、gym-pybullet-drones 快速原型；
- **可行性佐证**：Neuro-LIFT 已把"事件相机+SNN"飞上真机（Bebop2）。

### 5.2 前沿没把握的（无成熟先例，按实验对待）
- **果蝇双翅飞行全身仿真**：FlyGym 组件列表未含飞行/双翅模块（**未能在线核实**有无飞行扩展）；扑翼气动平均模型的参数需自行标定，"像不像果蝇飞行"没有现成基准；
- **CPG/翅肌运动神经元**：在 FAFB 表外（VNC），本地拿不到；合成 CPG 是权宜之计，VNC 连接组获取途径本次**未能在线核实**；
- **efference copy 通路**：生物学有旁证，但子图内无数据支撑，纯合成——补偿量调不好会引入新的偏差；
- **闭环稳定性训练**："拓扑固定、只训权重"的约束下，新增 1–3 万节点后闭环能否稳定飞，**没有任何先例可援**；flybrain 的"涌现"也只在无物理身体的演示里成立；
- **生物 SNN 的 sim-to-real**：Neuro-LIFT 证明了"能飞"，但"果蝇连接组决策 + 四旋翼"这个具体组合的差距曲线未知；
- **flybrain 权重/编码细节**未公开（README 未写明），不能默认与我们的权重规则一致。

### 5.3 预计难点 Top 3
1. **闭环稳定性训练**：动作→感觉→脑→动作的回路里，微小的编码偏差会被自运动放大（误触发→乱飞→更乱的光流）。缓解：efference copy 先行、能量正则、冻结拓扑只微调、逐级放权（先自运动开关对照，再全闭环）。
2. **sim-to-real 差距**：合成线索→仿真 SensorGen→真机传感器三层分布差。缓解：每层域随机化+真机数据回灌；L2 HIL 提前暴露延迟/丢包问题。
3. **神经元群的"数据-功能"错位**：FAFB 只有脑，功能必需的 CPG/MN 在 VNC；连接组拓扑≠生理权重（单只雌蝇、突触数只是代理）。缓解：合成模块明示边界（结论里永远带"合成数据/代理权重"限定语，沿用 `REPORT.md` 的科学限定语制度）。

---

## 6. 附录：本次在线核实明细（URL → 核实到的内容）

| URL | 核实结果 |
|---|---|
| https://pypi.org/project/flygym/ | flygym 2.1.0（2026-06-24 发布），Apache-2.0，NeuroMechFly v2 全套组件描述（正文见 §1.1） |
| https://raw.githubusercontent.com/NeLy-EPFL/flygym/main/README.md | 与 PyPI 同源内容；2.x 重写、10x/300x 加速、1.x 迁移至 flygym-gymnasium |
| https://arxiv.org/search/?query=FlyGym | 搜索页无可检索结果（页面仅 Tips）——FlyGym 论文以 PyPI 页给出的 Nature Methods 链接为准 |
| https://api.github.com/repos/snedea/flybrain + https://raw.githubusercontent.com/snedea/flybrain/main/readme.md | MIT；139,255 神经元 / 2.7M 连接（FlyWire FAFB v783）；Web Worker LIF + WebGL 面板；159 star/32 fork；2026-08-13 最后推送；fork 自 heyseth/worm-sim |
| https://api.github.com/repos/heyseth/worm-sim | MIT；425 star/37 fork；2018 创建、2026-06 推送；C. elegans 302 神经元浏览器仿真 |
| https://api.github.com/repos/heyeseth/worm-sim | **404**（该拼写不存在；正确为 `heyseth`） |
| https://raw.githubusercontent.com/openworm/OpenWorm/master/README.md + /LICENSE | c302+Sibernetic 联跑；MIT |
| https://mujoco.org/ + https://raw.githubusercontent.com/google-deepmind/mujoco/main/LICENSE | Apache-2.0；muscle/tendon 执行器清单 |
| https://raw.githubusercontent.com/google/brax/main/README.md + /LICENSE | Apache-2.0；官方声明物理侧转 MJX/MuJoCo Warp |
| https://raw.githubusercontent.com/dimforge/rapier/master/README.md | Apache-2.0；Rust 2D/3D；Python 绑定开发中 |
| https://pypi.org/project/brian2/ | CeCILL-2.1；2.10.1（2025-12-05） |
| https://raw.githubusercontent.com/nest/nest-simulator/master/README.md | GPLv2+；v3.7 |
| https://raw.githubusercontent.com/genn-team/genn/master/README.md + /LICENSE | LGPL-2.1；5.4.0；CUDA/HIP 代码生成 |
| https://pypi.org/project/spikingjelly/ | PyPI 标注 Other/Proprietary；稳定版 2023-03，2.0.0rc1 2026-08-29 |
| https://arxiv.org/abs/2501.19259 | Neuro-LIFT：Bebop2 真机、事件相机+SNN+LLM；IJCNN 2025；CC BY-NC-ND 4.0 |
| https://github.com/AmoghJoshi/Neuro-LIFT | **404，代码仓库未能在线核实** |
| https://raw.githubusercontent.com/utiasDSL/gym-pybullet-drones/main/README.md | gymnasium/SB3/Betaflight SITL；License **未能在线核实** |
| https://raw.githubusercontent.com/PX4/PX4-Autopilot/main/LICENSE | BSD 3-Clause |
| https://raw.githubusercontent.com/ArduPilot/ardupilot/master/COPYING.txt | GPLv3 |

**术语速查**：GF=巨纤维（DNp01，逃逸触发）；LPLC2=视觉膨胀敏感投射神经元；JO-B/JO-C=Johnston's 器机械感受（风/声/重力）；CX=中央复合体（航向/转向）；CPG=中枢模式发生器（节律）；DN=下行神经元；VNC=腹神经索（本表数据**不含**）；efference copy=传出副本（自运动补偿）；HIL=硬件在环；SITL=软件在环；DVS=事件相机。
