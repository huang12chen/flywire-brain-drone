# 数据与第三方声明 / Data & Third-Party Notices

## 神经连接组数据（Neural Connectome Data）

本仓库的 `web\snn_data.js`、`snn_trained.json` 等数据产物**衍生自 FlyWire FAFB 全脑连接组**（单只雌性成年黑腹果蝇）：

- **来源 / Source**: FlyWire Codex, `https://codex.flywire.ai`（dataset=fafb, v783）
- **引用 / Citation**: Winding, M., et al. "The neuronal wiring diagram of an adult brain." *Nature* 634, 124–138 (2024). Dorkenwald, S., et al. *Nature* 634, 103–111 (2024).
- **说明 / Note**: "Google open source" often seen alongside this data refers to **Neuroglancer** (the viewer software by Google); the data itself is produced by the FlyWire Consortium (Princeton University, HHMI Janelia, MRC LMB, Cambridge).
- 本仓库仅含逃逸相关子回路的**衍生子集**（LPLC2/JO/GF/下游共 1871 神经元、45524 条连接的加权摘要），非原始全脑数据。使用衍生数据请同样遵守 FlyWire 数据使用规范并引用上述文献。
- The files here are a **derived subcircuit summary** (not the raw whole-brain data). If you use it, please follow FlyWire data-use norms and cite the papers above.

## 第三方代码 / Third-Party Code

| 组件 | 许可 | 说明 |
|---|---|---|
| three.js r147 (`web\vendor\three.min.js`) | MIT | © 2010-2022 three.js authors |
| 参考项目 snedea/flybrain | MIT | 全脑浏览器仿真的工程思路参考（未拷贝代码） |
| 参考项目 heyseth/worm-sim | MIT | 前身线虫仿真（未拷贝代码） |

## 免责 / Disclaimer

- 飞行动力学参数为**演示级**，未做真机标定，不可直接用于真实飞行器。
- 本项目输出的控制决策**不构成**任何真实无人机的飞行控制建议；真机应用必须保留人工接管与安全护栏。
- 训练指标来自固定种子合成数据评估（复现命令与哈希锚点见 README）。
