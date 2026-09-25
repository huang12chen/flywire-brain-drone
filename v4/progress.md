# progress —— v4 阶段 A 会话日志

## 会话 1（当前）
- 恢复上下文：读 train_snn.py / web\evaluate.js / web\make_web_data.py / web\smoke_test.js / web\snn_runtime.js /
  metrics.json / REPORT.md；核实图规模 1871/45524、CPU 16 核。
- 建 v4\ 与 task_plan.md / findings.md / progress.md。
- 写 v4\train_snn_v4.py（修 1 处 SyntaxError：global 声明位置）。
- **测速完成**（v4\speed_test.json）：训练批次 0.119s/批（B=64）→ 5 万样本单 epoch ≈94s、25 轮 ≈0.65h/种子
  → **不缩减，保留 5 万训练样本**；B=1 评估前向 6ms/样本。
- 3 种子分离后台启动（Start-Process，PID 50288/49120/52544；线程 8/4/4），日志 v4\train_seedXXXX.log：
  训练 50000（威胁 ~24k）/ 验证 1200（威胁 568，与 v3 同批✓）/ OOD 2000（威胁 978，分布已偏移✓）。
- 写 v4\TRAIN_STATE.md、v4\make_web_data_v4.py、v4\evaluate_v4.js。
- 基线哈希核对：web\results.json SHA256 = 0E1339A088EF3B5820DCA342B989249403BE41D5A3F7EDB9DDB1D1A91C217E9E（与要求一致）。
- **性能排障**（详见 TRAIN_STATE.md 调度沿革 + batch_probe.py）：
  - 3 进程并行（8/4/4，OpenMP 自旋）500–575s/epoch；独占但自旋策略下仍 629s/epoch → 排除"进程间竞争"假设；
  - 探针实测 `OMP_WAIT_POLICY=PASSIVE, KMP_BLOCKTIME=0` 同批计算快 3–5 倍 → 根因 = OpenMP 自旋等待抢核；
  - 设置固化进 train_snn_v4.py；最终配置：**逐种子独跑 12 线程，持续实测 252s/epoch（0.29s/批）**，
    5 万样本 25 轮 ≈ 1.75h/种子，在任务预期「可能数小时/种子」内 → **维持 5 万样本不缩减**（测速结果写报告 §2）。
  - 短突发测速 0.119s/批（94s/epoch 外推）不可外推到持续负载，教训记入报告。
- **评估口径自检通过**：v4\evaluate_v4.js 以 v3 数据（web\snn_data.js 只读）逐格逐位复现 web\results.json 全部数值
  （0.801±0.020 / 0.926±0.030 / 10.6 / 7.65；夜 0.634、高速 FAR 0.545、暴风 30.7°…）→ 口径与 v3 完全一致。
  自检产物 v4\selftest_v3pipeline.json（32s 跑完全矩阵）。
- v4\ 里出现他人投放的《完整果蝇设计方案.md》（v4 闭环身-脑调研设计）——不属阶段 A 范围，仅记录不动它。
- 调度定案：20240521 独占优先（进行中）→ 完成后 20240522/20240523 并行（8+8 线程）；预计总墙钟 ~4.5h。
- 下一步：等训练完成 → make_web_data_v4 + evaluate_v4（每种子）→ report_tables.py → REPORT_v4.md → 收尾自检。

