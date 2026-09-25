# v4 阶段 A —— 训练状态与恢复手册（TRAIN_STATE.md）

> 更新时间：见文末「状态快照」。**铁律**：v4\ 以外只读；`web\results.json` SHA256 必须保持
> `0E1339A088EF3B5820DCA342B989249403BE41D5A3F7EDB9DDB1D1A91C217E9E`。

## 1. 测速结论（决定数据规模）
`v4\speed_test.json`（`py -3.13 -X utf8 v4\train_snn_v4.py --speed-test`）：
- 训练批次（B=64，含前向+反向+step）短突发均值 **0.119 s**；编码 64 样本 4.0 ms；
- 外推单 epoch（5 万样本 + 1200 验证损失）≈ 94 s → 25 轮 ≈ 0.65 h/种子；B=1 评估前向 6.0 ms/样本。

**实测修正（重要）**：持续训练时短突发值不可外推——首轮 3 进程并行实测 ~500 s/epoch、单进程独占仍 ~560–630 s/epoch。
根因定位（`v4\batch_probe.py`）：**OpenMP 默认自旋等待（KMP_BLOCKTIME=200ms）在持续/多进程负载下互相抢核**；
设 `OMP_WAIT_POLICY=PASSIVE, KMP_BLOCKTIME=0` 后同一批计算降到 **0.15–0.21 s/批**（即使与其他进程竞争）。
该设置已固化进 `train_snn_v4.py`（import torch 前 setdefault）。→ **仍按 5 万训练样本执行，不缩减**。

## 2. 三个种子 = 三个分离后台进程（第 2 轮启动：统一命令、从头训练、OMP 已修）
统一前置：`$env:PYTHONPATH = "<项目根>\pylibs"`，工作目录 = 项目根。

| 种子 | PID（第 2 轮） | 线程 | 启动命令 | 日志 | 状态文件 |
|---|---|---|---|---|---|
| 20240521 | py 58704 | 6 | `py -3.13 -X utf8 -u v4\train_snn_v4.py --seed 20240521 --threads 6` | `v4\train_seed20240521.log`（stderr: `.err.log`） | `v4\status_seed20240521.json` |
| 20240522 | py 64164 | 5 | `py -3.13 -X utf8 -u v4\train_snn_v4.py --seed 20240522 --threads 5` | `v4\train_seed20240522.log` | `v4\status_seed20240522.json` |
| 20240523 | py 54492 | 5 | `py -3.13 -X utf8 -u v4\train_snn_v4.py --seed 20240523 --threads 5` | `v4\train_seed20240523.log` | `v4\status_seed20240523.json` |

**调度沿革（如实记录）**：
- 第 1 轮（8/4/4 线程并行）：~500 s/epoch，OpenMP 自旋竞争所致；其 3 个第 1 轮检查点已删除，第 1 轮数值见下方存档。
- 中途实验：终止 20240522/20240523 让 20240521 独占 → 仍 629 s/epoch（证明不是进程间竞争而是自旋策略）；
  `v4\batch_probe.py` + `OMP_WAIT_POLICY=PASSIVE` 定位根因后，**全部从头重跑**（统一命令、统一线程数，保证一条命令可复现）。
- 第 1 轮 train/val 损失存档（已被第 2 轮覆盖，仅作记录）：20240521 ep1 0.9732/0.7874、ep2 0.8772/0.7642；
  20240522 ep1 0.9389/0.7599；20240523 ep1 0.9741/0.8191。

（启动方式：PowerShell `Start-Process ... -RedirectStandardOutput v4\train_seedXXXX.log -RedirectStandardError v4\train_seedXXXX.err.log -WindowStyle Hidden -PassThru`；
进程与本会话解耦，会话被打断不影响训练。）

## 3. 检查点与恢复命令
- 每轮训练结束写检查点：`v4\ckpt_seedXXXX.pt`（含 model/opt/scheduler/hist/best_state/epoch）；
- **恢复（断点续训，逐位可复现）**：
  ```powershell
  $proj = '<项目根>'
  $env:PYTHONPATH = "$proj\pylibs"; Set-Location $proj
  Start-Process -FilePath py -ArgumentList @('-3.13','-X','utf8','-u','v4\train_snn_v4.py','--seed','20240521','--threads','8','--resume') `
    -WorkingDirectory $proj -RedirectStandardOutput "$proj\v4\train_seed20240521.log" `
    -RedirectStandardError "$proj\v4\train_seed20240521.err.log" -WindowStyle Hidden -PassThru
  ```
  （换 `--seed` 即可恢复对应种子；训练批次 RNG 按 `seed+100+epoch` 逐轮播种，续训不改变结果。）
- 训练完成后产物：`v4\snn_trained_seedXXXX.json`、`v4\metrics_seedXXXX.json`、`v4\training_curve_seedXXXX.png`。

## 4. 训练完成后的评估流水线（勿覆盖 web\ 产物）
```powershell
# 主模型（种子 20240521）→ v4\snn_trained_v4.json + v4\snn_data_v4.js
py -3.13 -X utf8 v4\make_web_data_v4.py --seed 20240521
# 另两个种子 → v4\snn_trained_v4_seedXXXX.json + v4\snn_data_v4_seedXXXX.js
py -3.13 -X utf8 v4\make_web_data_v4.py --seed 20240522 --tag _seed20240522
py -3.13 -X utf8 v4\make_web_data_v4.py --seed 20240523 --tag _seed20240523
# 实验矩阵评估（输出 v4\results_v4.json 等）
F:\Node\node.exe v4\evaluate_v4.js
F:\Node\node.exe v4\evaluate_v4.js --data v4/snn_data_v4_seed20240522.js --out v4/results_v4_seed20240522.json
F:\Node\node.exe v4\evaluate_v4.js --data v4/snn_data_v4_seed20240523.js --out v4/results_v4_seed20240523.json
# 收尾自检
F:\Node\node.exe web\smoke_test.js
Get-FileHash web\results.json -Algorithm SHA256
```

## 5. 超参数（v4 阶段 A）
N_TRAIN=50000 / N_VAL=1200（固定 rng(20240523)，= v3 同批）/ N_OOD=2000（固定 rng(20240603)，偏移分布）
噪声 σ~U(0,0.05)（可调 --noise-std）/ 模态 dropout p=0.12（--drop-p）/ wd=1e-4（--wd）
早停：盯验证损失 patience=6、最少 8 轮（--patience）/ 检查点选优 = val_loss 最小
BATCH=64、EPOCHS=25、LR=2e-3+余弦退火（与 v3 一致）；架构/标签/决策窗/损失/激活边界与 v3 逐位一致。

## 状态快照（最后更新：种子 20240521 完成后）
- ✅ 种子 20240521：训练完成（跑满 25 轮，best ep25 val=0.6861），已导出
  `v4\snn_trained_seed20240521.json` / `metrics_seed20240521.json` / `training_curve_seed20240521.png`；
  已打包 `v4\snn_trained_v4.json` + `v4\snn_data_v4.js`，已评估 `v4\results_v4.json`。
- ⏳ 种子 20240522（PID 25848）、20240523（PID 218844）：分离后台训练中（8 线程、AboveNormal、OMP PASSIVE），
  各写 `v4\train_seedXXX.log` / `status_seedXXX.json` / `ckpt_seedXXX.pt`，预计 ~3h 跑完 25 轮。
- ✅ 收尾自检（已做一次）：`web\smoke_test.js` 4/4 全过 exit=0；`web\results.json` SHA256 =
  `0E1339A088EF3B5820DCA342B989249403BE41D5A3F7EDB9DDB1D1A91C217E9E` ✅。两枚种子完成后需**再复验一次**。
- ✅ `v4\REPORT_v4.md` 初稿已交付（种子 1 全量数字 + 3 种子 mean±std 占位）。

## 4b. 两枚种子完成后的「5 分钟收尾」（任何会话可直接照抄执行）
```powershell
$proj = '<项目根>'; Set-Location $proj
# 1) 打包另两个种子模型
py -3.13 -X utf8 v4\make_web_data_v4.py --seed 20240522 --tag _seed20240522
py -3.13 -X utf8 v4\make_web_data_v4.py --seed 20240523 --tag _seed20240523
# 2) 实验矩阵评估（各 ~45s）
F:\Node\node.exe v4\evaluate_v4.js --data v4/snn_data_v4_seed20240522.js --out v4/results_v4_seed20240522.json
F:\Node\node.exe v4\evaluate_v4.js --data v4/snn_data_v4_seed20240523.js --out v4/results_v4_seed20240523.json
# 3) 汇总 3 种子 mean±std(ddof=1) + 过拟合体检 + 分场景表 → v4\report_tables.json + 控制台 Markdown
py -3.13 -X utf8 v4\report_tables.py
# 4) 收尾自检：冒烟全过 + 哈希不变
F:\Node\node.exe web\smoke_test.js
(Get-FileHash web\results.json -Algorithm SHA256).Hash   # 必须 = 0E1339A088EF3B5820DCA342B989249403BE41D5A3F7EDB9DDB1D1A91C217E9E
# 5) 把 report_tables.py 的表格数字回填 v4\REPORT_v4.md 占位处（T1/T2/T3b/T4）
```
> 若哈希对不上：**不要自行处理**，先报总管（可能有并发写 web\）。
