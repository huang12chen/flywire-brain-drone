# -*- coding: utf-8 -*-
"""
train_snn_v4.py —— v4 阶段 A：数据升级 + 防过拟合（在 v3 train_snn.py 基础上**只改数据与正则**）
============================================================================================
与 v3 train_snn.py 的差异（除此之外逐位一致）：
  1) 训练样本 N_TRAIN 5000 -> 50000（可用 --n-train 调；测速后可能降到 2–3 万）；
  2) 输入增强（仅训练时）：视觉/风觉通道发放概率加高斯噪声 σ~U(0, NOISE_STD=0.05)（可调），
     并以 P_DROP=0.12 概率整通道置零（随机缺视觉或缺风觉，各 6%）；
  3) OOD 测试集 2000 条：明显偏移的参数单独生成（见 sample_threat_ood / build_ood_dataset），
     任何训练/验证环节不得使用（仅在最终评估里单列）；
  4) 3 个训练种子（20240521/20240522/20240523），验证/OOD 集跨种子固定，报 mean±std(ddof=1)；
  5) 正则化：Adam weight_decay=1e-4；早停盯**验证损失**（PATIENCE=6，最少 MIN_EPOCHS=8）；
     每轮记录 train/val 损失曲线（过拟合体检 = train-val 差距）；每轮存检查点，支持 --resume 断点续训；
  6) 检查点选优准则：验证损失最小（v3 为每 5 轮 score 选优）—— 差异会在 REPORT_v4.md 如实声明。

不变项（与 train_snn.py 完全一致，保证可比性）：
  架构（EscapeSNN/LIF β=0.85, VTH=1.0, T=12, dt=1ms, 软复位）、拓扑 mask、in_gain、读出头、
  标签规则（ṙ<0 且 ttc<50ms 且 r<25cm；esc=−r̂）、决策窗（触发=GF 首次放电步<12）、
  损失函数（0.5·BCE + 0.5·方向余弦(带 yTrig 掩码) + 2e-3·rate·T + 0.6·GF放电BCE(pos_weight=2.5)）、
  激活边界（>0）、Adam+余弦退火、BATCH=64、EPOCHS=25、LR=2e-3。

用法：
  py -3.13 v4\\train_snn_v4.py --seed 20240521                  # 训练一个种子
  py -3.13 v4\\train_snn_v4.py --seed 20240521 --resume         # 断点续训
  py -3.13 v4\\train_snn_v4.py --speed-test                     # 只测速
"""
import os
import sys
import json
import math
import time
import copy
import random
import argparse

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))      # v4\
ROOT = os.path.dirname(BASE)                            # 项目根（v3 冻结区，只读）
os.environ.setdefault('PYTHONPATH', os.path.join(ROOT, 'pylibs'))
# [v4] OpenMP 被动等待（必须在 import torch 前设置）：默认自旋等待在多进程/持续负载下会互相抢核，
#      实测同一批计算从 0.6–0.8s/批 降到 0.15–0.21s/批（见 REPORT_v4.md §2 测速）。
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
os.environ.setdefault('KMP_BLOCKTIME', '0')
import sys
sys.path.insert(0, os.path.join(ROOT, 'pylibs'))        # 兜底：分离后台进程也能找到 numpy/torch

import torch
import torch.nn as nn

try:  # snntorch 的代理梯度；若未安装则用等价的自实现 fast-sigmoid
    from snntorch import surrogate as _surr
    spike_grad = _surr.fast_sigmoid()
    USING = 'snntorch surrogate.fast_sigmoid'
except Exception:
    class _FastSigmoid(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return (x > 0).float()   # 边界 >0 与 snntorch FastSigmoid 一致

        @staticmethod
        def backward(ctx, grad):
            (x,) = ctx.saved_tensors
            slope = 2.0
            return grad * (1.0 / (1.0 + slope * x.abs()) ** 2)

    def spike_grad(x):
        return _FastSigmoid.apply(x)
    USING = 'manual fast-sigmoid'

# ---------------- 超参数（大写常量与 v3 train_snn.py 逐位一致，除非注明 v4 新增） ----------------
DT_MS = 1.0
T_STEPS = 12          # 仿真窗口 12 ms（GF 逃逸反应潜伏期量级）
BETA = 0.85           # 膜电位衰减
VTH = 1.0             # 阈值
BATCH = 64
EPOCHS = 25
LR = 2e-3
N_TRAIN_DEFAULT = 50000   # [v4] v3=5000
N_VAL = 1200
N_OOD = 2000              # [v4] OOD 测试集规模
RATE_MAX_HZ = 200.0   # 输入编码最大放电率
BASE_RATE_HZ = 5.0
TAU_TTC_MS = 50.0     # 触发判据：碰撞时间 < 50 ms
R_TRIGGER_CM = 25.0   # 且距离 < 25 cm

# ---------------- [v4] 数据增强与正则超参数 ----------------
NOISE_STD = 0.05      # 高斯噪声上限（每样本 σ~U(0,NOISE_STD)），0–0.05 可调
P_DROP = 0.12         # 模态 dropout：整通道置零概率（10–15% 区间；缺视觉/缺风觉各 6%）
WD = 1e-4             # 权重衰减
PATIENCE = 6          # 早停耐心（盯验证损失）
MIN_EPOCHS = 8        # 早停前最少训练轮数
VAL_LOSS_SEED = 424242    # 验证损失的固定编码种子（每轮重置 → 损失曲线可比、可复跑）
VAL_SET_SEED = 20240523   # 验证集固定种子（= v3 的 SEED+2；3 个训练种子共用同一批验证样本）
OOD_SET_SEED = 20240603   # OOD 集固定种子（3 个训练种子共用）


def load_graph():
    with open(os.path.join(ROOT, 'escape_network_sparse.json'), encoding='utf-8') as f:
        g = json.load(f)
    n = g['meta']['num_nodes']
    src = np.array([e['src'] for e in g['edges']], dtype=np.int64)
    dst = np.array([e['dst'] for e in g['edges']], dtype=np.int64)
    w = np.array([e['weight'] for e in g['edges']], dtype=np.float32)
    # 按突触后神经元的总入强度归一化，使初始动力学稳定
    abs_sum = np.zeros(n, dtype=np.float32)
    np.add.at(abs_sum, dst, np.abs(w))
    w = w / (abs_sum[dst] + 1e-6) * 1.2
    return g, n, src, dst, w


# ---------------- 合成数据集：物理公式生成"威胁状态 -> 线索 -> 标签"（与 v3 逐位一致） ----------------
def sample_threat(rng):
    """返回某一时刻的威胁几何状态（单位 cm, cm/ms -> 换算为 m/s 展示）"""
    d0 = rng.uniform(3.0, 35.0)                 # 当前距离 cm
    az = rng.uniform(-math.pi, math.pi)         # 相对果蝇朝向的方位角
    el = rng.uniform(-0.6, 0.6)
    speed = rng.uniform(0.3, 12.0)              # m/s
    miss = rng.uniform(0.0, 8.0)                # 脱靶量 cm
    s_size = rng.uniform(0.3, 3.0)              # 威胁物半径 cm
    # 视线方向（威胁相对果蝇的位置单位向量）
    rhat = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    rvec = rhat * d0
    # 速度方向：瞄准果蝇附近（miss 造成切向分量）
    to_fly = -rhat
    tangent = np.array([-rhat[1], rhat[0], 0.0])
    tn = tangent / (np.linalg.norm(tangent) + 1e-9)
    vdir = to_fly * math.sqrt(max(0.0, 1 - (miss / (d0 + 1e-9)) ** 2)) + tn * (miss / (d0 + 1e-9))
    vdir = vdir / (np.linalg.norm(vdir) + 1e-9)
    v_cm_per_ms = vdir * (speed * 100.0 / 1000.0)   # m/s -> cm/ms
    return dict(d=d0, az=az, el=el, s=s_size, rhat=rhat, rvec=rvec, v=v_cm_per_ms, speed=speed)


def sample_threat_ood(rng):
    """[v4] OOD 分布（明显偏移，任何训练/验证环节不得使用）：
      * 威胁速度上限 12 m/s 且快威胁占比加大：60% ~ U(7,12) + 40% ~ U(3,12)（训练分布为 U(0.3,12)）
      * 角度分布不同：az ~ U(-2.0, 2.0)（前向偏置，训练为 U(-π,π)）、el ~ U(-0.9, 0.9)（训练为 U(-0.6,0.6)）
      * 距离更宽 d ~ U(3, 40) cm（训练为 U(3,35)）、脱靶量更大 miss ~ U(0, 10) cm（训练为 U(0,8)）
      * 风场更强：风压 ×2 再叠加环境风 U(0.01, 0.03) cm/ms（随机方向，样本内恒定）
      * 传感器噪声 ×2：编码时高斯噪声 σ~U(0, 2·NOISE_STD)（见 run_eval 的 noise_sigma_max）
      标签规则不变（ṙ<0 且 ttc<50ms 且 r<25cm；esc=−r̂）。"""
    d0 = rng.uniform(3.0, 40.0)
    if rng.random() < 0.6:
        speed = rng.uniform(7.0, 12.0)
    else:
        speed = rng.uniform(3.0, 12.0)
    az = rng.uniform(-2.0, 2.0)
    el = rng.uniform(-0.9, 0.9)
    miss = rng.uniform(0.0, 10.0)
    s_size = rng.uniform(0.3, 3.0)
    rhat = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    rvec = rhat * d0
    to_fly = -rhat
    tangent = np.array([-rhat[1], rhat[0], 0.0])
    tn = tangent / (np.linalg.norm(tangent) + 1e-9)
    vdir = to_fly * math.sqrt(max(0.0, 1 - (miss / (d0 + 1e-9)) ** 2)) + tn * (miss / (d0 + 1e-9))
    vdir = vdir / (np.linalg.norm(vdir) + 1e-9)
    v_cm_per_ms = vdir * (speed * 100.0 / 1000.0)
    return dict(d=d0, az=az, el=el, s=s_size, rhat=rhat, rvec=rvec, v=v_cm_per_ms, speed=speed)


def cues_and_labels(st):
    """由物理公式计算两类线索与监督标签（与 v3 逐位一致）。
    视觉膨胀率：theta = 2*atan(s/r)  ->  dtheta/dt = -2 s r' / (r^2 + s^2)   (rad/s)
    风压：球体势流近似 u = C s^2 v / r^2，方向沿威胁速度方向                    (任意单位, 归一化后用)
    """
    r = st['d']
    s = st['s']
    rdot = float(np.dot(st['v'], st['rhat']))       # cm/ms，接近时为负
    looming = max(0.0, -2.0 * s * rdot / (r ** 2 + s ** 2)) * 1000.0   # rad/s
    u = (s ** 2) * np.linalg.norm(st['v']) / (r ** 2 + 1e-9)           # cm/ms 任意单位
    wind_vec = u * (st['v'] / (np.linalg.norm(st['v']) + 1e-9))

    ttc_ms = 1e9 if rdot >= 0 else r / (-rdot) / 1.0  # r(cm)/|rdot|(cm/ms) = ms
    trigger = 1.0 if (rdot < 0 and ttc_ms < TAU_TTC_MS and r < R_TRIGGER_CM) else 0.0
    esc_dir = -st['rhat'] / (np.linalg.norm(st['rhat']) + 1e-9)   # 逃逸方向 = 背离威胁（威胁在 +r̂ 方向）
    return looming, wind_vec, trigger, esc_dir, ttc_ms


def encode_spikes(rng, looming, wind_vec, threat_az, pd_pref, wind_pref, n_steps,
                  noise_sigma=0.0, drop_mode=0):
    """泊松发放率编码 -> [T, N_in] 0/1 脉冲（v3 主体逐位一致；v4 新增两个可选参数，默认行为与 v3 相同）
    [v4] noise_sigma>0：每个输入通道的发放概率叠加 N(0, noise_sigma) 高斯噪声后截断到 [0, 0.9]；
                      noise_sigma=0 时不消耗 RNG → 与 v3 编码逐位一致。
    [v4] drop_mode：1=整条视觉通道置零 / 2=整条风觉通道置零 / 0=不置零（模态 dropout，仅训练用）。
    LPLC2 群体：按偏好方位角编码"威胁方位角"（方位信息 -> 群体码）
    JO   群体：按敏感轴编码风矢量方向（风的方向本身携带威胁方向信息）
    """
    n_vis = pd_pref.shape[0]
    n_in = n_vis + wind_pref.shape[0]
    loom_n = min(1.0, looming / (looming + 2.0))          # 饱和归一
    wind_mag = np.linalg.norm(wind_vec)
    wind_n = min(1.0, wind_mag / (wind_mag + 0.012))   # Johnston's 器为高灵敏机械感受器（半饱和常数取小）

    rates = np.full(n_in, BASE_RATE_HZ, dtype=np.float32)

    # 视觉通道（索引 0 .. n_vis-1）：偏好方位与威胁方位匹配度整流
    cos_v = np.cos(pd_pref[:, 0] - threat_az)
    vis_drive = loom_n * np.clip(cos_v, 0.0, None) * RATE_MAX_HZ * 0.8
    rates[:n_vis] += vis_drive

    # 风觉通道：风矢量在敏感轴上的投影整流
    wdir = wind_vec / (wind_mag + 1e-9)
    proj = wind_pref @ wdir
    wind_drive = wind_n * np.clip(proj, 0.0, None) * RATE_MAX_HZ * 0.8
    rates[n_vis:] += wind_drive

    p = np.clip(rates * DT_MS / 1000.0, 0.0, 0.9)
    if noise_sigma > 0.0:   # [v4] 传感器输入高斯噪声（0 时不走该分支，RNG 流与 v3 一致）
        p = np.clip(p + rng.normal(0.0, noise_sigma, size=n_in), 0.0, 0.9)
    spikes = (rng.random((n_steps, n_in)) < p[None, :]).astype(np.float32)
    if drop_mode == 1:
        spikes[:, :n_vis] = 0.0
    elif drop_mode == 2:
        spikes[:, n_vis:] = 0.0
    return spikes


def build_dataset(n, seed):
    """[v3 逐位一致] 训练/验证分布。"""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        st = sample_threat(rng)
        looming, wind_vec, trig, esc, ttc = cues_and_labels(st)
        samples.append(dict(looming=looming, wind=wind_vec, trig=trig, esc=esc,
                            az=st['az'], ttc=ttc, speed=st['speed'], d=st['d']))
    return samples, rng


def build_ood_dataset(n, seed):
    """[v4] OOD 分布（明显偏移）；标签规则与训练/验证完全一致。"""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        st = sample_threat_ood(rng)
        looming, wind_vec, trig, esc, ttc = cues_and_labels(st)
        # 风场更强：风压线索 ×2 再叠加环境风（随机方向、样本内恒定）
        wind_cue = wind_vec * 2.0
        amb_dir = rng.normal(size=3)
        amb_dir = amb_dir / (np.linalg.norm(amb_dir) + 1e-9)
        wind_cue = wind_cue + amb_dir * rng.uniform(0.01, 0.03)
        samples.append(dict(looming=looming, wind=wind_cue, trig=trig, esc=esc,
                            az=st['az'], ttc=ttc, speed=st['speed'], d=st['d']))
    return samples


class EscapeSNN(nn.Module):
    """以 FlyWire 稀疏拓扑为固定 mask 的 LIF 递归网络；仅训练边权与两个读出头。（与 v3 逐位一致）"""

    def __init__(self, n, src, dst, w0, in_vision, in_wind, hub_idx, out_idx):
        super().__init__()
        self.n = n
        self.src = torch.tensor(src)
        self.dst = torch.tensor(dst)
        self.register_buffer('srcb', self.src)
        self.register_buffer('dstb', self.dst)
        self.w = nn.Parameter(torch.tensor(w0))
        self.register_buffer('w_mask', torch.ones_like(self.w))

        self.in_vision = torch.tensor(in_vision, dtype=torch.long)
        self.in_wind = torch.tensor(in_wind, dtype=torch.long)
        self.hub_idx = torch.tensor(hub_idx, dtype=torch.long)
        self.out_idx = torch.tensor(out_idx, dtype=torch.long)

        self.in_gain = nn.Parameter(torch.ones(self.in_vision.numel() + self.in_wind.numel()) * 1.0)
        self.head_trig = nn.Linear(self.hub_idx.numel(), 1)
        self.head_dir = nn.Linear(self.out_idx.numel(), 3)

    def forward(self, spikes):
        """spikes: [T, B, N_in] -> trig_logit [B], dir_pred [B,3], rate [], v_hub_max [B],
        first_step [B]（GF 首次放电步，未放电为 T）, hub_spk [B]（GF 放电总数）"""
        T, B, _ = spikes.shape
        device = spikes.device
        v = torch.zeros(self.n, B, device=device)
        s = torch.zeros(self.n, B, device=device)
        v_hub_sum = torch.zeros(B, self.hub_idx.numel(), device=device)
        v_out_sum = torch.zeros(B, self.out_idx.numel(), device=device)
        v_hub_max = torch.full((B,), -1e9, device=device)
        first_step = torch.full((B,), float(T), device=device)
        fired_any = torch.zeros(B, dtype=torch.bool, device=device)
        hub_spk = torch.zeros(B, device=device)
        spike_count = 0.0

        for t in range(T):
            msg = (self.w * self.w_mask)[:, None] * s[self.srcb]         # [E, B]
            cur = torch.zeros(self.n, B, device=device)
            cur = cur.index_add(0, self.dstb, msg)                       # 稀疏递归电流

            xin = spikes[t] * self.in_gain[None, :]                      # [B, N_in]
            cur[self.in_vision] += xin[:, : self.in_vision.numel()].T
            cur[self.in_wind] += xin[:, self.in_vision.numel():].T

            v = BETA * v + cur - VTH * s
            s = spike_grad(v - VTH)
            spike_count = spike_count + s.mean()
            v_hub_sum += v[self.hub_idx].T
            v_out_sum += v[self.out_idx].T

            hv = v[self.hub_idx].T                       # [B, n_hub]
            v_hub_max = torch.maximum(v_hub_max, hv.max(dim=1).values)
            hs = s[self.hub_idx].T.sum(dim=1) > 0        # 本步 GF 是否放电
            first_step = torch.where(hs & ~fired_any,
                                     torch.full_like(first_step, float(t + 1)), first_step)
            fired_any = fired_any | hs
            hub_spk = hub_spk + s[self.hub_idx].T.sum(dim=1)

        trig_logit = self.head_trig(v_hub_sum / T).squeeze(-1)
        dir_pred = self.head_dir(v_out_sum / T)
        return trig_logit, dir_pred, spike_count / T, v_hub_max, first_step, hub_spk


def make_pref(n_vis, n_wind, seed):
    rng = np.random.default_rng(seed)
    pd_pref = np.zeros((n_vis, 2), dtype=np.float32)
    pd_pref[:, 0] = np.linspace(-math.pi, math.pi, n_vis, endpoint=False)  # 偏好方位
    pd_pref[:, 1] = rng.uniform(0.5, 1.0, n_vis)                            # 增益异质性（保留参数：当前未接入前向，维持两端 RNG 顺序一致，勿删）
    wind_pref = rng.normal(size=(n_wind, 3)).astype(np.float32)
    wind_pref /= np.linalg.norm(wind_pref, axis=1, keepdims=True)
    return pd_pref, wind_pref


def compute_loss(trig_logit, dir_pred, rate, v_hub_max, y_trig, y_esc, device):
    """[v3 损失式逐位一致] 抽出共用（训练与验证损失用同一式，过拟合体检才可比）。"""
    bce = nn.functional.binary_cross_entropy_with_logits(trig_logit, y_trig)
    dn = dir_pred / (dir_pred.norm(dim=1, keepdim=True) + 1e-8)
    cos_loss = (1.0 - (dn * y_esc).sum(dim=1))
    dir_loss = (cos_loss * y_trig).sum() / (y_trig.sum() + 1e-6)
    # 直接对"GF 是否放电"这一机制化触发判据做监督（v_hub_max 越过 VTH 即放电）
    # pos_weight=2.5：错失威胁的代价 > 误报（生物逃逸的不对称代价），提高对威胁的敏感度
    gf_loss = nn.functional.binary_cross_entropy_with_logits(
        (v_hub_max - VTH) * 4.0, y_trig,
        pos_weight=torch.tensor(2.5, device=device))
    return 0.5 * bce + 0.5 * dir_loss + 2e-3 * rate * T_STEPS + 0.6 * gf_loss


def encode_batch(rng, samples, idx, pd_pref, wind_pref, augment, noise_std):
    """[v4] 编码一个 batch；augment=True 时施加高斯噪声 + 模态 dropout（仅训练用）。"""
    n_vis = pd_pref.shape[0]
    n_in = n_vis + wind_pref.shape[0]
    B = len(idx)
    sp = np.zeros((T_STEPS, B, n_in), dtype=np.float32)
    trigs = np.zeros(B, dtype=np.float32)
    escs = np.zeros((B, 3), dtype=np.float32)
    for k, j in enumerate(idx):
        st = samples[j]
        sigma = 0.0
        drop_mode = 0
        if augment:
            if noise_std > 0:
                sigma = float(rng.uniform(0.0, noise_std))     # 每样本 σ~U(0, NOISE_STD)
            if rng.random() < P_DROP:                          # 模态 dropout：整通道置零
                drop_mode = 1 if rng.random() < 0.5 else 2
        sp[:, k, :] = encode_spikes(rng, st['looming'], st['wind'], st['az'], pd_pref, wind_pref,
                                    T_STEPS, noise_sigma=sigma, drop_mode=drop_mode)
        trigs[k] = st['trig']
        escs[k] = st['esc']
    return sp, trigs, escs


def run_eval(model, samples, pd_pref, wind_pref, device, mode='fusion', noise_sigma_max=0.0):
    """mode: fusion / vision_only / wind_only —— 与 v3 run_eval 逐位一致（B=1、rng(777)、每模式独立重置），
    [v4] 仅新增 noise_sigma_max（评估编码高斯噪声上限；验证=0 与 v3 同口径，OOD=2*NOISE_STD）。"""
    model.eval()
    rng = np.random.default_rng(777)
    n_vis = pd_pref.shape[0]
    hits = 0
    head_ok = 0
    gf_ok = 0
    ang_errs = []
    latencies = []
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for st in samples:
            sigma = float(rng.uniform(0.0, noise_sigma_max)) if noise_sigma_max > 0 else 0.0
            sp = encode_spikes(rng, st['looming'], st['wind'], st['az'], pd_pref, wind_pref, T_STEPS,
                               noise_sigma=sigma)
            if mode == 'vision_only':
                sp[:, n_vis:] = 0.0
            if mode == 'wind_only':
                sp[:, :n_vis] = 0.0
            x = torch.tensor(sp[:, None, :], dtype=torch.float32, device=device)  # [T, B=1, N_in]
            trig, dirp, _, _, first_step, _ = model(x)
            pred_head = (torch.sigmoid(trig)[0] >= 0.5).float().item()
            pred_gf = 1.0 if first_step[0].item() < T_STEPS else 0.0
            if pred_head == st['trig']:
                head_ok += 1
            if pred_gf == st['trig']:
                gf_ok += 1
            if pred_gf > 0.5 and st['trig'] > 0.5:
                tp += 1
            elif pred_gf > 0.5:
                fp += 1
            elif st['trig'] > 0.5:
                fn += 1
            else:
                tn += 1
            if st['trig'] > 0.5:
                cur_ang = float('nan')
                d = dirp[0].cpu().numpy()
                if np.linalg.norm(d) > 1e-6:
                    cos = float(np.dot(d / np.linalg.norm(d), st['esc']))
                    cur_ang = math.degrees(math.acos(np.clip(cos, -1, 1)))
                    ang_errs.append(cur_ang)
                if pred_gf >= 0.5:
                    latencies.append(first_step[0].item() * DT_MS)
                    if cur_ang < 30.0:
                        hits += 1
    n_pos = sum(1 for st in samples if st['trig'] > 0.5)
    succ = hits / max(1, n_pos)
    return {
        'trigger_acc_gf_spike': gf_ok / max(1, len(samples)),
        'trigger_acc_head': head_ok / max(1, len(samples)),
        'escape_success_rate': succ,
        'dir_mae_deg': float(np.mean(ang_errs)) if ang_errs else float('nan'),
        'gf_first_spike_ms': float(np.mean(latencies)) if latencies else float('nan'),
        'threat_recall': tp / max(1, tp + fn),            # [v4 新增诊断] P(GF放电|yTrig=1)
        'false_alarm_rate': fp / max(1, fp + tn),         # [v4 新增诊断] P(GF放电|yTrig=0)
        'n_samples': len(samples),
        'n_threat': n_pos,
    }


def val_loss_of(model, val_samples, pd_pref, wind_pref, device):
    """[v4] 验证损失：同损失式、干净编码（无增强）、固定编码种子（每轮重置 → 曲线只反映模型变化）。"""
    model.eval()
    rng = np.random.default_rng(VAL_LOSS_SEED)
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(val_samples), BATCH):
            idx = list(range(i, min(i + BATCH, len(val_samples))))
            sp, trigs, escs = encode_batch(rng, val_samples, idx, pd_pref, wind_pref,
                                           augment=False, noise_std=0.0)
            x = torch.tensor(sp, device=device)
            y_trig = torch.tensor(trigs, device=device)
            y_esc = torch.tensor(escs, device=device)
            trig_logit, dir_pred, rate, v_hub_max, _, _ = model(x)
            loss = compute_loss(trig_logit, dir_pred, rate, v_hub_max, y_trig, y_esc, device)
            tot += float(loss) * len(idx)
            cnt += len(idx)
    return tot / max(1, cnt)


def main():
    global P_DROP
    ap = argparse.ArgumentParser(description='v4 阶段 A 训练（数据升级 + 防过拟合）')
    ap.add_argument('--seed', type=int, default=20240521)
    ap.add_argument('--n-train', type=int, default=N_TRAIN_DEFAULT)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--noise-std', type=float, default=NOISE_STD)
    ap.add_argument('--drop-p', type=float, default=P_DROP)
    ap.add_argument('--wd', type=float, default=WD)
    ap.add_argument('--patience', type=int, default=PATIENCE)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--threads', type=int, default=0, help='torch CPU 线程数（0=默认；多种子并行时用它限流）')
    ap.add_argument('--speed-test', action='store_true')
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    if args.speed_test:
        return speed_test(args)

    P_DROP = args.drop_p
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device('cpu')
    g, n, src, dst, w0 = load_graph()
    in_v = g['input_vision_indices']
    in_w = g['input_wind_indices']
    hub = g['hub_gf_indices']
    out = g['output_indices']
    n_vis, n_wind = len(in_v), len(in_w)
    print(f'[v4] 图：{n} 节点 / {len(src)} 边 | 视觉入 {n_vis} | 风觉入 {n_wind} | GF {len(hub)} | 输出 {len(out)}')
    print(f'代理梯度: {USING}')
    print(f'[v4] 训练种子={SEED} | N_TRAIN={args.n_train} | 高斯噪声 σ~U(0,{args.noise_std}) | '
          f'模态 dropout p={P_DROP} | wd={args.wd} | 早停 patience={args.patience} (盯验证损失)')

    t_data = time.time()
    pd_pref, wind_pref = make_pref(n_vis, n_wind, SEED)          # [v4] pref 随训练种子（v3: make_pref(...,SEED)）
    train_samples, _ = build_dataset(args.n_train, SEED + 1)     # 训练集随种子
    val_samples, _ = build_dataset(N_VAL, VAL_SET_SEED)          # [v4] 验证集跨种子固定（= v3 同批样本）
    ood_samples = build_ood_dataset(N_OOD, OOD_SET_SEED)         # [v4] OOD 集跨种子固定，训练/验证禁用
    n_pos_tr = sum(1 for s in train_samples if s['trig'] > 0.5)
    n_pos_va = sum(1 for s in val_samples if s['trig'] > 0.5)
    n_pos_ood = sum(1 for s in ood_samples if s['trig'] > 0.5)
    print(f'数据生成 {time.time() - t_data:.1f}s | 训练 {len(train_samples)}（威胁 {n_pos_tr}）| '
          f'验证 {len(val_samples)}（威胁 {n_pos_va}）| OOD {len(ood_samples)}（威胁 {n_pos_ood}）')

    model = EscapeSNN(n, src, dst, w0, in_v, in_w, hub, out).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=args.wd)   # [v4] weight_decay=1e-4
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=2e-4)

    ckpt_path = os.path.join(BASE, f'ckpt_seed{SEED}.pt')
    status_path = os.path.join(BASE, f'status_seed{SEED}.json')
    hist = []
    best_state, best_val, best_ep, bad = None, float('inf'), 0, 0
    start_ep = 1
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        scheduler.load_state_dict(ck['sched'])
        hist = ck['hist']
        best_state, best_val, best_ep, bad = ck['best_state'], ck['best_val'], ck['best_ep'], ck['bad']
        start_ep = ck['epoch'] + 1
        print(f'== 断点续训：从第 {start_ep} 轮继续（历史 {len(hist)} 轮，最优 val={best_val:.4f}@ep{best_ep}）==')

    t0 = time.time()
    for ep in range(start_ep, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(SEED + 100 + ep)             # 每轮独立种子（续训也逐位可复现）
        order = rng.permutation(len(train_samples))
        tot = 0.0
        for i in range(0, len(train_samples), BATCH):
            idx = order[i: i + BATCH]
            B = len(idx)
            sp, trigs, escs = encode_batch(rng, train_samples, idx, pd_pref, wind_pref,
                                           augment=True, noise_std=args.noise_std)   # [v4] 输入增强
            x = torch.tensor(sp, device=device)
            y_trig = torch.tensor(trigs, device=device)
            y_esc = torch.tensor(escs, device=device)

            trig_logit, dir_pred, rate, v_hub_max, _, _ = model(x)
            loss = compute_loss(trig_logit, dir_pred, rate, v_hub_max, y_trig, y_esc, device)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * B
        scheduler.step()
        train_loss = tot / len(train_samples)
        vloss = val_loss_of(model, val_samples, pd_pref, wind_pref, device)          # [v4] 验证损失
        hist.append({'epoch': ep, 'loss': train_loss, 'val_loss': vloss,
                     'lr': scheduler.get_last_lr()[0], 'time_s': round(time.time() - t0, 1)})
        improved = vloss < best_val - 1e-6
        if improved:
            best_val, best_ep = vloss, ep
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        print(f'epoch {ep:2d}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={vloss:.4f}  '
              f'lr={scheduler.get_last_lr()[0]:.5f}  gap={train_loss - vloss:+.4f}  '
              f'({"best" if improved else f"bad={bad}/{args.patience}"})  ({time.time() - t0:.0f}s)', flush=True)

        torch.save({'epoch': ep, 'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': scheduler.state_dict(), 'hist': hist, 'best_state': best_state,
                    'best_val': best_val, 'best_ep': best_ep, 'bad': bad}, ckpt_path)
        with open(status_path, 'w', encoding='utf-8') as f:
            json.dump({'seed': SEED, 'epoch': ep, 'epochs_max': args.epochs, 'train_loss': train_loss,
                       'val_loss': vloss, 'best_val': best_val, 'best_epoch': best_ep, 'bad': bad,
                       'n_train': len(train_samples), 'elapsed_s': round(time.time() - t0, 1),
                       'state': 'running'}, f, ensure_ascii=False, indent=2)

        if bad >= args.patience and ep >= MIN_EPOCHS:
            print(f'== 早停：验证损失连续 {args.patience} 轮未改善（最优 ep{best_ep} val={best_val:.4f}）==', flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'\n已恢复最优权重（val_loss={best_val:.4f} @ ep{best_ep}）')

    print('\n===== 对照评估（同一验证集 1200 条，干净编码，与 v3 同口径） =====')
    results = {}
    for mode in ('fusion', 'vision_only', 'wind_only'):
        results[mode] = run_eval(model, val_samples, pd_pref, wind_pref, device, mode=mode)

    print(f'===== OOD 评估（{N_OOD} 条偏移分布，传感器噪声×2；训练/验证从未使用） =====')
    ood_results = {}
    for mode in ('fusion', 'vision_only', 'wind_only'):
        ood_results[mode] = run_eval(model, ood_samples, pd_pref, wind_pref, device, mode=mode,
                                     noise_sigma_max=2.0 * args.noise_std)

    priming = priming_test(model, pd_pref, wind_pref, device)

    fmt = '{:<12}{:>16}{:>16}{:>14}{:>14}{:>14}'.format(
        '组别', '触发准确率(GF)', '触发准确率(头)', '避障成功率', '方向误差(°)', 'GF潜伏期(ms)')
    for tag, block in (('验证集', results), ('OOD集', ood_results)):
        print(f'\n[{tag}]')
        print(fmt)
        for mode, r in block.items():
            print('{:<12}{:>16.3f}{:>16.3f}{:>14.3f}{:>14.1f}{:>14.2f}'.format(
                mode, r['trigger_acc_gf_spike'], r['trigger_acc_head'],
                r['escape_success_rate'], r['dir_mae_deg'], r['gf_first_spike_ms']))
    print('\n弱线索预激活(priming)测试 —— GF 放电比例与首次放电潜伏期：')
    for tag, r in priming.items():
        print('  {:<14} fire_rate={:.2f}  first_spike={:.2f} ms'.format(
            tag, r['fire_rate'], r['first_spike_ms']))

    export_model(model, results, ood_results, priming, hist, SEED, args)
    with open(status_path, 'w', encoding='utf-8') as f:
        json.dump({'seed': SEED, 'epoch': len(hist), 'epochs_max': args.epochs,
                   'best_val': best_val, 'best_epoch': best_ep, 'state': 'done'}, f,
                  ensure_ascii=False, indent=2)
    print(f'\n导出完成: v4/snn_trained_seed{SEED}.json / v4/metrics_seed{SEED}.json / v4/training_curve_seed{SEED}.png')


def priming_test(model, pd_pref, wind_pref, device, repeats=60):
    """中等强度的视觉/风觉线索单独或叠加时，统计 GF 是否放电与首次放电潜伏期。（与 v3 逐位一致）"""
    rng = np.random.default_rng(4242)
    n_vis = pd_pref.shape[0]
    out = {}
    for tag, mul_v, mul_w in (('weak_vision', 0.35, 0.0), ('weak_wind', 0.0, 0.35), ('weak_both', 0.35, 0.35)):
        fired = 0
        lats = []
        for _ in range(repeats):
            loom = 3.0 * mul_v if mul_v > 0 else 0.0
            wind = np.array([0.05 * mul_w, 0.0, 0.0]) if mul_w > 0 else np.zeros(3)
            sp = encode_spikes(rng, loom, wind, 0.0, pd_pref, wind_pref, T_STEPS)
            x = torch.tensor(sp[:, None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                _, _, _, _, first_step, _ = model(x)
            if first_step[0].item() < T_STEPS:
                fired += 1
                lats.append(first_step[0].item() * DT_MS)
        out[tag] = {'fire_rate': fired / repeats,
                    'first_spike_ms': float(np.mean(lats)) if lats else float('nan')}
    return out


def export_model(model, results, ood_results, priming, hist, seed, args):
    w = (model.w * model.w_mask).detach().cpu().numpy()
    edges = [[int(a), int(b), round(float(c), 6)] for a, b, c in zip(model.srcb.cpu().numpy(),
                                                                    model.dstb.cpu().numpy(), w)]
    payload = {
        'meta': {
            'dt_ms': DT_MS, 't_steps': T_STEPS, 'beta': BETA, 'threshold': VTH,
            'weight_init': 'W0 = sign(nt)*log1p(syn_count) / per-post |W| sum * 1.2（FlyWire 拓扑 mask 固定）',
            'note': '权重为合成任务上微调结果；拓扑与极性来自 FlyWire，非生理实测权重',
            'v4_stage': 'A（数据升级+防过拟合）',
            'train_seed': int(seed),
            'pref_seed': int(seed),   # make_pref 用的种子（导出网页数据时必须用同一种子重建 pd_pref/wind_pref）
            'dir_flipped': False,
        },
        'num_nodes': int(model.n),
        'input_vision_indices': model.in_vision.cpu().tolist(),
        'input_wind_indices': model.in_wind.cpu().tolist(),
        'hub_gf_indices': model.hub_idx.cpu().tolist(),
        'output_indices': model.out_idx.cpu().tolist(),
        'in_gain': [round(float(x), 6) for x in model.in_gain.detach().cpu().numpy()],
        'head_trig': {
            'w': [round(float(x), 6) for x in model.head_trig.weight.detach().cpu().numpy().ravel()],
            'b': round(float(model.head_trig.bias.item()), 6),
        },
        'head_dir': {
            'w': [[round(float(x), 6) for x in row] for row in model.head_dir.weight.detach().cpu().numpy()],
            'b': [round(float(x), 6) for x in model.head_dir.bias.detach().cpu().numpy()],
        },
        'edges': edges,
    }
    with open(os.path.join(BASE, f'snn_trained_seed{seed}.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f)

    def _nan_to_none(o):  # NaN -> null：metrics JSON 保持严格 JSON 合法（可被 Node require）
        if isinstance(o, float) and o != o:
            return None
        if isinstance(o, dict):
            return {k: _nan_to_none(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_nan_to_none(x) for x in o]
        return o
    metrics = _nan_to_none({
        'results': results,
        'ood_results': ood_results,
        'priming': priming,
        'train_loss': hist,
        'config': {'train_seed': seed, 'pref_seed': seed, 'n_train': args.n_train, 'n_val': N_VAL,
                   'n_ood': N_OOD, 'epochs_max': args.epochs, 'noise_std': args.noise_std,
                   'p_drop': P_DROP, 'wd': args.wd, 'patience': args.patience,
                   'val_set_seed': VAL_SET_SEED, 'ood_set_seed': OOD_SET_SEED,
                   'val_loss_seed': VAL_LOSS_SEED, 'batch': BATCH, 'lr': LR,
                   'checkpoint_selection': 'val_loss 最小（v3: 每5轮 score 最优）'},
    })
    with open(os.path.join(BASE, f'metrics_seed{seed}.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        eps = [h['epoch'] for h in hist]
        plt.figure(figsize=(6, 3.5))
        plt.plot(eps, [h['loss'] for h in hist], marker='o', label='train loss')
        plt.plot(eps, [h['val_loss'] for h in hist], marker='s', label='val loss')
        plt.xlabel('epoch'); plt.ylabel('loss'); plt.title(f'v4 SNN training curves (seed {seed})')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(BASE, f'training_curve_seed{seed}.png'), dpi=120)
    except Exception as e:
        print('(绘图跳过:', e, ')')


def speed_test(args):
    """[v4] 测速：数据生成 / 编码 / 训练批次 / 验证损失 / B=1 评估，各计时并外推单 epoch 与总时长。"""
    SEED = args.seed
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device('cpu')
    g, n, src, dst, w0 = load_graph()
    in_v, in_w = g['input_vision_indices'], g['input_wind_indices']
    hub, out = g['hub_gf_indices'], g['output_indices']
    n_vis, n_wind = len(in_v), len(in_w)
    pd_pref, wind_pref = make_pref(n_vis, n_wind, SEED)
    model = EscapeSNN(n, src, dst, w0, in_v, in_w, hub, out).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=args.wd)
    n_params = sum(p.numel() for p in model.parameters())

    rep = {}
    t = time.time(); s2k, _ = build_dataset(2000, SEED + 1); rep['gen_2000_s'] = round(time.time() - t, 2)
    t = time.time(); build_ood_dataset(500, OOD_SET_SEED); rep['gen_ood_500_s'] = round(time.time() - t, 2)

    rng = np.random.default_rng(SEED + 777)
    idx = list(range(64))
    t = time.time()
    for _ in range(100):
        encode_batch(rng, s2k, idx, pd_pref, wind_pref, augment=True, noise_std=args.noise_std)
    rep['encode_batch64_ms'] = round((time.time() - t) / 100 * 1000, 1)

    # 训练批次（第 1 批为预热，不计）
    for it in range(9):
        idx = list(range(64))
        sp, trigs, escs = encode_batch(rng, s2k, idx, pd_pref, wind_pref, augment=True, noise_std=args.noise_std)
        x = torch.tensor(sp, device=device)
        y_trig = torch.tensor(trigs, device=device); y_esc = torch.tensor(escs, device=device)
        t = time.time()
        trig_logit, dir_pred, rate, v_hub_max, _, _ = model(x)
        loss = compute_loss(trig_logit, dir_pred, rate, v_hub_max, y_trig, y_esc, device)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        dt = time.time() - t
        if it == 0:
            continue
        rep.setdefault('train_batch_s', []).append(round(dt, 3))
    rep['train_batch_s_mean'] = round(float(np.mean(rep['train_batch_s'])), 3)

    t = time.time(); val_loss_of(model, s2k[:512], pd_pref, wind_pref, device)
    rep['valloss_512_s'] = round(time.time() - t, 2)

    with torch.no_grad():
        x1 = torch.tensor(np.zeros((T_STEPS, 1, n_vis + n_wind), dtype=np.float32), device=device)
        t = time.time()
        for _ in range(50):
            model(x1)
        rep['eval_fwd_b1_ms'] = round((time.time() - t) / 50 * 1000, 1)

    nb = rep['train_batch_s_mean']
    per_epoch = {}
    for N in (50000, 30000, 20000):
        per_epoch[str(N)] = round(N / BATCH * nb + N_VAL / BATCH * (rep['valloss_512_s'] * BATCH / 512), 1)
    rep['per_epoch_s'] = per_epoch
    rep['total_25ep_h'] = {k: round(v * 25 / 3600, 2) for k, v in per_epoch.items()}
    rep['eval_wall_min'] = round((N_VAL * 3 + N_OOD * 3) * rep['eval_fwd_b1_ms'] / 1000 / 60, 1)
    rep['n_params'] = n_params
    rep['torch_threads'] = torch.get_num_threads()
    rep['config'] = {'batch': BATCH, 'T': T_STEPS, 'E': int(len(src)), 'n': int(n)}
    with open(os.path.join(BASE, 'speed_test.json'), 'w', encoding='utf-8') as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
