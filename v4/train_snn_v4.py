# -*- coding: utf-8 -*-
"""
train_snn_v4.py —— v4 Phase A: Data Upgrade + Anti-Overfitting (only changes data & regularization from v3 train_snn.py)
============================================================================================
Differences from v3 train_snn.py (bit-identical otherwise):
  1) Training samples N_TRAIN 5000 -> 50000 (adjustable via --n-train; may drop to 20-30k after speed test);
  2) Input augmentation (training only): Gaussian noise σ~U(0, NOISE_STD=0.05) added to visual/wind firing probabilities (adjustable),
     and P_DROP=0.12 probability of zeroing out entire modality (randomly drop vision or wind, 6% each);
  3) OOD test set of 2000 samples: clearly shifted parameters generated separately (see sample_threat_ood / build_ood_dataset),
     never used in any training/validation环节 (only listed separately in final evaluation);
  4) 3 training seeds (20240521/20240522/20240523), validation/OOD sets fixed across seeds, report mean±std(ddof=1);
  5) Regularization: Adam weight_decay=1e-4; early stopping monitors **validation loss** (PATIENCE=6, minimum MIN_EPOCHS=8);
     Each epoch records train/val loss curves (overfitting health check = train-val gap); checkpoint saved each epoch, supports --resume for checkpoint continuation;
  6) Checkpoint selection criterion: minimum validation loss (v3 selects best score every 5 epochs) — difference will be honestly declared in REPORT_v4.md.

Unchanged items (bit-identical to train_snn.py for comparability):
  Architecture (EscapeSNN/LIF β=0.85, VTH=1.0, T=12, dt=1ms, soft reset), topology mask, in_gain, readout heads,
  Label rule (ṙ<0 and ttc<50ms and r<25cm; esc=−r̂), decision window (trigger=GF first spike step<12),
  Loss function (0.5·BCE + 0.5·direction cosine (with yTrig mask) + 2e-3·rate·T + 0.6·GF firing BCE(pos_weight=2.5)),
  Activation boundary (>0), Adam+cosine annealing, BATCH=64, EPOCHS=25, LR=2e-3.

Usage:
  py -3.13 v4\\train_snn_v4.py --seed 20240521                  # Train one seed
  py -3.13 v4\\train_snn_v4.py --seed 20240521 --resume         # Resume from checkpoint
  py -3.13 v4\\train_snn_v4.py --speed-test                     # Speed test only
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
ROOT = os.path.dirname(BASE)                            # Project root (v3 frozen zone, read-only)
os.environ.setdefault('PYTHONPATH', os.path.join(ROOT, 'pylibs'))
# [v4] OpenMP passive waiting (must be set before importing torch): default spin-wait competes for cores under multi-process/continuous load,
#      measured speedup from 0.6-0.8s/batch down to 0.15-0.21s/batch (see REPORT_v4.md §2 speed test).
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
os.environ.setdefault('KMP_BLOCKTIME', '0')
import sys
sys.path.insert(0, os.path.join(ROOT, 'pylibs'))        # Fallback: detached background processes can also find numpy/torch

import torch
import torch.nn as nn

try:  # snntorch surrogate gradients; if not installed, use equivalent self-implemented fast-sigmoid
    from snntorch import surrogate as _surr
    spike_grad = _surr.fast_sigmoid()
    USING = 'snntorch surrogate.fast_sigmoid'
except Exception:
    class _FastSigmoid(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return (x > 0).float()   # Boundary >0 consistent with snntorch FastSigmoid

        @staticmethod
        def backward(ctx, grad):
            (x,) = ctx.saved_tensors
            slope = 2.0
            return grad * (1.0 / (1.0 + slope * x.abs()) ** 2)

    def spike_grad(x):
        return _FastSigmoid.apply(x)
    USING = 'manual fast-sigmoid'

# ---------------- Hyperparameters (uppercase constants bit-identical to v3 train_snn.py unless noted as v4 new) ----------------
DT_MS = 1.0
T_STEPS = 12          # Simulation window 12 ms (GF escape response latency order of magnitude)
BETA = 0.85           # Membrane potential decay
VTH = 1.0             # Threshold
BATCH = 64
EPOCHS = 25
LR = 2e-3
N_TRAIN_DEFAULT = 50000   # [v4] v3=5000
N_VAL = 1200
N_OOD = 2000              # [v4] OOD test set size
RATE_MAX_HZ = 200.0   # Maximum input encoding firing rate
BASE_RATE_HZ = 5.0
TAU_TTC_MS = 50.0     # Trigger criterion: time-to-collision < 50 ms
R_TRIGGER_CM = 25.0   # and distance < 25 cm

# ---------------- [v4] Data augmentation & regularization hyperparameters ----------------
NOISE_STD = 0.05      # Gaussian noise upper bound (per-sample σ~U(0,NOISE_STD)), 0-0.05 adjustable
P_DROP = 0.12         # Modality dropout: probability of zeroing entire channel (10-15% range; drop vision/drop wind 6% each)
WD = 1e-4             # Weight decay
PATIENCE = 6          # Early stopping patience (monitors validation loss)
MIN_EPOCHS = 8        # Minimum training epochs before early stopping
VAL_LOSS_SEED = 424242    # Fixed encoding seed for validation loss (reset each epoch → comparable, reproducible loss curves)
VAL_SET_SEED = 20240523   # Fixed validation set seed (= v3's SEED+2; 3 training seeds share the same validation samples)
OOD_SET_SEED = 20240603   # Fixed OOD set seed (shared by 3 training seeds)


def load_graph():
    with open(os.path.join(ROOT, 'escape_network_sparse.json'), encoding='utf-8') as f:
        g = json.load(f)
    n = g['meta']['num_nodes']
    src = np.array([e['src'] for e in g['edges']], dtype=np.int64)
    dst = np.array([e['dst'] for e in g['edges']], dtype=np.int64)
    w = np.array([e['weight'] for e in g['edges']], dtype=np.float32)
    # Normalize by total incoming weight of postsynaptic neuron to stabilize initial dynamics
    abs_sum = np.zeros(n, dtype=np.float32)
    np.add.at(abs_sum, dst, np.abs(w))
    w = w / (abs_sum[dst] + 1e-6) * 1.2
    return g, n, src, dst, w


# ---------------- Synthetic dataset: physics-formula-generated "threat state -> cues -> labels" (bit-identical to v3) ----------------
def sample_threat(rng):
    """Return threat geometry state at a given moment (units cm, cm/ms -> converted to m/s for display)"""
    d0 = rng.uniform(3.0, 35.0)                 # Current distance cm
    az = rng.uniform(-math.pi, math.pi)         # Azimuth angle relative to fly orientation
    el = rng.uniform(-0.6, 0.6)
    speed = rng.uniform(0.3, 12.0)              # m/s
    miss = rng.uniform(0.0, 8.0)                # Miss distance cm
    s_size = rng.uniform(0.3, 3.0)              # Threat object radius cm
    # Line-of-sight direction (unit vector from threat relative to fly)
    rhat = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    rvec = rhat * d0
    # Velocity direction: aimed near the fly (miss creates tangential component)
    to_fly = -rhat
    tangent = np.array([-rhat[1], rhat[0], 0.0])
    tn = tangent / (np.linalg.norm(tangent) + 1e-9)
    vdir = to_fly * math.sqrt(max(0.0, 1 - (miss / (d0 + 1e-9)) ** 2)) + tn * (miss / (d0 + 1e-9))
    vdir = vdir / (np.linalg.norm(vdir) + 1e-9)
    v_cm_per_ms = vdir * (speed * 100.0 / 1000.0)   # m/s -> cm/ms
    return dict(d=d0, az=az, el=el, s=s_size, rhat=rhat, rvec=rvec, v=v_cm_per_ms, speed=speed)


def sample_threat_ood(rng):
    """[v4] OOD distribution (clearly shifted, never used in any training/validation):
      * Threat speed cap 12 m/s with increased fast-threat ratio: 60% ~ U(7,12) + 40% ~ U(3,12) (training dist: U(0.3,12))
      * Different angle distribution: az ~ U(-2.0, 2.0) (forward-biased, training: U(-π,π)), el ~ U(-0.9, 0.9) (training: U(-0.6,0.6))
      * Wider distance d ~ U(3, 40) cm (training: U(3,35)), larger miss ~ U(0, 10) cm (training: U(0,8))
      * Stronger wind field: wind pressure ×2 plus ambient wind U(0.01, 0.03) cm/ms (random direction, constant within sample)
      * Sensor noise ×2: Gaussian noise σ~U(0, 2·NOISE_STD) at encoding (see run_eval noise_sigma_max)
      Label rule unchanged (ṙ<0 and ttc<50ms and r<25cm; esc=−r̂)."""
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
    """Compute two types of cues and supervision labels from physics formulas (bit-identical to v3).
    Visual expansion rate: theta = 2*atan(s/r)  ->  dtheta/dt = -2 s r' / (r^2 + s^2)   (rad/s)
    Wind pressure: sphere potential flow approximation u = C s^2 v / r^2, direction along threat velocity  (arbitrary units, used after normalization)
    """
    r = st['d']
    s = st['s']
    rdot = float(np.dot(st['v'], st['rhat']))       # cm/ms, negative when approaching
    looming = max(0.0, -2.0 * s * rdot / (r ** 2 + s ** 2)) * 1000.0   # rad/s
    u = (s ** 2) * np.linalg.norm(st['v']) / (r ** 2 + 1e-9)           # cm/ms arbitrary units
    wind_vec = u * (st['v'] / (np.linalg.norm(st['v']) + 1e-9))

    ttc_ms = 1e9 if rdot >= 0 else r / (-rdot) / 1.0  # r(cm)/|rdot|(cm/ms) = ms
    trigger = 1.0 if (rdot < 0 and ttc_ms < TAU_TTC_MS and r < R_TRIGGER_CM) else 0.0
    esc_dir = -st['rhat'] / (np.linalg.norm(st['rhat']) + 1e-9)   # Escape direction = away from threat (threat is in +r̂ direction)
    return looming, wind_vec, trigger, esc_dir, ttc_ms


def encode_spikes(rng, looming, wind_vec, threat_az, pd_pref, wind_pref, n_steps,
                  noise_sigma=0.0, drop_mode=0):
    """Poisson rate encoding -> [T, N_in] 0/1 spikes (v3 body bit-identical; v4 adds two optional parameters, default behavior same as v3)
    [v4] noise_sigma>0: Gaussian noise N(0, noise_sigma) added to each input channel firing probability then clipped to [0, 0.9];
                      noise_sigma=0 skips RNG → bit-identical to v3 encoding.
    [v4] drop_mode: 1=zero entire vision channel / 2=zero entire wind channel / 0=no zeroing (modality dropout, training only).
    LPLC2 population: encodes "threat azimuth" by preferred azimuth (azimuth info -> population code)
    JO population: encodes wind vector direction by sensitive axis (wind direction itself carries threat direction info)
    """
    n_vis = pd_pref.shape[0]
    n_in = n_vis + wind_pref.shape[0]
    loom_n = min(1.0, looming / (looming + 2.0))          # Saturating normalization
    wind_mag = np.linalg.norm(wind_vec)
    wind_n = min(1.0, wind_mag / (wind_mag + 0.012))   # Johnston's organ is a high-sensitivity mechanoreceptor (half-saturation constant set small)

    rates = np.full(n_in, BASE_RATE_HZ, dtype=np.float32)

    # Vision channel (indices 0 .. n_vis-1): preferred azimuth vs threat azimuth match rectified
    cos_v = np.cos(pd_pref[:, 0] - threat_az)
    vis_drive = loom_n * np.clip(cos_v, 0.0, None) * RATE_MAX_HZ * 0.8
    rates[:n_vis] += vis_drive

    # Wind channel: wind vector projection onto sensitive axis rectified
    wdir = wind_vec / (wind_mag + 1e-9)
    proj = wind_pref @ wdir
    wind_drive = wind_n * np.clip(proj, 0.0, None) * RATE_MAX_HZ * 0.8
    rates[n_vis:] += wind_drive

    p = np.clip(rates * DT_MS / 1000.0, 0.0, 0.9)
    if noise_sigma > 0.0:   # [v4] Sensor input Gaussian noise (skips branch when 0, RNG stream identical to v3)
        p = np.clip(p + rng.normal(0.0, noise_sigma, size=n_in), 0.0, 0.9)
    spikes = (rng.random((n_steps, n_in)) < p[None, :]).astype(np.float32)
    if drop_mode == 1:
        spikes[:, :n_vis] = 0.0
    elif drop_mode == 2:
        spikes[:, n_vis:] = 0.0
    return spikes


def build_dataset(n, seed):
    """[v3 bit-identical] Training/validation distribution."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        st = sample_threat(rng)
        looming, wind_vec, trig, esc, ttc = cues_and_labels(st)
        samples.append(dict(looming=looming, wind=wind_vec, trig=trig, esc=esc,
                            az=st['az'], ttc=ttc, speed=st['speed'], d=st['d']))
    return samples, rng


def build_ood_dataset(n, seed):
    """[v4] OOD distribution (clearly shifted); label rules identical to training/validation."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        st = sample_threat_ood(rng)
        looming, wind_vec, trig, esc, ttc = cues_and_labels(st)
        # Stronger wind field: wind pressure cue ×2 plus ambient wind (random direction, constant within sample)
        wind_cue = wind_vec * 2.0
        amb_dir = rng.normal(size=3)
        amb_dir = amb_dir / (np.linalg.norm(amb_dir) + 1e-9)
        wind_cue = wind_cue + amb_dir * rng.uniform(0.01, 0.03)
        samples.append(dict(looming=looming, wind=wind_cue, trig=trig, esc=esc,
                            az=st['az'], ttc=ttc, speed=st['speed'], d=st['d']))
    return samples


class EscapeSNN(nn.Module):
    """LIF recurrent network with FlyWire sparse topology as fixed mask; only trains edge weights and two readout heads. (Bit-identical to v3)"""

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
        first_step [B] (GF first spike step, T if not fired), hub_spk [B] (GF total spike count)"""
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
            cur = cur.index_add(0, self.dstb, msg)                       # Sparse recurrent current

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
            hs = s[self.hub_idx].T.sum(dim=1) > 0        # Whether GF fires this step
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
    pd_pref[:, 0] = np.linspace(-math.pi, math.pi, n_vis, endpoint=False)  # Preferred azimuth
    pd_pref[:, 1] = rng.uniform(0.5, 1.0, n_vis)                            # Gain heterogeneity (retained param: not connected to forward pass, keeps RNG order consistent, do not delete)
    wind_pref = rng.normal(size=(n_wind, 3)).astype(np.float32)
    wind_pref /= np.linalg.norm(wind_pref, axis=1, keepdims=True)
    return pd_pref, wind_pref


def compute_loss(trig_logit, dir_pred, rate, v_hub_max, y_trig, y_esc, device):
    """[v3 loss formula bit-identical] Factored out for shared use (training and validation use the same formula for overfitting health check comparability)."""
    bce = nn.functional.binary_cross_entropy_with_logits(trig_logit, y_trig)
    dn = dir_pred / (dir_pred.norm(dim=1, keepdim=True) + 1e-8)
    cos_loss = (1.0 - (dn * y_esc).sum(dim=1))
    dir_loss = (cos_loss * y_trig).sum() / (y_trig.sum() + 1e-6)
    # Direct supervision on "whether GF fires" mechanism-based trigger criterion (fires when v_hub_max exceeds VTH)
    # pos_weight=2.5: cost of missing threat > false alarm (asymmetric cost of biological escape), increases threat sensitivity
    gf_loss = nn.functional.binary_cross_entropy_with_logits(
        (v_hub_max - VTH) * 4.0, y_trig,
        pos_weight=torch.tensor(2.5, device=device))
    return 0.5 * bce + 0.5 * dir_loss + 2e-3 * rate * T_STEPS + 0.6 * gf_loss


def encode_batch(rng, samples, idx, pd_pref, wind_pref, augment, noise_std):
    """[v4] Encode a batch; when augment=True, applies Gaussian noise + modality dropout (training only)."""
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
                sigma = float(rng.uniform(0.0, noise_std))     # Per-sample σ~U(0, NOISE_STD)
            if rng.random() < P_DROP:                          # Modality dropout: zero entire channel
                drop_mode = 1 if rng.random() < 0.5 else 2
        sp[:, k, :] = encode_spikes(rng, st['looming'], st['wind'], st['az'], pd_pref, wind_pref,
                                    T_STEPS, noise_sigma=sigma, drop_mode=drop_mode)
        trigs[k] = st['trig']
        escs[k] = st['esc']
    return sp, trigs, escs


def run_eval(model, samples, pd_pref, wind_pref, device, mode='fusion', noise_sigma_max=0.0):
    """mode: fusion / vision_only / wind_only — bit-identical to v3 run_eval (B=1, rng(777), each mode independently reset),
    [v4] only adds noise_sigma_max (evaluation encoding Gaussian noise upper bound; validation=0 same as v3, OOD=2*NOISE_STD)."""
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
        'threat_recall': tp / max(1, tp + fn),            # [v4 new diagnostic] P(GF fires|yTrig=1)
        'false_alarm_rate': fp / max(1, fp + tn),         # [v4 new diagnostic] P(GF fires|yTrig=0)
        'n_samples': len(samples),
        'n_threat': n_pos,
    }


def val_loss_of(model, val_samples, pd_pref, wind_pref, device):
    """[v4] Validation loss: same loss formula, clean encoding (no augmentation), fixed encoding seed (reset each epoch → curves reflect only model changes)."""
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
    ap = argparse.ArgumentParser(description='v4 Phase A training (data upgrade + anti-overfitting)')
    ap.add_argument('--seed', type=int, default=20240521)
    ap.add_argument('--n-train', type=int, default=N_TRAIN_DEFAULT)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--noise-std', type=float, default=NOISE_STD)
    ap.add_argument('--drop-p', type=float, default=P_DROP)
    ap.add_argument('--wd', type=float, default=WD)
    ap.add_argument('--patience', type=int, default=PATIENCE)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--threads', type=int, default=0, help='torch CPU thread count (0=default; use to throttle with multi-seed parallel)')
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
    print(f'[v4] Graph: {n} nodes / {len(src)} edges | vision input {n_vis} | wind input {n_wind} | GF {len(hub)} | output {len(out)}')
    print(f'Surrogate gradient: {USING}')
    print(f'[v4] train_seed={SEED} | N_TRAIN={args.n_train} | Gaussian noise σ~U(0,{args.noise_std}) | '
          f'modality dropout p={P_DROP} | wd={args.wd} | early stopping patience={args.patience} (monitors val loss)')

    t_data = time.time()
    pd_pref, wind_pref = make_pref(n_vis, n_wind, SEED)          # [v4] pref uses training seed (v3: make_pref(...,SEED))
    train_samples, _ = build_dataset(args.n_train, SEED + 1)     # Training set follows seed
    val_samples, _ = build_dataset(N_VAL, VAL_SET_SEED)          # [v4] Validation set fixed across seeds (= same samples as v3)
    ood_samples = build_ood_dataset(N_OOD, OOD_SET_SEED)         # [v4] OOD set fixed across seeds, not used in training/validation
    n_pos_tr = sum(1 for s in train_samples if s['trig'] > 0.5)
    n_pos_va = sum(1 for s in val_samples if s['trig'] > 0.5)
    n_pos_ood = sum(1 for s in ood_samples if s['trig'] > 0.5)
    print(f'Data generation {time.time() - t_data:.1f}s | train {len(train_samples)} (threat {n_pos_tr}) | '
          f'val {len(val_samples)} (threat {n_pos_va}) | OOD {len(ood_samples)} (threat {n_pos_ood})')

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
        print(f'== Resume from checkpoint: starting from epoch {start_ep} (history {len(hist)} epochs, best val={best_val:.4f}@ep{best_ep}) ==')

    t0 = time.time()
    for ep in range(start_ep, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(SEED + 100 + ep)             # Independent seed per epoch (reproducible even with resume)
        order = rng.permutation(len(train_samples))
        tot = 0.0
        for i in range(0, len(train_samples), BATCH):
            idx = order[i: i + BATCH]
            B = len(idx)
            sp, trigs, escs = encode_batch(rng, train_samples, idx, pd_pref, wind_pref,
                                           augment=True, noise_std=args.noise_std)   # [v4] Input augmentation
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
        vloss = val_loss_of(model, val_samples, pd_pref, wind_pref, device)          # [v4] Validation loss
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
            print(f'== Early stopping: validation loss did not improve for {args.patience} consecutive epochs (best ep{best_ep} val={best_val:.4f}) ==', flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'\nRestored best weights (val_loss={best_val:.4f} @ ep{best_ep})')

    print('\n===== Benchmark evaluation (same 1200-sample validation set, clean encoding, same protocol as v3) =====')
    results = {}
    for mode in ('fusion', 'vision_only', 'wind_only'):
        results[mode] = run_eval(model, val_samples, pd_pref, wind_pref, device, mode=mode)

    print(f'===== OOD evaluation ({N_OOD} shifted-distribution samples, sensor noise×2; never used in training/validation) =====')
    ood_results = {}
    for mode in ('fusion', 'vision_only', 'wind_only'):
        ood_results[mode] = run_eval(model, ood_samples, pd_pref, wind_pref, device, mode=mode,
                                     noise_sigma_max=2.0 * args.noise_std)

    priming = priming_test(model, pd_pref, wind_pref, device)

    fmt = '{:<12}{:>16}{:>16}{:>14}{:>14}{:>14}'.format(
        'Group', 'Trigger Acc(GF)', 'Trigger Acc(Head)', 'Escape Success', 'Direction Error(°)', 'GF Latency(ms)')
    for tag, block in (('Val Set', results), ('OOD Set', ood_results)):
        print(f'\n[{tag}]')
        print(fmt)
        for mode, r in block.items():
            print('{:<12}{:>16.3f}{:>16.3f}{:>14.3f}{:>14.1f}{:>14.2f}'.format(
                mode, r['trigger_acc_gf_spike'], r['trigger_acc_head'],
                r['escape_success_rate'], r['dir_mae_deg'], r['gf_first_spike_ms']))
    print('\nWeak cue priming test — GF firing rate and first spike latency:')
    for tag, r in priming.items():
        print('  {:<14} fire_rate={:.2f}  first_spike={:.2f} ms'.format(
            tag, r['fire_rate'], r['first_spike_ms']))

    export_model(model, results, ood_results, priming, hist, SEED, args)
    with open(status_path, 'w', encoding='utf-8') as f:
        json.dump({'seed': SEED, 'epoch': len(hist), 'epochs_max': args.epochs,
                   'best_val': best_val, 'best_epoch': best_ep, 'state': 'done'}, f,
                  ensure_ascii=False, indent=2)
    print(f'\nExport complete: v4/snn_trained_seed{SEED}.json / v4/metrics_seed{SEED}.json / v4/training_curve_seed{SEED}.png')


def priming_test(model, pd_pref, wind_pref, device, repeats=60):
    """Statistics on whether GF fires and first spike latency when visual/wind cues are presented individually or together at medium intensity. (Bit-identical to v3)"""
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
            'weight_init': 'W0 = sign(nt)*log1p(syn_count) / per-post |W| sum * 1.2 (FlyWire topology mask fixed)',
            'note': 'Weights are fine-tuned on the synthetic task; topology and polarity from FlyWire, not physiological measurements',
            'v4_stage': 'A (data upgrade + anti-overfitting)',
            'train_seed': int(seed),
            'pref_seed': int(seed),   # Seed used by make_pref (must use same seed to reconstruct pd_pref/wind_pref for web data export)
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

    def _nan_to_none(o):  # NaN -> null: metrics JSON stays strictly valid JSON (can be Node require'd)
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
                   'checkpoint_selection': 'min val_loss (v3: best score every 5 epochs)'},
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
        print('(Plot skipped:', e, ')')


def speed_test(args):
    """[v4] Speed test: data generation / encoding / training batch / validation loss / B=1 evaluation, each timed and extrapolated for single epoch and total duration."""
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

    # Training batch (first iteration is warmup, not counted)
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
