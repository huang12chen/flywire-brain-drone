# -*- coding: utf-8 -*-
"""
train_snn.py — Train a "visual-wind bimodal escape" LIF spiking neural network using sparse connectivity matrices exported from FlyWire
============================================================================================================================================
Objective notes (to avoid overpromising):
  * Connectome comes from a single female adult whole brain (FlyWire/FAFB); synapse counts are only proxies for anatomical connection strengths, not physiological weights;
  * Training data are "threat cue -> escape command" mappings synthesized programmatically from physics formulas (visual looming rate dθ/dt + antenna wind pressure),
    not electrophysiological measurements; sensor encoding parameters (max firing rate 200Hz, half-saturation constant, etc.) are engineering values at literature scales.
    Therefore this experiment demonstrates whether an SNN constrained by real connection topology can complete this control task, and the quantitative differences
    among three input conditions (pure vision / pure wind / fusion) on the same task, rather than making quantitative predictions about organism behavior.
  * The topology mask is fixed (biological connections only allow weight tuning, not arbitrary rewiring).

Model (one step per millisecond, total T=12 ms):
  v[t] = beta * v[t-1] + I_rec[t] + I_in[t] - Vth * s[t-1]      (LIF, soft reset)
  s[t] = H( v[t] - Vth )                                        (surrogate gradient: fast-sigmoid)
  I_rec[t] = W_e ⊗ s[t-1]   (message passing: index_add, equivalent to sparse matrix multiply, efficient on CPU)
  Output: trigger logit = Linear(GF membrane potential) ; escape direction = Linear(output layer DN membrane potential) -> R^3

Loss: 0.5*BCE(trigger) + 0.5*direction cosine loss + 2e-3*mean spike count*T(energy cost) + 0.6*GF spike BCE(pos_weight=2.5)
      (Full loss formula refers to REPORT §1.6 and the loss computation section in this file)
"""
import os
import json
import math
import time
import copy
import random

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('PYTHONPATH', os.path.join(BASE, 'pylibs'))

import torch
import torch.nn as nn

try:  # snntorch surrogate gradient; if not installed, use equivalent self-implemented fast-sigmoid
    from snntorch import surrogate as _surr
    spike_grad = _surr.fast_sigmoid()
    USING = 'snntorch surrogate.fast_sigmoid'
except Exception:
    class _FastSigmoid(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return (x > 0).float()   # boundary >0 consistent with snntorch FastSigmoid

        @staticmethod
        def backward(ctx, grad):
            (x,) = ctx.saved_tensors
            slope = 2.0
            return grad * (1.0 / (1.0 + slope * x.abs()) ** 2)

    def spike_grad(x):
        return _FastSigmoid.apply(x)
    USING = 'manual fast-sigmoid'

SEED = 20240521
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------- Hyperparameters ----------------
DT_MS = 1.0
T_STEPS = 12          # Simulation window 12 ms (GF escape response latency scale)
BETA = 0.85           # Membrane potential decay
VTH = 1.0             # Threshold
BATCH = 64
EPOCHS = 25
LR = 2e-3
N_TRAIN = 5000
N_VAL = 1200
RATE_MAX_HZ = 200.0   # Max firing rate for input encoding
BASE_RATE_HZ = 5.0
TAU_TTC_MS = 50.0     # Trigger criterion: time-to-collision < 50 ms
R_TRIGGER_CM = 25.0   # and distance < 25 cm


def load_graph():
    with open(os.path.join(BASE, 'escape_network_sparse.json'), encoding='utf-8') as f:
        g = json.load(f)
    n = g['meta']['num_nodes']
    src = np.array([e['src'] for e in g['edges']], dtype=np.int64)
    dst = np.array([e['dst'] for e in g['edges']], dtype=np.int64)
    w = np.array([e['weight'] for e in g['edges']], dtype=np.float32)
    # Normalize by total incoming strength of postsynaptic neurons to stabilize initial dynamics
    abs_sum = np.zeros(n, dtype=np.float32)
    np.add.at(abs_sum, dst, np.abs(w))
    w = w / (abs_sum[dst] + 1e-6) * 1.2
    return g, n, src, dst, w


# ---------------- Synthetic dataset: physics formula generates "threat state -> cues -> labels" ----------------
def sample_threat(rng):
    """Return the threat geometry state at a given moment (units cm, cm/ms -> converted to m/s for display)"""
    d0 = rng.uniform(3.0, 35.0)                 # Current distance cm
    az = rng.uniform(-math.pi, math.pi)         # Azimuth angle relative to fly orientation
    el = rng.uniform(-0.6, 0.6)
    speed = rng.uniform(0.3, 12.0)              # m/s
    miss = rng.uniform(0.0, 8.0)                # Miss distance cm
    s_size = rng.uniform(0.3, 3.0)              # Threat object radius cm
    # Line of sight direction (unit vector of threat position relative to fly)
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


def cues_and_labels(st):
    """Compute two types of cues and supervision labels from physics formulas.
    Visual looming rate: theta = 2*atan(s/r)  ->  dtheta/dt = -2 s r' / (r^2 + s^2)   (rad/s)
    Wind pressure: sphere potential flow approximation u = C s^2 v / r^2, direction along threat velocity direction   (arbitrary units, used after normalization)
    """
    r = st['d']
    s = st['s']
    rdot = float(np.dot(st['v'], st['rhat']))       # cm/ms, negative when approaching
    looming = max(0.0, -2.0 * s * rdot / (r ** 2 + s ** 2)) * 1000.0   # rad/s
    u = (s ** 2) * np.linalg.norm(st['v']) / (r ** 2 + 1e-9)           # cm/ms arbitrary units
    wind_vec = u * (st['v'] / (np.linalg.norm(st['v']) + 1e-9))

    ttc_ms = 1e9 if rdot >= 0 else r / (-rdot) / 1.0  # r(cm)/|rdot|(cm/ms) = ms
    trigger = 1.0 if (rdot < 0 and ttc_ms < TAU_TTC_MS and r < R_TRIGGER_CM) else 0.0
    esc_dir = -st['rhat'] / (np.linalg.norm(st['rhat']) + 1e-9)   # Escape direction = away from threat (threat is in +rhat direction)
    return looming, wind_vec, trigger, esc_dir, ttc_ms


def encode_spikes(rng, looming, wind_vec, threat_az, pd_pref, wind_pref, n_steps):
    """Poisson firing rate encoding -> [T, N_in] 0/1 spikes
    LPLC2 population: encodes "threat azimuth" by preferred azimuth (azimuth information -> population code)
    JO   population: encodes wind vector direction by sensitive axis (wind direction itself carries threat direction information)
    """
    n_vis = pd_pref.shape[0]
    n_in = n_vis + wind_pref.shape[0]
    loom_n = min(1.0, looming / (looming + 2.0))          # Saturated normalization
    wind_mag = np.linalg.norm(wind_vec)
    wind_n = min(1.0, wind_mag / (wind_mag + 0.012))   # Johnston's organ is a high-sensitivity mechanoreceptor (half-saturation constant set small)

    rates = np.full(n_in, BASE_RATE_HZ, dtype=np.float32)

    # Visual channel (indices 0 .. n_vis-1): rectified match between preferred azimuth and threat azimuth
    cos_v = np.cos(pd_pref[:, 0] - threat_az)
    vis_drive = loom_n * np.clip(cos_v, 0.0, None) * RATE_MAX_HZ * 0.8
    rates[:n_vis] += vis_drive

    # Wind channel: rectified projection of wind vector onto sensitive axis
    wdir = wind_vec / (wind_mag + 1e-9)
    proj = wind_pref @ wdir
    wind_drive = wind_n * np.clip(proj, 0.0, None) * RATE_MAX_HZ * 0.8
    rates[n_vis:] += wind_drive

    p = np.clip(rates * DT_MS / 1000.0, 0.0, 0.9)
    spikes = (rng.random((n_steps, n_in)) < p[None, :]).astype(np.float32)
    return spikes


def build_dataset(n, seed):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        st = sample_threat(rng)
        looming, wind_vec, trig, esc, ttc = cues_and_labels(st)
        samples.append(dict(looming=looming, wind=wind_vec, trig=trig, esc=esc,
                            az=st['az'], ttc=ttc, speed=st['speed'], d=st['d']))
    return samples, rng


class EscapeSNN(nn.Module):
    """LIF recurrent network with fixed FlyWire sparse topology as mask; only edge weights and two readout heads are trained."""

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
        first_step [B] (GF first spike step, T if no spike), hub_spk [B] (GF total spike count)"""
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
    pd_pref[:, 1] = rng.uniform(0.5, 1.0, n_vis)                            # Gain heterogeneity (retained parameter: currently not connected to forward pass, keeps both ends of RNG order consistent, do not delete)
    wind_pref = rng.normal(size=(n_wind, 3)).astype(np.float32)
    wind_pref /= np.linalg.norm(wind_pref, axis=1, keepdims=True)
    return pd_pref, wind_pref


def run_eval(model, samples, pd_pref, wind_pref, device, mode='fusion'):
    """mode: fusion / vision_only / wind_only"""
    model.eval()
    rng = np.random.default_rng(777)
    n_vis = pd_pref.shape[0]
    hits = 0
    head_ok = 0
    gf_ok = 0
    ang_errs = []
    latencies = []
    with torch.no_grad():
        for st in samples:
            sp = encode_spikes(rng, st['looming'], st['wind'], st['az'], pd_pref, wind_pref, T_STEPS)
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
        'n_samples': len(samples),
        'n_threat': n_pos,
    }


def main():
    device = torch.device('cpu')
    g, n, src, dst, w0 = load_graph()
    in_v = g['input_vision_indices']
    in_w = g['input_wind_indices']
    hub = g['hub_gf_indices']
    out = g['output_indices']
    n_vis, n_wind = len(in_v), len(in_w)
    print(f'Graph: {n} nodes / {len(src)} edges | Vision input {n_vis} | Wind input {n_wind} | GF {len(hub)} | Output {len(out)}')
    print(f'Surrogate gradient: {USING}')

    pd_pref, wind_pref = make_pref(n_vis, n_wind, SEED)
    train_samples, _ = build_dataset(N_TRAIN, SEED + 1)
    val_samples, _ = build_dataset(N_VAL, SEED + 2)

    model = EscapeSNN(n, src, dst, w0, in_v, in_w, hub, out).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=2e-4)
    best_state, best_score = None, -1.0

    in_idx = np.array(in_v + in_w, dtype=np.int64)
    t0 = time.time()
    hist = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        rng = np.random.default_rng(SEED + 100 + ep)
        order = rng.permutation(len(train_samples))
        tot = 0.0
        for i in range(0, len(train_samples), BATCH):
            idx = order[i: i + BATCH]
            B = len(idx)
            sp = np.zeros((T_STEPS, B, n_vis + n_wind), dtype=np.float32)
            trigs = np.zeros(B, dtype=np.float32)
            escs = np.zeros((B, 3), dtype=np.float32)
            for k, j in enumerate(idx):
                st = train_samples[j]
                sp[:, k, :] = encode_spikes(rng, st['looming'], st['wind'], st['az'], pd_pref, wind_pref, T_STEPS)
                trigs[k] = st['trig']
                escs[k] = st['esc']
            x = torch.tensor(sp, device=device)
            y_trig = torch.tensor(trigs, device=device)
            y_esc = torch.tensor(escs, device=device)

            trig_logit, dir_pred, rate, v_hub_max, _, _ = model(x)
            bce = nn.functional.binary_cross_entropy_with_logits(trig_logit, y_trig)
            dn = dir_pred / (dir_pred.norm(dim=1, keepdim=True) + 1e-8)
            cos_loss = (1.0 - (dn * y_esc).sum(dim=1))
            dir_loss = (cos_loss * y_trig).sum() / (y_trig.sum() + 1e-6)
            # Directly supervise the mechanistic trigger criterion of "whether GF fires" (fires when v_hub_max crosses VTH)
            # pos_weight=2.5: cost of missing threat > false positive (asymmetric cost of biological escape), increases sensitivity to threats
            gf_loss = nn.functional.binary_cross_entropy_with_logits(
                (v_hub_max - VTH) * 4.0, y_trig,
                pos_weight=torch.tensor(2.5, device=device))
            loss = 0.5 * bce + 0.5 * dir_loss + 2e-3 * rate * T_STEPS + 0.6 * gf_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * B
        scheduler.step()
        hist.append({'epoch': ep, 'loss': tot / len(train_samples),
                     'lr': scheduler.get_last_lr()[0]})
        print(f'epoch {ep:2d}/{EPOCHS}  loss={tot / len(train_samples):.4f}  '
              f'lr={scheduler.get_last_lr()[0]:.5f}  ({time.time() - t0:.0f}s)')

        # Quick evaluation every 5 epochs on a validation subset, save best weights (prevent overfitting)
        if ep % 5 == 0 or ep == EPOCHS:
            r = run_eval(model, val_samples[:400], pd_pref, wind_pref, device, mode='fusion')
            score = r['escape_success_rate'] + 0.25 * r['trigger_acc_gf_spike']
            print(f'         [Val] GF_trigger={r["trigger_acc_gf_spike"]:.3f} '
                  f'success_rate={r["escape_success_rate"]:.3f} direction_error={r["dir_mae_deg"]:.1f}°  score={score:.3f}')
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'\nRestored best weights (score={best_score:.3f})')

    print('\n===== Comparative evaluation (same validation set) =====')
    results = {}
    for mode in ('fusion', 'vision_only', 'wind_only'):
        results[mode] = run_eval(model, val_samples, pd_pref, wind_pref, device, mode=mode)

    # Multimodal pre-activation (priming): GF first spike latency under moderate-strength cues
    priming = priming_test(model, pd_pref, wind_pref, device)

    fmt = '{:<12}{:>16}{:>16}{:>14}{:>14}{:>14}'.format(
        'Group', 'Trigger Acc(GF)', 'Trigger Acc(Head)', 'Avoidance SR', 'Dir Error(°)', 'GF Latency(ms)')
    print(fmt)
    for mode, r in results.items():
        print('{:<12}{:>16.3f}{:>16.3f}{:>14.3f}{:>14.1f}{:>14.2f}'.format(
            mode, r['trigger_acc_gf_spike'], r['trigger_acc_head'],
            r['escape_success_rate'], r['dir_mae_deg'], r['gf_first_spike_ms']))
    print('\nWeak-cue priming test — GF firing ratio and first spike latency:')
    for tag, r in priming.items():
        print('  {:<14} fire_rate={:.2f}  first_spike={:.2f} ms'.format(
            tag, r['fire_rate'], r['first_spike_ms']))

    export_model(model, results, priming, hist)
    print('\nExport complete: snn_trained.json / metrics.json')


def priming_test(model, pd_pref, wind_pref, device, repeats=60):
    """Under moderate-strength visual/wind cues alone or combined, record whether GF fires and first spike latency."""
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


def export_model(model, results, priming, hist):
    w = (model.w * model.w_mask).detach().cpu().numpy()
    edges = [[int(a), int(b), round(float(c), 6)] for a, b, c in zip(model.srcb.cpu().numpy(),
                                                                    model.dstb.cpu().numpy(), w)]
    payload = {
        'meta': {
            'dt_ms': DT_MS, 't_steps': T_STEPS, 'beta': BETA, 'threshold': VTH,
            'weight_init': 'W0 = sign(nt)*log1p(syn_count) / per-post |W| sum * 1.2 (FlyWire topology mask fixed)',
            'note': 'Weights are fine-tuned on the synthetic task; topology and polarity come from FlyWire, not physiological measurements',
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
    with open(os.path.join(BASE, 'snn_trained.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f)

    def _nan_to_none(o):  # NaN -> null: keep metrics.json strictly JSON-valid (can be required by Node)
        if isinstance(o, float) and o != o:
            return None
        if isinstance(o, dict):
            return {k: _nan_to_none(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_nan_to_none(x) for x in o]
        return o
    metrics = _nan_to_none({'results': results, 'priming': priming, 'train_loss': hist})
    with open(os.path.join(BASE, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3.5))
        plt.plot([h['epoch'] for h in hist], [h['loss'] for h in hist], marker='o')
        plt.xlabel('epoch'); plt.ylabel('loss'); plt.title('SNN training loss')
        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'training_curve.png'), dpi=120)
    except Exception as e:
        print('(Plotting skipped:', e, ')')


if __name__ == '__main__':
    main()
