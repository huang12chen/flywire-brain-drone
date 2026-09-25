# -*- coding: utf-8 -*-
"""
train_snn_v4c.py —— v4 Phase C: Long Decision Window (T_STEPS 64 ≈ 64ms integration)
===================================================================
Based on train_snn_v4b.py, only change: T_STEPS from 12 to 64.
Everything else unchanged (50k samples, seed 20240521, augmentation + fake threats + TTC regression).

Usage:
  py -3.13 v4\\train_snn_v4c.py --seed 20240521
"""
import os, sys, json, math, time, copy, random, argparse
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
os.environ.setdefault('PYTHONPATH', os.path.join(ROOT, 'pylibs'))
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
os.environ.setdefault('KMP_BLOCKTIME', '0')
sys.path.insert(0, os.path.join(ROOT, 'pylibs'))

import torch
import torch.nn as nn

try:
    from snntorch import surrogate as _surr
    spike_grad = _surr.fast_sigmoid()
    USING = 'snntorch surrogate.fast_sigmoid'
except Exception:
    class _FastSigmoid(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return (x > 0).float()
        @staticmethod
        def backward(ctx, grad):
            (x,) = ctx.saved_tensors
            return grad * (1.0 / (1.0 + 2.0 * x.abs()) ** 2)
    def spike_grad(x):
        return _FastSigmoid.apply(x)
    USING = 'manual fast-sigmoid'

# ── Hyperparameters (v4c only change: T_STEPS=64) ─────────────────────────────────
DT_MS = 1.0
T_STEPS = 64            # [v4c] Extended from 12 to 64 (~64ms integration window)
BETA = 0.85
VTH = 1.0
BATCH = 64
EPOCHS = 25
LR = 2e-3
N_TRAIN_DEFAULT = 50000
N_VAL = 1200
N_OOD = 2000
RATE_MAX_HZ = 200.0
BASE_RATE_HZ = 5.0
TAU_TTC_MS = 50.0
R_TRIGGER_CM = 25.0

NOISE_STD = 0.05
P_DROP = 0.12
WD = 1e-4
PATIENCE = 6
MIN_EPOCHS = 8
VAL_LOSS_SEED = 424242
VAL_SET_SEED = 20240523
OOD_SET_SEED = 20240603

# [v4b] Fake threat injection
P_FAKE = 0.10
FAKE_D_MIN = 30.0
FAKE_AZ_OFFSET = 1.05

# [v4b] Loss weights
W_REAL_BCE = 0.3
W_TTC_MSE = 0.2
TTC_NORM = 50.0
TTC_MAX_MS = 100.0


def load_graph():
    with open(os.path.join(ROOT, 'escape_network_sparse.json'), encoding='utf-8') as f:
        g = json.load(f)
    n = g['meta']['num_nodes']
    src = np.array([e['src'] for e in g['edges']], dtype=np.int64)
    dst = np.array([e['dst'] for e in g['edges']], dtype=np.int64)
    w = np.array([e['weight'] for e in g['edges']], dtype=np.float32)
    abs_sum = np.zeros(n, dtype=np.float32)
    np.add.at(abs_sum, dst, np.abs(w))
    w = w / (abs_sum[dst] + 1e-6) * 1.2
    return g, n, src, dst, w


def sample_threat(rng):
    d0 = rng.uniform(3.0, 35.0)
    az = rng.uniform(-math.pi, math.pi)
    el = rng.uniform(-0.6, 0.6)
    speed = rng.uniform(0.3, 12.0)
    miss = rng.uniform(0.0, 8.0)
    s_size = rng.uniform(0.3, 3.0)
    rhat = np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])
    rvec = rhat * d0
    to_fly = -rhat
    tangent = np.array([-rhat[1], rhat[0], 0.0])
    tn = tangent / (np.linalg.norm(tangent) + 1e-9)
    vdir = to_fly * math.sqrt(max(0.0, 1-(miss/(d0+1e-9))**2)) + tn*(miss/(d0+1e-9))
    vdir = vdir / (np.linalg.norm(vdir) + 1e-9)
    v_cm_per_ms = vdir * (speed*100.0/1000.0)
    return dict(d=d0, az=az, el=el, s=s_size, rhat=rhat, rvec=rvec, v=v_cm_per_ms, speed=speed)


def sample_threat_ood(rng):
    d0 = rng.uniform(3.0, 40.0)
    speed = rng.uniform(7.0, 12.0) if rng.random() < 0.6 else rng.uniform(3.0, 12.0)
    az = rng.uniform(-2.0, 2.0)
    el = rng.uniform(-0.9, 0.9)
    miss = rng.uniform(0.0, 10.0)
    s_size = rng.uniform(0.3, 3.0)
    rhat = np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])
    rvec = rhat * d0
    to_fly = -rhat
    tangent = np.array([-rhat[1], rhat[0], 0.0])
    tn = tangent / (np.linalg.norm(tangent) + 1e-9)
    vdir = to_fly * math.sqrt(max(0.0, 1-(miss/(d0+1e-9))**2)) + tn*(miss/(d0+1e-9))
    vdir = vdir / (np.linalg.norm(vdir) + 1e-9)
    v_cm_per_ms = vdir * (speed*100.0/1000.0)
    return dict(d=d0, az=az, el=el, s=s_size, rhat=rhat, rvec=rvec, v=v_cm_per_ms, speed=speed)


def cues_and_labels(st):
    r = st['d']
    s = st['s']
    rdot = float(np.dot(st['v'], st['rhat']))
    looming = max(0.0, -2.0*s*rdot/(r**2+s**2))*1000.0
    u = (s**2)*np.linalg.norm(st['v'])/(r**2+1e-9)
    wind_vec = u*(st['v']/(np.linalg.norm(st['v'])+1e-9))
    ttc_ms = 1e9 if rdot >= 0 else r/(-rdot)/1.0
    trigger = 1.0 if (rdot < 0 and ttc_ms < TAU_TTC_MS and r < R_TRIGGER_CM) else 0.0
    esc_dir = -st['rhat']/(np.linalg.norm(st['rhat'])+1e-9)
    return looming, wind_vec, trigger, esc_dir, ttc_ms


def sample_fake_threat(rng):
    if rng.random() < 0.5:
        d0 = rng.uniform(FAKE_D_MIN, 50.0)
        az = rng.uniform(-math.pi, math.pi)
        speed = rng.uniform(5.0, 12.0)
        miss = rng.uniform(0.0, 3.0)
    else:
        d0 = rng.uniform(5.0, 20.0)
        az_offset = rng.choice([-1, 1]) * rng.uniform(FAKE_AZ_OFFSET, math.pi)
        az = rng.uniform(-math.pi, math.pi)
        az = (az + az_offset) % (2*math.pi) - math.pi
        speed = rng.uniform(8.0, 15.0)
        miss = rng.uniform(0.0, 8.0)
    el = rng.uniform(-0.3, 0.3)
    s_size = rng.uniform(0.5, 2.5)
    rhat = np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])
    rvec = rhat * d0
    to_fly = -rhat
    tangent = np.array([-rhat[1], rhat[0], 0.0])
    tn = tangent / (np.linalg.norm(tangent)+1e-9)
    vdir = to_fly * math.sqrt(max(0.0, 1-(miss/(d0+1e-9))**2)) + tn*(miss/(d0+1e-9))
    vdir = vdir / (np.linalg.norm(vdir)+1e-9)
    v_cm_per_ms = vdir * (speed*100.0/1000.0)
    return dict(d=d0, az=az, el=el, s=s_size, rhat=rhat, rvec=rvec, v=v_cm_per_ms, speed=speed)


def encode_spikes(rng, looming, wind_vec, threat_az, pd_pref, wind_pref, n_steps,
                  noise_sigma=0.0, drop_mode=0):
    n_vis = pd_pref.shape[0]
    n_in = n_vis + wind_pref.shape[0]
    loom_n = min(1.0, looming/(looming+2.0))
    wind_mag = np.linalg.norm(wind_vec)
    wind_n = min(1.0, wind_mag/(wind_mag+0.012))
    rates = np.full(n_in, BASE_RATE_HZ, dtype=np.float32)
    cos_v = np.cos(pd_pref[:, 0]-threat_az)
    vis_drive = loom_n * np.clip(cos_v, 0.0, None) * RATE_MAX_HZ * 0.8
    rates[:n_vis] += vis_drive
    wdir = wind_vec / (wind_mag+1e-9)
    proj = wind_pref @ wdir
    wind_drive = wind_n * np.clip(proj, 0.0, None) * RATE_MAX_HZ * 0.8
    rates[n_vis:] += wind_drive
    p = np.clip(rates*DT_MS/1000.0, 0.0, 0.9)
    if noise_sigma > 0.0:
        p = np.clip(p + rng.normal(0.0, noise_sigma, size=n_in), 0.0, 0.9)
    spikes = (rng.random((n_steps, n_in)) < p[None, :]).astype(np.float32)
    if drop_mode == 1:
        spikes[:, :n_vis] = 0.0
    elif drop_mode == 2:
        spikes[:, n_vis:] = 0.0
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


def build_ood_dataset(n, seed):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        st = sample_threat_ood(rng)
        looming, wind_vec, trig, esc, ttc = cues_and_labels(st)
        wind_cue = wind_vec * 2.0
        amb_dir = rng.normal(size=3)
        amb_dir = amb_dir / (np.linalg.norm(amb_dir)+1e-9)
        wind_cue = wind_cue + amb_dir * rng.uniform(0.01, 0.03)
        samples.append(dict(looming=looming, wind=wind_cue, trig=trig, esc=esc,
                            az=st['az'], ttc=ttc, speed=st['speed'], d=st['d']))
    return samples


class EscapeSNN(nn.Module):
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
        self.in_gain = nn.Parameter(torch.ones(self.in_vision.numel()+self.in_wind.numel())*1.0)
        self.head_trig = nn.Linear(self.hub_idx.numel(), 1)
        self.head_dir = nn.Linear(self.out_idx.numel(), 3)
        self.head_real = nn.Linear(self.hub_idx.numel(), 1)
        self.head_ttc = nn.Linear(self.hub_idx.numel(), 1)

    def forward(self, spikes):
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
            msg = (self.w*self.w_mask)[:, None]*s[self.srcb]
            cur = torch.zeros(self.n, B, device=device)
            cur = cur.index_add(0, self.dstb, msg)
            xin = spikes[t]*self.in_gain[None, :]
            cur[self.in_vision] += xin[:, :self.in_vision.numel()].T
            cur[self.in_wind] += xin[:, self.in_vision.numel():].T
            v = BETA*v + cur - VTH*s
            s = spike_grad(v - VTH)
            spike_count = spike_count + s.mean()
            v_hub_sum += v[self.hub_idx].T
            v_out_sum += v[self.out_idx].T
            hv = v[self.hub_idx].T
            v_hub_max = torch.maximum(v_hub_max, hv.max(dim=1).values)
            hs = s[self.hub_idx].T.sum(dim=1) > 0
            first_step = torch.where(hs & ~fired_any,
                                     torch.full_like(first_step, float(t+1)), first_step)
            fired_any = fired_any | hs
            hub_spk = hub_spk + s[self.hub_idx].T.sum(dim=1)

        hub_feat = v_hub_sum / T
        trig_logit = self.head_trig(hub_feat).squeeze(-1)
        dir_pred = self.head_dir(v_out_sum / T)
        real_logit = self.head_real(hub_feat).squeeze(-1)
        ttc_pred = self.head_ttc(hub_feat).squeeze(-1)
        return trig_logit, dir_pred, spike_count/T, v_hub_max, first_step, hub_spk, real_logit, ttc_pred


def make_pref(n_vis, n_wind, seed):
    rng = np.random.default_rng(seed)
    pd_pref = np.zeros((n_vis, 2), dtype=np.float32)
    pd_pref[:, 0] = np.linspace(-math.pi, math.pi, n_vis, endpoint=False)
    pd_pref[:, 1] = rng.uniform(0.5, 1.0, n_vis)
    wind_pref = rng.normal(size=(n_wind, 3)).astype(np.float32)
    wind_pref /= np.linalg.norm(wind_pref, axis=1, keepdims=True)
    return pd_pref, wind_pref


def compute_loss(trig_logit, dir_pred, rate, v_hub_max, y_trig, y_esc,
                 real_logit, ttc_pred, y_real, y_ttc, device):
    bce = nn.functional.binary_cross_entropy_with_logits(trig_logit, y_trig)
    dn = dir_pred / (dir_pred.norm(dim=1, keepdim=True)+1e-8)
    cos_loss = (1.0 - (dn*y_esc).sum(dim=1))
    dir_loss = (cos_loss*y_trig).sum() / (y_trig.sum()+1e-6)
    gf_loss = nn.functional.binary_cross_entropy_with_logits(
        (v_hub_max - VTH)*4.0, y_trig,
        pos_weight=torch.tensor(2.5, device=device))
    base = 0.5*bce + 0.5*dir_loss + 2e-3*rate*T_STEPS + 0.6*gf_loss
    real_bce = nn.functional.binary_cross_entropy_with_logits(real_logit, y_real)
    ttc_mse = nn.functional.mse_loss(ttc_pred, y_ttc)
    return base + W_REAL_BCE*real_bce + W_TTC_MSE*ttc_mse


def encode_batch(rng, samples, idx, pd_pref, wind_pref, augment, noise_std):
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
                sigma = float(rng.uniform(0.0, noise_std))
            if rng.random() < P_DROP:
                drop_mode = 1 if rng.random() < 0.5 else 2
        sp[:, k, :] = encode_spikes(rng, st['looming'], st['wind'], st['az'], pd_pref, wind_pref,
                                    T_STEPS, noise_sigma=sigma, drop_mode=drop_mode)
        trigs[k] = st['trig']
        escs[k] = st['esc']
    return sp, trigs, escs


def val_loss_of(model, val_samples, pd_pref, wind_pref, device):
    model.eval()
    rng = np.random.default_rng(VAL_LOSS_SEED)
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(val_samples), BATCH):
            idx = list(range(i, min(i+BATCH, len(val_samples))))
            sp, trigs, escs = encode_batch(rng, val_samples, idx, pd_pref, wind_pref,
                                           augment=False, noise_std=0.0)
            x = torch.tensor(sp, device=device)
            y_trig = torch.tensor(trigs, device=device)
            y_esc = torch.tensor(escs, device=device)
            y_real = torch.ones(len(idx), device=device)
            y_ttc = torch.tensor([min(s['ttc'], TTC_MAX_MS)/TTC_NORM for s in
                                  [val_samples[j] for j in idx]], device=device)
            trig_logit, dir_pred, rate, v_hub_max, _, _, real_logit, ttc_pred = model(x)
            loss = compute_loss(trig_logit, dir_pred, rate, v_hub_max, y_trig, y_esc,
                                real_logit, ttc_pred, y_real, y_ttc, device)
            tot += float(loss)*len(idx)
            cnt += len(idx)
    return tot / max(1, cnt)


def run_eval(model, samples, pd_pref, wind_pref, device, mode='fusion', noise_sigma_max=0.0):
    model.eval()
    rng = np.random.default_rng(777)
    n_vis = pd_pref.shape[0]
    hits = head_ok = gf_ok = 0
    ang_errs, latencies = [], []
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
            x = torch.tensor(sp[:, None, :], dtype=torch.float32, device=device)
            trig, dirp, _, _, first_step, _, _, _ = model(x)
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
                d = dirp[0].cpu().numpy()
                cur_ang = float('nan')
                if np.linalg.norm(d) > 1e-6:
                    cos = float(np.dot(d/np.linalg.norm(d), st['esc']))
                    cur_ang = math.degrees(math.acos(np.clip(cos, -1, 1)))
                    ang_errs.append(cur_ang)
                if pred_gf >= 0.5:
                    latencies.append(first_step[0].item()*DT_MS)
                    if cur_ang < 30.0:
                        hits += 1
    n_pos = sum(1 for st in samples if st['trig'] > 0.5)
    succ = hits / max(1, n_pos)
    return {
        'trigger_acc_gf_spike': gf_ok/max(1, len(samples)),
        'trigger_acc_head': head_ok/max(1, len(samples)),
        'escape_success_rate': succ,
        'dir_mae_deg': float(np.mean(ang_errs)) if ang_errs else float('nan'),
        'gf_first_spike_ms': float(np.mean(latencies)) if latencies else float('nan'),
        'threat_recall': tp/max(1, tp+fn),
        'false_alarm_rate': fp/max(1, fp+tn),
        'n_samples': len(samples),
        'n_threat': n_pos,
    }


def priming_test(model, pd_pref, wind_pref, device, repeats=60):
    rng = np.random.default_rng(4242)
    n_vis = pd_pref.shape[0]
    out = {}
    for tag, mul_v, mul_w in (('weak_vision', 0.35, 0.0), ('weak_wind', 0.0, 0.35), ('weak_both', 0.35, 0.35)):
        fired = 0
        lats = []
        for _ in range(repeats):
            loom = 3.0*mul_v if mul_v > 0 else 0.0
            wind = np.array([0.05*mul_w, 0.0, 0.0]) if mul_w > 0 else np.zeros(3)
            sp = encode_spikes(rng, loom, wind, 0.0, pd_pref, wind_pref, T_STEPS)
            x = torch.tensor(sp[:, None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                _, _, _, _, first_step, _, _, _ = model(x)
            if first_step[0].item() < T_STEPS:
                fired += 1
                lats.append(first_step[0].item()*DT_MS)
        out[tag] = {'fire_rate': fired/repeats, 'first_spike_ms': float(np.mean(lats)) if lats else float('nan')}
    return out


def export_model(model, results, ood_results, priming, hist, seed, args):
    w = (model.w*model.w_mask).detach().cpu().numpy()
    edges = [[int(a), int(b), round(float(c), 6)] for a, b, c in zip(
        model.srcb.cpu().numpy(), model.dstb.cpu().numpy(), w)]
    payload = {
        'meta': {
            'dt_ms': DT_MS, 't_steps': T_STEPS, 'beta': BETA, 'threshold': VTH,
            'weight_init': 'W0 = sign(nt)*log1p(syn_count) / per-post |W| sum * 1.2',
            'note': 'Weights are fine-tuned on the synthetic task; topology and polarity from FlyWire, not physiological measurements',
            'v4_stage': 'C (long decision window T_STEPS=64)',
            'train_seed': int(seed),
            'pref_seed': int(seed),
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
        'head_real': {
            'w': [round(float(x), 6) for x in model.head_real.weight.detach().cpu().numpy().ravel()],
            'b': round(float(model.head_real.bias.item()), 6),
        },
        'head_ttc': {
            'w': [round(float(x), 6) for x in model.head_ttc.weight.detach().cpu().numpy().ravel()],
            'b': round(float(model.head_ttc.bias.item()), 6),
        },
        'edges': edges,
    }
    with open(os.path.join(BASE, f'snn_trained_seed{seed}.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f)

    def _nan_to_none(o):
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
                    'p_fake': P_FAKE, 'w_real_bce': W_REAL_BCE, 'w_ttc_mse': W_TTC_MSE,
                    'checkpoint_selection': 'min val_loss', 't_steps': T_STEPS},
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
        plt.xlabel('epoch'); plt.ylabel('loss'); plt.title(f'v4c SNN training curves (seed {seed}, T={T_STEPS})')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(BASE, f'training_curve_seed{seed}.png'), dpi=120)
    except Exception as e:
        print('(Plot skipped:', e, ')')


def main():
    global P_DROP
    ap = argparse.ArgumentParser(description='v4 Phase C: long decision window T_STEPS=64')
    ap.add_argument('--seed', type=int, default=20240521)
    ap.add_argument('--n-train', type=int, default=N_TRAIN_DEFAULT)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--noise-std', type=float, default=NOISE_STD)
    ap.add_argument('--drop-p', type=float, default=P_DROP)
    ap.add_argument('--wd', type=float, default=WD)
    ap.add_argument('--patience', type=int, default=PATIENCE)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--threads', type=int, default=0)
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    P_DROP = args.drop_p
    SEED = args.seed
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device('cpu')
    g, n, src, dst, w0 = load_graph()
    in_v, in_w = g['input_vision_indices'], g['input_wind_indices']
    hub, out = g['hub_gf_indices'], g['output_indices']
    n_vis, n_wind = len(in_v), len(in_w)
    print(f'[v4c] Graph: {n} nodes / {len(src)} edges | vision input {n_vis} | wind input {n_wind} | GF {len(hub)} | output {len(out)}')
    print(f'Surrogate gradient: {USING}')
    print(f'[v4c] T_STEPS={T_STEPS} (long decision window ~64ms integration) | rest same as v4b')

    t_data = time.time()
    pd_pref, wind_pref = make_pref(n_vis, n_wind, SEED)
    train_samples, _ = build_dataset(args.n_train, SEED+1)
    val_samples, _ = build_dataset(N_VAL, VAL_SET_SEED)
    ood_samples = build_ood_dataset(N_OOD, OOD_SET_SEED)
    n_pos_tr = sum(1 for s in train_samples if s['trig'] > 0.5)
    n_pos_va = sum(1 for s in val_samples if s['trig'] > 0.5)
    n_pos_ood = sum(1 for s in ood_samples if s['trig'] > 0.5)
    print(f'Data generation {time.time()-t_data:.1f}s | train {len(train_samples)} (threat {n_pos_tr}) | '
          f'val {len(val_samples)} (threat {n_pos_va}) | OOD {len(ood_samples)} (threat {n_pos_ood})')

    model = EscapeSNN(n, src, dst, w0, in_v, in_w, hub, out).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=args.wd)
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
        start_ep = ck['epoch']+1
        print(f'== Resume from checkpoint: starting from epoch {start_ep} (history {len(hist)} epochs, best val={best_val:.4f}@ep{best_ep}) ==')

    n_fake_per_batch = max(1, int(BATCH * P_FAKE))
    t0 = time.time()
    for ep in range(start_ep, args.epochs+1):
        model.train()
        rng = np.random.default_rng(SEED+100+ep)
        order = rng.permutation(len(train_samples))
        tot = 0.0
        for i in range(0, len(train_samples), BATCH):
            idx = list(order[i: i+BATCH])
            B = len(idx)
            sp, trigs, escs = encode_batch(rng, train_samples, idx, pd_pref, wind_pref,
                                           augment=True, noise_std=args.noise_std)
            n_fake = min(n_fake_per_batch, B)
            for fi in range(n_fake):
                fst = sample_fake_threat(rng)
                f_looming, f_wind, f_trig, f_esc, f_ttc = cues_and_labels(fst)
                fsp = encode_spikes(rng, f_looming, f_wind, fst['az'], pd_pref, wind_pref,
                                    T_STEPS, noise_sigma=float(rng.uniform(0.0, args.noise_std)))
                sp[:, B-1-fi, :] = fsp
                trigs[B-1-fi] = 0.0
                escs[B-1-fi] = 0.0

            x = torch.tensor(sp, device=device)
            y_trig = torch.tensor(trigs, device=device)
            y_esc = torch.tensor(escs, device=device)
            y_real = torch.ones(B, device=device)
            y_real[B-n_fake:] = 0.0
            y_ttc_list = []
            for jk in range(B):
                if jk < B - n_fake:
                    st = train_samples[idx[jk]]
                    y_ttc_list.append(min(st['ttc'], TTC_MAX_MS) / TTC_NORM)
                else:
                    y_ttc_list.append(TTC_MAX_MS / TTC_NORM)
            y_ttc = torch.tensor(y_ttc_list, device=device, dtype=torch.float32)

            trig_logit, dir_pred, rate, v_hub_max, _, _, real_logit, ttc_pred = model(x)
            loss = compute_loss(trig_logit, dir_pred, rate, v_hub_max, y_trig, y_esc,
                                real_logit, ttc_pred, y_real, y_ttc, device)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach())*B
        scheduler.step()
        train_loss = tot / len(train_samples)
        vloss = val_loss_of(model, val_samples, pd_pref, wind_pref, device)
        hist.append({'epoch': ep, 'loss': train_loss, 'val_loss': vloss,
                     'lr': scheduler.get_last_lr()[0], 'time_s': round(time.time()-t0, 1)})
        improved = vloss < best_val - 1e-6
        if improved:
            best_val, best_ep = vloss, ep
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        print(f'epoch {ep:2d}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={vloss:.4f}  '
              f'lr={scheduler.get_last_lr()[0]:.5f}  gap={train_loss-vloss:+.4f}  '
              f'({"best" if improved else f"bad={bad}/{args.patience}"})  ({time.time()-t0:.0f}s)', flush=True)

        torch.save({'epoch': ep, 'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': scheduler.state_dict(), 'hist': hist, 'best_state': best_state,
                    'best_val': best_val, 'best_ep': best_ep, 'bad': bad}, ckpt_path)
        with open(status_path, 'w', encoding='utf-8') as f:
            json.dump({'seed': SEED, 'epoch': ep, 'epochs_max': args.epochs, 'train_loss': train_loss,
                       'val_loss': vloss, 'best_val': best_val, 'best_epoch': best_ep, 'bad': bad,
                       'n_train': len(train_samples), 'elapsed_s': round(time.time()-t0, 1),
                       'state': 'running'}, f, ensure_ascii=False, indent=2)

        if bad >= args.patience and ep >= MIN_EPOCHS:
            print(f'== Early stopping: validation loss did not improve for {args.patience} consecutive epochs (best ep{best_ep} val={best_val:.4f}) ==', flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'\nRestored best weights (val_loss={best_val:.4f} @ ep{best_ep})')

    print('\n===== Benchmark evaluation (same 1200-sample validation set, clean encoding) =====')
    results = {}
    for mode in ('fusion', 'vision_only', 'wind_only'):
        results[mode] = run_eval(model, val_samples, pd_pref, wind_pref, device, mode=mode)

    print(f'===== OOD evaluation ({N_OOD} shifted-distribution samples, sensor noise×2) =====')
    ood_results = {}
    for mode in ('fusion', 'vision_only', 'wind_only'):
        ood_results[mode] = run_eval(model, ood_samples, pd_pref, wind_pref, device, mode=mode,
                                     noise_sigma_max=2.0*args.noise_std)

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
        print('  {:<14} fire_rate={:.2f}  first_spike={:.2f} ms'.format(tag, r['fire_rate'], r['first_spike_ms']))

    export_model(model, results, ood_results, priming, hist, SEED, args)
    with open(status_path, 'w', encoding='utf-8') as f:
        json.dump({'seed': SEED, 'epoch': len(hist), 'epochs_max': args.epochs,
                   'best_val': best_val, 'best_epoch': best_ep, 'state': 'done'}, f,
                  ensure_ascii=False, indent=2)
    print(f'\nExport complete: v4/snn_trained_seed{SEED}.json / v4/metrics_seed{SEED}.json / v4/training_curve_seed{SEED}.png')


if __name__ == '__main__':
    main()
