# -*- coding: utf-8 -*-
"""batch_probe.py —— 定位训练批次耗时异常（与 speed_test 同法计时，输出到 stdout）"""
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(ROOT, 'pylibs'))
sys.path.insert(0, BASE)

import numpy as np
import torch
import train_snn_v4 as T

THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
torch.set_num_threads(THREADS)
print('threads =', torch.get_num_threads())

g, n, src, dst, w0 = T.load_graph()
in_v, in_w = g['input_vision_indices'], g['input_wind_indices']
hub, out = g['hub_gf_indices'], g['output_indices']
model = T.EscapeSNN(n, src, dst, w0, in_v, in_w, hub, out)
opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
pd_pref, wind_pref = T.make_pref(len(in_v), len(in_w), 20240521)
samples, _ = T.build_dataset(512, 1)
rng = np.random.default_rng(0)

model.train()
for it in range(8):
    sp, trigs, escs = T.encode_batch(rng, samples, list(range(64)), pd_pref, wind_pref, True, 0.05)
    x = torch.tensor(sp)
    y_trig = torch.tensor(trigs)
    y_esc = torch.tensor(escs)
    t = time.time()
    trig_logit, dir_pred, rate, v_hub_max, _, _ = model(x)
    loss = T.compute_loss(trig_logit, dir_pred, rate, v_hub_max, y_trig, y_esc, torch.device('cpu'))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    print(f'batch {it}: {time.time() - t:.3f}s', flush=True)
