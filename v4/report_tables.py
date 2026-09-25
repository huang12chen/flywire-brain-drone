# -*- coding: utf-8 -*-
"""
report_tables.py —— v4 report table aggregator (read-only inputs, outputs go to v4\ only)
================================================================================
Inputs (all read-only):
  v4\\metrics_seed{20240521,20240522,20240523}.json    (train_snn_v4.py export: val 1200 + OOD 2000)
  v4\\results_v4.json, v4\\results_v4_seed20240522.json, v4\\results_v4_seed20240523.json (evaluate_v4.js)
  web\\results.json (v3 baseline, read-only)
  v4\\speed_test.json
Outputs:
  v4\\report_tables.json (all numbers, including per-seed details for reproducibility) + console Markdown tables

Statistics: mean ± std across 3 training seeds (sample std ddof=1);
per-scenario table additionally provides "main model (20240521) 5 evaluation seeds mean±std" column, comparable to v3 web\\results.json ±std.
"""
import os
import sys
import json
import math
import hashlib

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(ROOT, 'pylibs'))

SEEDS = [20240521, 20240522, 20240523]
EXPECTED_WEB_RESULTS_SHA256 = '0E1339A088EF3B5820DCA342B989249403BE41D5A3F7EDB9DDB1D1A91C217E9E'


def mean_std(xs):
    xs = [x for x in xs if x is not None and isinstance(x, (int, float)) and not math.isnan(x)]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0, len(xs)
    ss = sum((x - m) ** 2 for x in xs)
    return m, math.sqrt(ss / (len(xs) - 1)), len(xs)


def pair(xs, digits):
    m, s, n = mean_std(xs)
    if m is None:
        return 'n/a'
    return f'{m:.{digits}f}±{s:.{digits}f}'


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def main():
    out = {'seeds': SEEDS, 'std_definition': 'Sample std ddof=1, over 3 training seeds'}

    # ---------- 0. Integrity check ----------
    web_results = os.path.join(ROOT, 'web', 'results.json')
    sha = hashlib.sha256(open(web_results, 'rb').read()).hexdigest().upper()
    out['integrity'] = {
        'web_results_sha256': sha,
        'expected': EXPECTED_WEB_RESULTS_SHA256,
        'match': sha == EXPECTED_WEB_RESULTS_SHA256,
    }

    # ---------- 1. Main table / OOD table (training-side 1200 val / 2000 OOD) ----------
    metrics = {s: load(os.path.join(BASE, f'metrics_seed{s}.json')) for s in SEEDS}
    rows = {}
    for split in ('results', 'ood_results'):
        rows[split] = {}
        for mode in ('fusion', 'vision_only', 'wind_only'):
            rows[split][mode] = {}
            for key, dig in (('trigger_acc_gf_spike', 3), ('trigger_acc_head', 3),
                             ('escape_success_rate', 3), ('dir_mae_deg', 1), ('gf_first_spike_ms', 2),
                             ('threat_recall', 3), ('false_alarm_rate', 3)):
                vals = [metrics[s][split][mode][key] for s in SEEDS]
                m, sd, n = mean_std(vals)
                rows[split][mode][key] = {'mean': m, 'std': sd, 'n': n, 'per_seed': dict(zip(map(str, SEEDS), vals))}
            rows[split][mode]['n_threat'] = metrics[SEEDS[0]][split][mode]['n_threat']
    out['val_1200'] = rows['results']
    out['ood_2000'] = rows['ood_results']

    # ---------- 2. Overfitting health check (train/val loss curves) ----------
    fit = {}
    for s in SEEDS:
        hist = metrics[s]['train_loss']
        best = min(hist, key=lambda h: h['val_loss'])
        last = hist[-1]
        gaps = [h['loss'] - h['val_loss'] for h in hist]
        fit[str(s)] = {
            'epochs_run': len(hist),
            'best_epoch': best['epoch'],
            'best_val_loss': best['val_loss'],
            'train_loss_at_best': best['loss'],
            'gap_at_best': best['loss'] - best['val_loss'],
            'final_train_loss': last['loss'],
            'final_val_loss': last['val_loss'],
            'final_gap': last['loss'] - last['val_loss'],
            'gap_last5_mean': sum(gaps[-5:]) / min(5, len(gaps)),
            'train_loss_ep1': hist[0]['loss'],
            'val_loss_ep1': hist[0]['val_loss'],
        }
    out['overfit_check'] = fit

    # ---------- 3. Per-scenario (evaluate.js matrix; main model 5 eval seeds vs v3 same protocol + 3 training seeds cross-model) ----------
    res_v4 = {s: load(os.path.join(BASE, 'results_v4.json' if s == 20240521 else f'results_v4_seed{s}.json'))
              for s in SEEDS}
    res_v3 = load(web_results)

    def cell(res, cid):
        for g in ('A', 'B', 'C', 'D'):
            for c in res['groups'][g]:
                if c['id'] == cid:
                    return c
        return None

    scen = {}
    for cid, label in (('B1', '(meadow)'), ('B2', '(storm)'), ('B3', '(night)'),
                       ('C1', ' slow'), ('C2', ' mid'), ('C3', ' high'),
                       ('A1', ' fusion'), ('A2', ''), ('A3', '')):
        entry = {'label': label, 'v3': {}, 'v4_main_seed20240521': {}, 'v4_3trainseeds': {}}
        c3 = cell(res_v3, cid)
        c4m = cell(res_v4[20240521], cid)
        for key in ('trigger_acc', 'threat_recall', 'false_alarm_rate', 'escape_success_rate',
                    'dir_mae_deg', 'gf_first_spike_ms'):
            entry['v3'][key] = c3['metrics'][key]
            entry['v4_main_seed20240521'][key] = c4m['metrics'][key]
            per_model = [cell(res_v4[s], cid)['metrics'][key]['mean'] for s in SEEDS]
            m, sd, n = mean_std(per_model)
            entry['v4_3trainseeds'][key] = {'mean': m, 'std': sd, 'n': n,
                                            'per_seed': dict(zip(map(str, SEEDS), per_model))}
        entry['n_threat'] = {'v3': c3['n_threat_total'], 'v4': c4m['n_threat_total']}
        scen[cid] = entry
    out['scenarios'] = scen

    # ---------- 4. Group D (physical integration metrics, separate columns) ----------
    dgrp = {}
    for cid, label in (('D1', ''), ('D2', '')):
        e = {'label': label, 'v3': {}, 'v4_main_seed20240521': {}, 'v4_3trainseeds': {}}
        c3 = cell(res_v3, cid)
        c4m = cell(res_v4[20240521], cid)
        for key in ('physical_success_rate', 'stationary_safe_rate', 'action_saved_rate', 'escape_success_rate'):
            e['v3'][key] = c3['metrics'][key]
            e['v4_main_seed20240521'][key] = c4m['metrics'][key]
            per_model = [cell(res_v4[s], cid)['metrics'][key]['mean'] for s in SEEDS]
            m, sd, n = mean_std(per_model)
            e['v4_3trainseeds'][key] = {'mean': m, 'std': sd, 'n': n,
                                        'per_seed': dict(zip(map(str, SEEDS), per_model))}
        dgrp[cid] = e
    out['group_D'] = dgrp

    # ---------- 5. Determinism checks ----------
    out['checks'] = {str(s): res_v4[s]['checks'] for s in SEEDS}

    # ---------- 6. Speed test ----------
    try:
        out['speed_test'] = load(os.path.join(BASE, 'speed_test.json'))
    except Exception:
        out['speed_test'] = None

    # ---------- 7. Priming ----------
    out['priming'] = {str(s): metrics[s]['priming'] for s in SEEDS}

    with open(os.path.join(BASE, 'report_tables.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ---------- Console Markdown ----------
    def fmt_pair(d, digits):
        return f"{d['mean']:.{digits}f}±{d['std']:.{digits}f}" if d and d.get('mean') is not None else 'n/a'

    print('### T1 Validation Set 1200 samples (3 training seeds mean±std, ddof=1)')
    print('| Group | Trigger Acc(GF) | Trigger Acc(Head) | Escape Success | Direction Error(°) | GF Latency(ms) |')
    print('|---|---|---|---|---|---|')
    for mode, zh in (('fusion', ''), ('vision_only', ''), ('wind_only', '')):
        r = rows['results'][mode]
        print(f"| {zh} | {fmt_pair(r['trigger_acc_gf_spike'],3)} | {fmt_pair(r['trigger_acc_head'],3)} | "
              f"{fmt_pair(r['escape_success_rate'],3)} | {fmt_pair(r['dir_mae_deg'],1)} | {fmt_pair(r['gf_first_spike_ms'],2)} |")

    print('\n### T2 OOD Set 2000 samples (3 training seeds mean±std, ddof=1)')
    print('| Group | Trigger Acc(GF) | Escape Success | Direction Error(°) | GF Latency(ms) | Recall | False Alarm FAR |')
    print('|---|---|---|---|---|---|---|')
    for mode, zh in (('fusion', ''), ('vision_only', ''), ('wind_only', '')):
        r = rows['ood_2000'][mode]
        print(f"| {zh} | {fmt_pair(r['trigger_acc_gf_spike'],3)} | {fmt_pair(r['escape_success_rate'],3)} | "
              f"{fmt_pair(r['dir_mae_deg'],1)} | {fmt_pair(r['gf_first_spike_ms'],2)} | "
              f"{fmt_pair(r['threat_recall'],3)} | {fmt_pair(r['false_alarm_rate'],3)} |")

    print('\n### T3 Per-scenario (recall / false alarm FAR; main model 20240521 5 eval seeds mean±std, same protocol as v3)')
    print('| Scenario | v3 Recall | v4 Recall | v3 FAR | v4 FAR | v3 Dir° | v4 Dir° |')
    print('|---|---|---|---|---|---|---|')
    for cid in ('B1', 'B2', 'B3', 'C1', 'C2', 'C3'):
        e = scen[cid]
        print(f"| {e['label']} | {fmt_pair(e['v3']['threat_recall'],3)} | {fmt_pair(e['v4_main_seed20240521']['threat_recall'],3)} | "
              f"{fmt_pair(e['v3']['false_alarm_rate'],3)} | {fmt_pair(e['v4_main_seed20240521']['false_alarm_rate'],3)} | "
              f"{fmt_pair(e['v3']['dir_mae_deg'],1)} | {fmt_pair(e['v4_main_seed20240521']['dir_mae_deg'],1)} |")

    print('\n### T3b Per-scenario across 3 training seeds (each model cell is 5 eval seed mean, then mean±std ddof=1 across 3 training seeds)')
    print('| Scenario | Recall | False Alarm FAR | Trigger Acc | Escape Success | Direction Error(°) |')
    print('|---|---|---|---|---|---|')
    for cid in ('B1', 'B2', 'B3', 'C3'):
        e = scen[cid]['v4_3trainseeds']
        print(f"| {scen[cid]['label']} | {fmt_pair(e['threat_recall'],3)} | {fmt_pair(e['false_alarm_rate'],3)} | "
              f"{fmt_pair(e['trigger_acc'],3)} | {fmt_pair(e['escape_success_rate'],3)} | {fmt_pair(e['dir_mae_deg'],1)} |")

    print('\n### T4 Overfitting Health Check')
    print('| Seed | Epochs Run | Best Epoch | Best Val | Train@Best | Gap@Best | Final Train | Final Val | Final Gap |')
    print('|---|---|---|---|---|---|---|---|---|')
    for s in SEEDS:
        f = fit[str(s)]
        print(f"| {s} | {f['epochs_run']} | {f['best_epoch']} | {f['best_val_loss']:.4f} | {f['train_loss_at_best']:.4f} | "
              f"{f['gap_at_best']:+.4f} | {f['final_train_loss']:.4f} | {f['final_val_loss']:.4f} | {f['final_gap']:+.4f} |")

    print(f"\n[Integrity] web\\results.json SHA256 = {sha}  match expected = {out['integrity']['match']}")
    print('[Output] v4/report_tables.json')


if __name__ == '__main__':
    main()
