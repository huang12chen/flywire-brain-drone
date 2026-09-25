# -*- coding: utf-8 -*-
"""
report_tables.py —— v4 报告表格汇总器（只读输入，产物只落 v4\）
================================================================================
输入（全部只读）：
  v4\\metrics_seed{20240521,20240522,20240523}.json    （train_snn_v4.py 导出：验证 1200 + OOD 2000）
  v4\\results_v4.json, v4\\results_v4_seed20240522.json, v4\\results_v4_seed20240523.json（evaluate_v4.js）
  web\\results.json（v3 基线，只读）
  v4\\speed_test.json
输出：
  v4\\report_tables.json（全部数字，含每种子明细，便于复跑核实） + 控制台 Markdown 表格

统计口径：跨 3 个训练种子的 mean ± std（样本标准差 ddof=1）；
分场景表另给「主模型(20240521) 5 评估种子 mean±std」列，与 v3 web\\results.json 的 ±std 同口径可比。
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
    out = {'seeds': SEEDS, 'std_definition': '样本标准差 ddof=1，对 3 个训练种子'}

    # ---------- 0. 完整性检查 ----------
    web_results = os.path.join(ROOT, 'web', 'results.json')
    sha = hashlib.sha256(open(web_results, 'rb').read()).hexdigest().upper()
    out['integrity'] = {
        'web_results_sha256': sha,
        'expected': EXPECTED_WEB_RESULTS_SHA256,
        'match': sha == EXPECTED_WEB_RESULTS_SHA256,
    }

    # ---------- 1. 主表 / OOD 表（训练端 1200 验证 / 2000 OOD） ----------
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

    # ---------- 2. 过拟合体检（train/val 损失曲线） ----------
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

    # ---------- 3. 分场景（evaluate.js 矩阵；主模型 5 评估种子 vs v3 同口径 + 3 训练种子跨模型） ----------
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
    for cid, label in (('B1', '晴(meadow)'), ('B2', '暴风(storm)'), ('B3', '黑夜(night)'),
                       ('C1', '低速 slow'), ('C2', '中速 mid'), ('C3', '高速 high'),
                       ('A1', '融合 fusion'), ('A2', '纯视觉'), ('A3', '纯风觉')):
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

    # ---------- 4. D 组（物理积分口径，分列不混用） ----------
    dgrp = {}
    for cid, label in (('D1', '果蝇身体'), ('D2', '无人机身体')):
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

    # ---------- 5. 确定性检查 ----------
    out['checks'] = {str(s): res_v4[s]['checks'] for s in SEEDS}

    # ---------- 6. 测速 ----------
    try:
        out['speed_test'] = load(os.path.join(BASE, 'speed_test.json'))
    except Exception:
        out['speed_test'] = None

    # ---------- 7. priming ----------
    out['priming'] = {str(s): metrics[s]['priming'] for s in SEEDS}

    with open(os.path.join(BASE, 'report_tables.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ---------- 控制台 Markdown ----------
    def fmt_pair(d, digits):
        return f"{d['mean']:.{digits}f}±{d['std']:.{digits}f}" if d and d.get('mean') is not None else 'n/a'

    print('### T1 验证集 1200 条（3 训练种子 mean±std, ddof=1）')
    print('| 组别 | 触发准确率(GF) | 触发准确率(头) | 方向成功率 | 方向误差(°) | GF潜伏期(ms) |')
    print('|---|---|---|---|---|---|')
    for mode, zh in (('fusion', '融合'), ('vision_only', '纯视觉'), ('wind_only', '纯风觉')):
        r = rows['results'][mode]
        print(f"| {zh} | {fmt_pair(r['trigger_acc_gf_spike'],3)} | {fmt_pair(r['trigger_acc_head'],3)} | "
              f"{fmt_pair(r['escape_success_rate'],3)} | {fmt_pair(r['dir_mae_deg'],1)} | {fmt_pair(r['gf_first_spike_ms'],2)} |")

    print('\n### T2 OOD 集 2000 条（3 训练种子 mean±std, ddof=1）')
    print('| 组别 | 触发准确率(GF) | 方向成功率 | 方向误差(°) | GF潜伏期(ms) | 召回 | 误报FAR |')
    print('|---|---|---|---|---|---|---|')
    for mode, zh in (('fusion', '融合'), ('vision_only', '纯视觉'), ('wind_only', '纯风觉')):
        r = rows['ood_2000'][mode]
        print(f"| {zh} | {fmt_pair(r['trigger_acc_gf_spike'],3)} | {fmt_pair(r['escape_success_rate'],3)} | "
              f"{fmt_pair(r['dir_mae_deg'],1)} | {fmt_pair(r['gf_first_spike_ms'],2)} | "
              f"{fmt_pair(r['threat_recall'],3)} | {fmt_pair(r['false_alarm_rate'],3)} |")

    print('\n### T3 分场景（召回 / 误报 FAR；主模型 20240521 的 5 评估种子 mean±std，与 v3 同口径）')
    print('| 场景 | v3 召回 | v4 召回 | v3 FAR | v4 FAR | v3 方向° | v4 方向° |')
    print('|---|---|---|---|---|---|---|')
    for cid in ('B1', 'B2', 'B3', 'C1', 'C2', 'C3'):
        e = scen[cid]
        print(f"| {e['label']} | {fmt_pair(e['v3']['threat_recall'],3)} | {fmt_pair(e['v4_main_seed20240521']['threat_recall'],3)} | "
              f"{fmt_pair(e['v3']['false_alarm_rate'],3)} | {fmt_pair(e['v4_main_seed20240521']['false_alarm_rate'],3)} | "
              f"{fmt_pair(e['v3']['dir_mae_deg'],1)} | {fmt_pair(e['v4_main_seed20240521']['dir_mae_deg'],1)} |")

    print('\n### T3b 分场景跨 3 训练种子（各模型格内 5 评估种子均值，再对 3 训练种子取 mean±std ddof=1）')
    print('| 场景 | 召回 | 误报 FAR | 触发准确率 | 方向成功率 | 方向误差(°) |')
    print('|---|---|---|---|---|---|')
    for cid in ('B1', 'B2', 'B3', 'C3'):
        e = scen[cid]['v4_3trainseeds']
        print(f"| {scen[cid]['label']} | {fmt_pair(e['threat_recall'],3)} | {fmt_pair(e['false_alarm_rate'],3)} | "
              f"{fmt_pair(e['trigger_acc'],3)} | {fmt_pair(e['escape_success_rate'],3)} | {fmt_pair(e['dir_mae_deg'],1)} |")

    print('\n### T4 过拟合体检')
    print('| 种子 | 训练轮数 | 最优轮 | best val | train@best | gap@best | 末轮 train | 末轮 val | 末轮 gap |')
    print('|---|---|---|---|---|---|---|---|---|')
    for s in SEEDS:
        f = fit[str(s)]
        print(f"| {s} | {f['epochs_run']} | {f['best_epoch']} | {f['best_val_loss']:.4f} | {f['train_loss_at_best']:.4f} | "
              f"{f['gap_at_best']:+.4f} | {f['final_train_loss']:.4f} | {f['final_val_loss']:.4f} | {f['final_gap']:+.4f} |")

    print(f"\n[完整性] web\\results.json SHA256 = {sha}  与期望一致 = {out['integrity']['match']}")
    print('[产物] v4/report_tables.json')


if __name__ == '__main__':
    main()
