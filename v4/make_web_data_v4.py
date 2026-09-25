# -*- coding: utf-8 -*-
r"""
make_web_data_v4.py —— v4 版数据打包（复制自 web\make_web_data.py，只改输入/输出路径与 pref 种子）
================================================================================
输入：v4\\snn_trained_seed{seed}.json（train_snn_v4.py 的导出）
输出（默认 seed=20240521，--tag 为空）：
    v4\\snn_trained_v4.json   + v4\\snn_data_v4.js
输出（带 --tag _seed20240522 等）：
    v4\\snn_trained_v4{tag}.json + v4\\snn_data_v4{tag}.js

与 v3 web\\make_web_data.py 的差异：
  1) 绝不触碰 web\\snn_data.js / snn_trained.json（全部产物落 v4\\）；
  2) pd_pref/wind_pref 的重建种子取自模型 meta.pref_seed（v3 固定 SEED=20240521；
     v4 的 make_pref 随训练种子变，必须用同一种子重建，否则风觉编码不一致）。

用法：
  py -3.13 v4\\make_web_data_v4.py --seed 20240521
  py -3.13 v4\\make_web_data_v4.py --seed 20240522 --tag _seed20240522
  py -3.13 v4\\make_web_data_v4.py --seed 20240521 --flip-dir   # 方向头取反（应急，需人工确认）
"""
import os
import sys
import json
import argparse

BASE = os.path.dirname(os.path.abspath(__file__))       # v4\
ROOT = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(ROOT, 'pylibs'))        # 兜底：无需依赖外部 PYTHONPATH

import numpy as np


def make_pref(n_vis, n_wind, seed):
    rng = np.random.default_rng(seed)
    pd_pref = np.zeros((n_vis, 2), dtype=np.float32)
    pd_pref[:, 0] = np.linspace(-np.pi, np.pi, n_vis, endpoint=False)
    pd_pref[:, 1] = rng.uniform(0.5, 1.0, n_vis)  # 保留参数（当前未接入前向），维持与训练端 RNG 顺序一致，勿删
    wind_pref = rng.normal(size=(n_wind, 3)).astype(np.float32)
    wind_pref /= np.linalg.norm(wind_pref, axis=1, keepdims=True)
    return pd_pref, wind_pref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=20240521, help='训练种子（决定读哪个 snn_trained_seedXXX.json）')
    ap.add_argument('--tag', type=str, default='', help="输出文件名后缀，如 _seed20240522；默认 '' -> snn_trained_v4.json / snn_data_v4.js")
    ap.add_argument('--flip-dir', action='store_true')
    args = ap.parse_args()

    model_path = os.path.join(BASE, f'snn_trained_seed{args.seed}.json')
    with open(model_path, encoding='utf-8') as f:
        data = json.load(f)

    n_vis = len(data['input_vision_indices'])
    n_wind = len(data['input_wind_indices'])
    pref_seed = int(data.get('meta', {}).get('pref_seed', 20240521))
    pd_pref, wind_pref = make_pref(n_vis, n_wind, pref_seed)
    data['pd_pref'] = [[round(float(a), 6), round(float(b), 6)] for a, b in pd_pref]
    data['wind_pref'] = [[round(float(x), 6) for x in r] for r in wind_pref]
    data['meta']['dir_flipped'] = bool(data['meta'].get('dir_flipped', False))  # 显式导出方向翻转标记

    if args.flip_dir:
        hd = data['head_dir']
        hd['w'] = [[-x for x in row] for row in hd['w']]
        hd['b'] = [-x for x in hd['b']]
        data['meta']['dir_flipped'] = True
        print('已对方向读出头取反（--flip-dir）')

    out_json = os.path.join(BASE, f'snn_trained_v4{args.tag}.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    out = os.path.join(BASE, f'snn_data_v4{args.tag}.js')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('/* 由 make_web_data_v4.py 生成：训练好的 SNN 权重 + 传感器编码参数（v4） */\n')
        f.write('window.SNN_DATA = ')
        json.dump(data, f, separators=(',', ':'))
        f.write(';\n')
    print(f'输入 {model_path}（train_seed={data["meta"].get("train_seed")} pref_seed={pref_seed}）')
    print(f'已生成 {out}  ({os.path.getsize(out) / 1e6:.2f} MB)')
    print(f'已生成 {out_json}')
    print(f'节点 {data["num_nodes"]} | 边 {len(data["edges"])} | 视觉入 {n_vis} | 风觉入 {n_wind}')


if __name__ == '__main__':
    main()
