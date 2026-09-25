# -*- coding: utf-8 -*-
"""
make_web_data.py — 生成 web/snn_data.js（把训练好的模型 + 传感器编码参数打包成网页可直接加载的 JS）
同时把 pd_pref / wind_pref 回写进 snn_trained.json，保证数据自洽。
（pd_pref/wind_pref 由 make_pref(210, 325, SEED=20240521) 重现，与训练端完全一致）

用法：
  python make_web_data.py              # 正常打包
  python make_web_data.py --flip-dir   # 把方向读出头取反（用于方向标签写反的旧模型应急修正）
"""
import os
import sys
import json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20240521


def make_pref(n_vis, n_wind, seed):
    rng = np.random.default_rng(seed)
    pd_pref = np.zeros((n_vis, 2), dtype=np.float32)
    pd_pref[:, 0] = np.linspace(-np.pi, np.pi, n_vis, endpoint=False)
    pd_pref[:, 1] = rng.uniform(0.5, 1.0, n_vis)  # 保留参数（当前未接入前向），维持与训练端 RNG 顺序一致，勿删
    wind_pref = rng.normal(size=(n_wind, 3)).astype(np.float32)
    wind_pref /= np.linalg.norm(wind_pref, axis=1, keepdims=True)
    return pd_pref, wind_pref


def main():
    model_path = os.path.join(BASE, 'snn_trained.json')
    with open(model_path, encoding='utf-8') as f:
        data = json.load(f)

    n_vis = len(data['input_vision_indices'])
    n_wind = len(data['input_wind_indices'])
    pd_pref, wind_pref = make_pref(n_vis, n_wind, SEED)
    data['pd_pref'] = [[round(float(a), 6), round(float(b), 6)] for a, b in pd_pref]
    data['wind_pref'] = [[round(float(x), 6) for x in r] for r in wind_pref]
    data['meta']['dir_flipped'] = bool(data['meta'].get('dir_flipped', False))  # 显式导出方向翻转标记

    if '--flip-dir' in sys.argv:
        hd = data['head_dir']
        hd['w'] = [[-x for x in row] for row in hd['w']]
        hd['b'] = [-x for x in hd['b']]
        data['meta']['dir_flipped'] = True
        print('已对方向读出头取反（--flip-dir）')

    with open(model_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    out = os.path.join(BASE, 'web', 'snn_data.js')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('/* 由 make_web_data.py 生成：训练好的 SNN 权重 + 传感器编码参数 */\n')
        f.write('window.SNN_DATA = ')
        json.dump(data, f, separators=(',', ':'))
        f.write(';\n')
    print(f'已生成 {out}  ({os.path.getsize(out) / 1e6:.2f} MB)')
    print(f'节点 {data["num_nodes"]} | 边 {len(data["edges"])} | 视觉入 {n_vis} | 风觉入 {n_wind}')


if __name__ == '__main__':
    main()
