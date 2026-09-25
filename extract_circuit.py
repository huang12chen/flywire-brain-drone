# -*- coding: utf-8 -*-
"""
extract_circuit.py — 从 FlyWire 本地下载文件中提取「视觉膨胀 + 触角风觉 -> 巨纤维(GF) -> 下行/运动输出」子网络
==================================================================================================
输入（只读，位于 DATA_DIR）：
  connections_princeton.csv.gz      列: pre_root_id, post_root_id, neuropil, syn_count, nt_type
                                    —— 注意：同一对 (pre,post) 在不同 neuropil 会有多行，需要合并求和
  consolidated_cell_types.csv.gz    列: root_id, primary_type, additional_type(s)
                                    —— 注意：巨纤维 GF 不是 primary_type='GF'，而是 primary_type='DNp01'
                                       且 additional_type(s) 含 'Giant_Fiber, GF'
  classification.csv.gz             列: root_id, flow, super_class, class, sub_class, ...
                                    —— super_class: sensory/visual_projection/descending/motor ...
  neurons.csv.gz                    列: root_id, group, nt_type, ... （神经元级递质预测，作兜底）

输出（写到脚本所在目录）：
  escape_network_sparse.json        稀疏图（节点表 + 边表 + 权重），供 SNN 训练脚本使用
  nodes.csv / edges.csv             便于人工核对的表格

子图构造原则（与对话中 SOP 的差别，均以真实数据字段为准）：
  1. 视觉输入 = primary_type == 'LPLC2'（视觉膨胀/宽场运动敏感视觉投射神经元）
  2. 风觉输入 = primary_type in {'JO-B', 'JO-C'}（Johnston's 机械感受器，风/声/重力）
  3. 逃逸中枢 = additional_type(s) 含 Giant_Fiber/GF（即 DNp01，左右各一）
  4. 核心中间层 = 位于「输入 -> GF」短路径（≤3 条突触边，即最多 2 个中间神经元）上的神经元
     —— GF 逃逸反应潜伏期仅数毫秒，生物学上就是少突触通路；同时避免 2 跳∩2 跳并集爆炸到上万节点
  5. 输出层   = GF 正向 2 跳内、super_class ∈ {descending, motor} 的神经元（下行/脑内运动神经元）
  6. 边 = 上述节点集合内部的全部连接；权重 W = sign(递质) * log1p(syn_count)
     递质符号（果蝇）：ACH 兴奋(+1)；GABA 抑制(-1)；GLUT 在昆虫为抑制(-1)；DA/SER/OCT 为调质(+1 并标记)
"""
import os
import json
import gzip
import csv
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r'<FlyWire数据目录>'

PATH_CONN = os.path.join(DATA_DIR, 'connections_princeton.csv.gz')
PATH_TYPES = os.path.join(DATA_DIR, 'consolidated_cell_types.csv.gz')
PATH_CLASS = os.path.join(DATA_DIR, 'classification.csv.gz')
PATH_NEURONS = os.path.join(DATA_DIR, 'neurons.csv.gz')

VISION_TYPES = {'LPLC2'}
WIND_TYPES = {'JO-B', 'JO-C'}
HUB_TOKENS = ('giant_fiber', 'gf')     # 在 additional_type(s) 中匹配
HUB_PRIMARY = 'DNp01'                  # FlyWire 中巨纤维的 primary_type
OUTPUT_SUPER = {'descending', 'motor'}
NT_SIGN = {'ACH': 1.0, 'GABA': -1.0, 'GLUT': -1.0, 'DA': 1.0, 'SER': 1.0, 'OCT': 1.0}
MODULATORY_NT = {'DA', 'SER', 'OCT'}

import pandas as pd
import numpy as np


def log(msg):
    print(msg, flush=True)


def load_tables():
    log('[1/6] 读取本地 CSV（只读）...')
    cells = pd.read_csv(PATH_TYPES)
    cls = pd.read_csv(PATH_CLASS, usecols=['root_id', 'super_class', 'class', 'sub_class', 'flow'])
    ntn = pd.read_csv(PATH_NEURONS, usecols=['root_id', 'nt_type', 'nt_type_score'])

    cells['additional_type(s)'] = cells['additional_type(s)'].fillna('')
    cls = cls.drop_duplicates('root_id')

    conn = pd.read_csv(
        PATH_CONN,
        usecols=['pre_root_id', 'post_root_id', 'neuropil', 'syn_count', 'nt_type'],
        dtype={'pre_root_id': 'int64', 'post_root_id': 'int64', 'syn_count': 'int32',
               'neuropil': 'category', 'nt_type': 'category'},
    )
    log(f'      细胞 {len(cells)} 个 | 连接原始行 {len(conn)} 条（按 neuropil 拆分，需合并）')
    return cells, cls, ntn, conn


def find_nodes(cells):
    log('[2/6] 定位输入层 / 逃逸中枢节点...')
    vision_ids = set(cells.loc[cells['primary_type'].isin(VISION_TYPES), 'root_id'])
    wind_ids = set(cells.loc[cells['primary_type'].isin(WIND_TYPES), 'root_id'])

    add = cells['additional_type(s)'].str.lower()
    hub_mask = add.str.contains('giant_fiber', regex=False) | (cells['primary_type'] == HUB_PRIMARY)
    hub_ids = set(cells.loc[hub_mask, 'root_id'])

    log(f'      视觉输入 LPLC2 : {len(vision_ids)} 个')
    log(f'      风觉输入 JO-B/C: {len(wind_ids)} 个')
    log(f'      巨纤维 GF(DNp01): {len(hub_ids)} 个  root_id={sorted(hub_ids)}')
    assert hub_ids, '未找到巨纤维 GF，请检查数据文件'
    return vision_ids, wind_ids, hub_ids


def aggregate_edges(conn):
    log('[3/6] 合并同一对神经元在不同 neuropil 的连接行...')
    g = conn.groupby(['pre_root_id', 'post_root_id'], sort=False)
    edges = g['syn_count'].sum().reset_index(name='syn_count')
    # 取该对连接中突触数最多的那一行的递质类型作为主递质
    idx = conn.groupby(['pre_root_id', 'post_root_id'], sort=False)['syn_count'].idxmax()
    nt = conn.loc[idx, ['pre_root_id', 'post_root_id', 'nt_type']].rename(columns={'nt_type': 'nt_major'})
    edges = edges.merge(nt, on=['pre_root_id', 'post_root_id'], how='left')
    log(f'      合并后唯一有向边 {len(edges)} 条')
    return edges


def hop(edges, seeds, direction='forward', exclude=None):
    """在聚合后的边表上做 1 跳扩展，返回新增节点集合。"""
    if not seeds:
        return set()
    if direction == 'forward':
        mask = edges['pre_root_id'].isin(seeds)
        got = set(edges.loc[mask, 'post_root_id'])
    else:
        mask = edges['post_root_id'].isin(seeds)
        got = set(edges.loc[mask, 'pre_root_id'])
    if exclude:
        got -= exclude
    return got


def build_subgraph(edges, cls_map, vision_ids, wind_ids, hub_ids):
    log('[4/6] 以 GF 为中心做双向 2 跳抽稀（保留在 输入->GF 通路上的神经元）...')
    inputs = vision_ids | wind_ids

    fwd1 = hop(edges, inputs, 'forward') - inputs
    fwd2 = hop(edges, fwd1, 'forward') - inputs - fwd1
    rev1 = hop(edges, hub_ids, 'backward') - hub_ids
    rev2 = hop(edges, rev1, 'backward') - hub_ids - rev1

    # 输入到 GF 的核心通路：只保留位于「输入 -> GF」≤3 边短路径上的神经元
    #   1 个中间神经元: input -> x -> GF                (x ∈ fwd1 ∩ rev1)
    #   2 个中间神经元: input -> x -> y -> GF           (x ∈ fwd1 ∩ rev2, y ∈ fwd2 ∩ rev1)
    core = ((fwd1 & (rev1 | rev2)) | (fwd2 & rev1)) | hub_ids

    # GF 下游输出层：正向 2 跳内的 下行/运动 神经元
    down1 = hop(edges, hub_ids, 'forward') - hub_ids
    down2 = hop(edges, down1, 'forward') - hub_ids - down1
    out_ids = {n for n in (down1 | down2) if cls_map.get(n, '') in OUTPUT_SUPER}

    node_set = (inputs | core | out_ids | hub_ids)
    log(f'      1 跳/2 跳候选: fwd {len(fwd1)}/{len(fwd2)}  rev {len(rev1)}/{len(rev2)}')
    log(f'      核心通路神经元 {len(core)} 个 | GF 下行输出 {len(out_ids)} 个 | 节点合计 {len(node_set)} 个')
    return node_set, core, out_ids


def main():
    cells, cls, ntn, conn = load_tables()

    cls_map = dict(zip(cls['root_id'], cls['super_class']))
    type_map = dict(zip(cells['root_id'], cells['primary_type']))
    add_map = dict(zip(cells['root_id'], cells['additional_type(s)']))
    neuron_nt = dict(zip(ntn['root_id'], ntn['nt_type']))

    vision_ids, wind_ids, hub_ids = find_nodes(cells)
    edges = aggregate_edges(conn)
    del conn

    node_set, core_ids, out_ids = build_subgraph(edges, cls_map, vision_ids, wind_ids, hub_ids)

    log('[5/6] 导出子图边与权重 W = sign(NT) * log1p(syn_count)...')
    sub_edges = edges[edges['pre_root_id'].isin(node_set) & edges['post_root_id'].isin(node_set)].copy()
    sub_edges = sub_edges[sub_edges['syn_count'] >= 3].copy()  # 剔除 <3 的弱连接（与 FlyWire 筛选口径一致）

    # 递质：优先用连接行的主递质，缺失则用神经元级预测，再缺失记为 ACH
    nt_major = sub_edges['nt_major'].astype(str)
    fallback = sub_edges['pre_root_id'].map(neuron_nt)
    nt_final = nt_major.where(nt_major.isin(NT_SIGN.keys()), fallback.astype(str))
    nt_final = nt_final.where(nt_final.isin(NT_SIGN.keys()), 'ACH')
    sub_edges['nt'] = nt_final
    sub_edges['sign'] = nt_final.map(NT_SIGN)
    sub_edges['modulatory'] = nt_final.isin(MODULATORY_NT)
    sub_edges['weight'] = sub_edges['sign'] * np.log1p(sub_edges['syn_count'].astype(float))

    nodes = sorted(node_set)
    node2idx = {n: i for i, n in enumerate(nodes)}

    def role_of(n):
        if n in hub_ids:
            return 'hub_gf'
        if n in vision_ids:
            return 'input_vision'
        if n in wind_ids:
            return 'input_wind'
        if n in out_ids:
            return 'output'
        return 'interneuron'

    node_records = [{
        'idx': node2idx[n],
        'root_id': int(n),
        'primary_type': type_map.get(n, ''),
        'additional_type': add_map.get(n, ''),
        'super_class': cls_map.get(n, ''),
        'role': role_of(n),
    } for n in nodes]

    edge_records = [{
        'src': node2idx[int(r.pre_root_id)],
        'dst': node2idx[int(r.post_root_id)],
        'syn_count': int(r.syn_count),
        'nt': str(r.nt),
        'sign': float(r.sign),
        'weight': float(r.weight),
    } for r in sub_edges.itertuples()]

    log('[6/6] 写出 escape_network_sparse.json / nodes.csv / edges.csv ...')
    graph = {
        'meta': {
            'source': 'FlyWire FAFB (connections_princeton / consolidated_cell_types / classification / neurons)',
            'vision_types': sorted(VISION_TYPES),
            'wind_types': sorted(WIND_TYPES),
            'hub_rule': "additional_type(s) contains 'Giant_Fiber' or primary_type=='DNp01'",
            'weight_rule': 'W = sign(nt) * log1p(syn_count); sign: ACH+1 GABA-1 GLUT-1 (DA/SER/OCT +1, modulatory)',
            'num_nodes': len(node_records),
            'num_edges': len(edge_records),
        },
        'input_vision_indices': [node2idx[n] for n in sorted(vision_ids) if n in node2idx],
        'input_wind_indices': [node2idx[n] for n in sorted(wind_ids) if n in node2idx],
        'hub_gf_indices': [node2idx[n] for n in sorted(hub_ids) if n in node2idx],
        'output_indices': [node2idx[n] for n in sorted(out_ids) if n in node2idx],
        'nodes': node_records,
        'edges': edge_records,
    }
    with open(os.path.join(BASE, 'escape_network_sparse.json'), 'w', encoding='utf-8') as f:
        json.dump(graph, f)

    pd.DataFrame(node_records).to_csv(os.path.join(BASE, 'nodes.csv'), index=False)
    pd.DataFrame(edge_records).to_csv(os.path.join(BASE, 'edges.csv'), index=False)

    role_cnt = Counter(n['role'] for n in node_records)
    log('')
    log('==== 提取完成 ====')
    log(f"节点 {len(node_records)} 个 {dict(role_cnt)}")
    log(f"边   {len(edge_records)} 条 (syn_count>=3)")
    log(f"输出 escape_network_sparse.json -> {BASE}")


if __name__ == '__main__':
    main()
