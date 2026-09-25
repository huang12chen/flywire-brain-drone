# -*- coding: utf-8 -*-
"""
extract_circuit.py — Extract the "visual looming + antenna wind sense -> giant fiber (GF) -> descending/motor output" subnetwork from FlyWire local download files
==================================================================================================
Input (read-only, located in DATA_DIR):
  connections_princeton.csv.gz      Columns: pre_root_id, post_root_id, neuropil, syn_count, nt_type
                                     Note: the same pair (pre,post) may have multiple rows in different neuropils; need to merge and sum
  consolidated_cell_types.csv.gz    Columns: root_id, primary_type, additional_type(s)
                                     Note: giant fiber GF is not primary_type='GF', but primary_type='DNp01'
                                        and additional_type(s) contains 'Giant_Fiber, GF'
  classification.csv.gz             Columns: root_id, flow, super_class, class, sub_class, ...
                                     super_class: sensory/visual_projection/descending/motor ...
  neurons.csv.gz                    Columns: root_id, group, nt_type, ... (neuron-level neurotransmitter prediction, used as fallback)

Output (written to the script's directory):
  escape_network_sparse.json        Sparse graph (node table + edge table + weights), for use by SNN training scripts
  nodes.csv / edges.csv             Human-readable tables for manual verification

Subgraph construction principles (differences from the SOP discussed earlier; all based on real data fields):
  1. Visual input = primary_type == 'LPLC2' (visual looming/wide-field motion-sensitive visual projection neurons)
  2. Wind sense input = primary_type in {'JO-B', 'JO-C'} (Johnston's organ mechanoreceptors, wind/sound/gravity)
  3. Escape hub = additional_type(s) contains Giant_Fiber/GF (i.e., DNp01, one on each side)
  4. Core intermediate layer = neurons on the short path from input to GF (<=3 synaptic edges, i.e., at most 2 interneurons)
     The GF escape response latency is only a few milliseconds; biologically this is a paucisynaptic pathway; also avoids the 2-hop union explosion to tens of thousands of nodes
  5. Output layer = neurons within 2 forward hops from GF with super_class in {descending, motor} (descending/intracerebral motor neurons)
  6. Edges = all connections within the above node set; weight W = sign(neurotransmitter) * log1p(syn_count)
     Neurotransmitter signs (Drosophila): ACH excitatory (+1); GABA inhibitory (-1); GLUT inhibitory in insects (-1); DA/SER/OCT are modulatory (+1 and marked)
"""
import os
import json
import gzip
import csv
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r'<FlyWire data directory>'

PATH_CONN = os.path.join(DATA_DIR, 'connections_princeton.csv.gz')
PATH_TYPES = os.path.join(DATA_DIR, 'consolidated_cell_types.csv.gz')
PATH_CLASS = os.path.join(DATA_DIR, 'classification.csv.gz')
PATH_NEURONS = os.path.join(DATA_DIR, 'neurons.csv.gz')

VISION_TYPES = {'LPLC2'}
WIND_TYPES = {'JO-B', 'JO-C'}
HUB_TOKENS = ('giant_fiber', 'gf')     # matched in additional_type(s)
HUB_PRIMARY = 'DNp01'                  # primary_type of giant fiber in FlyWire
OUTPUT_SUPER = {'descending', 'motor'}
NT_SIGN = {'ACH': 1.0, 'GABA': -1.0, 'GLUT': -1.0, 'DA': 1.0, 'SER': 1.0, 'OCT': 1.0}
MODULATORY_NT = {'DA', 'SER', 'OCT'}

import pandas as pd
import numpy as np


def log(msg):
    print(msg, flush=True)


def load_tables():
    log('[1/6] Reading local CSV files (read-only)...')
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
    log(f'      Cells: {len(cells)} | Raw connection rows: {len(conn)} (split by neuropil, need merging)')
    return cells, cls, ntn, conn


def find_nodes(cells):
    log('[2/6] Locating input layer / escape hub nodes...')
    vision_ids = set(cells.loc[cells['primary_type'].isin(VISION_TYPES), 'root_id'])
    wind_ids = set(cells.loc[cells['primary_type'].isin(WIND_TYPES), 'root_id'])

    add = cells['additional_type(s)'].str.lower()
    hub_mask = add.str.contains('giant_fiber', regex=False) | (cells['primary_type'] == HUB_PRIMARY)
    hub_ids = set(cells.loc[hub_mask, 'root_id'])

    log(f'      Visual input LPLC2: {len(vision_ids)}')
    log(f'      Wind sense input JO-B/C: {len(wind_ids)}')
    log(f'      Giant fiber GF (DNp01): {len(hub_ids)}  root_id={sorted(hub_ids)}')
    assert hub_ids, 'Giant fiber GF not found, please check data files'
    return vision_ids, wind_ids, hub_ids


def aggregate_edges(conn):
    log('[3/6] Merging connection rows for the same neuron pair across different neuropils...')
    g = conn.groupby(['pre_root_id', 'post_root_id'], sort=False)
    edges = g['syn_count'].sum().reset_index(name='syn_count')
    # Use the neurotransmitter type from the row with the most synapses as the primary neurotransmitter for this pair
    idx = conn.groupby(['pre_root_id', 'post_root_id'], sort=False)['syn_count'].idxmax()
    nt = conn.loc[idx, ['pre_root_id', 'post_root_id', 'nt_type']].rename(columns={'nt_type': 'nt_major'})
    edges = edges.merge(nt, on=['pre_root_id', 'post_root_id'], how='left')
    log(f'      Unique directed edges after merging: {len(edges)}')
    return edges


def hop(edges, seeds, direction='forward', exclude=None):
    """Perform 1-hop expansion on the aggregated edge table, returning the set of newly discovered nodes."""
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
    log('[4/6] Performing bidirectional 2-hop pruning centered on GF (keeping neurons on the input->GF pathway)...')
    inputs = vision_ids | wind_ids

    fwd1 = hop(edges, inputs, 'forward') - inputs
    fwd2 = hop(edges, fwd1, 'forward') - inputs - fwd1
    rev1 = hop(edges, hub_ids, 'backward') - hub_ids
    rev2 = hop(edges, rev1, 'backward') - hub_ids - rev1

    # Core pathway from input to GF: only keep neurons on the <=3 edge short path from input to GF
    #   1 interneuron: input -> x -> GF                (x in fwd1 intersect rev1)
    #   2 interneurons: input -> x -> y -> GF          (x in fwd1 intersect rev2, y in fwd2 intersect rev1)
    core = ((fwd1 & (rev1 | rev2)) | (fwd2 & rev1)) | hub_ids

    # GF downstream output layer: descending/motor neurons within 2 forward hops
    down1 = hop(edges, hub_ids, 'forward') - hub_ids
    down2 = hop(edges, down1, 'forward') - hub_ids - down1
    out_ids = {n for n in (down1 | down2) if cls_map.get(n, '') in OUTPUT_SUPER}

    node_set = (inputs | core | out_ids | hub_ids)
    log(f'      1-hop/2-hop candidates: fwd {len(fwd1)}/{len(fwd2)}  rev {len(rev1)}/{len(rev2)}')
    log(f'      Core pathway neurons: {len(core)} | GF descending output: {len(out_ids)} | Total nodes: {len(node_set)}')
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

    log('[5/6] Exporting subgraph edges with weights W = sign(NT) * log1p(syn_count)...')
    sub_edges = edges[edges['pre_root_id'].isin(node_set) & edges['post_root_id'].isin(node_set)].copy()
    sub_edges = sub_edges[sub_edges['syn_count'] >= 3].copy()  # Remove weak connections with syn_count<3 (consistent with FlyWire filtering criteria)

    # Neurotransmitter: prefer the primary NT from the connection row; if missing, use neuron-level prediction; if still missing, default to ACH
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

    log('[6/6] Writing escape_network_sparse.json / nodes.csv / edges.csv ...')
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
    log('==== Extraction complete ====')
    log(f"Nodes: {len(node_records)} {dict(role_cnt)}")
    log(f"Edges: {len(edge_records)} (syn_count>=3)")
    log(f"Output escape_network_sparse.json -> {BASE}")


if __name__ == '__main__':
    main()
