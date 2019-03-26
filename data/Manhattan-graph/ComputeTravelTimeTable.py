"""
compute the travel time table for edges in Manhattan
"""

import time
import sys
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm

sys.path.append('../..')

from lib.Route import get_duration


def load_Manhattan_graph():
    edges = pd.read_csv('edges.csv')
    nodes = pd.read_csv('nodes.csv')
    travel_time_edges = pd.read_csv('travel-time-sun.csv', index_col=0).mean(1)
    G = nx.DiGraph()
    num_edges = edges.shape[0]
    rng = tqdm(edges.iterrows(), total=num_edges, ncols=100, desc='Loading Manhattan Graph')
    for i, edge in rng:
        src = edge['source']
        sink = edge['sink']
        travel_time = round(travel_time_edges.iloc[i], 2)
        G.add_edge(src, sink, weight=travel_time)

        src_pos = np.array([nodes.iloc[src - 1]["lng"], nodes.iloc[src - 1]["lat"]])
        sink_pos = np.array([nodes.iloc[sink - 1]["lng"], nodes.iloc[sink - 1]["lat"]])
        G.add_node(src, pos=src_pos)
        G.add_node(sink, pos=sink_pos)

    # pos = nx.shell_layout(G)
    # nx.draw_networkx_nodes(G, pos, node_size=700)
    # nx.draw_networkx_edges(G, pos, width=6)
    # nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
    # plt.axis('off')
    # plt.show()

    return G


def compute_table_nx(nodes_id, travel_time_table):
    G = load_Manhattan_graph()
    for o in tqdm(nodes_id):
        for d in tqdm(nodes_id):
            duration, path = nx.bidirectional_dijkstra(G, o, d)
            try:
                travel_time_table.iloc[o - 1, d - 1] = duration
            except nx.NetworkXNoPath:
                print('no path between', o, d)

        if o == 2:
            print('')
            print(travel_time_table.head(2))
            break


def compute_table_OSRM(nodes, nodes_id, travel_time_table):
    for o in tqdm(nodes_id):
        olng = nodes.iloc[o - 1]["lng"]
        olat = nodes.iloc[o - 1]["lat"]
        for d in tqdm(nodes_id):
            dlng = nodes.iloc[d - 1]["lng"]
            dlat = nodes.iloc[d - 1]["lat"]
            duration = get_duration(olng, olat, dlng, dlat)
            if duration is not None:
                travel_time_table.iloc[o - 1, d - 1] = duration

        if o == 2:
            print('')
            print(travel_time_table.head(2))
            break


if __name__ == '__main__':
    nodes = pd.read_csv('nodes.csv')
    nodes_id = list(range(1, nodes.shape[0] + 1))
    # travel_time_table = pd.DataFrame(-np.ones((num_nodes, num_nodes)), index=nodes_id, columns=nodes_id)
    # travel_time_table.to_csv('travel-time-table.csv')

    travel_time_table = pd.read_csv('travel-time-table.csv', index_col=0)
    print(travel_time_table.head(2))
    print(travel_time_table.shape[0])
    print(travel_time_table.shape[1])

    # compute_table_OSRM(nodes, nodes_id, travel_time_table)

    compute_table_nx(nodes_id, travel_time_table)

    # travel_time_table.to_csv('travel-time-table-1.csv')