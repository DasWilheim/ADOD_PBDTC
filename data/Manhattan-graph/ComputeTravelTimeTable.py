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

# from lib.Route import get_duration_from_osrm


def load_Manhattan_graph():
    edges = pd.read_csv('edges.csv')
    nodes = pd.read_csv('nodes.csv')
    travel_time_edges = pd.read_csv('time-sat.csv', index_col=0).mean(1)
    G = nx.DiGraph()
    num_edges = edges.shape[0]
    rng = tqdm(edges.iterrows(), total=num_edges, ncols=100, desc='Loading Manhattan Graph')
    for i, edge in rng:
        u = edge['source']
        v = edge['v']
        travel_time = round(travel_time_edges.iloc[i], 2)
        G.add_edge(u, v, weight=travel_time)

        u_pos = np.array([nodes.iloc[u - 1]['lng'], nodes.iloc[u - 1]['lat']])
        v_pos = np.array([nodes.iloc[v - 1]['lng'], nodes.iloc[v - 1]['lat']])
        G.add_node(u, pos=u_pos)
        G.add_node(v, pos=v_pos)

    # pos = nx.shell_layout(G)
    # nx.draw_networkx_nodes(G, pos, node_size=700)
    # nx.draw_networkx_edges(G, pos, width=6)
    # nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
    # plt.axis('off')
    # plt.show()

    return G


def compute_table_nx(nodes_id, travel_time_table):
    G = load_Manhattan_graph()

    time1 = time.time()
    length = dict(nx.all_pairs_dijkstra_path_length(G, cutoff=None, weight='weight'))
    print('...running time : %.05f seconds' % (time.time() - time1))

    for o in tqdm(nodes_id):
        for d in tqdm(nodes_id):
            try:
                duration = round(length[o][d], 2)
                travel_time_table.iloc[o - 1, d - 1] = duration
            except nx.NetworkXNoPath:
                print('no path between', o, d)


def compute_table_OSRM(nodes, nodes_id, travel_time_table):
    for o in tqdm(nodes_id):
        olng = nodes.iloc[o - 1]['lng']
        olat = nodes.iloc[o - 1]['lat']
        for d in tqdm(nodes_id):
            dlng = nodes.iloc[d - 1]['lng']
            dlat = nodes.iloc[d - 1]['lat']
            duration = get_duration_from_osrm(olng, olat, dlng, dlat)
            if duration is not None:
                travel_time_table.iloc[o - 1, d - 1] = round(duration, 2)


def compute_shortest_path_table(nodes, G):
    time1 = time.time()
    len_path = dict(nx.all_pairs_dijkstra(G, cutoff=None, weight='weight'))
    print('all_pairs_dijkstra running time : %.05f seconds' % (time.time() - time1))
    # nodes = pd.read_csv('nodes.csv')
    nodes_id = list(range(1, nodes.shape[0] + 1))
    num_nodes = len(nodes_id)
    shortest_path_table = pd.DataFrame(([['-1']*num_nodes]*num_nodes), index=nodes_id, columns=nodes_id)
    for o in tqdm(nodes_id):
        for d in tqdm(nodes_id):
            try:
                # duration = round(len_path[o][0][d], 2)
                path = len_path[o][1][d]
                if len(path) == 1 or len(path) == 2:
                    continue
                if len(path) == 3:
                    sub_path = path[1]
                else:
                    u_1 = path[1]
                    v_1 = path[-2]
                    sub_path = u_1 * 10000 + v_1
                shortest_path_table.iloc[o - 1, d - 1] = sub_path
            except nx.NetworkXNoPath:
                print('no path between', o, d)
    # shortest_path_table.to_csv('shortest-path-table.csv')
    return shortest_path_table


def compute_k_shortest_path_table(nodes, G, NOD_SPT):
    pass


def store_map_as_pickle_file():
    G = load_Manhattan_graph()
    with open('map.pickle', 'wb') as f:
        pickle.dump(G, f)


if __name__ == '__main__':
    # # for travel time table
    # nodes = pd.read_csv('nodes.csv')
    # nodes_id = list(range(1, nodes.shape[0] + 1))
    # # travel_time_table = pd.DataFrame(-np.ones((num_nodes, num_nodes)), index=nodes_id, columns=nodes_id)
    # # travel_time_table.to_csv('time-table-empty.csv')
    #
    # travel_time_table = pd.read_csv('time-table-empty.csv', index_col=0)
    # # print(travel_time_table.head(2))
    # # print(travel_time_table.shape[0])
    # # print(travel_time_table.shape[1])
    #
    # compute_table_OSRM(nodes, nodes_id, travel_time_table)
    #
    # # compute_table_nx(nodes_id, travel_time_table)
    #
    # travel_time_table.to_csv('time-table-osrm.csv')

    # travel_time_table = pd.read_csv('time-table-osrm.csv', index_col=0)
    # # print(travel_time_table.iloc[3826, 3833])
    # # print(travel_time_table.iloc[3910, 3920])
    # print(travel_time_table.iloc[5:10, 1800:2000])
    # travel_time_table = pd.read_csv('time-table-sat.csv', index_col=0)
    # print(travel_time_table.iloc[5:10, 1800:2000])

    # for routing server
    # store_map_as_pickle_file()

    with open('map.pickle', 'rb') as f:
        G = pickle.load(f)

    aa = time.time()
    path = nx.dijkstra_path(G, 1, 10)
    print(path)
    print('aa running time:', (time.time() - aa))

    bb = time.time()
    path = nx.bidirectional_dijkstra(G, 1, 4000)
    print(path)
    print('bb running time:', (time.time() - bb))





