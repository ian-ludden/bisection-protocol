import networkx as nx
import numpy as np
from pprint import pprint
import sys

from numpy.lib.scimath import log2

"""
Determines whether a partition, given as a rows x cols numpy array 
of integer part assignments from 1 to n, is reachable via the bisection protocol. 
"""
def is_partition_feasible(assignment_grid):

    n = int(np.max(assignment_grid))
    num_rows = len(assignment_grid)
    num_cols = len(assignment_grid[0])

    district_edges = set()

    # Build district adjacency graph
    for i in range(num_rows):
        for j in range(num_cols):
            unit_assignment = assignment_grid[i, j]
            if i < num_rows - 1: # Check unit below
                south_neighbor_assignment = assignment_grid[i + 1, j]
                if south_neighbor_assignment != unit_assignment:
                    district_edges.add((unit_assignment, south_neighbor_assignment))
            if j < num_cols - 1:
                east_neighbor_assignment = assignment_grid[i, j + 1]
                if east_neighbor_assignment != unit_assignment:
                    district_edges.add((unit_assignment, east_neighbor_assignment))

    
    district_adj_graph = nx.Graph(list(district_edges))
    print(district_adj_graph)
    for node in district_adj_graph:
        print(node, ':', [neighbor for neighbor in district_adj_graph.neighbors(node)])

    return is_bisectable_graph(district_adj_graph)


"""
Determines whether an undirected graph can be 
a district adjacency graph produced by the bisection protocol. 

Parameter: 
===============
    graph - a networkx Graph with nodes representing (super)districts

Returns:
===============
    True if the given graph can be produced by the bisection protocol, 
    False otherwise. 
"""
def is_bisectable_graph(graph):
    if graph.number_of_nodes() < 2: # Base case
        return True

    log2_num_nodes = np.log2(graph.number_of_nodes())
    is_power_two_num_nodes = (log2_num_nodes == int(log2_num_nodes))


    return False


if __name__ == '__main__':
    if len(sys.argv) < 4:
        exit('Not enough arguments. Usage: python bisection_feasibility_grid_graphs.py [partitions enumeration filename] [rows] [cols]')

    filename = sys.argv[1]
    num_rows = int(sys.argv[2])
    num_cols = int(sys.argv[3])

    partitions = np.loadtxt(filename, dtype=int, delimiter=',')
    
    infeasible_indices = []

    for i in range(len(partitions)):
        partition_data = partitions[i].reshape((num_rows, num_cols))
        pprint(partition_data)

        is_feasible = is_partition_feasible(partition_data)

        if not is_feasible:
            infeasible_indices.append(i)

