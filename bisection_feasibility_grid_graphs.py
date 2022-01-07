from itertools import permutations
from math import ceil
import networkx as nx
import numpy as np
from pprint import pprint
import sys


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
    # for node in district_adj_graph:
    #     print(node, ':', [neighbor for neighbor in district_adj_graph.neighbors(node)])

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
    num_nodes = graph.number_of_nodes()
    if num_nodes < 2: # Base case
        return True

    if num_nodes > 10:
        raise Exception("Too many nodes to enumerate matchings in search of feasible bisection.")

    log2_num_nodes = np.log2(num_nodes)
    is_power_two_num_nodes = (log2_num_nodes == int(log2_num_nodes))

    if not is_power_two_num_nodes:
        nearest_power_two = 2**(ceil(log2_num_nodes))
        num_bisections_last_round = (nearest_power_two - num_nodes)

        #TODO Enumerate possible last-round matchings (of size num_bisections_last_round)
        raise Exception("Have not yet implemented bisection feasibility check for non-power-of-two order graphs")
        return False

    
    perfect_matchings = enumerate_perfect_matchings(graph)
    
    # print('The graph has', len(perfect_matchings), 'perfect matchings.')

    for matching in perfect_matchings:
        contracted_graph = graph.copy()
        for edge in matching:
            contracted_graph = nx.contracted_edge(contracted_graph, edge, self_loops=False)
        
        # Just need one perfect matching to work
        if is_bisectable_graph(contracted_graph):
            return True
    
    # If no perfect matching works, then we must conclude the graph cannot be reached via bisection
    return False

"""
Enumerates the perfect matchings of the given graph.
Brute-force implementation, but not easy to improve. 

TODO : Consider testing with DeFord's FKT implementation, which *counts* perfect matchings. 

Parameters:
===============
    graph - a networkx Graph

Returns:
===============
    a list of sets of edges which are perfect matchings

"""
def enumerate_perfect_matchings(graph):
    perfect_matchings = []
    if graph.number_of_nodes() % 2 == 1:
        # No perfect matchings possible if odd number of nodes
        return perfect_matchings
    
    edges = sorted([edge for edge in graph.edges])
    candidate_matchings = list(permutations(edges, r = graph.number_of_nodes() // 2))
    for matching in candidate_matchings:
        if nx.is_perfect_matching(graph, matching):
            perfect_matchings.append(matching)

    unique_perfect_matchings = []

    # Remove duplicates
    is_new_matching = [True for i in range(len(perfect_matchings))]
    for i in range(len(perfect_matchings)):
        if is_new_matching[i]:
            unique_perfect_matchings.append(perfect_matchings[i])
        else:
            continue

        for j in range(i + 1, len(perfect_matchings)):
            edges_i = perfect_matchings[i]
            edges_j = perfect_matchings[j]
            is_same_matching = True
            for edge_j in edges_j:
                is_same_matching = is_same_matching and edge_j in edges_i
            
            if is_same_matching: 
                is_new_matching[j] = False

    return unique_perfect_matchings


if __name__ == '__main__':
    if len(sys.argv) < 4:
        exit('Not enough arguments. Usage: python bisection_feasibility_grid_graphs.py [partitions enumeration filename] [rows] [cols]')

    filename = sys.argv[1]
    num_rows = int(sys.argv[2])
    num_cols = int(sys.argv[3])

    partitions = np.loadtxt(filename, dtype=int, delimiter=',')
    
    infeasible_indices = []

    for i in range(len(partitions)):
        if i % 10 == 0:
            print('Checking partition ', i, '...', sep='')
        partition_data = partitions[i].reshape((num_rows, num_cols))
        # pprint(partition_data)

        is_feasible = is_partition_feasible(partition_data)

        if not is_feasible:
            infeasible_indices.append(i)
    
    print('\nFinished iterating over partitions of the {} x {} grid.'.format(num_rows, num_cols))
    print('Of the', len(partitions), 'partitions,', len(infeasible_indices), 'are infeasible:')
    pprint(infeasible_indices)

