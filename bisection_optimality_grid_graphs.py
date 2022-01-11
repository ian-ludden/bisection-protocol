from datetime import datetime
# from itertools import permutations
import os
import networkx as nx
import numpy as np
from pprint import pprint
import psutil
import sys

# Start of DistrictPlan class

class DistrictPlan:
    """
    # TODO
    """
    def __init__(self, assignment_grid):
        self.num_districts = int(np.max(assignment_grid))
        self.num_rows = len(assignment_grid)
        self.num_cols = len(assignment_grid[0])
        self.num_units = self.num_rows * self.num_cols

        self.assignment = {}
        index = 1
        for i in range(num_rows):
            for j in range(num_cols):
                self.assignment[index] = assignment_grid[i, j]
                index += 1

        district_edges = set()
        self.cut_edges = set()

        # Build district adjacency graph and set of cut edges between districts
        index = 0
        for i in range(num_rows):
            for j in range(num_cols):
                index += 1
                unit_assignment = assignment_grid[i, j]
                if i < num_rows - 1: # Check unit below
                    south_neighbor_assignment = assignment_grid[i + 1, j]
                    if south_neighbor_assignment != unit_assignment:
                        district_edges.add((unit_assignment, south_neighbor_assignment))
                        south_neighbor_index = index + num_cols
                        self.cut_edges.add(tuple(sorted([index, south_neighbor_index])))

                if j < num_cols - 1:
                    east_neighbor_assignment = assignment_grid[i, j + 1]
                    if east_neighbor_assignment != unit_assignment:
                        district_edges.add((unit_assignment, east_neighbor_assignment))
                        east_neighbor_index = index + 1
                        self.cut_edges.add(tuple(sorted([index, east_neighbor_index])))

        self.district_graph = nx.Graph(list(district_edges))


    """
    Computes the edges cut by the given bipartition. 

    Parameters:
    ===========
        bipartition - the set of district indices for one half of the bipartition
    
    Returns:
    ===========
        forced_cut_edges - the set of cut edges in the district plan which are also cut by the bipartition
    """
    def edges_cut_by_bipartition(self, bipartition):
        forced_cut_edges = set()
        for cut_edge in self.cut_edges:
            districts = [self.assignment[i] for i in cut_edge]
            if (districts[0] in bipartition and districts[1] not in bipartition) or \
               (districts[1] in bipartition and districts[0] not in bipartition):
               forced_cut_edges.add(cut_edge)
        return forced_cut_edges

# End of DistrictPlan class

"""
Enumerates the balanced, connected bipartitions of the given graph.

NOTE: Only implemented for 4 nodes. 

Parameters:
===============
    graph - a networkx Graph

Returns:
===============
    a list of sets of nodes such that each set is one side of a bipartition

"""
def enumerate_bipartitions(graph):
    if graph.number_of_nodes() != 4:
        raise NotImplementedError("Only implemented for 4 nodes.")
    
    bipartitions = []
    # if graph.number_of_nodes() % 2 == 1:
    #     # No balanced bipartitions possible if odd number of nodes
    #     return bipartitions
    
    nodes = sorted([node for node in graph.nodes])
    candidate_node_sets = [[nodes[0], nodes[1]], [nodes[0], nodes[2]], [nodes[0], nodes[3]]] # All possible pairs that contain 1

    # Check feasibility of each candidate bipartition, i.e., connectedness of each half
    for candidate_node_set in candidate_node_sets:
        other_set = set(nodes).difference(set(candidate_node_set))
        if tuple(candidate_node_set) in graph.edges and tuple(other_set) in graph.edges:
            bipartitions.append(set(candidate_node_set))
    
    # # More general implementation for more than 4 nodes:
    # for this_half in candidate_node_sets:
    #     other_half = list(set(nodes).difference(set(this_half)))
    #     if nx.is_connected(nx.subgraph(graph, this_half)) and nx.is_connected(nx.subgraph(graph, other_half)):
    #         bipartitions.append(set(this_half))
    
    return bipartitions


if __name__ == '__main__':
    if len(sys.argv) < 4:
        exit('Not enough arguments. Usage: python bisection_feasibility_grid_graphs.py [partitions enumeration filename] [rows] [cols]')

    filename = sys.argv[1]
    num_rows = int(sys.argv[2])
    num_cols = int(sys.argv[3])

    partitions = np.loadtxt(filename, dtype=int, delimiter=',')
    
    plans = []

    for i in range(len(partitions)):
        if i % 100000 == 0:
            print('Loading partition ', i, '...', sep='')
            now = datetime.now()
            print('\tCurrent time:', now.strftime("%H:%M:%S"))
            # Memory check
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            print('Current memory usage: {:.2f} MB'.format(mem_info.rss / (1024 ** 2)))
        
        partition_data = partitions[i].reshape((num_rows, num_cols))

        plan = DistrictPlan(partition_data)
        plans.append(plan)

    # List of sets of cut edges representing feasible bisections in the first round
    first_round_strategies = []

    for i in range(len(plans)):
        plan = plans[i]

        for node in plan.district_graph.nodes:
            print(node, ':', [neighbor for neighbor in plan.district_graph.neighbors(node)])

        bipartitions = enumerate_bipartitions(plan.district_graph)
        for bipartition in bipartitions:
            bipartition_cut_edges = plan.edges_cut_by_bipartition(bipartition)
            
            if bipartition_cut_edges not in first_round_strategies:
                first_round_strategies.append(bipartition_cut_edges)
    


    # for i in range(len(plans)):
    #     plan = plans[I]
        
        # if not bipartition_cut_edges.issubset(other_plan.cut_edges):
        #     continue # Bisection is incompatible with 
                
