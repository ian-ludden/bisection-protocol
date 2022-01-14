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

        voter_grid - a grid of the same shape as assignment_grid 
                     indicating the voter distribution (+1 for player 1, 0 for player 2)
    """
    def __init__(self, assignment_grid, voter_grid):
        self.num_districts = int(np.max(assignment_grid))
        self.num_rows = len(assignment_grid)
        self.num_cols = len(assignment_grid[0])
        self.num_units = self.num_rows * self.num_cols
        self.units_per_district = self.num_units // self.num_districts

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

        # Compute wins for player 1, wins for player 2, and ties
        votes = voter_grid.flatten()
        self.wins1 = 0
        self.wins2 = 0
        self.ties = 0
        for i in range(1, self.num_districts + 1):
            unit_indices = [index for index in range(1, self.num_units + 1) if self.assignment[index] == i]
            votes1 = sum([votes[unit_index - 1] for unit_index in unit_indices])
            votes2 = self.units_per_district - votes1
            if votes1 > votes2: self.wins1 += 1
            elif votes2 > votes1: self.wins2 += 1
            else: self.ties += 1

        self.player1_utility = self.wins1 - self.wins2 # Net utility for player 1


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

    """
    Returns representation of DistrictPlan.
    """
    def __repr__(self) -> str:
        str_rep = ""
        index = 0
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                index += 1
                str_rep += str(self.assignment[index])
            str_rep += "\n" # End row
        str_rep += '\nPlayer 1 receives ' + str(self.player1_utility) + ' net utility from ' + str(self.wins1) + ' win(s), '\
                + str(self.wins2) + ' loss(es), and ' + str(self.ties) + ' tie(s).\n'
        return str_rep

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
    if len(sys.argv) < 6:
        exit('Not enough arguments. Usage: python bisection_feasibility_grid_graphs.py [partitions enumeration filename] [rows] [cols] [voter distribution filename] [first player (\'R\' or \'D\')]')

    partition_filename = sys.argv[1]
    num_rows = int(sys.argv[2])
    num_cols = int(sys.argv[3])
    voters_filename = sys.argv[4]
    first_player = sys.argv[5]

    partitions = np.loadtxt(partition_filename, dtype=int, delimiter=',')
    voter_grid = np.genfromtxt(voters_filename, dtype=int, delimiter=1)

    # Invert voter grid if R is bisecting first
    if first_player == "R":
        voter_grid = np.ones(voter_grid.shape) - voter_grid
    
    print("Voter grid ({} first):".format(first_player))
    pprint(voter_grid)
    print()

    plans = []

    for i in range(len(partitions)):
        if i % 100000 == 0:
            print('Loading partition ', i, '...', sep='')
            now = datetime.now()
            print('\tCurrent time:', now.strftime("%H:%M:%S"))
            # Memory check
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            print('\tCurrent memory usage: {:.2f} MB'.format(mem_info.rss / (1024 ** 2)))
        
        partition_data = partitions[i].reshape((num_rows, num_cols))

        plan = DistrictPlan(partition_data, voter_grid)
        plans.append(plan)

    # List of sets of cut edges representing feasible bisections in the first round
    first_round_strategies = [] # Alternatively, could iterate over partitions of grid into two equal-size components

    for i in range(len(plans)):
        plan = plans[i]

        if i % 100000 == 0:
            print('Looking for strategies in partition ', i, '...', sep='')

        # for node in plan.district_graph.nodes:
        #     print(node, ':', [neighbor for neighbor in plan.district_graph.neighbors(node)])

        bipartitions = enumerate_bipartitions(plan.district_graph)
        for bipartition in bipartitions:
            bipartition_cut_edges = plan.edges_cut_by_bipartition(bipartition)
            
            if bipartition_cut_edges not in first_round_strategies:
                first_round_strategies.append(bipartition_cut_edges)
    
    print('Found', len(first_round_strategies), 'distinct first-round strategies.')

    # Search for best first-round strategy given optimal second-round response
    best_player1_utility = -1. * plans[0].num_districts
    count_optimal_strategies = 0
    latest_best_strategy_index = -1
    resulting_plan_index = -1

    for strategy_index, first_round_bisection in enumerate(first_round_strategies):
        worst_player1_utility = plans[0].num_districts
        latest_best_response_index = -1

        if strategy_index % 10000 == 0:
            print('Considering first-round strategy with index', strategy_index)
            now = datetime.now()
            print('\tCurrent time:', now.strftime("%H:%M:%S"))
            # Memory check
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            print('\tCurrent memory usage: {:.2f} MB'.format(mem_info.rss / (1024 ** 2)))

        for i in range(len(plans)):
            plan = plans[i]
        
            if not first_round_bisection.issubset(plan.cut_edges):
                continue # Bisection is incompatible with partition

            player1_utility = plan.player1_utility
            if player1_utility < worst_player1_utility:
                latest_best_response_index = i
                worst_player1_utility = player1_utility
                if worst_player1_utility <= best_player1_utility:
                    break # First-round strategy is dominated

        # Check whether this first-round strategy beats (or ties) best known
        if worst_player1_utility == best_player1_utility:
            count_optimal_strategies += 1
            latest_best_strategy_index = strategy_index
            resulting_plan_index = latest_best_response_index
        
        if worst_player1_utility > best_player1_utility:
            count_optimal_strategies = 1
            best_player1_utility = worst_player1_utility
            latest_best_strategy_index = strategy_index
            resulting_plan_index = latest_best_response_index

        # TODO delete this break; just accelerating search for 6x6 instances with 50% vote-share and 4 districts, 
        # since ties are impossible (districts have 9 units each) and neither player can win all 4 (need 20 to win 4)
        if best_player1_utility >= 2:
            break 
    
    print('Optimal play gives player 1 utility {}.'.format(best_player1_utility))
    print('There is (are)', count_optimal_strategies, 'optimal first-round bisection(s).')
    print('Best (or one of the best) first-round bisection:')
    pprint(first_round_strategies[latest_best_strategy_index])
    print('Best (or one of the best) second-round response plans (index {}):'.format(resulting_plan_index))
    print(plans[resulting_plan_index])
    print('With cut edges:')
    pprint(plans[resulting_plan_index].cut_edges)


                
