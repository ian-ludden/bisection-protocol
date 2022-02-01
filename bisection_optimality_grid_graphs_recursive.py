from datetime import datetime
from timeit import default_timer as timer
from enum import unique
from math import floor, log10
from itertools import combinations
import os
import networkx as nx
import numpy as np
from pprint import pprint
import psutil
import sys

DEBUG = True

def nodeset_to_string(nodeset):
    return ",".join(sorted(str(node) for node in nodeset))


# Start of DistrictPlan class

class DistrictPlan:
    """
    # TODO

        voter_grid - a grid of the same shape as assignment_grid 
                     indicating the voter distribution (+1 for player 1, 0 for player 2)
    """
    def __init__(self, assignment_grid):
        self.num_districts = int(np.max(assignment_grid))
        self.num_rows = len(assignment_grid)
        self.num_cols = len(assignment_grid[0])
        self.num_units = self.num_rows * self.num_cols
        self.units_per_district = self.num_units // self.num_districts

        self.assignment = {}
        index = 1
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                self.assignment[index] = assignment_grid[i, j]
                index += 1

        district_edges = set()
        self.cut_edges = set()

        # Build district adjacency graph and set of cut edges between districts
        index = 0
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                index += 1
                unit_assignment = assignment_grid[i, j]
                if i < self.num_rows - 1: # Check unit below
                    south_neighbor_assignment = assignment_grid[i + 1, j]
                    if south_neighbor_assignment != unit_assignment:
                        district_edges.add((unit_assignment, south_neighbor_assignment))
                        south_neighbor_index = index + self.num_cols
                        self.cut_edges.add(tuple(sorted([index, south_neighbor_index])))

                if j < self.num_cols - 1:
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


"""
Enumerates all balanced (within 1 node) partitions of a graph into 
two connected components. 

Parameters:
===============
    graph - undirected networkx Graph

Returns:
===============
    side1_node_sets - list of combinations of nodes for side 1 
                      of size floor(graph.number_of_nodes() / 2) 
                      that induce two connected components
"""
def enumerate_bisections(graph):
    side1_node_sets = []
    size1 = floor(graph.number_of_nodes() / 2.)
    
    nodes = sorted([node for node in graph.nodes])

    candidates_side1 = list(combinations(nodes, size1))
    for candidate_side1 in candidates_side1:
        complement_nodes = set(nodes).difference(candidate_side1)
        if nx.is_connected(nx.subgraph(graph, candidate_side1)) and nx.is_connected(nx.subgraph(graph, complement_nodes)):
            side1_node_sets.append(candidate_side1)
    
    return side1_node_sets



class BisectionInstance:

    def __init__(self, partition_filename, voters_filename, num_rows, num_cols, first_player="D"):
        partitions = np.loadtxt(partition_filename, dtype=int, delimiter=',')
        
        voter_grid = np.genfromtxt(voters_filename, dtype=int, delimiter=1)
        self.memoized_utilities = {}
        self.plans = []
        self.num_rows = num_rows
        self.num_cols = num_cols
        
        if DEBUG:
            print("Voter grid (1 is D, 0 is R):")
            pprint(voter_grid)
            print()

        self.votes = voter_grid.flatten()

        # Build plans
        for i in range(len(partitions)):
            if DEBUG and i % (10**(floor(log10(len(partitions))) - 1)) == 0: # Check progress every 10% or so
                print('Loading partition ', i, '...', sep='')
                now = datetime.now()
                print('\tCurrent time:', now.strftime("%H:%M:%S"))
                # Memory check
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                print('\tCurrent memory usage: {:.2f} MB'.format(mem_info.rss / (1024 ** 2)))
            
            partition_data = partitions[i].reshape((self.num_rows, self.num_cols))

            plan = DistrictPlan(partition_data)
            self.plans.append(plan)

        self.num_districts = self.plans[0].num_districts
        self.units_per_district = self.plans[0].units_per_district

        # Build unit adjacency graph
        unit_edges = set()
        for u in range(1, self.num_rows * self.num_cols + 1):
            u_row = (u - 1) // self.num_cols
            u_col = (u - 1) % self.num_cols
            for v in range(u, self.num_rows * self.num_cols + 1):
                v_row = (v - 1) // self.num_cols
                v_col = (v - 1) % self.num_cols
                
                if abs(u_row - v_row) + abs(u_col - v_col) == 1:
                    unit_edges.add((u, v))

        self.graph = nx.Graph(unit_edges)


    """
    Computes the utility of a single district, 
    formed from the given set of nodes (units), 
    from the perspective of the current player. 

    Parameters:
    ===========
        nodeset - set of nodes (unit indices)
        current_player - "D" or "R"
    
    Returns:
    ===========
        1 if current player has simple majority, 
        -1 if other player has simple majority, 
        0 otherwise (tie)
    """
    def district_utility(self, nodeset, current_player):
        votes_d = sum([self.votes[node - 1] for node in nodeset])
        votes_current_player = votes_d if current_player == "D" else len(nodeset) - votes_d
        if votes_current_player > len(nodeset) / 2:
            return 1
        elif votes_current_player < len(nodeset) / 2:
            return -1
        else:
            return 0


    """
    Lists unique assignments of the given set of nodes (units)
    to *whole* districts, among all feasible district plans. 

    Iterates over all possible plans, so could take a while.

    Parameters:
    ===========
        nodeset - set of node (unit) indices

    Returns:
    ===========
        list of unique assignments of the given set of nodes, 
        represented as dictionaries mapping node IDs to districts, 
        with districts relabeled to be consecutive integers starting from 1
    """
    def enumerate_unique_plans(self, nodeset):
        if len(nodeset) == self.num_rows * self.num_cols: # Shortcut for first (root) call
            return [plan.assignment for plan in self.plans]

        unique_plans = []
        
        for plan in self.plans:
            asst = plan.assignment
            restricted_plan = {unit : asst[unit] for unit in nodeset}
            
            vals = set(restricted_plan.values())
            if len(vals) > (len(nodeset) // self.units_per_district): # Must form whole districts
                continue

            if restricted_plan not in unique_plans:
                unique_plans.append(restricted_plan)

        return unique_plans


    """
    Recursively computes an optimal bisection.

    Parameters:
    ===============
        nodeset - the set of nodes (units) in the piece to be bisected 
        current_player - "D", corresponding to 1s in self.votes, or "R", corresponding to 0s

    Returns:
    ===============
        best_util - the highest possible first-player utility under optimal play
    """
    def recursive_optimal_bisection(self, nodeset, current_player):
        nodeset_str = nodeset_to_string(nodeset)
        if nodeset_str in self.memoized_utilities: 
            return self.memoized_utilities[nodeset_str]

        if len(nodeset) <= self.units_per_district:
            return self.district_utility(nodeset, current_player)

        # districts_in_piece = len(nodeset) // self.units_per_district
        
        next_player = "D" if current_player == "R" else "R"
        unique_plans = self.enumerate_unique_plans(nodeset)

        if len(nodeset) == (self.num_rows * self.num_cols):
            print("Found", len(unique_plans), "unique plans.")

        best_utility = -1 * (len(nodeset) // self.units_per_district)

        # Iterate over distinct plans (partitions) for the piece
        for unique_plan in unique_plans:
            district_edges = set()
            for node_u in unique_plan:
                district_u = unique_plan[node_u]
                for node_v in self.graph.neighbors(node_u):
                    if node_v not in nodeset: continue
                    district_v = unique_plan[node_v]
                    if district_u != district_v:
                        district_edges.add(tuple(sorted([district_u, district_v])))

            aux_graph = nx.Graph(district_edges)

            if aux_graph.number_of_nodes() <= 0:
                continue

            # Check simple utility bound before considering bisections
            votes_d = sum(self.votes[node - 1] for node in nodeset)
            votes_current_player = votes_d if current_player == "D" else len(nodeset) - votes_d
            max_wins = votes_current_player // ((self.units_per_district // 2) + 1)
            if max_wins <= best_utility:
                continue # Can't beat best utility seen so far, so skip this plan

            unique_bisections = enumerate_bisections(aux_graph)
            
            # Iterate over distinct bisections of the piece that fit with the current plan
            for side1_tuple in unique_bisections:
                side1 = set(side1_tuple)
                nodeset1 = set([node for node in nodeset if unique_plan[node] in side1])
                nodeset2 = nodeset.difference(nodeset1)

                opponent_util1 = self.recursive_optimal_bisection(nodeset1, next_player)
                opponent_util2 = self.recursive_optimal_bisection(nodeset2, next_player)

                utility = -1 * (opponent_util1 + opponent_util2) # Negate, since zero-sum
                if utility > best_utility:
                    best_utility = utility
                    if len(nodeset) == self.num_rows * self.num_cols: # Top level/root
                        self.best_first_round_sides = [nodeset1, nodeset2]


        # Memoize best utility
        self.memoized_utilities[nodeset_to_string(nodeset)] = best_utility

        return best_utility


if __name__ == '__main__':
    if len(sys.argv) < 5:
        exit('Not enough arguments. Usage: python bisection_feasibility_grid_graphs.py [partitions enumeration filename] [rows] [cols] [voter distribution filename] [first player (\'R\' or \'D\', optional with default \'D\')]')

    partition_filename = sys.argv[1]
    num_rows = int(sys.argv[2])
    num_cols = int(sys.argv[3])
    voters_filename = sys.argv[4]
    
    now = datetime.now()
    print('Start time:', now.strftime("%H:%M:%S"))
    start = timer()

    if len(sys.argv) >= 6:
        first_player = sys.argv[5]
        bisection_instance = BisectionInstance(partition_filename, voters_filename, num_rows, num_cols, first_player)
    else:
        first_player = "D"
        bisection_instance = BisectionInstance(partition_filename, voters_filename, num_rows, num_cols)

    full_nodeset = set([i for i in range(1, num_rows * num_cols + 1)])
    best_utility_first_player = bisection_instance.recursive_optimal_bisection(full_nodeset, first_player)

    print("Best utility achieved by player", first_player, "when going first:", best_utility_first_player,'\n')    
    end = timer()
    now = datetime.now()
    print('End time:', now.strftime("%H:%M:%S"))
    print('Elapsed time: {:.3f} seconds'.format(end - start))
    print()

    # TODO: Recover sequence of optimal bisections
    # if bisection_instance.best_first_round_sides:
    #     pprint(bisection_instance.best_first_round_sides)


                
