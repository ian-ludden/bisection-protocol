from datetime import datetime
from timeit import default_timer as timer
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
    Concise representation of a district plan as an assignment dictionary 
    mapping units to districts, along with aggregate election results. 

    Parameters:
    ===========
        assignment_grid - a 2-D grid of units' district labels (1 to k)

        voter_grid - a grid of the same shape as assignment_grid 
                     indicating the voter distribution (+1 for player D, 0 for player R)
        
        first_player - either "D" or "R"
    """
    def __init__(self, assignment_grid, voter_grid, first_player):
        num_districts = int(np.max(assignment_grid))
        num_rows = len(assignment_grid)
        num_cols = len(assignment_grid[0])
        num_units = num_rows * num_cols
        units_per_district = num_units // num_districts
        self.first_player = first_player
        
        votes = voter_grid.flatten()
        if first_player == "R":
            votes = [1 - vote for vote in votes] # Flip
        votes_p1_districts = np.zeros((num_districts,))

        self.assignment = {}
        index = 1
        for i in range(num_rows):
            for j in range(num_cols):
                assigned_district = assignment_grid[i, j]
                self.assignment[index] = assigned_district
                votes_p1_districts[assigned_district - 1] += (1 if votes[index - 1] == 1 else 0)
                index += 1

        self.wins_p1 = 0
        self.ties = 0
        for k in range(num_districts):
            if votes_p1_districts[k] > units_per_district // 2:
                self.wins_p1 += 1
            if (units_per_district % 2 == 0) and (votes_p1_districts[k] == units_per_district // 2):
                self.ties += 1

        self.wins_p2 = num_districts - self.wins_p1 - self.ties
        self.p1_utility = self.wins_p1 - self.wins_p2


    """
    Returns representation of DistrictPlan.
    """
    def __repr__(self) -> str:
        str_rep = ""
        # index = 0
        # for i in range(self.num_rows):
        #     for j in range(self.num_cols):
        #         index += 1
        #         str_rep += str(self.assignment[index])
        #     str_rep += "\n" # End row
        str_rep += '\nFirst player, ' + self.first_player +', receives ' + str(self.p1_utility) + ' net utility from ' + str(self.wins_p1) + ' win(s), '\
                + str(self.wins_p2) + ' loss(es), and ' + str(self.ties) + ' tie(s).\n'
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

            plan = DistrictPlan(partition_data, voter_grid, first_player)
            self.plans.append(plan)

        self.num_districts = int(np.max(partitions[0]))
        self.units_per_district = (self.num_rows * self.num_cols) // self.num_districts

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

        For top/root level, returns list of DistrictPlan objects (assignment dictionaries with election/utility data)
    """
    def enumerate_unique_plans(self, nodeset):
        if len(nodeset) == self.num_rows * self.num_cols: # Shortcut for first (root) call
            return self.plans

        unique_plans = []
        
        for plan in self.plans:
            asst = plan.assignment
            restricted_plan = {unit : asst[unit] for unit in nodeset}
            
            vals = set(restricted_plan.values())
            if len(vals) > (len(nodeset) // self.units_per_district): # Must form whole districts
                continue

            # TODO: relabel districts to be consecutive integers starting from 1
            # (not urgent, just will speed things up by a small constant factor probably)

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

        districts_in_piece = len(nodeset) // self.units_per_district

        if len(nodeset) <= self.units_per_district:
            return self.district_utility(nodeset, current_player)

        is_top_level = len(nodeset) == self.num_rows * self.num_cols
        
        next_player = "D" if current_player == "R" else "R"
        unique_plans = self.enumerate_unique_plans(nodeset)

        # Sort by decreasing first player utility at top/root level of recursion tree
        if is_top_level:
            unique_plans.sort(key=lambda plan : plan.p1_utility, reverse=True)

        if is_top_level:
            print("\nFound", len(unique_plans), "unique plans for full nodeset.")

        best_utility = -1 * (len(nodeset) // self.units_per_district)

        # Iterate over distinct plans (partitions) for the piece
        for plan_index, unique_plan in enumerate(unique_plans):
            if is_top_level and unique_plan.p1_utility < best_utility:
                break

            if is_top_level:
                unique_plan = unique_plan.assignment # Reduce DistrictPlan to its assignment dictionary

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
            # (first player draws all, with no contiguity constraints) 
            votes_d = sum(self.votes[node - 1] for node in nodeset)
            votes_current_player = votes_d if current_player == "D" else len(nodeset) - votes_d
            
            max_wins = votes_current_player // ((self.units_per_district // 2) + 1)
            max_ties_given_max_wins = 0
            if self.units_per_district % 2 == 0:
                max_ties_given_max_wins = (votes_current_player - (max_wins * ((self.units_per_district // 2) + 1))) // (self.units_per_district // 2)
            max_utility = max_wins - (districts_in_piece - max_wins - max_ties_given_max_wins)

            if best_utility >= max_utility:
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
                    if is_top_level:
                        print('New best utility at top level:', best_utility)
                        self.best_first_round_sides = [nodeset1, nodeset2]
                        self.best_plan = unique_plans[plan_index]
                
                    # Check utility bound
                    if best_utility >= max_utility:
                        break # Can't beat best utility seen so far, so continue on from this plan


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
    if bisection_instance.best_first_round_sides:
        pprint(bisection_instance.best_first_round_sides)
        
    if bisection_instance.best_plan:
        pprint(bisection_instance.best_plan)
        pprint(bisection_instance.best_plan.assignment)

