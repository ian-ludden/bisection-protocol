import csv
import math
import numpy as np
from pprint import pprint
import sys

from bisectionUtils import csvToArray

######################################################################
# Author: 	Ian Ludden
# Date: 	14 May 2019
# 
# Computes the efficiency gap of district plans resulting from 
# optimal "play" of the bisection protocol. 
######################################################################
DEBUG = False

if len(sys.argv) >= 3:
	thresholdsFilename = str(sys.argv[1])
	optAseatsFilename = str(sys.argv[2])
else: # Load default files for testing purposes
	thresholdsFilename = 'thresholds_1_to_32.csv'
	optAseatsFilename = 'optAseats_1_to_32.csv'

# n = 0 # TODO: n, s as parameters?
# s = 0

# Read in tables of t_{n,j} and K_{n,j} as defined in bisectionDP.py.
t = csvToArray(thresholdsFilename)
K = csvToArray(optAseatsFilename)

# TODO: sweep s from 0 to n with some specified resolution?

def computeDistrictPlan(n, s, A):
	"""Recursively compute the vote-shares of all districts.

	Keyword arguments:
	n -- the number of districts
	s -- the total vote-share of the cutting player
	A -- the index of the cutting player (1 or 2)

	Returns a list of Player 1 vote-shares in each district. 
	"""
	B = 3 - A

	# Compute sizes of 'left' (P0) and 'right' (P1) sides of cut. 
	P0 = math.floor(n/2)
	P1 = math.ceil(n/2)

	# Error checks.
	if s > n:
		raise ValueError('Cannot have s > n.')

	# 0. Base case.
	if n == 1:
		return [s] if A == 1 else [1 - s]

	# 1. Determine how many seats Player A will win (j).
	j = 0
	for i in range(n+1):
		if t[n,i] <= s:
			j = i
		else:
			break
	if DEBUG:
		print('j={}'.format(j))

	# 2. Find # seats Player A will win from P0, P1.
	seatsFromP0 = int(K[n,j])
	seatsFromP1 = j - seatsFromP0

	if DEBUG:
		print('seatsFromP0={}'.format(seatsFromP0))
		print('seatsFromP1={}'.format(seatsFromP1))

	# 3. Compute Player B's vote-shares for each side.
	s0 = P0
	if seatsFromP0 > 0:
		s0 = t[P0, P0 - seatsFromP0 + 1]

	s1 = P1
	if seatsFromP1 > 0:
		s1 = t[P1, P1 - seatsFromP1 + 1]

	if DEBUG:
		print('s0={0:.5f}'.format(s0))
		print('s1={0:.5f}'.format(s1))

	# We may assume any extra Player A vote-share is evenly divided.
	# Note that the way leftovers are handled does not affect the
	# efficiency gap. 
	leftovers = (n - s) - (s0 + s1)
	s0 = s0 + leftovers / 2
	s1 = s1 + leftovers / 2
	
	# ... Unless, of course, an even split would overfill one side.
	if s0 > P0:
		s1 = s1 + (s0 - P0)
		s0 = P0
	if s1 > P1:
		s0 = s0 + (s1 - P1)
		s1 = P1

	# 4. Recurse on each side.
	if DEBUG:
		print('Calling computeDistrictPlan({0}, {1:.5f}, {2})'.format(P0, s0, B))
		print('Calling computeDistrictPlan({0}, {1:.5f}, {2})'.format(P1, s1, B))
	voteShares0 = computeDistrictPlan(P0, s0, B)
	voteShares1 = computeDistrictPlan(P1, s1, B)

	return voteShares0 + voteShares1


def calcEfficiencyGap(voteShares):
	"""Computes the efficiency gap of a districting plan, 
	as given by a list of Player 1's vote-shares in each district.
	"""
	wastedVotes1 = 0
	wastedVotes2 = 0

	for s in voteShares:
		w1 = s - 0.5 if s >= 0.5 else s
		w2 = (1 - s) - 0.5 if s < 0.5 else 1 - s
		wastedVotes1 = wastedVotes1 + w1
		wastedVotes2 = wastedVotes2 + w2

	return (wastedVotes1 - wastedVotes2) / len(voteShares)


# def calcWastedVotes(n, s, A):
# 	"""Recursively compute how many votes are wasted for each player.

# 	Keyword arguments:
# 	n -- the number of districts
# 	s -- the vote-share of the cutting player
# 	A -- the index of the cutting player (1 or 2)

# 	Returns [wasted votes Player 1, wasted votes Player 2]
# 	"""
# 	B = 3 - A

# 	# Compute sizes of 'left' (P0) and 'right' (P1) sides of cut. 
# 	P0 = floor(n/2)
# 	P1 = ceil(n/2)

# 	# Error checks.
# 	if s > n:
# 		raise ValueError('Cannot have s > n.')

# 	# 0. Base case.
# 	if n == 1:
# 		if s >= 0.5:
# 				wastedVotes = [s - 0.5, 1 - s]
# 			else:
# 				wastedVotes = [s, 1 - s - 0.5]
# 		if A == 2:
# 			list.reverse(wastedVotes)
# 		return wastedVotes


# 	# 1. Determine how many seats Player A will win (j).
# 	j = 0
# 	for i in range(n+1):
# 		if t[n,i] <= s:
# 			j = i
# 		else:
# 			break

# 	# 2. Find # seats Player A will win from P0.
# 	seatsFromP0 = K[n,j]

# 	# 3. Compute Player B's vote-shares for each side.
# 	s0 = 0
# 	if seatsFromP0 > 0:
# 		s0 = t[P0, P0 - seatsFromP0 + 1]

# 	s1 = 0
# 	if seatsFromP1 > 0:
# 		s1 = t[P1, P1 - seatsFromP1 + 1]

# 	# 4. Recurse on each side.
# 	[w01, w02] = calcWastedVotes(P0, s0, B)
# 	[w11, w12] = calcWastedVotes(P1, s1, B)

# 	return [w01 + w11, w02 + w12]



