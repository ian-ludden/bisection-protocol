import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import sys

from bisectionUtils import calcEfficiencyGap, csvToArray

######################################################################
# Author: 	Ian Ludden
# Date: 	14 May 2019
# 
# Computes the efficiency gap of district plans resulting from 
# optimal "play" of the bisection protocol. 
######################################################################
DEBUG = False

# Load default files with thresholds and opt seats.
thresholdsFilename = 'thresholds_1_to_32.csv'
optAseatsFilename = 'optAseats_1_to_32.csv'

# Read in tables of t_{n,j} and K_{n,j} as defined in bisectionDP.py.
t = csvToArray(thresholdsFilename)
K = csvToArray(optAseatsFilename)

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
		raise ValueError('n={0}, s={1:.5f}. Cannot have s > n.'.format(n, s))

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
	
	# ... Unless, of course, an even split makes s0 or s1 invalid.
	if s0 > P0:
		s1 = s1 + (s0 - P0)
		s0 = P0
	if s1 > P1:
		s0 = s0 + (s1 - P1)
		s1 = P1
	if s0 < 0:
		s1 = s1 + s0
		s0 = 0
	if s1 < 0:
		s0 = s0 + s1
		s1 = 0

	# 4. Recurse on each side.
	if DEBUG:
		print('Calling computeDistrictPlan({0}, {1:.5f}, {2})'.format(P0, s0, B))
		print('Calling computeDistrictPlan({0}, {1:.5f}, {2})'.format(P1, s1, B))
	voteShares0 = computeDistrictPlan(P0, s0, B)
	voteShares1 = computeDistrictPlan(P1, s1, B)

	return voteShares0 + voteShares1


if __name__ == '__main__':
	if len(sys.argv) < 2:
		exit('Not enough arguments.')
	n = int(sys.argv[1])

	# Increase to make plot more accurate
	resolution = 53
	
	# Range of vote-share values is slightly offset to avoid 
	# ambiguities of landing exactly on a threshold
	sSweep = np.linspace(0.101, n-0.099, resolution*n)
	A = 1

	effGaps = np.zeros(sSweep.shape)
	for i in range(len(sSweep)):
		try:
			voteShares = computeDistrictPlan(n, sSweep[i], A)
		except ValueError:
			print('Error computing district plan for n={0}, s={1:.5f}.'.format(n, sSweep[i]))
			raise
		effGaps[i] = calcEfficiencyGap(voteShares)

	normalizedS = (np.concatenate(([0.0], sSweep, [n]))) / n
	effGapsPercent = (np.concatenate(([-0.5], effGaps, [0.5]))) * 100.0
	titleText = 'Efficiency Gaps for n = {0}'.format(n)

	fig, axarr = plt.subplots(nrows=2, sharex=True)
	fig.suptitle(titleText)

	# Plot Seat-share and Efficiency Gap in separate plots
	# (since y-axis is different for each)
	yThresholds = np.array(range(n+1))
	yThresholds = np.repeat(yThresholds, 2)

	xThresholds = np.repeat(t[n,:n+1], 2)
	xThresholds = xThresholds[2:]
	xThresholds = np.insert(xThresholds, 0, 0)
	xThresholds = np.append(xThresholds, n)

	axarr[0].plot(xThresholds/n, yThresholds/n)
	axarr[0].set(ylabel='Seat-share')
	axarr[1].plot(normalizedS, effGapsPercent)
	axarr[1].set(xlabel='Fractional Vote-share', ylabel='Efficency Gap (%)')
	axarr[0].grid()
	axarr[1].grid()
	fig.savefig('plotEffGapsAndSeatShareBisection_{0}_res{1}.png'.format(n,resolution))
	plt.show()




