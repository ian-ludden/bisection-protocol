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
# Date: 	16 May 2019
# 
# Computes the efficiency gap of district plans resulting from 
# optimal play of the I-cut-you-choose protocol introducdd by 
# Pegden et al. (2017). 
# 
# Source: https://arxiv.org/abs/1710.08781
######################################################################
DEBUG = False

# Load tresholds from default file.
thresholdsFilename = 'icycThresholds_1_to_32.csv'
t = csvToArray(thresholdsFilename)

def computeDistrictPlan(n, s, A):
	"""Recursively compute the vote-shares of all districts.

	Keyword arguments:
	n -- the number of districts
	s -- the total vote-share of the cutting player
	A -- the index of the cutting player (1 or 2)

	Returns a list of Player 1 vote-shares in each district. 
	"""
	B = 3 - A

	# Error checks.
	if s > n:
		raise ValueError('n={0}, s={1:.5f}. Cannot have s > n.'.format(n, s))

	# 0. Base case.
	if n == 1:
		return [s] if A == 1 else [1 - s]

	# 1. Implement Lemmas 3.4, 3.6 from Pegden et al. (2017).
	s1 = (s) if (A == 1) else (n - s)
	voteShare = 0

	if A == 1 and s1 < n / 2:
		# P2 wins a district in which they are fully packed. 
		pass # s1 = s1
	elif A == 2 and s1 >= n / 2:
		# Player 1 wins a district in which they are fully packed.
		voteShare = 1
	else: 
		# The stronger player draws n identical districts. 
		voteShare = s1 / n

	s1 = s1 - voteShare

	# 2. Recurse.
	if DEBUG:
		print('Calling computeDistrictPlan({0}, {1:.5f}, {2})'.format(n - 1, s1, B))
	otherVoteShares = computeDistrictPlan(n - 1, s1, B)

	return np.insert(otherVoteShares, 0, voteShare)


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
	fig.savefig('plotEffGapsAndSeatShareICYC_{0}_res{1}.png'.format(n,resolution))
	plt.show()

