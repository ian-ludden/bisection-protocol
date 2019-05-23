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
# optimal play of the I-cut-you-freeze protocol introduced by 
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
	s -- the total vote-share of Player 1 (always)
	A -- the index of the cutting player (1 or 2)

	Returns a list of Player 1 vote-shares in each district. 
	"""
	if DEBUG:
		print('computeDistrictPlan({0},{1:.3f},{2})'.format(n, s, A))
	
	B = 3 - A

	# Error checks.
	if s > n:
		raise ValueError('n={0}, s={1:.5f}. Cannot have s > n.'.format(n, s))

	# 0. Base case.
	if n == 1:
		return [s]

	# 1. Implement Lemmas 3.4, 3.6 from Pegden et al. (2017).

	# Player 1's vote-share in the frozen district.
	voteShare = 0

	if A == 1 and s < n / 2:
		# P2 wins a district in which they are fully packed. 
		pass # s = s
	elif A == 2 and s >= n / 2:
		# Player 1 wins a district in which they are fully packed.
		voteShare = 1
	else: 
		# The stronger player draws n identical districts. 
		voteShare = s / n

	s = s - voteShare

	# 2. Recurse.
	if DEBUG:
		print('Calling computeDistrictPlan({0}, {1:.5f}, {2})'.format(n - 1, s, B))
	otherVoteShares = computeDistrictPlan(n - 1, s, B)

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
	titleText = 'Efficiency Gap vs. Vote-share, n = {0}'.format(n)

	fig, axarr = plt.subplots(nrows=2, sharex=True, figsize=(8,8))
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
	axarr[0].set_yticks(np.arange(0, 1.25, step=0.25))
	axarr[1].plot(normalizedS, effGapsPercent)
	axarr[1].set(xlabel='Vote-share', ylabel='Efficency Gap (%)')
	yticks1 = [-50, -25, -8, 0, 8, 25, 50]
	axarr[1].set_yticks(yticks1)
	axarr[0].grid()
	axarr[1].grid()
	plt.xticks(np.arange(0, 1.25, step=0.25))
	fig.savefig('plotEffGapsAndSeatShareICYF_{0}_res{1}.png'.format(n,resolution))
	plt.show()

