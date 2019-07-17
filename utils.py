import csv
import os
import math
import numpy as np
from pprint import pprint
from scipy.special import factorial2
import sys

######################################################################
# Author: 	Ian Ludden
# Date: 	14 May 2019
# 
# Utility functions for analyzing the bisection and I-cut-you-freeze 
# districting protocols.  
######################################################################

def csvToArray(filename):
	"""Reads the given CSV file into an array. 
	   The top left entry is in A[1, 1].
	"""
	if filename is None:
		return np.zeros((100,100))

	with open(filename, 'r') as csvFile:
		reader = csv.reader(csvFile)
		data = list(reader)
	maxIndex = len(data) + 1
	A = np.zeros((maxIndex,maxIndex))
	i = 1
	for row in data:
		j = 1
		for elem in row:
			if len(elem) > 0:
				A[i,j] = float(elem)
			j = j + 1
		i = i + 1
	return A


def arrayToCSV(A, filename, precision=1):
	"""Saves the given array to a CSV file with the given name.
	   Optional precision parameter specifies how many places
	   after decimal to write.
	"""
	np.savetxt(filename, A, fmt='%.{0}f'.format(precision), delimiter=',')


def calcICYFThreshold(n, k):
	"""Computes the vote-share needed to win k of n districts
	   under the I-cut-you-freeze protocol.
	"""
	if n == 0:
		return 0
	if n == 1:
		return 0.5 if k == 1 else -1

	nIsEven = 1 - (n % 2)

	if k <= n / 2:
		threshold1 = factorial2(n - 1) / factorial2(n - 2)
		threshold2 = factorial2(2 * k + (1 - nIsEven) - 2) / factorial2(2 * k + (1 - nIsEven) - 3)
		threshold = threshold1 * threshold2 / 2.0
	else:
		threshold1 = factorial2(n) / factorial2(n - 1)
		threshold2 = factorial2(2 * (n - k) - (1 - nIsEven) + 1) / factorial2(2 * (n - k) - (1 - nIsEven))
		threshold = threshold1 * threshold2 / 2.0
		threshold = n - threshold

	return threshold


def getThresholds(protocol, nMax):
	"""Returns all the thresholds from t_{0,0}
	   through t_{nMax,nMax} for the given protocol. 
	   For bisection protocol, returns second argument 
	   of table of optimal number of seats to win from 
	   first (smaller) side. 

	   If default thresholds file is missing or 
	   doesn't go high enough, all thresholds are 
	   computed. 
	"""
	nUB = nMax + 2

	thresholdsFilename = protocol['defaultThresholdsFilename']
	optAseatsFilename = protocol['defaultOptAFilename']

	if os.path.exists(thresholdsFilename):
		thresholds = csvToArray(thresholdsFilename)
		if protocol['abbrev'] == 'B': # bisection
			if os.path.exists(optAseatsFilename):
				optAseats = csvToArray(optAseatsFilename)
				if thresholds.shape[0] >= nUB: # thresholds file is big enough
					return [thresholds, optAseats]
		else: # I-cut-you-freeze
			optAseats = np.zeros((nUB,nUB))
			if thresholds.shape[0] >= nUB: # thresholds file is big enough
				return [thresholds, optAseats]

	# If execution reaches here, must compute all thresholds. 
	print('Computing thresholds for {0} protocol.'.format(protocol['name']))
	# Large negative number for initialization
	INIT_VAL = -100.
	t = np.ones((nUB, nUB)) * INIT_VAL
	# t is the table of thresholds t_{n,j}
	# Large positive value for penalizing impossible seat-share splits
	BIG_VAL = np.iinfo(np.int16).max

	if protocol['abbrev'] == 'B': # bisection
		# Base cases:
		t[0,0] = 0.0
		t[1,0] = 0.0
		t[1,1] = 0.5

		# K := optimum 'a' shares k_{n,j}
		K = np.zeros((nUB, nUB))

		# Populate tables with dynamic programming
		for n in range(2, nUB):
			t[n,0] = 0.0
			a = int(np.floor(n/2))
			b = int(np.ceil(n/2))

			for j in range(1, n+1):
				aCost = a - t[a,a-j+1]
				if j > a:
					aCost = BIG_VAL
				bCost = b - t[b,b-j+1]
				if j > b:
					bCost = BIG_VAL
				minCost = min(aCost, bCost)
				if minCost == aCost:
					K[n,j] = j

				for k in range(1, a+1):
					aCost = BIG_VAL if k > a else a - t[a,a-k+1]
					bCost = BIG_VAL if j-k > b else b - t[b,b-(j-k)+1]
					cost = aCost + bCost
					if cost <= minCost: # Using '<=' to put as much in 'a' side as possible
						minCost = cost
						K[n,j] = k

				t[n,j] = minCost

	else: # I-cut-you-freeze
		for i in range(1, nUB):
			for j in range(1, i + 1):
				t[i,j] = calcICYFThreshold(i, j)

		K = None

	# Save new computed thresholds. 
	# For Bisection, all thresholds are integer multiples of 0.5, 
	# so default precision of 1 is fine, but 
	# ICYF often has repeating decimals, so precision of 10 is reasonable
	thresholdsFilename = 'thresholds{0}_1_to_{1}.csv'.format(protocol['abbrev'], nUB - 1)
	arrayToCSV(t[1:,1:], thresholdsFilename, precision=10)
	print('Saving {0} thresholds.'.format(protocol['name']))

	# For Bisection, also need to save optAseats table.
	if protocol['abbrev'] == 'B':
		print('Saving bisection optAseats.')
		optAseatsFilename = 'optAseatsB_1_to_{0}.csv'.format(nUB - 1)
		# All values are integers, so default precision of 1 is fine.
		arrayToCSV(t[1:,1:], optAseatsFilename)

	return [t, K]


def getDistrictPlanB(n, s, A, t, K):
	"""Recursively computes all district vote-shares
	   under optimal play of the bisection protocol.

	Keyword arguments:
	n -- the number of districts
	s -- the total vote-share of the cutting player
	A -- the index of the cutting player (1 or 2)
	t -- table of thresholds
	K -- table of the optimal number of seats to win from side A

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

	# 2. Find # seats Player A will win from P0, P1.
	seatsFromP0 = int(K[n,j])
	seatsFromP1 = j - seatsFromP0

	# 3. Compute Player B's vote-shares for each side.
	s0 = P0
	if seatsFromP0 > 0:
		s0 = t[P0, P0 - seatsFromP0 + 1]

	s1 = P1
	if seatsFromP1 > 0:
		s1 = t[P1, P1 - seatsFromP1 + 1]

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
	voteShares0 = getDistrictPlanB(P0, s0, B, t, K)
	voteShares1 = getDistrictPlanB(P1, s1, B, t, K)

	return voteShares0 + voteShares1


def getDistrictPlanICYF(n, s, A, t, K):
	"""Recursively compute the vote-shares of all districts 
	   under the I-cut-you-freeze protocol. 
	   Assumes threshold table t is available. 

	Keyword arguments:
	n -- the number of districts
	s -- the total vote-share of Player 1 (always)
	A -- the index of the cutting player (1 or 2)
	t -- table of thresholds
	K -- table of the optimal number of seats to win from side A (unused)

	Returns a list of Player 1 vote-shares in each district. 
	"""	
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
	otherVoteShares = getDistrictPlanICYF(n - 1, s, B, t, K)

	return np.insert(otherVoteShares, 0, voteShare)


def getDistrictPlan(n, s, A, protocolAbbrev, t, K):
	"""Gets the district vote-shares under the given protocol
	   using the appropriate helper function.
	"""
	fnName = 'getDistrictPlan{0}'.format(protocolAbbrev)
	return globals()[fnName](n, s, A, t, K)


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


def calcSainteLagueIndex(voteShares):
	"""Computes the Sainte-Laguë Index of the districting plan, 
	given as a list of Player 1's vote-shares in each district.

	Reference: M. Gallagher. "Proportionality, Disproportionality 
		and Electoral Systems." 1991.
	"""
	n = len(voteShares)
	seatSharePercent1 = sum((x >= 0.5) for x in voteShares) / n
	voteSharePercent1 = sum(voteShares) / n

	seatSharePercent2 = 1 - seatSharePercent1
	voteSharePercent2 = 1 - voteSharePercent1

	ssPercent = [seatSharePercent1, seatSharePercent2]
	vsPercent = [voteSharePercent1, voteSharePercent2]

	return sainteLagueHelper(seatSharesPercent=ssPercent, voteSharesPercent=vsPercent)


def sainteLagueHelper(seatSharesPercent, voteSharesPercent):
	"""Computes the Sainte-Laguë Index given party seat-shares and 
	vote-shares as percentages.
	Reference: M. Gallagher. "Proportionality, Disproportionality 
		and Electoral Systems." 1991.
	"""
	total = 0
	nPartiesSeats = len(seatSharesPercent)
	nPartiesVotes = len(voteSharesPercent)

	if nPartiesSeats != nPartiesVotes:
		raise InputError('Length of seatSharesPercent is {0}, but length of voteSharesPercent is {1}.'.format(nPartiesSeats, nPartiesVotes))

	for i in range(nPartiesSeats):
		diff = seatSharesPercent[i] - voteSharesPercent[i]
		total = total + (diff ** 2 / voteSharesPercent[i])

	return total


def calcPA(thresholds, n):
	"""Computes the partisan asymmetry (PA) of the seats-votes curve
	   f(v), as defined by the given thresholds (t_{n,0} through t_{n,n}). 
	   Returns the area between f(v) and 1-f(1-v).
	"""
	total = 0
	for j in range(1, n + 1):
		total = total + abs(thresholds[j] - (n - thresholds[n - j + 1]))

	# total = total * 1. / (n**2)
	total = total / n
	total = total / n
	return total


def calcCompet(voteShares, threshold=0.05):
	"""Computes the competitiveness of the given districting plan, 
	   measured as the number of highly competitive districts. 
	Keyword arguments:
	voteShares -- a list of Player 1's vote-shares in each district.
	threshold -- maximum margin considered competitive. 
				 Default value is 0.05 (+/- 5% from even 50%/50% split). 
	"""
	gaps = np.abs(np.subtract(voteShares, 0.5))
	return np.count_nonzero(gaps <= threshold)


def calcMeanMedian(voteShares):
	"""Computes the difference between the mean and median
	   vote-shares for Player 1. 
	   The mean-median difference for Player 2 is simply 
	   the negation of that of Player 1. 
	"""
	return np.mean(voteShares) - np.median(voteShares)