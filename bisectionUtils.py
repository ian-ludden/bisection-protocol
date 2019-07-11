import csv
import numpy as np

######################################################################
# Author: 	Ian Ludden
# Date: 	14 May 2019
# 
# Utility functions for scripts related to the bisection protocol.  
######################################################################

def csvToArray(filename):
	"""Reads the given CSV file into an array."""
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

def arrayToCSV(A, filename):
	"""Saves the given array to a CSV file with the given name."""
	np.savetxt(filename, A, fmt='%.1f', delimiter=',')	

def isCloseEnough(a, b):
	"""Returns True if a and b are within epsilon of each other."""
	EPSILON = 1e-10
	return abs(a - b) <= EPSILON

def listsMatch(l1, l2):
	"""Returns True if the lists are approx. equal when sorted."""
	if len(l1) != len(l2):
		return False

	l1.sort()
	l2.sort()

	for i in range(len(l1)):
		if not isCloseEnough(l1[i], l2[i]):
			return False

	return True

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

def calcCompet(voteShares, metric='max_margin', threshold=0.05):
	"""Computes the competitiveness of the given districting plan. 
	Keyword arguments:
	voteShares -- a list of Player 1's vote-shares in each district
	metric -- either 'max_margin' for maximum margin (absolute value) or 
			  'count_compet' for number of competitive districts. 
			  Default value is 'max_margin'.
	threshold -- maximum margin considered competitive. 
				 Only used for 'count_compet' metric. 
				 Default value is 0.05 (+/- 5% from even 50%/50% split). 
	"""
	gaps = np.abs(np.subtract(voteShares, 0.5))
	# print('total: {0:.2f}'.format(np.sum(voteShares)))
	# print('vote-shares: {0}'.format(voteShares))
	# print('gaps: {0}\n'.format(gaps))
	if metric == 'max_margin':
		return np.max(gaps)
	else:
		return np.count_nonzero(gaps <= threshold)

def calcMeanMedian(voteShares):
	"""Computes the difference between the mean and median
	   vote-shares for Player 1. 
	   The mean-median difference for Player 2 is simply 
	   the negation of that of Player 1. 
	"""
	return np.mean(voteShares) - np.median(voteShares)