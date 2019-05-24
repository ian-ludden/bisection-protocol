import numpy as np
from pprint import pprint
from scipy.special import factorial2

from bisectionUtils import arrayToCSV

######################################################################
# Author: 	Ian Ludden
# Date: 	16 May 2019
# 
# Implements Theorem 2.4 from Pegden et al. (2017) to compute the 
# thresholds t_{n,j} (my definition; they call it the minimum s
# such that \sigma(n,s) >= k). 
# 
# Source: https://arxiv.org/abs/1710.08781
######################################################################

def calcThreshold(n, k):
	"""Computes the vote-share needed to win k of n districts."""
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


# The maximum number of districts
nUB = 54

t = np.zeros((nUB, nUB))
for i in range(1, nUB):
	for j in range(1, i + 1):
		t[i,j] = calcThreshold(i, j)

arrayToCSV(t[1:,1:], 'icycThresholds_1_to_{0}.csv'.format(nUB-1))