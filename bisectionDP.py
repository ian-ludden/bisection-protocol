import numpy as np

######################################################################
# Author: 	Ian Ludden
# Date: 	02 May 2019
# 
# Implements a DP for computing all the thresholds t_{n,j}, as well as
# the optimum number of seats K_{n,j} to win from the smaller piece. 
######################################################################

INIT_VAL = -100
BIG_VAL = np.iinfo(np.int16).max

nUB = 33

# t is the table of thresholds t_{n,j}
t = INIT_VAL*np.ones((nUB,nUB)) # Initialize to large negative number
t[0,0] = 0.0
t[1,0] = 0.0
t[1,1] = 0.5

# K := optimum 'a' shares k_{n,j}
K = np.zeros((nUB,nUB))

for n in range(2,nUB):
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

thresholds = t[1:,1:]
optAseats = K[1:,1:]

np.savetxt('thresholds_1_to_{0}.csv'.format(nUB-1), thresholds, fmt='%.1f', delimiter=',')
np.savetxt('optAseats_1_to_{0}.csv'.format(nUB-1), optAseats, fmt='%.1f', delimiter=',')

