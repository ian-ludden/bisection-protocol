import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import sys

from bisectionUtils import calcEfficiencyGap, calcSainteLagueIndex, csvToArray
from effGap import computeDistrictPlan

######################################################################
# Author: 	Ian Ludden
# Date: 	16 May 2019
# 
# Computes the disproportionality (Sainte-Laguë index) of district
# plans resulting from optimal play of the bisection protocol. 
######################################################################
DEBUG = False

# Load default files with thresholds and opt seats.
thresholdsFilename = 'thresholds_1_to_32.csv'
optAseatsFilename = 'optAseats_1_to_32.csv'

# Read in tables of t_{n,j} and K_{n,j} as defined in bisectionDP.py.
t = csvToArray(thresholdsFilename)
K = csvToArray(optAseatsFilename)

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

	sainteLagues= np.zeros(sSweep.shape)
	for i in range(len(sSweep)):
		try:
			voteShares = computeDistrictPlan(n, sSweep[i], A)
		except ValueError:
			print('Error computing district plan for n={0}, s={1:.5f}.'.format(n, sSweep[i]))
			raise
		sainteLagues[i] = calcSainteLagueIndex(voteShares)

	normalizedS = (np.concatenate(([0.0], sSweep, [n]))) / n

	# If s = 0 or s = n, then Sainte-Laguë index is 0.
	sainteLaguesPercent = (np.concatenate(([0], sainteLagues, [0]))) * 100.0
	
	titleText = 'Bisection Protocol: Seat-Share and Proportionality for n = {0}'.format(n)

	fig, axarr = plt.subplots(nrows=2, sharex=True)
	fig.suptitle(titleText)

	# Plot Seat-share and Sainte-Laguë Index in separate plots
	# (since y-axis is different for each)
	yThresholds = np.array(range(n+1))
	yThresholds = np.repeat(yThresholds, 2)

	xThresholds = np.repeat(t[n,:n+1], 2)
	xThresholds = xThresholds[2:]
	xThresholds = np.insert(xThresholds, 0, 0)
	xThresholds = np.append(xThresholds, n)

	axarr[0].plot(xThresholds/n, yThresholds/n)
	axarr[0].set(ylabel='Seat-share')
	axarr[1].plot(normalizedS, sainteLaguesPercent)
	axarr[1].set(xlabel='Fractional Vote-share', ylabel='Sainte-Laguë Index (%)')
	axarr[0].grid()
	axarr[1].grid()
	fig.savefig('plotSLIndexAndSeatShareBisection_{0}_res{1}.png'.format(n,resolution))
	plt.show()




