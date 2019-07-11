import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import sys

from bisectionUtils import calcCompet, csvToArray
from effGap import computeDistrictPlan

######################################################################
# Author: 	Ian Ludden
# Date: 	09 July 2019
# 
# Computes the competitiveness of district plans resulting from 
# optimal "play" of the bisection protocol. 
######################################################################
DEBUG = False
isCompressed = False # Flag for packing constraint

competMetric = 'count_compet' # See calcCompet function in bisectionUtils
competThreshold = 0.05 # Maximum margin considered competitive

# Load default files with thresholds and opt seats.
thresholdsFilename = 'thresholds_1_to_53.csv'
optAseatsFilename = 'optAseats_1_to_53.csv'

# Read in tables of t_{n,j} and K_{n,j} as defined in bisectionDP.py.
t = csvToArray(thresholdsFilename)
K = csvToArray(optAseatsFilename)

if __name__ == '__main__':
	if len(sys.argv) < 2:
		exit('Not enough arguments.')
	n = int(sys.argv[1])

	# If provided, third argument is the packing parameter (and toggles its use).
	if len(sys.argv) >= 3:
		isCompressed = True
		delta = float(sys.argv[2])
		gamma = 0.5 - delta # Lower bound on vote-share in district

	# Increase to make plot more accurate
	resolution = 53
	
	# Range of vote-share values is slightly offset to avoid 
	# ambiguities of landing exactly on a threshold
	sSweep = np.linspace(0.0101, n-0.0099, resolution*n)
	A = 1

	competVals = np.zeros(sSweep.shape)
	for i in range(len(sSweep)):
		try:
			voteShares = computeDistrictPlan(n, sSweep[i], A)
			if isCompressed:
				for j in range(len(voteShares)):
					voteShares[j] = voteShares[j] * (1 - 2 * gamma) + gamma
			competVals[i] = calcCompet(voteShares, competMetric, competThreshold)
		except ValueError:
			print('Error computing district plan for n={0}, s={1:.5f}.'.format(n, sSweep[i]))
			raise

	normalizedS = sSweep / n
	if isCompressed:
		normalizedS = normalizedS * (1 - 2 * gamma) + gamma

	# titleText = 'Competitiveness vs. Vote-share, n = {0}'.format(n)
	titleText = ''

	fig, axarr = plt.subplots(nrows=2, sharex=True, figsize=(8,8))
	fig.suptitle(titleText)

	# Plot Seat-share and Competitiveness in separate plots
	# (since y-axis is different for each)
	yThresholds = np.array(range(n+1))
	yThresholds = np.repeat(yThresholds, 2)

	xThresholds = np.repeat(t[n,:n+1], 2)
	xThresholds = xThresholds[2:]
	xThresholds = np.insert(xThresholds, 0, 0)
	xThresholds = np.append(xThresholds, n)

	if isCompressed:
		xThresholds = xThresholds * (1 - 2 * gamma) + (gamma * n)

	axarr[0].plot(xThresholds/n, yThresholds/n)
	axarr[0].set(ylabel='Seat-share')
	axarr[0].set_yticks(np.arange(0, 1.25, step=0.25))
	axarr[0].set_xticks(np.arange(0, 1.25, step=0.25))
	axarr[1].plot(normalizedS, competVals)
	if competMetric == 'max_margin':
		axarr[1].set(xlabel='Vote-share', ylabel='Maximum Margin')
		axarr[1].set_ylim(bottom=0, top=0.5)
	else:
		axarr[1].set(xlabel='Vote-share', ylabel='No. Competitive Districts')
		axarr[1].set_ylim(bottom=-n, top=n)
	# yticks1 = np.arange(-n, n+1)
	# axarr[1].set_yticks(yticks1)
	axarr[1].set_xticks(np.arange(0, 1.25, step=0.25))
	axarr[0].grid()
	axarr[1].grid()
	plt.xlim(0, 1)

	# Change font sizes
	for ax in axarr:
		for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + 
					 ax.get_xticklabels() + ax.get_yticklabels()):
			item.set_fontsize(16)

	if isCompressed:
		fig.savefig('competPackingB{0}quarter.pdf'.format(n))
	else:
		fig.savefig('competB{0}.pdf'.format(n))

	plt.show()
