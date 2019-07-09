import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import sys

from bisectionUtils import calcEfficiencyGap, calcSainteLagueIndex, csvToArray
from effGapICYC import computeDistrictPlan

######################################################################
# Author: 	Ian Ludden
# Date: 	16 May 2019
# 
# Computes the disproportionality (Sainte-Laguë index) of district
# plans resulting from optimal play of the I-cut-you-choose protocol. 
######################################################################
DEBUG = False

# Load tresholds from default file.
thresholdsFilename = 'icycThresholds_1_to_32.csv'
t = csvToArray(thresholdsFilename)

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
	
	# titleText = 'I-cut-you-choose Protocol: Seat-Share and Proportionality for n = {0}'.format(n)
	titleText = ''

	fig, axarr = plt.subplots(nrows=2, sharex=True, figsize=(8,8))
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
	axarr[0].set_yticks(np.arange(0, 1.25, step=0.25))
	axarr[1].plot(normalizedS, sainteLaguesPercent)
	axarr[1].set(xlabel='Vote-share', ylabel='Sainte-Laguë Index (%)')
	axarr[1].set_yticks(np.arange(0, 35, step=5))
	axarr[0].grid()
	axarr[1].grid()
	plt.xticks(np.arange(0, 1.25, step=0.25))

	# Change font sizes
	for ax in axarr:
		for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + 
					 ax.get_xticklabels() + ax.get_yticklabels()):
			item.set_fontsize(16)

	fig.savefig('slICYF{0}.pdf'.format(n))
	plt.show()




