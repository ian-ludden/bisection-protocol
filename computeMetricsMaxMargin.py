import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import sys
import utils

######################################################################
# Author: 	Ian Ludden
# Date: 	22 July 2019
# 
# computeSeatsMaxMargin.py
# 
# Computes the predicted seat-shares for different states under 
# different max margin packing constraints. 
######################################################################
DEFAULT_RESOLUTION = 53

protocols = {
	'B': {
		'name': 'Bisection',
		'abbrev': 'B',
		'defaultThresholdsFilename': 'thresholdsB_1_to_54.csv', 
		'defaultOptAFilename': 'optAseatsB_1_to_54.csv'
	}, 
	'ICYF': {
		'name': 'I-cut-you-freeze',
		'abbrev': 'ICYF',
		'defaultThresholdsFilename': 'thresholdsICYF_1_to_54.csv', 
		'defaultOptAFilename': None
	}
}

metrics = {
	'EG': {
		'name': 'Efficiency Gap',
		'fn': 'calcEfficiencyGap',
		'units': '(%)',
		'val_at_zero': -0.5,
		'val_at_n': 0.5,
		'yticks': [-50, -25, -8, 0, 8, 25, 50],
		'ymin': -50,
		'ymax': 50
	},
	'MM': {
		'name': 'Mean-Median Difference',
		'fn': 'calcMeanMedian',
		'units': '(%)',
		'val_at_zero': 0.,
		'val_at_n': 0.,
		'yticks': [-50, -25, 0, 25, 50],
		'ymin': -50,
		'ymax': 50
	},
	'PA': {
		'name': 'Partisan Asymmetry',
		'fn': 'calcPA',
		'units': None,
		'val_at_zero': -0.5,
		'val_at_n': 0.5,
		'yticks': np.linspace(0, 0.5, 6),
		'ymin': 0,
		'ymax': 0.5
	},
	'SL': {
		'name': 'Sainte-Laguë Index',
		'fn': 'calcSainteLagueIndex',
		'units': '(%)',
		'val_at_zero': 0.0,
		'val_at_n': 0.0,
		'yticks': np.arange(0, 65, step=5),
		'ymin': 0,
		'ymax': 60
	},
	'CP': {
		'name': 'Competitiveness',
		'fn': 'calcCompet',
		'units': '(No. Districts)',
		'val_at_zero': 0,
		'val_at_n': 0,
		'yticks': lambda n: np.arange(0, (np.ceil(n / 5) + 1) * 5, step=5), 
		'ymin': -2,
		'ymax': lambda n: n
	}
}

if __name__ == '__main__':
	if len(sys.argv) < 2:
		exit('Not enough arguments. Must provide input filename.')

	inputFilepath = str(sys.argv[1])
	with open(inputFilepath, 'r') as f:
		reader = csv.reader(f)
		headers1 = next(reader)
		headers2 = next(reader)
		data = list(reader)

	tB, KB = utils.getThresholds(protocols['B'], 53)
	tIcyf, KIcyf = utils.getThresholds(protocols['ICYF'], 53)

	print('State,Party Drawing First,Predicted R Seat-Share,,Sainte-Laguë index,,Efficiency Gap,,No. Competitive Districts (5% margin),,')
	print(',,ICYF,Bisection,ICYF,Bisection,ICYF,Bisection,ICYF,Bisection,')

	i = 0
	while i < len(data) - 1:
		# pprint(data[i])
		# pprint(data[i+1])
		row = data[i]
		stateName = str(row[1])
		# Number of districts
		n = int(row[2])
		# Republican vote-share
		sR = float(row[3]) * n
		# Max margins for Republicans drawing first
		maxMarginIcyfR = float(row[5])
		maxMarginBisectionR = float(row[6])

		# Next row is margins for Democrats drawing first
		row2 = data[i + 1]
		# Democrat vote-share
		sD = n - sR
		# Max margins for Republican
		maxMarginIcyfD = float(row[5])
		maxMarginBisectionD = float(row[6])

		# Increase resolution to make plot more accurate
		resolution = DEFAULT_RESOLUTION

		# Range of vote-share values is slightly offset to avoid 
		# ambiguities of landing exactly on a threshold
		sSweep = np.linspace(0.101, n-0.099, resolution*n)
		
		# Player 1 always goes first
		A = 1

		# For gammas and voteShares:
		# [0] == ICYF R
		# [1] == Bisection R
		# [2] == ICYF D
		# [3] == Bisection D
		gammas = []
		gammas.append(0.5 - maxMarginIcyfR)
		gammas.append(0.5 - maxMarginBisectionR)
		gammas.append(0.5 - maxMarginIcyfD)
		gammas.append(0.5 - maxMarginBisectionD)
		voteShares = []
		voteShares.append(utils.getDistrictPlan(n, (sR - gammas[0] * n) / (1 - 2 * gammas[0]), A, 'ICYF', tIcyf, KIcyf))
		voteShares.append(utils.getDistrictPlan(n, (sR - gammas[1] * n) / (1 - 2 * gammas[1]), A, 'B', tB, KB))
		voteShares.append(utils.getDistrictPlan(n, (sD - gammas[2] * n) / (1 - 2 * gammas[2]), A, 'ICYF', tIcyf, KIcyf))
		voteShares.append(utils.getDistrictPlan(n, (sD - gammas[3] * n) / (1 - 2 * gammas[3]), A, 'B', tB, KB))

		# Compress vote-shares to appropriate range (0.5 +/- max margin)
		for index in range(len(voteShares)):
			for j in range(len(voteShares[index])):
				voteShares[index][j] = voteShares[index][j] * (1 - 2 * gammas[index]) + gammas[index]

		# Compute seat-shares as percentages
		seatShares = []
		for index in range(len(voteShares)):
			origSeatShare = np.sum(np.array(voteShares[index]) >= 0.5)
			seatShares.append(origSeatShare * 100. / n)

		# Compute Sainte-Laguë indices, Efficiency Gaps, and No. Competitive Districts
		slIndices = []
		effGaps = []
		competCounts = []
		# TODO: Add partisan asymmetry?
		for index in range(len(voteShares)):
			slIndices.append(utils.calcSainteLagueIndex(voteShares[index]) * 100.)
			effGaps.append(utils.calcEfficiencyGap(voteShares[index]) * 100.)
			competCounts.append(utils.calcCompet(voteShares[index]))

		# Flip EG for R, seat-shares for D
		print('{0},R,{1:.2f}%,{2:.2f}%,{3:.4f}%,{4:.4f}%,{5:.2f}%,{6:.2f}%,{7},{8}'.format(stateName, seatShares[0], seatShares[1], slIndices[0], slIndices[1], -1. * effGaps[0], -1. * effGaps[1], competCounts[0], competCounts[1]))
		print('{0},D,{1:.2f}%,{2:.2f}%,{3:.4f}%,{4:.4f}%,{5:.2f}%,{6:.2f}%,{7},{8}'.format(stateName, 100. - seatShares[2], 100. - seatShares[3], slIndices[2], slIndices[3], effGaps[2], effGaps[3], competCounts[2], competCounts[3]))

		# Increment index by 2 to go to next state
		i += 2




