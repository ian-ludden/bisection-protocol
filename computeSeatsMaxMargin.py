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

	print('State,Party Drawing First,Predicted Seat-Shares')
	print(',,ICYF,Bisection')

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

		seatShares = []
		for index in range(len(voteShares)):
			seatShares.append(np.sum(np.array(voteShares[index]) >= 0.5))

		print('{2},R,{0:.2f}%,{1:.2f}%'.format(seatShares[0] * 100. / n, seatShares[1] * 100. / n, stateName))
		print('{2},D,{0:.2f}%,{1:.2f}%'.format(100. - seatShares[2] * 100. / n, 100. - seatShares[3] * 100. / n, stateName))

		# Increment index by 2 to go to next state
		i += 2




