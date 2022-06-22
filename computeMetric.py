import json
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import seaborn as sns

import sys

import utils

######################################################################
# Author: 	Ian Ludden
# Date: 	11 July 2019
# 
# Computes the given metric of the given protocol across 
# all vote-shares for the given settings. 
# Optionally saves the data to a csv file and shows/saves a plot. 
# See settings.json for examples of valid settings. 
######################################################################
N_MAX = 150
VERBOSE = True
FIG_DIR = 'plots/'

protocols = {
	'B': {
		'name': 'Bisection',
		'abbrev': 'B',
		'defaultThresholdsFilename': 'thresholdsB_1_to_151.csv', 
		'defaultOptAFilename': 'optAseatsB_1_to_151.csv'
	}, 
	'ICYF': {
		'name': 'I-cut-you-freeze',
		'abbrev': 'ICYF',
		'defaultThresholdsFilename': 'thresholdsICYF_1_to_151.csv', 
		'defaultOptAFilename': None
	}
}

metrics = {
	'EG': {
		'name': 'Efficiency Gap',
		'fn': 'calcEfficiencyGap',
		'units': '(%)',
		'yticks': [-50, -25, -15, -8, 0, 8, 15, 25, 50],
		'ymin': -50,
		'ymax': 50, 
		'cmap': 'RdBu_r', 
		'norm': colors.BoundaryNorm(boundaries=[-50, -25, -15, -8, -4, 4, 8, 15, 25, 50], ncolors=256)
	},
	'MM': {
		'name': 'Mean-Median Difference',
		'fn': 'calcMeanMedian',
		'units': '(%)',
		'yticks': [-50, -25, 0, 25, 50],
		'ymin': -50,
		'ymax': 50
	},
	'PA': {
		'name': 'Partisan Asymmetry',
		'fn': 'calcPA',
		'units': None,
		'yticks': np.linspace(0, 0.50, 6),
		'ymin': 0,
		'ymax': 0.4, 
		'cmap': 'Reds', 
		'norm': colors.BoundaryNorm(boundaries=[0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40], ncolors=256)
	},
	'SL': {
		'name': 'Sainte-Laguë Index',
		'fn': 'calcSainteLagueIndex',
		'units': '(%)',
		'yticks': np.arange(0, 65, step=5),
		'ymin': 0,
		'ymax': 100, 
		'cmap': 'Greys', 
		'norm': colors.BoundaryNorm(boundaries=[0, 0.5, 1, 2, 5, 25, 100], ncolors=256)
	},
	'CP': {
		'name': 'Competitiveness',
		'fn': 'calcCompet',
		'units': '(No. Districts)',
		'yticks': lambda n: np.arange(0, (np.ceil(n / 5) + 1) * 5, step=5), 
		'ymin': -2,
		'ymax': lambda n: n
	}
}


if __name__ == '__main__':
	if len(sys.argv) < 2:
		exit('Not enough arguments. Must provide settings filename.')

	settingsFilepath = str(sys.argv[1])
	with open(settingsFilepath, 'r') as f:
		settings = json.load(f)['settings']

	# If provided, third argument is a specific setting name to use. 
	if len(sys.argv) >= 3:
		settingName = str(sys.argv[2])
	else:
		settingName = None

	# Adjust font settings
	plt.rcParams['font.family'] = ['Arial', 'sans-serif']
	plt.rcParams['font.size'] = 12

	for setting in settings:
		if settingName is not None and settingName != setting['name']:
			continue

		try:
			protocolAbbrev = setting['protocol']
			metricAbbrev = setting['metric']
		except KeyError:
			exit('Setting {0} is missing protocol and/or metric name.'.format(setting['name']))

		try:
			protocol = protocols[protocolAbbrev]
		except KeyError:
			exit('Setting {0} has invalid protocol name: {1}.'.format(setting['name'], setting['protocol']))

		try:
			metric = metrics[metricAbbrev]
		except KeyError:
			exit('Setting {0} has invalid metric name: {1}.'.format(setting['name'], setting['metric']))

		try:
			n = setting['n']
		except KeyError:
			exit('Setting {0} is missing number of districts (n).'.format(setting['name']))

		# Read in tables of thresholds/optimal seats to win from smaller side.
		t, K = utils.getThresholds(protocol, n)

		# If provided, packing_delta is the packing parameter (and toggles its use).
		isCompressed = 'packing_delta' in setting.keys()
		if isCompressed:
			delta = float(setting['packing_delta'])
			gamma = 0.5 - delta # Lower bound on vote-share in district


		# Vote-shares for which to compute metric values
		voteShares = setting['vote-shares']

		# Range of numbers of districts
		nSweep = np.arange(1, N_MAX + 1)
		
		# Player 1 always goes first
		A = 1

		metricVals = np.zeros((len(voteShares), len(nSweep)))
		for v_i, voteShare in enumerate(voteShares):
			for n_i in range(len(nSweep)):
				try:
					voteShareRaw = voteShare * nSweep[n_i]
					if isCompressed:
						# Scale down voteShareRaw so total is a voteShare fraction after later compression
						voteShareRaw -= gamma * nSweep[n_i]
						voteShareRaw /= (1 - 2 * gamma)

					districtVoteShares = utils.getDistrictPlan(nSweep[n_i], voteShareRaw, A, protocolAbbrev, t, K)
					if isCompressed:
						# Compress vote-shares to appropriate range (0.5 +/- delta)
						for j in range(len(districtVoteShares)):
							districtVoteShares[j] = districtVoteShares[j] * (1 - 2 * gamma) + gamma

					metricFn = getattr(utils, metric['fn'])
					if metricAbbrev == 'CP' and 'compet_threshold' in setting.keys():
						metricVals[v_i, n_i] = metricFn(districtVoteShares, threshold=setting['compet_threshold'])
					else:
						metricVals[v_i, n_i] = metricFn(districtVoteShares)
				except ValueError:
					print('Error computing district plan for n={0}, vote-share={1:.5f}.'.format(n_i, voteShare))
					raise

		filename = '{0}_{1}_{2}'.format(protocolAbbrev, metricAbbrev, N_MAX)
		if isCompressed:
			filename = '{0}_{1}_{2:.2f}'.format(filename, 'packing', delta)

		if metric['units'] == '(%)':
			metricVals = metricVals * 100.

		if setting['save_csv'] == 1:
			utils.arrayToCSV(metricVals, '{0}.csv'.format(filename), precision=4)

		# Plot Seat-share and Metric Vals in separate plots
		# (since y-axis is different for each)
		yThresholds = np.array(range(n+1))
		yThresholds = np.repeat(yThresholds, 2)

		xThresholds = np.repeat(t[n,:n+1], 2)
		xThresholds = xThresholds[2:]
		xThresholds = np.insert(xThresholds, 0, 0)
		xThresholds = np.append(xThresholds, n)

		if isCompressed:
			xThresholds = xThresholds * (1 - 2 * gamma) + (gamma * n)

		titleText = '{0} vs. Vote-share, n = {1}'.format(metric['name'], n)
		titleText = ''

		if setting['show_seats-votes'] == 1:
			raise NotImplementedError("The show_seats-votes flag has been removed.")
		
		fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(8,6))

		data_array = np.array(metricVals)

		# Add dummy zero column to shift xtick labels by 1 
		# (1 to 150 districts, not 0 to 149)
		blank_col = np.zeros((data_array.shape[0], 1))
		data_array = np.hstack((blank_col, data_array))

		# Flip so vote-shares increase from bottom to top
		np.flip(data_array, axis=0)
		
		# Compare observed min/max values to preset cutoffs
		if VERBOSE:
			print(', '.join([setting['name'], setting['protocol'], metricAbbrev]))
			print('Chosen vmin/vmax:\t', metric['ymin'], metric['ymax'])
			print('Data vmin/vmax:  \t', data_array.min(), data_array.max())
			print()

		center_val = 0 if metricAbbrev == 'EG' else None

		cmap_val = metric['cmap']
		norm_val = metric['norm']

		xtick_labels = 10 # Print district counts in increments of 10
		ytick_labels = np.flip(voteShares)
		ytick_labels = ['{:d}%'.format(int(100 * yt)) for yt in ytick_labels]

		cbar_label = '{} {}'.format(metric['name'], metric['units'] if metric['units'] else '')
		sns.heatmap(data_array, vmin=metric['ymin'], vmax=metric['ymax'], ax=ax,\
			center=center_val, cmap=cmap_val,\
			xticklabels=xtick_labels, yticklabels=ytick_labels,\
			norm=norm_val,\
			cbar_kws={'label': cbar_label})
		plt.xlabel('No. Districts')
		plt.ylabel('Player 1 Vote-share')

		if setting['save_plot'] == 1:
			fig.savefig('{0}{1}.pdf'.format(FIG_DIR, filename), bbox_inches='tight')

		if setting['show_plot'] == 1:
			plt.title(', '.join([setting['name'], setting['protocol'], metricAbbrev]))
			plt.show()


