import json
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
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
		exit('Not enough arguments. Must provide settings filename.')

	settingsFilepath = str(sys.argv[1])
	with open(settingsFilepath, 'r') as f:
		settings = json.load(f)['settings']

	# If provided, third argument is a specific setting name to use. 
	if len(sys.argv) >= 3:
		settingName = str(sys.argv[2])
	else:
		settingName = None

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
			fig, axarr = plt.subplots(nrows=2, sharex=True, figsize=(8,9))
			fig.suptitle(titleText)
			axarr[0].plot(xThresholds/n, yThresholds/n)
			axarr[0].set(ylabel='Seat-share')
			axarr[0].set_yticks(np.arange(0, 1.25, step=0.25))
			axarr[0].set_xticks(np.arange(0, 1.25, step=0.25))
			for v_i in range(len(voteShares)):
				axarr[1].plot(nSweep, metricVals[v_i, :], 'o')
			ylabelMetric = '{0} {1}'.format(metric['name'], metric['units'])
			axarr[1].set(xlabel='Vote-share', ylabel=ylabelMetric)
			yticks1 = metric['yticks']
			if metricAbbrev == 'CP':
				yticks1 = yticks1(n)
			axarr[1].set_yticks(yticks1)
			ymin = metric['ymin']
			ymax = metric['ymax']
			if metricAbbrev == 'CP':
				ymax = ymax(n)
			axarr[1].set_ylim(bottom=ymin, top=ymax)
			axarr[1].set_xticks(np.arange(0, N_MAX + 5, step=N_MAX // 10))
			axarr[0].grid()
			axarr[1].grid()
			axarr[0].set_xlim(0, 1)
			axarr[1].set_xlim(0, N_MAX + 2)
			lg = axarr[1].legend(['{:.1f}%'.format(voteShare * 100.) for voteShare in voteShares], fontsize=16, loc='upper right', ncol=2)
			lg.set_title("Player 1 Vote-Share", prop={'size': 16})

			if metricAbbrev == 'EG':
				axarr[1].fill_between(nSweep, -8, 8, alpha=0.15)

			# Change font sizes
			for ax in axarr:
				for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + 
							 ax.get_xticklabels() + ax.get_yticklabels()):
					item.set_fontsize(16)
		else:
			fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(8,6))
			markers = ['o', 'x', '+', '^', 'd', '*', 's']
			for v_i in range(len(voteShares)):
				ax.plot(nSweep, metricVals[v_i, :], markers[v_i % len(markers)])
			ylabelMetric = '{0} {1}'.format(metric['name'], metric['units'])
			ax.set(xlabel='No. Districts', ylabel=ylabelMetric)
			yticks1 = metric['yticks']
			if metricAbbrev == 'CP':
				yticks1 = yticks1(n)
			ax.set_yticks(yticks1)
			ymin = metric['ymin']
			ymax = metric['ymax']
			if metricAbbrev == 'CP':
				ymax = ymax(n)
			ax.set_ylim(bottom=ymin, top=ymax)
			ax.set_xticks(np.arange(0, N_MAX + 5, step=N_MAX // 10))
			ax.grid()
			lg = ax.legend(['{:.2f}%'.format(voteShare * 100.) for voteShare in voteShares], fontsize=16, loc='upper right', ncol=2)
			lg.set_title("Player 1 Vote-Share", prop={'size': 16})
			plt.xlim(0, N_MAX + 2)
			plt.gcf().subplots_adjust(bottom=0.2)

			if metricAbbrev == 'EG':
				ax.fill_between([0, N_MAX + 5], -8, 8, alpha=0.15)

			for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + 
						 ax.get_xticklabels() + ax.get_yticklabels()):
				item.set_fontsize(16)

		if setting['save_plot'] == 1:
			fig.savefig('{0}.pdf'.format(filename), bbox_inches='tight')

		if setting['show_plot'] == 1:
			plt.show()



