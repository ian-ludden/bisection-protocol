import json
import matplotlib
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

def computePA(setting):
	"""Computes the partisan asymmetry of both protocols for 
	   each number of districts up to the given n.
	   Supports packing constraint.
	"""
	n = setting['n']
	isCompressed = 'packing_delta' in setting.keys()
	if isCompressed:
		delta = float(setting['packing_delta'])
		gamma = 0.5 - delta # Lower bound on vote-share in district

	bPAs = np.zeros(n + 1)
	icyfPAs = np.zeros(n + 1)

	bThresholds, bOptAseats = utils.getThresholds(protocols['B'], n)
	icyfThresholds, icyfOptAseats = utils.getThresholds(protocols['ICYF'], n)

	for i in range(1, n + 1):
		bPAs[i] = utils.calcPA(bThresholds[i,:(i+1)], i)
		icyfPAs[i] = utils.calcPA(icyfThresholds[i,:(i+1)], i)

	xVals = np.arange(1, n + 1)

	filename = '{0}_{1}'.format(metricAbbrev, n)
	if isCompressed:
		filename = '{0}_{1}_{2:.2f}'.format(filename, 'packing', delta)

	if setting['save_csv'] == 1:
		utils.arrayToCSV(np.array([xVals, bPAs[1:], icyfPAs[1:]]).T, '{0}.csv'.format(filename), precision=4)

	# TODO: Plot
	# TODO: Handle packing constraint (compress)

	if setting['save_plot'] == 1:
		fig.savefig('{0}.pdf'.format(filename))

	if setting['show_plot'] == 1:
		plt.show()



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

		if metricAbbrev == 'PA':
			# PA is handled differently, since each n generates a single value
			# and we want to plot results for both protocols together. 
			computePA(setting)
			exit()

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


		# Increase resolution to make plot more accurate
		resolution = setting.get('res', DEFAULT_RESOLUTION)

		# Range of vote-share values is slightly offset to avoid 
		# ambiguities of landing exactly on a threshold
		sSweep = np.linspace(0.101, n-0.099, resolution*n)
		
		# Player 1 always goes first
		A = 1

		metricVals = np.zeros(sSweep.shape)
		for i in range(len(sSweep)):
			try:
				voteShares = utils.getDistrictPlan(n, sSweep[i], A, protocolAbbrev, t, K)
				if isCompressed:
					# Compress vote-shares to appropriate range (0.5 +/- delta)
					for j in range(len(voteShares)):
						voteShares[j] = voteShares[j] * (1 - 2 * gamma) + gamma

				metricFn = getattr(utils, metric['fn'])
				if metricAbbrev == 'CP' and 'compet_threshold' in setting.keys():
					metricVals[i] = metricFn(voteShares, threshold=setting['compet_threshold'])
				else:
					metricVals[i] = metricFn(voteShares)
			except ValueError:
				print('Error computing district plan for n={0}, s={1:.5f}.'.format(n, sSweep[i]))
				raise

		# Normalize to put vote-shares between 0 and 1, and add 0 and n to make plot pretty
		normalizedS = (np.concatenate(([0.0], sSweep, [n]))) / n
		if isCompressed:
			# Need to scale the normalized vote-shares too
			normalizedS = normalizedS * (1 - 2 * gamma) + gamma

		metricVals = np.concatenate(([metric['val_at_zero']], metricVals, [metric['val_at_n']]))

		filename = '{0}_{1}_{2}'.format(protocolAbbrev, metricAbbrev, n)
		if isCompressed:
			filename = '{0}_{1}_{2:.2f}'.format(filename, 'packing', delta)

		if metric['units'] == '(%)':
			metricVals = metricVals * 100

		if setting['save_csv'] == 1:
			utils.arrayToCSV(np.transpose(np.array([normalizedS, metricVals])), '{0}.csv'.format(filename), precision=4)

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
			fig, axarr = plt.subplots(nrows=2, sharex=True, figsize=(8,8))
			fig.suptitle(titleText)
			axarr[0].plot(xThresholds/n, yThresholds/n)
			axarr[0].set(ylabel='Seat-share')
			axarr[0].set_yticks(np.arange(0, 1.25, step=0.25))
			axarr[0].set_xticks(np.arange(0, 1.25, step=0.25))
			axarr[1].plot(normalizedS, metricVals)
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
			axarr[1].set_xticks(np.arange(0, 1.25, step=0.25))
			axarr[0].grid()
			axarr[1].grid()
			plt.xlim(0, 1)

			# Change font sizes
			for ax in axarr:
				for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + 
							 ax.get_xticklabels() + ax.get_yticklabels()):
					item.set_fontsize(16)
		else:
			fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(8,4))
			ax.plot(normalizedS, metricVals)
			ylabelMetric = '{0} {1}'.format(metric['name'], metric['units'])
			ax.set(xlabel='Vote-share', ylabel=ylabelMetric)
			yticks1 = metric['yticks']
			if metricAbbrev == 'CP':
				yticks1 = yticks1(n)
			ax.set_yticks(yticks1)
			ymin = metric['ymin']
			ymax = metric['ymax']
			if metricAbbrev == 'CP':
				ymax = ymax(n)
			ax.set_ylim(bottom=ymin, top=ymax)
			ax.set_xticks(np.arange(0, 1.25, step=0.25))
			ax.grid()
			plt.xlim(0, 1)
			plt.gcf().subplots_adjust(bottom=0.2)

			for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + 
						 ax.get_xticklabels() + ax.get_yticklabels()):
				item.set_fontsize(16)

		if setting['save_plot'] == 1:
			fig.savefig('{0}.pdf'.format(filename))

		if setting['show_plot'] == 1:
			plt.show()



