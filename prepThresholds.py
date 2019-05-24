import numpy as np
import sys
import csv

######################################################################
# Author: 	Ian Ludden
# Date: 	08 May 2019
# 
# This script takes as input the thresholds produced by bisectionDP.py 
# for the n-district bisection protocol. It outputs Mathematica code
# for representing seat-share as a function of vote-share and plotting 
# this function with its mirror. This allows for the partisan
# asymmetry of the protocol to be computed. 
######################################################################

filename = str(sys.argv[1])
with open(filename, 'r') as csvFile:
	reader = csv.reader(csvFile)
	n = 32
	for row in reader:
		sys.stdout.write('g[x_]:=(')
		n = n + 1
		sys.stdout.write('HeavisideTheta[x-{0}/{1}]'.format(row[0], n))
		for index in range(1, n):
			threshold = row[index]
			sys.stdout.write('+HeavisideTheta[x-{0}/{1}]'.format(threshold, n))
		sys.stdout.write(')/{0}\n'.format(n))

		# Print code for plotting g(x) with 1-g(1-x)
		sys.stdout.write('Plot[{{{{{0}g[x/{0}]}},{0} - {0} g[1-x/{0}] }},{{x,0,{0}}},Exclusions->{{False}}]\n'.format(n))

		# Print code for integrating to find the partisan asymmetry
		sys.stdout.write('NIntegrate[Abs[g[x] - (1 - g[1 - x])], {x, 0, 1}]\n\n')