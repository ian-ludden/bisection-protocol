import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import sys

from bisectionUtils import csvToArray

######################################################################
# Author: 	Ian Ludden
# Date: 	29 May 2019
# 
# Computes the partisan asymmetry of district plans resulting from 
# optimal "play" of the bisection and I-cut-you-freeze protocols. 
######################################################################

if __name__ == '__main__':
	# Extra arg means I-cut-you-freeze, no argument means bisection.
	thresholdsFilename = 'thresholds_1_to_53.csv'
	
	if len(sys.argv) > 1:
		thresholdsFilename = 'icycThresholds_1_to_53.csv'

	# Read in table of t_{n,j} as defined in bisectionDP.py.
	t = csvToArray(thresholdsFilename)

	for n in range(1, 54):
		pa = 0
		for j in range(1, n + 1):
			pa = pa + abs(t[n,j] - (n - t[n,n-j+1]))

		pa = pa / n
		pa = pa / n
		print('{0},{1:.4f}'.format(n, pa))
		# Recommend piping output to a CSV file
