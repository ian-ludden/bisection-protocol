import csv
import numpy as np

######################################################################
# Author: 	Ian Ludden
# Date: 	14 May 2019
# 
# Utility functions for scripts related to the bisection protocol.  
######################################################################

def csvToArray(filename):
	"""Reads the given CSV file into an array."""
	with open(filename, 'r') as csvFile:
		reader = csv.reader(csvFile)
		data = list(reader)
	maxIndex = len(data) + 1
	A = np.zeros((maxIndex,maxIndex))
	i = 1
	for row in data:
		j = 1
		for elem in row:
			if len(elem) > 0:
				A[i,j] = float(elem)
			j = j + 1
		i = i + 1
	return A

def arrayToCSV(A, filename):
	"""Saves the given array to a CSV file with the given name."""
	np.savetxt(filename, A, fmt='%.1f', delimiter=',')	

def isCloseEnough(a, b):
	"""Returns True if a and b are within epsilon of each other."""
	EPSILON = 1e-10
	return abs(a - b) <= EPSILON

def listsMatch(l1, l2):
	"""Returns True if the lists are approx. equal when sorted."""
	if len(l1) != len(l2):
		return False

	l1.sort()
	l2.sort()

	for i in range(len(l1)):
		if not isCloseEnough(l1[i], l2[i]):
			return False

	return True