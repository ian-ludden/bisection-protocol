from pprint import pprint
import sys

from computeMetric import protocols
import utils

######################################################################
# Author: 	Ian Ludden
# Date: 	17 Dec 2021
# 
# Print district vote-shares resulting from a given protocol and 
# given parameters.
######################################################################

if __name__ == '__main__':
	if len(sys.argv) < 4:
		exit('Not enough arguments. Usage: python district_plan.py [N] [s1] [protocolAbbrev]')

	n = int(sys.argv[1])
	s1 = float(sys.argv[2])
	protocolAbbrev = str(sys.argv[3])

	protocol = protocols[protocolAbbrev]

	t, K = utils.getThresholds(protocol, n)
	voteShares = utils.getDistrictPlan(n, s1, 1, protocolAbbrev, t, K)

	pprint(list(voteShares))
	print('Sorted: ')
	pprint(sorted(voteShares))
