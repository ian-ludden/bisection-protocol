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

	vs = voteShares.copy()
	while sum(vs) < n:
		candidate_jumps = [0.5 - v for v in vs if v < 0.5]
		if candidate_jumps:
			next_jump = min(candidate_jumps)
		else:
			next_jump = 1 - min(vs)
		print(next_jump)

		vs = [min(v + next_jump, 1) for v in vs]

		print(sum(vs), sum([v >= 0.5 for v in vs]))

	vs = voteShares.copy()
	while sum(vs) > 0:
		candidate_drops = [v - 0.5 for v in vs if v > 0.5]
		if candidate_drops:
			next_drop = min(candidate_drops)
		else:
			next_drop = max(vs)
		print(next_drop)

		vs = [max(v - next_drop, 0) for v in vs]

		print(sum(vs), sum([v > 0.5 for v in vs]))



