import unittest
import pprint
from effGap import calcEfficiencyGap, computeDistrictPlan
from bisectionUtils import isCloseEnough, listsMatch

class TestEffGapMethods(unittest.TestCase):
	
	def test_calc_efficiency_gap(self):
		voteShares = [0.5]
		effGap = calcEfficiencyGap(voteShares)
		trueEffGap = -0.5
		self.assertTrue(isCloseEnough(effGap, trueEffGap))

		voteShares = [0.7, 0.4]
		effGap = calcEfficiencyGap(voteShares)
		trueEffGap = 0.1
		self.assertTrue(isCloseEnough(effGap, trueEffGap))

		voteShares = [0.1, 0.6, 0.3, 0.8, 0.9, 0.4]
		effGap = calcEfficiencyGap(voteShares)
		trueEffGap = 1.0/30
		self.assertTrue(isCloseEnough(effGap, trueEffGap))

	def test_compute_district_plan(self):
		# One district
		n = 1
		s = 0.7
		A = 1
		voteShares = computeDistrictPlan(n, s, A)
		trueVoteShares = [0.7]
		self.assertTrue(listsMatch(voteShares, trueVoteShares))

		# Two districts
		n = 2
		s = 0.4
		A = 2
		voteShares = computeDistrictPlan(n, s, A)
		trueVoteShares = [0.8, 0.8]
		self.assertTrue(listsMatch(voteShares, trueVoteShares))

		# Three districts
		n = 3
		s = 1.3
		A = 1
		voteShares = computeDistrictPlan(n, s, A)
		trueVoteShares = [0.9, 0.2, 0.2]
		self.assertTrue(listsMatch(voteShares, trueVoteShares))

		s = 0.55
		A = 2
		voteShares = computeDistrictPlan(n, s, A)
		trueVoteShares = [0.475, 0.9875, 0.9875]
		self.assertTrue(listsMatch(voteShares, trueVoteShares))

		# Four districts
		n = 4
		s = 1.2
		A = 1
		voteShares = computeDistrictPlan(n, s, A)
		trueVoteShares = [0.3, 0.8, 0.05, 0.05]
		self.assertTrue(listsMatch(voteShares, trueVoteShares))

if __name__ == '__main__':
	unittest.main()
