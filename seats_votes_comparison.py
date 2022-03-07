import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import sys

from computeMetric import protocols
import utils

######################################################################
# Author:   Ian Ludden
# Date:     06 Jan 2022
# 
# Compare protocol seats-votes curve to seats-votes curve for  
# uniform partisan swing from a specific district plan 
# produced by the protocol. 
######################################################################

"""
Returns x and y coordinates of points defining the protocol seats-votes curve. 

Parameters:
===============
    n - number of districts (int)
    protocolAbbrev - 'B' for bisection or 'ICYF' for I-cut-you-freeze

Returns:
===============
    x - x-coordinates of protocol seats-votes curve, scaled to the interval [0, 1]
    y - y-coordinates of protocol seats-votes curve, scaled to the interval [0, 1]
"""
def getProtocolSeatsVotes(n, protocolAbbrev):
    protocol = protocols[protocolAbbrev]
    t, K = utils.getThresholds(protocol, n)

    x_distinct_vals = t[n, :(n+1)]
    x = np.repeat(x_distinct_vals, 2)[1:]
    x = np.append(x, n)
    y_distinct_vals = [i for i in range(n + 1)]
    y = np.repeat(y_distinct_vals, 2)

    return x / n, y / n



if __name__ == '__main__':
    if len(sys.argv) < 4:
        exit('Not enough arguments. Usage: python district_plan.py [N] [s1] [protocolAbbrev]')

    n = int(sys.argv[1])
    print('Parameter n =', n)
    s1 = float(sys.argv[2])
    print('Parameter s1 =', s1)
    protocolAbbrev = str(sys.argv[3])
    protocol = protocols[protocolAbbrev]
    print('Protocol =', protocol['name'])
    
    x_protocol, y_protocol = getProtocolSeatsVotes(n, protocolAbbrev)

    t, K = utils.getThresholds(protocol, n)
    voteShares = utils.getDistrictPlan(n, s1, 1, protocolAbbrev, t, K)
    
    x_plan, y_plan = utils.getPlanSeatsVotes(voteShares)
    x_plan_flip = [1. - xi for xi in reversed(x_plan)]
    y_plan_flip = [1. - yi for yi in reversed(y_plan)]

    print('District vote-shares:')
    pprint(list(voteShares))
    print('District vote-shares, sorted:')
    pprint(sorted(voteShares))

    print("Plan partisan asymmetry:", utils.calcPA(voteShares))
    print("Plan SL index:", utils.calcSainteLagueIndex(voteShares))
    print("Plan efficiency gap:", utils.calcEfficiencyGap(voteShares))

    plt.plot(x_protocol, y_protocol)
    plt.plot(x_plan, y_plan, '--')
    plt.plot(x_plan_flip, y_plan_flip, '-*')
    plt.title('Seats-votes curves for protocol and district plan, N = {}'.format(n))
    plt.legend(["{} Protocol".format(protocol['name']), "District Plan for s1 = {}".format(s1), "Player 2 perspective"])
    plt.xlabel('Player 1 Vote-share')
    plt.ylabel('Player 1 Seat-share')
    plt.show()
