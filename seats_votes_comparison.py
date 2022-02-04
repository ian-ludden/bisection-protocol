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

"""
Returns x and y coordinates of points defining the seats-votes curve 
of the district plan represented by the given fractional vote-shares. 

Parameter:
===============
    voteShares - list of fractional vote-shares in each district

Returns:
===============
    x - x-coordinates of plan seats-votes curve, scaled to the interval [0, 1]
    y - y-coordinates of plan seats-votes curve, scaled to the interval [0, 1]
"""
def getPlanSeatsVotes(voteShares):
    vs = sorted(voteShares)

    x = [0]
    y = [0]

    unique_vals, counts = np.unique(vs, return_counts=True)
    
    for index, val in enumerate(unique_vals):
        if val < 0.5:
            # Determine how much must be added to total vote-share to make these districts wins
            diff = 0.5 - val
            vs_hypothetical = [min(v + diff, 1.0) for v in vs]
            x_hypothetical = sum(vs_hypothetical)
            y_hypothetical_2 = sum(map(lambda v : v >= 0.5, vs_hypothetical))
            y_hypothetical_1 = y_hypothetical_2 - counts[index]

            x.append(x_hypothetical)
            y.append(y_hypothetical_1)
            x.append(x_hypothetical)
            y.append(y_hypothetical_2)

        if val >= 0.5:
            # Determine how much must be removed from total vote-share to make these districts losses
            diff = val - 0.5
            vs_hypothetical = [max(v - diff, 0.0) for v in vs]
            x_hypothetical = sum(vs_hypothetical)
            y_hypothetical_2 = sum(map(lambda v : v >= 0.5, vs_hypothetical))
            y_hypothetical_1 = y_hypothetical_2 - counts[index]

            x.append(x_hypothetical)
            y.append(y_hypothetical_1)
            x.append(x_hypothetical)
            y.append(y_hypothetical_2)
    
    x.append(n)
    y.append(n)

    x = sorted(x)
    y = sorted(y)

    print(x)
    print(y)

    return [xi / n for xi in x], [yi / n for yi in y]



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
    
    x_plan, y_plan = getPlanSeatsVotes(voteShares)

    print('District vote-shares:')
    pprint(list(voteShares))
    print('District vote-shares, sorted:')
    pprint(sorted(voteShares))

    plt.plot(x_protocol, y_protocol)
    plt.plot(x_plan, y_plan, '--')
    plt.title('Seats-votes curves for protocol and district plan, N = {}'.format(n))
    plt.legend(["{} Protocol".format(protocol['name']), "District Plan for s1 = {}".format(s1)])
    plt.xlabel('Player 1 Vote-share')
    plt.ylabel('Player 1 Seat-share')
    plt.show()
