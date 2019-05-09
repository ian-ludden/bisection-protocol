# bisection-protocol
Analysis of the Bisection Protocol for Political Districting. 

## Overview
In the *bisection protocol* for political districting, two players (the major parties) alternately divide all remaining pieces of the state in half (up to rounding) until each piece has a `1/n` fraction of the total population, where `n` is the number of districts needed.

Player 1 begins by dividing the state of measure `n` into two pieces, one of size `floor(n/2)` and the other of size `ceil(n/2)`. Player 2 then divides each of these in half (up to rounding). Player 1 then divides each of the four pieces created by Player 2 in half (up to rounding), and so on. When a created piece has size 1, it is frozen. The process terminates when every piece is frozen, and these `n` pieces form the districts. Intuitively, this process is most natural when `n` is a power of two, so some of the analysis will only consider these cases. 

### Thresholds: Definition and Recurrence
For integers `j` and `n`, with `j <= n`, let `t_{n,j}` be the minimum vote-share Player 1 must have to win at least `j` districts under optimal play by both players. 

The thresholds obey the following recurrence:

If `j = 0`, then `t_{n,j} = 0`. When `n=j=1`, `t_{n,j} = 1/2`. For `n >= 2` and `j >= 1`,

```
t_{n,j} = min_{k\in K} { ( a - t_{a,a-k+1} ) + ( b - t_{b,b-(j-k)+1} ) }

t_{n,j} = min_{k \in K} { n - t_{a,a-k+1} - t_{b,b-(j-k)+1} },
```
where `a = floor(n/2)` is the size of the smaller part after the cut, `b = ceil(n/2)` is the size of the larger, and `K = {k in integers : 0 <= k <= j, 0 <= k <= a, 0 <= j-k <= b}` is the set of feasible slates from the part of size `a`. 

The thresholds are computed via dynamic programming using this recurrence (`bisectionDP.py`).

These thresholds are then processed into Wolfram Mathematica code for plotting the seat-share vs. vote-share curves (`prepThresholds.py`). The Mathematica notebook `PartisanAsymmetryComputations.nb` contains the plots and integration to compute the partisan asymmetry of each curve, as defined in <a href='http://www.optimization-online.org/DB_HTML/2019/03/7123.html'>Swamy et al. (pre-print)</a>. 

