# Job 2, arm B: expectations written before its numbers (5 Nov 2026; arm A read, arm B not yet run)

Arm A (matched on the far probes alone; the local frames: 3 held-out test frames, MB-pol eps_inf 1.6870):

| edge, order | eps | cell M | mu | R | R iso | iso R spread / MB-pol's (r) | abs mu / MB-pol's (r) | probes 10/12/15 | T5 |
|---|---|---|---|---|---|---|---|---|---|
| 0.75, 0 | 1.8198 | 17.4 | 17.5 | 22.0 | 20.0 | 3.78 (0.45) | 1.146 (0.996) | 7.0/4.7/4.2 | 0.943 |
| 1.5, 0 | 1.7664 | 10.9 | 11.3 | 14.0 | 12.4 | 3.42 (0.57) | 1.078 (0.997) | 6.8/4.3/4.2 | 0.956 |
| 2.0, 0 | 1.7214 | 6.5 | 8.6 | 7.9 | 6.3 | 3.04 (0.62) | 1.027 (0.997) | 7.0/4.0/4.5 | 0.980 |
| 2.5, 0 | 1.6912 | 6.3 | 8.8 | 5.6 | 3.7 | 2.94 (0.62) | 0.992 (0.996) | 7.3/4.0/4.8 | 0.997 |
| 0.75, 2 | 1.7990 | 16.2 | 19.5 | 18.1 | 16.6 | 1.99 (0.02) | 1.172 (0.996) | 2.5/2.6/2.0 | 1.026 |
| 1.5, 2 | 1.7727 | 12.3 | 12.3 | 13.4 | 12.8 | 1.63 (0.02) | 1.105 (0.998) | 2.2/2.1/1.8 | 1.034 |
| 2.0, 2 | 1.7429 | 8.9 | 8.2 | 9.2 | 8.7 | 1.71 (-0.07) | 1.057 (0.998) | 2.3/2.1/2.1 | 1.042 |
| 2.5, 2 | 1.7200 | 6.1 | 6.9 | 6.4 | 5.8 | 1.74 (-0.12) | 1.023 (0.998) | 2.9/2.3/2.4 | 1.048 |

Reading of arm A: the molecules' induced dipoles follow MB-pol's almost exactly in direction and relative size
(r 0.996-0.998); their error is mostly one scale, which falls with the edge with eps_inf (|mu| excess about 0.7 of the
susceptibility's). The molecules' response to a uniform field does not follow MB-pol's from molecule to molecule: its
spread is 3-3.8 times MB-pol's at order 0 (r 0.45-0.62) and 1.6-2 times at order 2 (r about 0), nearly independent of
the edge. The read-out's polarisability is built from pairs with the neighbours (bandfields.response_energy_pair: the
isolated monomer's is zero), so it varies with the environment as MB-pol's fixed site polarisabilities do not; the
mutual edge hardly touches that.

Expected of arm B (each edge 1.5 / 2.0 / 2.5 A, orders 0 and 2; the second stage on the 10 local calibration frames,
weight 0.1):

B1. The calibration reaches the bulk response: eps_inf on the 3 held-out frames and on the 6 frames no calibration uses
    within 0.010 of MB-pol's, at every edge and order. Its edge dependence (1.5 -> 2.5 A) then <= 0.01, against 0.053
    (order 2) and 0.075 (order 0) in arm A.
B2. The far probes do not see the counterterm: the probes' part of the matching objective rises by less than 2x in
    the second stage; the probes' non-additive error at 10, 12, 15 A at most 1 point above arm A's at order 2 and 2
    points at order 0, at every edge; T5's slope within 0.90-1.10.
B3. The molecules' dipoles take the right scale: |mu| / MB-pol's within 0.98-1.02 at every edge and order, r stays
    >= 0.995, and the molecules' dipole error falls at every edge and order (at 2.0 A to <= 7 % at order 2, <= 8 % at
    order 0).
B4. The molecules' response spread is not the cell sums' to fix: its spread stays >= 1.5 times MB-pol's and its
    correlation with MB-pol's <= 0.7, at every edge and order. (Falsified if the calibration brings the spread below 1.5:
    then the sums do constrain the environment dependence.)
B5. The per-molecule response error (R, full tensor) falls at every edge and order but stays >= 4 %: what remains is
    the spread (B4).
