# Data dictionary

## Source
Abilene traffic matrices (Yin Zhang, 2004), weeks X06..X10
(first days 2004-05-01, 05-08, 05-15, 05-22, 05-29; consecutive weeks).
Topology: topo-2003-04-10.txt (12 routers, 15 physical cables = 30 directed links).

## hourly.npz
| Field   | Shape       | Real or generated | Description |
|---------|-------------|-------------------|-------------|
| hourly  | (840,12,12) | REAL (measured)   | [hour, source router, destination router]. Sum of twelve 5-minute measurements. Unit: relative (100 bytes per 5 min, packet sampling rate 100); NOT Mbps. |
| routers | (12,)       | REAL              | Router names in the same order as the matrix indices (order taken from demands.txt). |
| weeks   | (5,)        | REAL              | Source file names. |

## Processing
- Only the first of the 5 values per OD pair (realOD) is used. The other 4 are estimates (gravity / tomogravity) and are NOT used.
- The diagonal (router to itself) is kept in the file but removed when building the 132 OD pairs.
- 5-minute data summed to hourly. Seed = 42.

## Generated parameters (added later, config file)
Number of wavelengths, reach limit and the CSP assignment cap are GENERATED (see the table below). Link capacity is NOT used: the CSP counts wavelengths per link, not Mbps.

## Known limits
- Traffic destined to AS11537 (Abilene itself) is not included in the matrices.
- Routing was simulated with OSPF when the matrices were built.
- Data is from 2004.
- The time zone and exact start hour of each file are not stated in the readme.
  We assume each file starts at 00:00 on its listed date.

## Generated and design parameters (set in the code, no config file)
| Parameter | Value | Real or generated | Where |
|-----------|-------|-------------------|-------|
| Reach limit | 5000 km | GENERATED | integration.py (REACH_KM) |
| Wavelengths per link (W) | 3, 4 or 8 depending on the experiment | GENERATED | experiment scripts |
| Number of requests (K) | varies per experiment | GENERATED | experiment scripts |
| Routes per request | 3, then 5 if the CSP fails | design choice | integration.py |
| CSP assignment cap | 20,000 inside plan(); 100,000 in exp2_final.py | GENERATED | integration.py, exp2_final.py |
| Link length | great-circle (haversine) distance from router coordinates | DERIVED from real coordinates | search.py |
| Random seeds | 42 for data and bootstrap; NN seeds 0-4 | - | make_data.py, forecast.py |

## germany50 (experiment 1 only, 50-node network)
Source: TopoHub copy of the SNDlib germany50 network (GML file, data/germany50.gml).
50 nodes, 88 undirected cables (= 176 directed links in our graph).
Node coordinates (lat/lon): REAL. Link length: DERIVED with the same haversine function as Abilene.
No traffic data is used for this network (search experiment only).
Licence: not verified. Cite SNDlib and the TopoHub paper.
