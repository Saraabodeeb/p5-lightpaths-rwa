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
Number of wavelengths, lightpath capacity, reach limit: GENERATED, documented in the config.

## Known limits
- Traffic destined to AS11537 (Abilene itself) is not included in the matrices.
- Routing was simulated with OSPF when the matrices were built.
- Data is from 2004.
- The time zone and exact start hour of each file are not stated in the readme.
  We assume each file starts at 00:00 on its listed date.
