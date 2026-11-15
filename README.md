# P5 - Network Lightpaths: Route and Wavelength Assignment

Search + CSP + Neural Network on the Abilene traffic matrices.
Decision-support / simulation only: no real network is configured by this code.

## What the system does
1. A neural network forecasts next-hour traffic for the 132 sender-receiver pairs of the 12 Abilene routers; the top K flows become requests.
2. Search (A*, also BFS and UCS) finds up to 3 loop-free routes per request. Cost = great-circle distance in km; routes longer than the reach limit (5000 km, a generated parameter) are discarded.
3. A CSP assigns a (route, wavelength) to every request. Two requests sharing a directed link need different wavelengths. Backtracking, Forward Checking and AC-3 are separate switches, with counters (assignments, backtracks).
4. If the CSP fails, Search is re-run with 5 routes per request; if it still fails, the lowest-priority request is blocked and the reason is reported. If nothing can be served, a clear NO SOLUTION message is printed.

## Environment
Tested on macOS (Apple M1) with Python 3.12.8.

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

requirements.txt was produced with pip freeze from this environment.

## Data
- Abilene traffic matrices (Yin Zhang, 2004), weeks X06..X10 (2004-05-01 .. 2004-05-29, consecutive weeks). Only the real measurement of each sender-receiver pair is used; the other four values per pair in the files are estimates and are not used.
- Abilene topology: topo-2003-04-10.txt (12 routers, 15 cables = 30 directed links), from the same server.
- germany50 (50 nodes, 88 cables): GML file from TopoHub (copy of the SNDlib network), committed as data/germany50.gml. Used only in experiment 1. Licence not verified.
- data_dictionary.md lists every field and says what is real and what is generated (reach limit, number of wavelengths, CSP assignment cap).

make_data.py downloads the Abilene files into data/ if they are missing and builds data/processed/hourly.npz (840 hours x 12 x 12). The server is old and can be very slow (a 7 MB file took about 15 minutes in one test). On macOS with Python from python.org, a CERTIFICATE_VERIFY_FAILED error is fixed by running "Install Certificates.command" in the Python application folder.

Split (hours of the 840): 0-167 history only (needed for the same-hour-last-week feature), 168-623 train, 624-671 validation, 672-839 test (week X10).

## How to run
All commands are run from the project folder with the environment active.

    python make_data.py          # data (about 1 minute if the files are already in data/)
    python baselines.py          # simple forecasting baselines
    python nn_v2.py              # NN vs last-hour baseline, validation-based choice, 5 seeds
    python test_search.py        # search tests (known answer + all pairs vs networkx)
    python test_csp.py           # CSP tests: 5 normal, 3 edge, 2 impossible
    python test_system.py        # whole-system tests: 5 normal, 3 edge, 2 impossible
    python exp1.py               # experiment 1: BFS vs UCS vs A* (1-2 minutes)
    python exp2_final.py         # experiment 2: BT vs BT+FC vs BT+FC+AC3 (5-10 minutes)
    python exp3_final.py         # experiment 3: NN vs last hour inside the system (10-15 minutes)
    python run_demo.py           # integrated demo, writes demo_output.png
    python demo_live.py          # live demo: search trace, CSP with and without FC / AC-3, NN on vs off

The first call that needs the NN predictions trains 5 networks (about 1 minute) and caches them in data/processed/nn_test_preds.npy. Outputs of the tests and experiments are saved in results/.

## Files and modules
| Module | Files |
|---|---|
| data | topology.py, make_data.py, check.py, plot_total.py, data_dictionary.md |
| nn | forecast.py, baselines.py, nn_v1.py, nn_v2.py |
| search | search.py (BFS / UCS / A* with counters) |
| csp | csp.py (BT, FC, AC-3 with counters), oracle.py (independent OR-tools solver, used only to label cases as feasible or infeasible) |
| integration | integration.py (plan()), run_demo.py, demo_live.py |
| eval | exp1.py, exp2_final.py, exp3_final.py |
| tests | test_search.py, test_csp.py, test_system.py |

Exploratory scripts kept for history (superseded by the final experiment scripts): run_csp_demo.py, run_exp2.py, exp2b.py, run_exp3.py, run_exp3b.py, inspect_g50.py, look.py.

## Main results (details in results/)
- Search: A* and UCS costs match networkx on all 132 Abilene pairs and all 2450 germany50 pairs; the heuristic never overestimates. A* expands far fewer states than UCS on germany50 (6.9 vs 275.7 for the best route). BFS returns the fewest-hops route, which is 4% (Abilene) to 11% (germany50) longer in km. Counters count partial paths, because a state is a node together with the path so far.
- CSP: with 3 wavelengths, Forward Checking roughly halves the assignments and proves infeasibility in more cases than plain backtracking. With 4 or more wavelengths the three methods are close. AC-3 removed nothing in these instances, so it equals Forward Checking.
- NN: its top-K hit rate on the test week is within about 0.7 percentage points of the last-hour baseline. Inside the system it serves slightly more of the true top-20 at 4 wavelengths (+0.8 points) and slightly fewer of the true top-15 at 3 wavelengths (-1.4 points). The ceiling (perfect forecast) is clearly higher than both, so network capacity limits the result more than forecast quality.

## Limitations
- One test week (X10); the confidence intervals use blocks of 6 hours and do not cover other weeks.
- Data from 2004. Traffic to Abilene's own AS is not in the matrices, and routing was simulated with OSPF when they were built. Units are relative, not Mbps.
- The start hour and time zone of each file are not given; midnight on the listed date is assumed.
- Reach limit, number of wavelengths and the CSP assignment cap are generated parameters. The reach limit removes alternative routes and lowers capacity.
- A CSP result of "limit" means the assignment cap was reached: it is not a proof that no solution exists. In experiment 2, plain backtracking and Forward Checking failed to prove infeasibility in many cases that the independent solver proves infeasible.
- Passing tests show that plans are valid, not that they are optimal.
- Blocking a request means the group of requests could not be scheduled together; it does not mean that request alone has a problem.
