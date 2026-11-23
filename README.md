# ENT-A analysis script (pre-data)

Pre-specified analysis for a retrospective cleft-care cohort: general-anaesthesia (GA) episodes, missed combination opportunities, inter-specialty consultation timeliness and a deterministic counterfactual planner. **No patient data are in this repository.** Everything here runs on synthetic test data only.

* `analysis/ent_a_episode_analysis.py` — the analysis (input schema documented in its header).
* `analysis/make_test_data_SYNTHETIC.py` — generates synthetic CSVs (`python analysis/make_test_data_SYNTHETIC.py _synthetic_test_data 150`, seed 42).
* `analysis/test_edge_cases.py` — 8 edge-case tests.
* `SYNTHETIC_RUN_OUTPUT.txt` — output on the synthetic data (`--boot 200`). It is **not** study evidence.

```
pip install -r requirements.txt     # Python 3.11
python analysis/test_edge_cases.py
python analysis/make_test_data_SYNTHETIC.py _synthetic_test_data 150
python analysis/ent_a_episode_analysis.py _synthetic_test_data --boot 200
```
Caveats: the combinability matrix, 240-minute cap, scheduling windows and overhead time are drafts awaiting multidisciplinary-team sign-off. The code was written by the authors with AI assistance (see the manuscript's AI statement). Reproduction by the same authors on one machine only.
