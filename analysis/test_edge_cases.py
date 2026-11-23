"""SYNTHETIC unit-style edge-case tests for ent_a_episode_analysis.py (code testing only).
Usage: python test_edge_cases.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ent_a_episode_analysis as A  # noqa: E402

T = pd.Timestamp


def pts_frame(last_contact="2020-01-01", first_contact="2018-01-01"):
    return pd.DataFrame(dict(patient_id=["X1"], birth_date=[T("2018-01-01")], first_contact=[T(first_contact)],
                             last_contact=[T(last_contact)],
                             syndromic=[0], cleft_type=["CLP"], asa_class=[1], airway_risk=[0])
                        ).set_index("patient_id")


def proc(rows):
    df = pd.DataFrame(rows, columns=["patient_id", "date", "discipline", "procedure_code", "ga_flag",
                                     "episode_id", "duration_min"])
    df["date"] = pd.to_datetime(df.date)
    df["indication_date"] = pd.NaT
    return df


def test_procedure_after_last_contact_excluded():
    p = pts_frame("2019-06-30")
    pr = proc([["X1", "2019-03-01", "PLASTIC", "PALATOPLASTY", 1, "E0", 120],
               ["X1", "2019-07-15", "ENT", "VT", 1, "E1", 20]])  # after last_contact
    ep = A.ga_episodes(pr, p, 6)
    assert list(ep.episode_id) == ["E0"], ep
    assert A.planner(ep, p, 90) == 1


def test_missing_duration_not_merged():
    p = pts_frame()
    pr = proc([["X1", "2019-03-01", "PLASTIC", "PALATOPLASTY", 1, "E0", 120],
               ["X1", "2019-03-20", "ENT", "VT", 1, "E1", np.nan]])
    ep = A.ga_episodes(pr, p, 6)
    assert ep.duration.isna().sum() == 1
    assert not A.missed_flags(ep, p, 90).any()
    assert A.planner(ep, p, 90) == 2  # not merged in main analysis
    # Sensitivity: median-by-code imputation (VT median from another child's record) allows merge.
    extra = proc([["X2", "2019-01-01", "ENT", "VT", 1, "Z0", 20]])
    ep_imp = A.ga_episodes(A.impute_duration(pd.concat([pr, extra], ignore_index=True)), p, 6)
    assert ep_imp.duration.notna().all() and A.planner(ep_imp, p, 90) == 1


def test_short_followup_consult_censored():
    p = pts_frame("2019-03-31")
    c = pd.DataFrame(dict(consult_id=["C0", "C1"], patient_id=["X1", "X1"], from_discipline=["PLASTIC"] * 2,
                          to_discipline=["ENT"] * 2,
                          request_date=[T("2019-03-01"), T("2018-06-01")],  # C0: 30 d follow-up, open
                          response_date=[pd.NaT, T("2018-06-10")],
                          closure_date=[pd.NaT, T("2018-12-01")]))  # C1: eligible, closed after 90 d
    res, _ = A.consult_metrics(c, p)
    assert res["n_eligible_90d"] == 1 and res["n_censored_before_90d"] == 1
    assert res["not_closed_90d"] == 1.0  # only C1 in the denominator (closed at 183 d)
    assert res["km_closure"][90] == 0.0  # C0 censored at 30 d, not counted as an event/failure


def test_closure_after_last_contact_ignored():
    p = pts_frame("2019-03-31")
    c = pd.DataFrame(dict(consult_id=["C0"], patient_id=["X1"], from_discipline=["PLASTIC"],
                          to_discipline=["ENT"], request_date=[T("2019-03-01")],
                          response_date=[T("2019-03-05")], closure_date=[T("2019-05-01")]))
    res, _ = A.consult_metrics(c, p)
    assert res["km_closure"][90] == 0.0 and res["n_censored_before_90d"] == 1


def test_same_discipline_not_merged_via_session():
    # Two ENT episodes (VT re-insertion) must not merge, even through a dental session.
    p = pts_frame()
    pr = proc([["X1", "2019-03-01", "DENTAL", "DENTAL_GA", 1, "E0", 60],
               ["X1", "2019-03-20", "ENT", "VT", 1, "E1", 20],
               ["X1", "2019-04-10", "ENT", "VT", 1, "E2", 20]])
    ep = A.ga_episodes(pr, p, 6)
    assert A.planner(ep, p, 90) == 2
    ep2 = A.ga_episodes(proc([["X1", "2019-03-01", "ENT", "VT", 1, "E0", 20],
                              ["X1", "2019-03-01", "DENTAL", "DENTAL_GA", 1, "E0", 60],
                              ["X1", "2019-04-01", "ENT", "ABR_GA", 1, "E1", 40]]), p, 6)
    assert not A.missed_flags(ep2, p, 90).any()  # overlapping discipline (ENT) -> not missed


def test_missing_safety_data_not_eligible():
    p = pts_frame()
    p["asa_class"] = np.nan
    pr = proc([["X1", "2019-03-01", "PLASTIC", "PALATOPLASTY", 1, "E0", 120],
               ["X1", "2019-03-20", "ENT", "VT", 1, "E1", 20]])
    ep = A.ga_episodes(pr, p, 6)
    assert A.planner(ep, p, 90) == 2 and not A.missed_flags(ep, p, 90).any()


def test_episode_before_first_contact_and_persontime():
    p = pts_frame("2020-01-01", first_contact="2019-01-01")
    pr = proc([["X1", "2018-06-01", "PLASTIC", "LIP_REPAIR", 1, "E0", 90],  # before entry
               ["X1", "2019-03-01", "PLASTIC", "PALATOPLASTY", 1, "E1", 120]])
    ep = A.ga_episodes(pr, p, 6)
    assert list(ep.episode_id) == ["E1"]
    d = A.pts_years(p, 6)
    assert abs(d.iloc[0] - 365 / 365.25) < 1e-9


def test_no_response_restricted_to_eligible():
    p = pts_frame("2019-03-31")
    c = pd.DataFrame(dict(consult_id=["C0", "C1"], patient_id=["X1", "X1"], from_discipline=["PLASTIC"] * 2,
                          to_discipline=["ENT"] * 2, request_date=[T("2019-03-01"), T("2018-06-01")],
                          response_date=[pd.NaT, T("2018-06-10")], closure_date=[pd.NaT, pd.NaT]))
    res, _ = A.consult_metrics(c, p)
    assert res["no_response_90d"] == 0.0  # C0 (30 d follow-up) excluded, C1 responded


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} SYNTHETIC edge-case tests passed")
