"""SYNTHETIC test data for ent_a_episode_analysis.py -- code testing only, NOT study data.
Usage: python make_test_data_SYNTHETIC.py <out_dir> [n_children]"""
import os
import sys

import numpy as np
import pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else "_codetest_SYNTHETIC"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 150
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(42)
base = pd.Timestamp("2016-01-01")
P, R, C = [], [], []
# (code, discipline, mean age in days, probability, duration)
SCHED = [("LIP_REPAIR", "PLASTIC", 110, .8, 90), ("PALATOPLASTY", "PLASTIC", 380, .85, 120),
         ("VT", "ENT", 400, .6, 20), ("ABR_GA", "ENT", 300, .3, 40), ("EUA_NASENDO", "ENT", 1400, .2, 20),
         ("ADENOIDECTOMY", "ENT", 1700, .1, 30), ("DENTAL_GA", "DENTAL", 1300, .35, 90),
         ("LIP_REVISION", "PLASTIC", 1600, .2, 60), ("FISTULA_REPAIR", "PLASTIC", 1200, .1, 90),
         ("ORTHO_MINOR", "ORTHO", 2000, .1, 30)]
for i in range(n):
    pid = f"S{i:04d}"
    bd = base + pd.Timedelta(days=int(rng.integers(0, 2500)))
    ct = rng.choice(["CL", "CP", "CLP"], p=[.2, .3, .5])
    syn = int(rng.random() < .2)
    P.append(dict(patient_id=pid, birth_date=bd.date(), first_contact=(bd + pd.Timedelta(days=int(rng.integers(0, 200)))).date(), last_contact=min(bd + pd.Timedelta(days=int(rng.integers(900, 3300))), pd.Timestamp("2025-12-31")).date(),
                  syndromic=syn, cleft_type=ct, asa_class=int(rng.choice([1, 2, 3], p=[.5, .35, .15])),
                  airway_risk=int(rng.random() < (.3 if syn else .05))))
    k = 0
    for code, disc, age, prob, dur in SCHED:
        if code == "LIP_REPAIR" and ct == "CP" or code == "PALATOPLASTY" and ct == "CL":
            continue
        if rng.random() < prob:
            d = bd + pd.Timedelta(days=int(max(30, rng.normal(age, age * .25))))
            # occasionally already combined with the previous episode
            eid = f"{pid}_E{k - 1}" if k and code == "VT" and rng.random() < .4 else f"{pid}_E{k}"
            if eid == f"{pid}_E{k}":
                k += 1
            else:
                d = [r for r in R if r["episode_id"] == eid][0]["date"]
            R.append(dict(patient_id=pid, date=pd.Timestamp(d).date(), discipline=disc, procedure_code=code,
                          ga_flag=1, episode_id=eid, duration_min=dur))
    for j in range(int(rng.integers(0, 4))):
        rq = bd + pd.Timedelta(days=int(rng.integers(30, 1500)))
        resp = rq + pd.Timedelta(days=int(rng.exponential(25))) if rng.random() < .85 else pd.NaT
        clo = resp + pd.Timedelta(days=int(rng.exponential(40))) if pd.notna(resp) and rng.random() < .7 else pd.NaT
        fr, to = rng.choice(["PLASTIC", "ENT", "DENTAL", "SPEECH", "ORTHO"], 2, replace=False)
        C.append(dict(consult_id=f"{pid}_C{j}", patient_id=pid, from_discipline=fr, to_discipline=to,
                      request_date=rq.date(), response_date=resp, closure_date=clo))
pd.DataFrame(P).to_csv(os.path.join(out, "patients.csv"), index=False)
pd.DataFrame(R).to_csv(os.path.join(out, "procedures.csv"), index=False)
pd.DataFrame(C).to_csv(os.path.join(out, "consults.csv"), index=False)
print(f"SYNTHETIC: {len(P)} children, {len(R)} procedures, {len(C)} consults -> {out}")
