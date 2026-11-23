"""ENT-A pre-specified analysis: general-anaesthesia (GA) episodes, missed combination
opportunities, inter-specialty consultation metrics and counterfactual surgical-episode
planner simulation in multidisciplinary cleft care.

Usage:  python ent_a_episode_analysis.py <data_dir> [--age-cap 6] [--boot 1000]

INPUT SCHEMA (CSV, UTF-8, one row per record, de-identified study IDs only)
patients.csv
    patient_id        str   study identifier
    birth_date        date  YYYY-MM-DD
    first_contact     date  OPTIONAL; first contact with the centre (cohort entry / start of
                            person-time). If absent, birth_date is used with a warning.
    last_contact      date  last documented contact (end of observation)
    syndromic         0/1   syndromic diagnosis or Robin sequence
    cleft_type        str   CL | CP | CLP
    asa_class         int   highest pre-operative ASA class recorded for the child
    airway_risk       0/1   documented difficult airway / airway alert
    (missing asa_class or airway_risk -> child treated as NOT eligible for combination)
procedures.csv
    patient_id, date (YYYY-MM-DD), discipline (PLASTIC|ENT|DENTAL|ORTHO|OMFS),
    procedure_code (see COMBINABLE below), ga_flag (0/1), episode_id (str; procedures
    sharing one anaesthetic share an episode_id), duration_min (procedure minutes),
    indication_date (OPTIONAL; date the decision to perform the procedure was first
    documented -- enables the information-constrained planner variant)
consults.csv
    consult_id, patient_id, from_discipline, to_discipline, request_date,
    response_date (first specialist contact; blank if none), closure_date
    (documented outcome; blank if none), transfer_date (OPTIONAL; care transferred
    elsewhere -- competing event for closure)

The planner is a deterministic simulation applied to historical event sequences; it is
not an intervention. All parameters below are pre-specified and require MDT sign-off.
"""
import argparse
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---- Pre-specified parameters (TODO: MDT clinician sign-off) ----
PROC_CLASS = {
    "LIP_REPAIR": "PLASTIC", "PALATOPLASTY": "PLASTIC", "LIP_REVISION": "PLASTIC",
    "FISTULA_REPAIR": "PLASTIC", "ABG": "OMFS", "VT": "ENT", "ABR_GA": "ENT",
    "EUA_NASENDO": "ENT", "ADENOIDECTOMY": "ENT", "TONSILLECTOMY": "ENT",
    "DENTAL_GA": "DENTAL", "ORTHO_MINOR": "ORTHO",
}
# Unordered pairs that may share one anaesthetic (Table 1 of the manuscript).
COMBINABLE = {frozenset(p) for p in [
    ("LIP_REPAIR", "VT"), ("LIP_REPAIR", "ABR_GA"), ("LIP_REPAIR", "EUA_NASENDO"),
    ("PALATOPLASTY", "VT"), ("PALATOPLASTY", "ABR_GA"), ("PALATOPLASTY", "DENTAL_GA"),
    ("LIP_REVISION", "VT"), ("LIP_REVISION", "ABR_GA"), ("LIP_REVISION", "DENTAL_GA"),
    ("LIP_REVISION", "EUA_NASENDO"), ("FISTULA_REPAIR", "VT"), ("FISTULA_REPAIR", "ABR_GA"),
    ("FISTULA_REPAIR", "DENTAL_GA"), ("ABG", "VT"), ("ABG", "DENTAL_GA"),
    ("ABG", "ORTHO_MINOR"), ("VT", "ABR_GA"), ("VT", "EUA_NASENDO"), ("VT", "DENTAL_GA"),
    ("VT", "ORTHO_MINOR"), ("VT", "ADENOIDECTOMY"), ("ABR_GA", "EUA_NASENDO"),
    ("ABR_GA", "DENTAL_GA"), ("EUA_NASENDO", "DENTAL_GA"), ("EUA_NASENDO", "ADENOIDECTOMY"),
    ("DENTAL_GA", "ORTHO_MINOR"), ("DENTAL_GA", "ADENOIDECTOMY"),
]}
DURATION_CAP_MIN = 240      # maximum combined procedure time per anaesthetic
ASA_MAX = 2                 # combination allowed only if the child's highest ASA <= this
EXCLUDE_AIRWAY_RISK = True  # no combination when an airway alert is recorded
ANAES_OVERHEAD_MIN = 45     # induction/emergence/turnover saved per avoided episode
OVERHEAD_SENS = (30, 45, 60)  # sensitivity range for minutes saved per avoided episode
WINDOWS = (60, 90, 120)
PRIMARY_WINDOW = 90
CONSULT_CLOSE_DAYS = 90
STUDY_END = pd.Timestamp("2025-12-31")  # administrative end of observation


def fits_cap(d1, d2):
    """Merge allowed only if both durations are known and the sum respects the cap."""
    return pd.notna(d1) and pd.notna(d2) and d1 + d2 <= DURATION_CAP_MIN


def combinable(codes_a, codes_b):
    """All cross-pairs must be in the matrix; same-code pairs are always allowed."""
    return all(a == b or frozenset((a, b)) in COMBINABLE for a in codes_a for b in codes_b)


def safe(pat, asa_max=ASA_MAX, airway=EXCLUDE_AIRWAY_RISK):
    """Child-level eligibility; missing ASA or airway status is treated as not eligible."""
    if pd.isna(pat["asa_class"]) or (airway and pd.isna(pat["airway_risk"])):
        return False
    return pat["asa_class"] <= asa_max and not (airway and pat["airway_risk"] == 1)


def disjoint(d1, d2):
    """Episodes may merge only if they share no discipline (no same-team re-operations)."""
    return not (set(d1) & set(d2))


def load(d):
    p = pd.read_csv(os.path.join(d, "patients.csv"), parse_dates=["birth_date", "last_contact"])
    if "first_contact" in p:
        p["first_contact"] = pd.to_datetime(p.first_contact).fillna(p.birth_date)
    else:
        print("WARNING: first_contact missing -- person-time starts at birth (immortal-time risk)")
        p["first_contact"] = p.birth_date
    p["last_contact"] = p.last_contact.clip(upper=STUDY_END)
    pr = pd.read_csv(os.path.join(d, "procedures.csv"))
    pr["date"] = pd.to_datetime(pr.date)
    pr["indication_date"] = pd.to_datetime(pr["indication_date"]) if "indication_date" in pr else pd.NaT
    c = pd.read_csv(os.path.join(d, "consults.csv"))
    for col in ("request_date", "response_date", "closure_date", "transfer_date"):
        if col in c:
            c[col] = pd.to_datetime(c[col])
    n_miss = pr.loc[pr.ga_flag == 1, "duration_min"].isna().sum()
    if n_miss:
        print(f"NOTE: {n_miss} GA procedures lack duration_min (episodes ineligible for merging "
              f"in the main analysis; median-by-code imputation in sensitivity analysis)")
    return p.set_index("patient_id"), pr, c


def impute_duration(pr):
    """Sensitivity analysis: impute missing duration_min with the median of the same code."""
    pr = pr.copy()
    med = pr.groupby("procedure_code").duration_min.transform("median")
    pr["duration_min"] = pr.duration_min.fillna(med)
    return pr


def _sum_or_nan(s):
    # An episode's duration is unknown if any of its procedures lacks a duration (never 0).
    return s.sum() if s.notna().all() else np.nan


def ga_episodes(pr, pts, age_cap):
    """One row per GA episode within observation: before age_cap years and on or before the
    child's end of follow-up (last_contact, already clipped to STUDY_END) -- the same end date
    used for the person-time denominator."""
    ga = pr[pr.ga_flag == 1].merge(pts[["birth_date", "first_contact", "last_contact"]],
                                   left_on="patient_id", right_index=True)
    ga = ga[((ga.date - ga.birth_date).dt.days < age_cap * 365.25) & (ga.date <= ga.last_contact)
            & (ga.date >= ga.first_contact)]
    ep = ga.groupby(["patient_id", "episode_id"]).agg(
        date=("date", "min"), codes=("procedure_code", lambda s: tuple(sorted(set(s)))),
        disciplines=("discipline", lambda s: tuple(sorted(set(s)))),
        duration=("duration_min", _sum_or_nan), indication=("indication_date", "max")).reset_index()
    return ep.sort_values(["patient_id", "date"]).reset_index(drop=True)


def missed_flags(ep, pts, window, **kw):
    """Episode is 'missed' if another GA episode of the same child, with no discipline in
    common, within +/-window days, was combinable and safe under the duration cap."""
    flag = np.zeros(len(ep), bool)
    for pid, g in ep.groupby("patient_id"):
        if not safe(pts.loc[pid], **kw):
            continue
        for i, j in combinations(g.index, 2):
            a, b = ep.loc[i], ep.loc[j]
            if (abs((a.date - b.date).days) <= window and disjoint(a.disciplines, b.disciplines)
                    and combinable(a.codes, b.codes) and fits_cap(a.duration, b.duration)):
                flag[i] = flag[j] = True
    return flag


def planner(ep, pts, window, info_constrained=False, **kw):
    """Greedy chronological (first-fit) planner: each episode joins the earliest planned
    session whose anchor date is within `window` days, if it shares no discipline with the
    session, all procedures are combinable, the child is safe and the cap is respected.
    info_constrained=True additionally requires the episode's indication to have been
    documented on or before the session anchor date (no use of future information).
    First-fit is a heuristic, not an optimal assignment."""
    planned = 0
    for pid, g in ep.groupby("patient_id"):
        ok = safe(pts.loc[pid], **kw)
        sessions = []  # [anchor_date, codes set, duration, disciplines set]
        for _, e in g.iterrows():
            placed = False
            if ok:
                for s in sessions:
                    if info_constrained and not (pd.notna(e.indication) and e.indication <= s[0]):
                        continue
                    if ((e.date - s[0]).days <= window and disjoint(e.disciplines, s[3])
                            and combinable(s[1], e.codes) and fits_cap(s[2], e.duration)):
                        s[1] |= set(e.codes)
                        s[2] += e.duration
                        s[3] |= set(e.disciplines)
                        placed = True
                        break
            if not placed:
                sessions.append([e.date, set(e.codes), e.duration, set(e.disciplines)])
        planned += len(sessions)
    return planned


def km_closure(c, horizon=365):
    """Kaplan-Meier cumulative probability of consult closure; censoring at last contact."""
    t, e = c["t"].values, c["closed"].values
    surv, out = 1.0, {}
    for u in np.unique(t[e == 1]):
        n = (t >= u).sum()
        surv *= 1 - ((t == u) & (e == 1)).sum() / n
        out[u] = 1 - surv
    return {d: max([v for k, v in out.items() if k <= d], default=0.0) for d in (30, 90, horizon)}


def aalen_johansen(t, e, horizon=CONSULT_CLOSE_DAYS):
    """Cumulative incidence of closure (event 1) by `horizon`, transfer (event 2) competing."""
    surv, cif = 1.0, 0.0
    for u in np.unique(t[(e > 0) & (t <= horizon)]):
        n = (t >= u).sum()
        d1, d2 = ((t == u) & (e == 1)).sum(), ((t == u) & (e == 2)).sum()
        cif += surv * d1 / n
        surv *= 1 - (d1 + d2) / n
    return cif


def consult_metrics(c, pts):
    c = c.merge(pts[["last_contact"]], left_on="patient_id", right_index=True)
    # Same end of follow-up as the denominator: drop requests after last_contact and ignore
    # responses/closures/transfers documented after it (censored at last_contact).
    c = c[c.request_date <= c.last_contact].copy()
    for col in ("response_date", "closure_date", "transfer_date"):
        if col in c:
            c[col] = c[col].where(c[col] <= c.last_contact)
    c["resp_days"] = (c.response_date - c.request_date).dt.days.clip(lower=0)
    c["closed"] = c.closure_date.notna().astype(int)
    c["t"] = np.where(c.closed == 1, (c.closure_date - c.request_date).dt.days,
                      (c.last_contact - c.request_date).dt.days).clip(min=0)
    # Proportion not closed within 90 d is restricted to consults with >=90 d potential
    # follow-up; shorter, unclosed consults are censored (handled only by KM / Aalen-Johansen).
    c["fu_days"] = (c.last_contact - c.request_date).dt.days
    c["elig90"] = c.fu_days >= CONSULT_CLOSE_DAYS
    c["open90"] = ((c.closed == 0) | (c.t > CONSULT_CLOSE_DAYS)).where(c.elig90)
    res = {"n_consults": len(c), "median_resp_days": c.resp_days.median(),
           "iqr_resp": tuple(c.resp_days.quantile([.25, .75])), "no_response_90d": (c.resp_days.isna() | (c.resp_days > CONSULT_CLOSE_DAYS)).where(c.elig90).mean(),
           "n_eligible_90d": int(c.elig90.sum()),
           "n_censored_before_90d": int(((~c.elig90) & ~((c.closed == 1) & (c.t <= CONSULT_CLOSE_DAYS))).sum()),
           "not_closed_90d": c.open90.mean(), "km_closure": km_closure(c)}
    if "transfer_date" in c and c.transfer_date.notna().any():
        tr = (c.closed == 0) & c.transfer_date.notna()
        ev = np.where(c.closed == 1, 1, np.where(tr, 2, 0))
        tt = np.where(tr, (c.transfer_date - c.request_date).dt.days, c.t).clip(min=0)
        res["cif_closure_90d_competing"] = aalen_johansen(tt, ev)
    by_pair = c.groupby(["from_discipline", "to_discipline"]).agg(
        n=("consult_id", "size"), median_resp=("resp_days", "median"),
        q1_resp=("resp_days", lambda s: s.quantile(.25)), q3_resp=("resp_days", lambda s: s.quantile(.75)),
        not_closed_90d=("open90", "mean"))
    return res, by_pair


def pts_years(pts, age_cap):
    """Person-years from first contact to min(last contact, age cap); floor 0.1 y."""
    cap = pts.birth_date + pd.to_timedelta(age_cap * 365.25, "D")
    return ((np.minimum(pts.last_contact, cap) - pts.first_contact).dt.days / 365.25).clip(lower=.1)


def nb_model(ep, pts, age_cap):
    cnt = ep.groupby("patient_id").size().reindex(pts.index, fill_value=0)
    d = pts.copy()
    d["ga"] = cnt
    d["years"] = pts_years(d, age_cap)
    # Person-time from cohort entry (first contact) to the end of follow-up; GA episodes
    # before first contact are excluded in ga_episodes, so numerator and denominator match.

    X = pd.get_dummies(d[["syndromic", "cleft_type"]], columns=["cleft_type"], drop_first=True,
                       dtype=float)
    X = sm.add_constant(X.astype(float))
    # NB2 with dispersion (alpha) estimated by maximum likelihood; offset = log person-years.
    m = sm.NegativeBinomial(d.ga, X, offset=np.log(d.years), loglike_method="nb2").fit(disp=0)
    return d, m


def bootstrap(ep, pts, window, B, rng):
    """Child-level (cluster) percentile bootstrap of the planner's relative reduction and of
    the proportion of episodes flagged as missed at `window`."""
    ids = pts.index.values
    by_child = {i: g for i, g in ep.groupby("patient_id")}
    red, mis = [], []
    for _ in range(B):
        s = rng.choice(ids, len(ids), replace=True)
        parts = [by_child[i].assign(patient_id=f"{i}_{k}") for k, i in enumerate(s) if i in by_child]
        if not parts:
            continue
        e = pd.concat(parts, ignore_index=True)
        p = pts.loc[s].copy()
        p.index = [f"{i}_{k}" for k, i in enumerate(s)]
        red.append(1 - planner(e, p, window) / len(e))
        mis.append(e[f"missed_{window}"].mean())
    return np.percentile(red, [2.5, 97.5]), np.percentile(mis, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--age-cap", type=float, default=6)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    rng = np.random.default_rng(2569)
    pts, pr, c = load(a.data_dir)
    ep = ga_episodes(pr, pts, a.age_cap)
    print(f"GA episodes with missing duration (ineligible for merging): {ep.duration.isna().sum()}")
    d, m = nb_model(ep, pts, a.age_cap)
    print(f"Children {len(pts)}; GA episodes to age {a.age_cap:g}: {len(ep)}; "
          f"mean/child {d.ga.mean():.2f} (median {d.ga.median():.0f}, "
          f"IQR {d.ga.quantile(.25):.0f}-{d.ga.quantile(.75):.0f})")
    print("Negative binomial (NB2, alpha estimated) rate ratios, offset log person-years:")
    ci = m.conf_int()
    print(pd.DataFrame({"RR": np.exp(m.params), "lo": np.exp(ci[0]), "hi": np.exp(ci[1])})
          .round(2).to_string())
    pts["group"] = np.where(pts.syndromic == 1, "syndromic/Robin", "non-syndromic")
    for w in WINDOWS:
        ep[f"missed_{w}"] = missed_flags(ep, pts, w)
    cols = [f"missed_{w}" for w in WINDOWS]
    print("\nMissed combination opportunity (proportion of GA episodes):")
    g = ep.merge(pts[["group", "cleft_type"]], left_on="patient_id", right_index=True)
    print(g.groupby("group")[cols].mean().round(3).to_string())
    print(g.groupby("cleft_type")[cols].mean().round(3).to_string())
    print("Children with >=1 missed opportunity (proportion of children with >=1 GA episode):")
    print(g.groupby(["group", "patient_id"])[cols].any().groupby("group").mean().round(3).to_string())
    print(f"\nPlanner simulation (rules, window -> planned, avoided, theatre min at "
          f"{ANAES_OVERHEAD_MIN} [range {OVERHEAD_SENS[0]}-{OVERHEAD_SENS[-1]}], visits):")
    rules = {"base": {}, "ASA<=3": {"asa_max": 3}, "no airway exclusion": {"airway": False}}
    if ep.indication.notna().any():
        rules["information-constrained"] = {"info_constrained": True}
    for name, kw in rules.items():
        for w in WINDOWS:
            n = planner(ep, pts, w, **kw)
            av = len(ep) - n
            print(f"  {name:24s} {w:4d}d  {n:5d}  {av:4d} ({av / len(ep):.1%})  "
                  f"{av * ANAES_OVERHEAD_MIN:6d} [{av * OVERHEAD_SENS[0]}-{av * OVERHEAD_SENS[-1]}]  {av:4d}")
    ep_imp = ga_episodes(impute_duration(pr), pts, a.age_cap)
    n_imp = planner(ep_imp, pts, PRIMARY_WINDOW)
    print(f"  sensitivity: median-by-code duration imputation, {PRIMARY_WINDOW}d -> planned {n_imp}, "
          f"avoided {len(ep_imp) - n_imp} ({(len(ep_imp) - n_imp) / len(ep_imp):.1%})")
    print("Stratified estimates (exploratory effect modification), base rules, primary window:")
    for key in ("group", "cleft_type"):
        for lvl, pg in pts.groupby(key):
            eg = ep[ep.patient_id.isin(pg.index)]
            if len(eg):
                print(f"  {lvl:20s} avoided {1 - planner(eg, pg, PRIMARY_WINDOW) / len(eg):.1%} of {len(eg)}")
    (lo, hi), (mlo, mhi) = bootstrap(ep, pts, PRIMARY_WINDOW, a.boot, rng)
    print(f"  child-level bootstrap 95% CI ({PRIMARY_WINDOW}d, B={a.boot}): relative reduction "
          f"{lo:.1%} to {hi:.1%}; missed proportion {mlo:.1%} to {mhi:.1%}")
    res, by_pair = consult_metrics(c, pts)
    print("\nConsultations:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})
    print(by_pair.round(2).to_string())


if __name__ == "__main__":
    sys.exit(main())
