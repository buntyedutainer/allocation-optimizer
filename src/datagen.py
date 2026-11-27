"""Builds synthetic students, preferences and scores. The same seed always gives the same data."""
import numpy as np
import pandas as pd

SKILLS = ["Python", "Data analysis", "Web", "Design", "Writing", "Communication"]
RANK_POINTS = {1: 10, 2: 8, 3: 6, 4: 4, 5: 2}
W_PREF, W_FIT = 0.7, 0.3
UNLISTED_RANK = 6

PROJECTS = pd.DataFrame([
    ("P01", "Data Dashboard",        ["Data analysis", "Python"],               4, 0.9),
    ("P02", "Web Portal",            ["Web", "Design"],                         4, 0.5),
    ("P03", "Chatbot Helper",        ["Python", "Communication"],               3, 1.0),
    ("P04", "Learning Content",      ["Writing", "Design"],                     5, 0.3),
    ("P05", "Survey Analysis",       ["Data analysis", "Writing"],              5, 0.4),
    ("P06", "Mobile App",            ["Web", "Python"],                         3, 0.8),
    ("P07", "Community Outreach",    ["Communication", "Writing"],              6, 0.2),
    ("P08", "UX Research",           ["Design", "Communication"],               5, 0.5),
    ("P09", "Automation Scripts",    ["Python", "Data analysis"],               5, 0.6),
    ("P10", "Social Media Campaign", ["Writing", "Communication", "Design"],    4, 0.4),
], columns=["project_id", "title", "needs", "capacity", "appeal"]).set_index("project_id")


def make_students(n, rng):
    levels = rng.integers(0, 3, size=(n, len(SKILLS)))
    for i in range(n):
        strong = rng.choice(len(SKILLS), size=2, replace=False)
        levels[i, strong] = np.minimum(levels[i, strong] + 1, 3)
    ids = [f"S{i + 1:02d}" for i in range(n)]
    return pd.DataFrame(levels, index=ids, columns=SKILLS)


def fit_matrix(students, projects):
    fit = pd.DataFrame(index=students.index, columns=projects.index, dtype=float)
    for pid, row in projects.iterrows():
        fit[pid] = students[row["needs"]].mean(axis=1) / 3
    return fit


def make_preferences(fit, projects, rng, top_k=5):
    noise = rng.random(fit.shape)
    utility = 0.5 * projects["appeal"].values + 0.3 * fit.values + 0.2 * noise
    utility = pd.DataFrame(utility, index=fit.index, columns=fit.columns)
    rows = []
    for sid, u in utility.iterrows():
        best = u.sort_values(ascending=False).index[:top_k]
        for rank, pid in enumerate(best, start=1):
            rows.append((sid, pid, rank))
    return pd.DataFrame(rows, columns=["student_id", "project_id", "rank"])


def build_instance(seed, n_students=40):
    """One complete synthetic dataset: students, ranks, scores and capacities."""
    rng = np.random.default_rng(seed)
    students = make_students(n_students, rng)
    fit = fit_matrix(students, PROJECTS)
    prefs = make_preferences(fit, PROJECTS, rng)
    rank_table = (prefs.pivot(index="student_id", columns="project_id", values="rank")
                  .reindex(index=students.index, columns=PROJECTS.index).fillna(UNLISTED_RANK).astype(int))
    pref_points = rank_table.apply(lambda col: col.map(lambda r: RANK_POINTS.get(r, 0)))
    score = (W_PREF * pref_points + W_FIT * 10 * fit).round(2)
    return {"students": students, "score": score, "rank_table": rank_table,
            "capacity": PROJECTS["capacity"].copy()}
