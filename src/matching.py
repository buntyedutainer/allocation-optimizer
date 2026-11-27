"""Matching tools for the allocation project."""
import numpy as np
import pandas as pd

UNLISTED_RANK = 6          # the rank we give to a project the student did not list


def load():
    score = pd.read_csv("scores.csv", index_col=0)
    projects = pd.read_csv("projects.csv", index_col=0)
    prefs = pd.read_csv("preferences.csv")
    rank_table = (prefs.pivot(index="student_id", columns="project_id", values="rank")
                  .reindex(index=score.index, columns=score.columns).fillna(UNLISTED_RANK).astype(int))
    return score, projects, projects["capacity"].copy(), rank_table


def match_greedy(score, rank_table, capacity, order, listed_only=True):
    """Go through the students in 'order'. Each takes their best-scoring project that still has a place."""
    left = capacity.copy()
    result = {}
    for sid in order:
        options = score.loc[sid][left > 0]                          # projects with a free place
        if listed_only:
            on_list = options[rank_table.loc[sid, options.index] < UNLISTED_RANK]
            if len(on_list) > 0:                                    # otherwise: fall back to any free project
                options = on_list
        pick = options.idxmax()
        result[sid] = pick
        left[pick] -= 1
    return pd.Series(result)


def match_random(score, rank_table, capacity, order, rng, listed_only=True):
    """Go through the students in 'order'. Each gets a random project that still has a place."""
    left = capacity.copy()
    result = {}
    for sid in order:
        free = list(left[left > 0].index)
        if listed_only:
            on_list = [p for p in free if rank_table.loc[sid, p] < UNLISTED_RANK]
            if on_list:
                free = on_list
        pick = rng.choice(free)
        result[sid] = pick
        left[pick] -= 1
    return pd.Series(result)


def validate(assign, score, capacity):
    problems = []
    if set(assign.index) != set(score.index):
        problems.append("not every student is placed exactly once")
    over = [p for p, c in assign.value_counts().items() if c > capacity[p]]
    if over:
        problems.append(f"over capacity: {over}")
    return problems or ["valid"]


def outcomes(assign, score, rank_table):
    """One row per student: the project they got, its score, and the rank they gave it."""
    return pd.DataFrame({"project": assign,
                         "score": [score.loc[s, p] for s, p in assign.items()],
                         "rank": [rank_table.loc[s, p] for s, p in assign.items()]}, index=assign.index)


def summary(out):
    return {"average happiness": out["score"].mean(),
            "top-1 %": 100 * (out["rank"] == 1).mean(),
            "top-3 %": 100 * (out["rank"] <= 3).mean(),
            "not on their list %": 100 * (out["rank"] == UNLISTED_RANK).mean(),
            "worst rank": int(out["rank"].max()),
            "lowest happiness": out["score"].min()}


def match_optimal(score, rank_table, capacity, listed_only=True, penalty=100.0):
    """The assignment with the highest total happiness (Hungarian method on a seat table)."""
    from scipy.optimize import linear_sum_assignment
    seats = [p for p in capacity.index for _ in range(int(capacity[p]))]      # one column per seat
    values = score[seats].values.astype(float)                                # students x seats
    if listed_only:
        unlisted = (rank_table[seats].values == UNLISTED_RANK)
        values = values - penalty * unlisted               # a heavy fine for unlisted projects, so they are used only if forced
    rows, cols = linear_sum_assignment(values, maximize=True)
    return pd.Series({score.index[r]: seats[c] for r, c in zip(rows, cols)})


def objective(assign, score, rank_table, listed_only=True, penalty=100.0):
    """What the optimal method maximises: total happiness, minus a fine for each unlisted placement."""
    total = sum(score.loc[s, p] for s, p in assign.items())
    if listed_only:
        total -= penalty * sum(rank_table.loc[s, p] == UNLISTED_RANK for s, p in assign.items())
    return total
