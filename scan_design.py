from __future__ import division, print_function
from string import letters
import itertools
import numpy as np
from numpy.linalg import pinv
from scipy import stats
import pandas as pd
from joblib import Parallel, delayed

import moss
from moss import glm

import params


class Params(object):
    """Mock the cregg Params object so we don't depend on Psychopy."""
    def __init__(self, dict):
        for key, val in dict.iteritems():
            setattr(self, key, val)


def main():

    p = Params(params.scan)
    print("Optimizing event order")
    designs = optimize_focb(p)
    print("Found {:d} suitable designs".format(len(designs)))
    for design, label in zip(designs, letters):
        print("Optimizing event timing for schedule {}".format(label))
        schedule = optimize_efficiency(design, p)
        schedule.to_csv(p.design_base.format(label.lower()), index=False)


def optimize_focb(p):
    """Search a large space of design orders to find the best ones."""
    conds = make_conditions(p)
    rs = np.random.RandomState(p.focb_seed)

    designs = []
    for _ in xrange(p.focb_batches):

        design_generator = shuffle_conditions_generator(conds, rs,
                                                        p.focb_batch_size)
        candidates = Parallel(n_jobs=-1)(delayed(design_focb_cost)(d)
                                         for d in design_generator)
        keep = [(d, c) for d, c in candidates if c < p.focb_cost_tol]
        designs.extend(keep)

    costs = np.array([c for d, c in designs])
    order = np.argsort(costs)
    ordered_designs = [designs[i][0] for i in order]
    return ordered_designs[:p.n_designs]


def make_conditions(p):
    """Cross relevant variables to create a conditions matrix."""
    conditions = itertools.product(["hue", "ori"],
                                   [0, 1],
                                   p.hue_features,
                                   p.ori_features,
                                   ["easy", "hard"],
                                   ["easy", "hard"])

    condition_cols = ["context", "cue_idx",
                      "hue", "ori",
                      "hue_diff", "ori_diff"]

    conditions = pd.DataFrame(list(conditions),
                              columns=condition_cols)

    return conditions


def design_focb_cost(design):
    """Compute the sum FOCB cost across design elements."""
    return design, design.apply(series_focb_cost).sum()


def series_focb_cost(series):
    """Compute how far the FOCB matrix is from what would be ideal."""
    ideal = np.ones((2, 2)) * .5
    actual = moss.transition_probabilities(series).values
    return np.sum(np.square(ideal - actual))


def shuffle_conditions_generator(conditions, rs, n=1000):
    """A generator to return shuffled condition matrices."""
    for _ in xrange(n):
        shuffler = rs.permutation(conditions.index)
        yield conditions.reindex(shuffler).reset_index(drop=True)


# -----


def optimize_efficiency(design, p):

    rs = np.random.RandomState(p.eff_seed)

    best_schedule = None
    best_efficiency = 0

    for sched in candidate_schedule_generator(design, p, rs, p.eff_n_sched):
        _, eff = schedule_efficiency(sched,
                                     p.eff_fir_basis,
                                     p.eff_leadout_trs)
        if eff > best_efficiency:
            best_schedule = sched
            best_efficiency = eff

    return best_schedule


def candidate_schedule_generator(design, p, rs, n=1000):
    """Generator to make design schedules with randomized onsets."""
    n_trials = len(design)
    iti_trs = get_iti_distribution(n_trials, p, rs)
    for _ in xrange(n):
        schedule = design.copy()
        iti_trs = rs.permutation(iti_trs)
        schedule["iti_trs"] = iti_trs
        compute_onsets(schedule, p)
        yield schedule


def compute_onsets(design, p):
    """Given a design with ITI information, compute trial onset time."""
    onsets = (design.iti_trs + p.trs_per_trial).cumsum().shift(1).fillna(0)
    design["trial_time_tr"] = onsets


def get_iti_distribution(n_trials, p, rs):
    """Return a vector of ITIs (in TR units) for each trial."""
    x = np.arange(*p.eff_geom_support)
    iti_pmf = stats.geom(p.eff_geom_p, loc=p.eff_geom_loc).pmf(x)
    iti_counts = np.round((iti_pmf / iti_pmf.sum()) * n_trials)
    iti_counts[0] += (n_trials - iti_counts.sum())

    iti_trs = [np.repeat(x_i, c) for x_i, c in zip(x, iti_counts)]
    iti_trs = np.concatenate(iti_trs)
    return iti_trs


def schedule_efficiency(schedule, nbasis, leadout_trs):
    """Compute the FIR design matrix efficiency for a given schedule."""
    par = pd.DataFrame(dict(onset=schedule.trial_time_tr,
                            condition=schedule.context))

    fir = glm.FIR(tr=1, nbasis=nbasis, offset=-1)
    ntp = par.onset.max() + leadout_trs

    X = glm.DesignMatrix(par, fir, ntp,
                         hpf_cutoff=None,
                         tr=1, oversampling=1).design_matrix.values
    eff = 1 / np.trace(pinv(X.T.dot(X)))
    return schedule, eff


if __name__ == "__main__":
    main()
