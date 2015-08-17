
import itertools
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import moss
from moss import glm
import cregg


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
        keep =[(d, c) for d, c in candidates if c < p.focb_cost_tol]
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
