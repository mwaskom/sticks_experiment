"""Estimate stimulus strength for scanning session."""
import sys
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_color_codes("muted")

from psychopy.data import FitWeibull
import cregg

import warnings
warnings.simplefilter("ignore", FutureWarning)


def main(arglist):

    # Load the psychophysics parameters
    p = cregg.Params("psychophys")
    p.set_by_cmdline(arglist)

    # Find data files and combine across runs
    dfs = []
    for run in p.fit_runs:
        fname = p.log_base.format(subject=p.subject, run=run) + ".csv"
        df = pd.read_csv(fname).dropna()
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # Fit to the whole dataset to get a more robust initial guess
    initial_guess = FitWeibull(df.context_strength, df.correct).params

    # Initialize the output data structure
    stim_strength = dict(hue=dict(), ori=dict())

    # Initialize the plot
    f, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    axes = dict(zip(["hue", "ori"], axes))
    colors = dict(hue="m", ori="c")
    xx = np.linspace(0, .25, 100)

    for context, df_context in df.groupby("context"):

        # Fit the psychometric function
        model = FitWeibull(df_context.context_strength,
                           df_context.correct,
                           guess=initial_guess)

        # Plot the data and model fit
        ax = axes[context]
        sns.regplot("context_strength", "correct", data=df_context,
                    fit_reg=False, x_estimator=np.mean, ci=68,
                    color=colors[context], ax=ax)
        ax.plot(xx, model.eval(xx), colors[context], lw=2)
        ax.set(xlim=(0, .25), ylim=(.4, 1), title=context)

        for diff in ["easy", "hard"]:

            # Find the estimated strength values
            a = p.strength_acc_targets[diff]
            s = model.inverse(a)
            stim_strength[context][diff] = s

            # Plot them
            ax.plot([s, s], [.4, a], c=".5", ls="--")
            ax.plot([0, s], [a, a], c=".5", ls="--")

    # Finalize the plot
    axes["ori"].set(ylabel="")
    f.tight_layout()
    png_name = p.strength_file_base.format(subject=p.subject)
    f.savefig(png_name, filetype="png")

    # Write out the strength values for the scan
    json_fname = p.strength_file_base.format(subject=p.subject) + ".json"
    with open(json_fname, "w") as fid:
        json.dump(stim_strength, fid)


if __name__ == "__main__":
    main(sys.argv[1:])
