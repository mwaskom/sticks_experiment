
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_color_codes("muted")

from psychopy.data import FitWeibull
import cregg


def main(arglist):

    p = cregg.Params("psychophys")
    psychophys_temp = p.log_base + ".csv"

    p = cregg.Params("scan")
    p.set_by_cmdline(arglist)

    dfs = []
    for run in range(1, 4):
        fname = psychophys_temp.format(subject=p.subject, run=run)
        try:
            df = pd.read_csv(fname).dropna()
            dfs.append(df)
        except IOError:
            pass

    if not dfs:
        raise ValueError("No data files found for " + p.subject)

    df = pd.concat(dfs, ignore_index=True)

    initial_guess = FitWeibull(df.context_strength, df.correct).params

    stim_strength = dict(hue=dict(), ori=dict())

    f, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    axes = dict(zip(["hue", "ori"], axes))
    colors = dict(hue="m", ori="c")
    xx = np.linspace(0, .25, 100)

    for context, df_context in df.groupby("context"):

        model = FitWeibull(df_context.context_strength,
                           df_context.correct,
                           guess=initial_guess)

        ax = axes[context]
        sns.regplot("context_strength", "correct", data=df_context,
                    fit_reg=False, x_estimator=np.mean, ci=68,
                    color=colors[context], ax=ax)
        ax.plot(xx, model.eval(xx), colors[context], lw=2)

        ax.set(xlim=(0, .25), ylim=(.4, 1), title=context)

    axes["ori"].set(ylabel="")
    f.tight_layout()

    png_name = p.strength_file_base.format(subject=p.subject)
    f.savefig(png_name, filetype="png")


if __name__ == "__main__":
    main(sys.argv[1:])
