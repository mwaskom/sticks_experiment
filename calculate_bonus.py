from __future__ import print_function
import sys
import pandas as pd


def main(arglist):

    subj = arglist[0]
    template = "data/{}_scan_run{:02d}.csv"

    dfs = []
    for run in range(1, 13):
        try:
            run_df = pd.read_csv(template.format(subj, run))
            dfs.append(run_df)
        except IOError:
            print("No data file for run {}".format(run))
    df = pd.concat(dfs, ignore_index=True)

    acc = df.correct.mean()
    rt = df.rt.mean()

    print("Subject {}:".format(subj))
    print("Accuracy: {:.2f}".format(acc))
    print("Mean RT: {:.2f}".format(rt))

    bonus = acc / rt * 30
    print("Bonus: {:.2f}".format(bonus))

if __name__ == "__main__":
    main(sys.argv[1:])
