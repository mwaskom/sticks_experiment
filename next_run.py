import sys
from glob import glob

import cregg


def main(arglist):

    mode = arglist.pop(0)
    p = cregg.Params(mode)
    p.set_by_cmdline(arglist)

    data_temp = p.log_base.format(subject=p.subject, run=99) + ".csv"
    data_temp = data_temp.replace("99", "??")
    next_run = len(glob(data_temp)) + 1

    print("Next run for {} {} is # {}".format(p.subject, mode, next_run))

if __name__ == "__main__":
    main(sys.argv[1:])
