#! /usr/bin/env python
import os
import sys


def main(arglist):

    uload_cmd = ["rsync -azP",
                 "data/",
                 "mwaskom@mwmp:studies/sticks/behavior/",
                 ]

    dload_cmd = ["rsync -azP",
                 "mwaskom@mwmp:studies/sticks/behavior/",
                 "data/",
                 ]

    if not arglist:
        os.system(" ".join(uload_cmd))
    elif arglist[0].startswith("up"):
        os.system(" ".join(uload_cmd))
    elif arglist[0].startswith("down"):
        os.system(" ".join(dload_cmd))
    else:
        sys.exit("Argument doesn't make sense!")


if __name__ == "__main__":
    main(sys.argv[1:])
