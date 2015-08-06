import os
import sys
subj = sys.argv[1]
target = "mwaskom@mwmp:studies/squad/data/{}_behav".format(subj)
os.system("rsync -aP data/{}_* {}".format(subj, target))
