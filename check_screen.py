
import sys
from psychopy import visual, event
import cregg


def main(arglist):

    p = cregg.Params("scan")
    p.set_by_cmdline(arglist)
    win = cregg.launch_window(p)
    visual.Circle(win, p.array_radius,
                  edges=128,
                  lineColor="white",
                  lineWidth=2).draw()
    win.flip()
    event.waitKeys(keyList=p.quit_keys)


if __name__ == "__main__":
    main(sys.argv[1:])
