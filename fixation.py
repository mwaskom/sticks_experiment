
import sys
import cregg


def main(arglist):

    p = cregg.Params("scan")
    p.set_by_cmdline(arglist)
    win = cregg.launch_window(p)
    cregg.WaitText(win, "+", height=2, color="black",
                   advance_keys=[], quit_keys=p.quit_keys).draw()


if __name__ == "__main__":
    main(sys.argv[1:])
