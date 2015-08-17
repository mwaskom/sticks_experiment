from __future__ import division, print_function

import sys
import json
import numpy as np

from psychopy import core, visual, event
from psychopy.data import MultiStairHandler

from colormath.color_objects import LCHabColor, sRGBColor
from colormath.color_conversions import convert_color

import cregg


import warnings
warnings.simplefilter("ignore", FutureWarning)


def main(arglist):

    p = cregg.Params("calibrate")
    p.set_by_cmdline(arglist)

    # Open up the stimulus window
    win = cregg.launch_window(p)
    p.win_refresh_hz = win.refresh_hz

    # Determine the fixed and moving color parameters
    fixed_L = p.lightness
    C = p.chroma
    fixed_h, moving_h = p.stick_hues

    # Initialize the stimulus object
    patches = ColorPatches(win, p)

    # Initialize the staircase
    conditions = [{"stepType": "lin",
                   "nReversals": p.reversals,
                   "nUp": 1, "nDown": 1,
                   "stepSizes": p.step_sizes,
                   "startVal": val,
                   "label": label}
                  for (val, label) in zip(p.start_vals, ["low", "high"])]
    stairs = MultiStairHandler(nTrials=p.trials,
                               conditions=conditions)

    # Showt the instructions
    instruct = cregg.WaitText(win, p.instruct_text,
                              advance_keys=p.wait_keys,
                              quit_keys=p.quit_keys)
    instruct.draw()

    # Initialize the clock
    clock = core.Clock()

    # Start the log file
    log_cols = ["staircase", "moving_L", "choice", "time"]
    p.log_base = p.log_base.format(subject=p.subject, monitor=p.monitor_name)
    log = cregg.DataLog(p, log_cols)

    # Initialize a randomizer
    rs = np.random.RandomState()

    for moving_L, conditions in stairs:

        # Randomize the sides that each hue is shown on
        if rs.rand() < .5:
            # Show fixed color on the left and moving color on the right
            colors = (fixed_L, C, fixed_h), (moving_L, C, moving_h)
            # A "right" response will mean the moving color is brighter
            # This will be treated as "correct" and will adjust it down
            trial_resp_keys = p.resp_keys[:]
        else:
            # Show fixed color on the right and moving color on the left
            colors = (moving_L, C, moving_h), (fixed_L, C, fixed_h)
            # A "left" response will mean the moving color is brighter
            # This will be treated as "incorrect" and will adjust it up
            trial_resp_keys = p.resp_keys[::-1]

        # Update the colors of the patches and draw them
        patches.set_colors(*colors)
        patches.draw()
        win.flip()

        # Listen for the first valid keypress
        resp, time = event.waitKeys(keyList=p.resp_keys,
                                    timeStamped=clock)[0]
        resp_code = trial_resp_keys.index(resp)

        # Update the staircase object
        stairs.addResponse(resp_code)

        # Update the log
        log.add_data(dict(staircase=conditions["label"],
                          moving_L=moving_L,
                          choice=resp_code,
                          time=time))

        # Wait for the next trial
        win.flip()
        cregg.wait_check_quit(p.iti)

    # Compute the lightness to use for the moving hue
    low_reversals = stairs.staircases[0].reversalIntensities[-p.reversals:]
    high_reversals = stairs.staircases[1].reversalIntensities[-p.reversals:]
    reversals = np.r_[low_reversals, high_reversals]
    L = reversals.mean()

    # Save out a final with the final calibrated L
    cal_fname = p.color_file.format(subject=p.subject,
                                    monitor=p.monitor_name)
    with open(cal_fname, "w") as fid:
        json.dump(dict(calibrated_L=L), fid)

    # Print a summary
    print("Total trials: {:d}".format(stairs.totalTrials))
    print("Final luminance: {:.2f}".format(L))
    print("Std. dev. of reversal points: {:.2f}".format(reversals.std()))


class ColorPatches(object):

    def __init__(self, win, p):

        grid = np.linspace(-1, 1, 128)
        x, y = np.meshgrid(grid, grid)
        mask = np.where((x ** 2 + y ** 2 < 1) & (x < 0), 1, -1)
        masks = mask, mask[:, ::-1]

        patches = [visual.GratingStim(win,
                                      tex=None,
                                      size=p.patch_size,
                                      mask=mask)
                   for mask in masks]

        self.patches = patches

    def set_colors(self, left_lch, right_lch):

        left_rgb = self.lch_to_rgb(*left_lch)
        right_rgb = self.lch_to_rgb(*right_lch)

        self.patches[0].color = left_rgb
        self.patches[1].color = right_rgb

    def lch_to_rgb(self, L, C, h):
        """Convert the color values from Lch to (-1, 1) RGB."""
        lch = LCHabColor(L, C, h)
        rgb = convert_color(lch, sRGBColor).get_value_tuple()
        psychopy_rgb = np.array(rgb) * 2 - 1
        return psychopy_rgb

    def draw(self):

        for patch in self.patches:
            patch.draw()

if __name__ == "__main__":
    main(sys.argv[1:])
