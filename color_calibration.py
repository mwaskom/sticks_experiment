from __future__ import division, print_function

import sys
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

    fixed_L = p.lightness
    C = p.chroma
    fixed_h, moving_h = p.stick_hues

    patches = ColorPatches(win, p)

    conditions = [{"stepType": "lin",
                   "nReversals": p.reversals,
                   "nUp": 1, "nDown": 1,
                   "stepSizes": p.step_sizes,
                   "startVal": val} for val in p.start_vals]

    stairs = MultiStairHandler(conditions=conditions)

    rs = np.random.RandomState()

    for moving_L, _ in stairs:

        if rs.rand() < .5:
            colors = (fixed_L, C, fixed_h), (moving_L, C, moving_h)
            trial_resp_keys = p.resp_keys
        else:
            colors = (moving_L, C, moving_h), (fixed_L, C, fixed_h)
            trial_resp_keys = p.resp_keys[::-1]

        patches.set_colors(*colors)

        patches.draw()
        win.flip()

        resp = event.waitKeys(keyList=p.resp_keys)[0]
        resp_code = trial_resp_keys.index(resp)

        stairs.addResponse(resp_code)

        win.flip()

        cregg.wait_check_quit(.25) # TODO


    r = np.r_[stairs.staircases[0].reversalIntensities[-p.reversals:],
              stairs.staircases[0].reversalIntensities[-p.reversals:]].mean()
    print(r)

    stairs.saveAsPickle("stairs.pkl")
    
    

class ColorPatches(object):

    def __init__(self, win, p):

        grid = np.linspace(-1, 1, 128)
        x, y = np.meshgrid(grid, grid)
        mask = ((x ** 2 + y ** 2 < 1) & (x < 0)) * 2 - 1
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
