from __future__ import division, print_function
import sys
import numpy as np

from colormath.color_objects import LCHabColor, sRGBColor
from colormath.color_conversions import convert_color

from psychopy import visual, event, core

import cregg

s
def main(arglist):


    p = cregg.Params("calibrate")
    p.set_by_cmdline(arglist)

    # Open up the stimulus window
    win = cregg.launch_window(p)
    p.win_refresh_hz = win.refresh_hz

    checkerboard = Checkerboard(win, p)

    set_points = []

    expected = p.lightness

    for trial in xrange(p.repeats):

        approach = ["below", "above"][trial % 2]
        checkerboard.reset(expected, approach)

        cregg.WaitText(win, [approach],
                       advance_keys=p.wait_keys,
                       quit_keys=p.quit_keys).draw()

        while checkerboard.active:

            checkerboard.draw()
            win.flip()

        set_points.append(checkerboard.moving_l)
        expected = checkerboard.moving_l

    print(set_points)


class Checkerboard(object):

    def __init__(self, win, p):

        self.p = p

        self.stim = visual.GratingStim(win,
                                       sf=(p.patch_sf, p.patch_sf),
                                       mask=p.patch_mask,
                                       size=p.patch_size)

        s = p.arrow_size
        o = p.patch_size / 2 + p.arrow_offset
        pos = dict(up=(0, o), down=(0, -o))
        verts = dict(up=[(-s, 0), (0, s), (s, 0)],
                     down=[(-s, 0), (0, -s), (s, 0)])
        self.arrows = {dir: visual.ShapeStim(win,
                                             lineWidth=p.arrow_width,
                                             closeShape=False,
                                             vertices=verts[dir],
                                             pos=pos[dir])
                       for dir in ["up", "down"]}

        self.reset(self.p.lightness, "below")


    def reset(self, expected, approach_from):

        self.active = True

        self.arrow_frames = dict(up=0, down=0)
        self.stim_frames = 0

        self.rgb_fixed = self.lch_to_psychopy_rgb(self.p.lightness,
                                                  self.p.chroma,
                                                  self.p.stick_hues[0])

        s = 1 if approach_from == "above" else -1
        self.moving_l = expected + s * self.p.diff_start

        self.rgb_moving = self.lch_to_psychopy_rgb(self.moving_l,
                                                   self.p.chroma,
                                                   self.p.stick_hues[1])

    def update(self):

        down, up, finish = self.p.resp_keys

        keys = event.getKeys(self.p.resp_keys + self.p.quit_keys)
        if keys:
            key = keys[0]
            if key == down:
                self.adjust_moving("down")
                self.activate_arrow("down")
            elif key == up:
                self.adjust_moving("up")
                self.activate_arrow("up")
            elif key == finish:
                self.select_level()
            elif key in self.p.quit_keys:
                core.quit()

    def draw(self):

        self.update()

        for dir in ["up", "down"]:
            if self.arrow_frames[dir] > 0:
                self.arrows[dir].draw()
                self.arrow_frames[dir] -= 1

        self.stim.tex = self.tex

        if not self.stim_frames % self.p.flicker_every:
            self.stim.phase += (0, 0.5)

        self.stim_frames += 1

        self.stim.draw()

    def activate_arrow(self, which):

        self.arrow_frames[which] = self.p.arrow_life

    def adjust_moving(self, which):

        s = 1 if which == "up" else -1
        delta = s * self.p.diff_step
        self.moving_l += delta

        self.rgb_moving = self.lch_to_psychopy_rgb(self.moving_l,
                                                   self.p.chroma,
                                                   self.p.stick_hues[1])

    def select_level(self):

        self.active = False

    def lch_to_psychopy_rgb(self, l, c, h):

        lch = LCHabColor(l, c, h)
        rgb = convert_color(lch, sRGBColor).get_value_tuple()
        psychopy_rgb = np.array(rgb) * 2 - 1
        return psychopy_rgb

    @property
    def tex(self):

        f = self.rgb_fixed
        m = self.rgb_moving

        tex = np.ones((4, 4, 3))
        tex[0] = m, m, f, f
        tex[1] = m, m, f, f
        tex[2] = f, f, m, m
        tex[3] = f, f, m, m

        return tex


if __name__ == "__main__":
    main(sys.argv[1:])
