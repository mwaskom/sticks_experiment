from __future__ import division
import sys
import json
import itertools
from copy import copy
from textwrap import dedent

import numpy as np
import pandas as pd

from colormath.color_objects import LCHabColor, sRGBColor
from colormath.color_conversions import convert_color

from psychopy import core, visual, event
from psychopy.data import StairHandler
import cregg

import warnings
warnings.simplefilter("ignore", FutureWarning)


def main(arglist):

    # Get the experiment parameters
    mode = arglist.pop(0)
    p = cregg.Params(mode)
    p.set_by_cmdline(arglist)

    # Get a random state that is seeded by the subject ID
    state = cregg.subject_specific_state(p.subject, p.cbid)

    # Open up the stimulus window
    win = cregg.launch_window(p)
    p.win_refresh_hz = win.refresh_hz

    # Fixation point
    fix = visual.Circle(win, interpolate=True,
                        fillColor=p.fix_iti_color,
                        lineColor=p.fix_iti_color,
                        size=p.fix_size)

    # The main stimulus fields
    array = FieldArray(win, p)

    stims = dict(

        fix=fix,
        array=array,

    )


class FieldArray(object):

    def __init__(self, win, p):

        self.fields = (StickField(win, p, "left"),
                       StickField(win, p, "right"))

    def update(self, p_widths, p_lengths, p_colors, p_oris):

        self.update_positions()
        self.update_sizes(p_widths, p_lengths)
        self.update_colors(p_colors)
        self.update_oris(p_oris)

    def update_positions(self):

        for field in self.fields:
            field.update_positions()

    def update_sizes(self, p_w, p_l):

        for field, p_w_f, p_l_f in zip(self.fields, p_w, p_l):
            field.update_sizes(p_w_f, p_l_f)

    def update_colors(self, p):

        for field, p_f in zip(self.fields, p):
            field.update_colors(p_f)

    def update_oris(self, p):

        for field, p_f in zip(self.fields, p):
            field.update_oris(p_f)

    def draw(self):

        for field in self.fields:
            field.draw()


class StickField(object):

    def __init__(self, win, p, side):

        self.p = p
        self.n = p.sticks_per_field
        self.random = np.random.RandomState()

        x_pos = p.field_offset * dict(left=-1, right=1)[side]

        self.edge = visual.Circle(win, p.field_radius, 128,
                                  pos=(x_pos, 0),
                                  fillColor=p.window_color,
                                  lineColor="white")

        self.sticks = visual.ElementArrayStim(win,
                                              nElements=self.n,
                                              fieldPos=(x_pos, 0),
                                              elementTex=None,
                                              elementMask="sqr",
                                              interpolate=True,
                                              texRes=128)

    def update(self, p_lengths, p_widths, p_colors, p_oris):

        self.update_positions()
        self.update_sizes(p_lengths, p_widths)
        self.update_colors(p_colors)
        self.update_oris(p_oris)

    def update_positions(self):

        theta = self.random.uniform(0, 2 * np.pi, self.n)
        r = self.p.field_radius * np.sqrt(self.random.uniform(0, 1, self.n))
        x, y = r * np.cos(theta), r * np.sin(theta)
        self.sticks.setXYs(np.c_[x, y])

    def update_sizes(self, p_w, p_l):

        thick, thin = self.p.widths
        widths = np.r_[np.repeat(thick, self.n * p_w),
                       np.repeat(thin, self.n * (1 - p_w))]
        widths = self.random.permutation(widths)

        long, short = self.p.lengths
        lengths = np.r_[np.repeat(long, self.n * p_l),
                        np.repeat(short, self.n * (1 - p_l))]
        lengths = self.random.permutation(lengths)

        sizes = np.c_[widths, lengths]
        self.sticks.setSizes(sizes)

    def update_colors(self, p):

        red, green = self.p.colors
        colors = np.r_[np.tile(red, self.n * p).reshape(-1, 3),
                       np.tile(green, self.n * (1 - p)).reshape(-1, 3)]

        colors = self.random.permutation(colors)
        self.sticks.setColors(colors)

    def update_oris(self, p):

        left, right = self.p.oris
        oris = np.r_[np.repeat(left, self.n * p),
                     np.repeat(right, self.n * (1 - p))]

        oris = self.random.permutation(oris)
        self.sticks.setOris(oris)

    def draw(self):

        if self.p.debug:
            self.edge.draw()
        self.sticks.draw()
