from __future__ import division
import sys
import json
import itertools
from copy import copy
from textwrap import dedent

import numpy as np
from scipy.spatial.distance import cdist
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

    # The main stimulus arrays
    array = StickArray(win, p)

    stims = dict(

        fix=fix,
        array=array,

    )


class StickArray(object):
    """Array of "sticks" that vary on four dimensions."""
    def __init__(self, win, p):

        self.win = win
        self.p = p
        self.random = np.random.RandomState()

        self.rgb_colors = self.lch_to_rgb(p)

        self.edge = visual.Circle(win, p.array_radius, 128,
                                  fillColor=p.window_color,
                                  lineColor="white")

        self.new_array()

    def new_array(self):
        """Initialize new stick positions."""
        initial_xys = self.initial_positions()
        n_sticks = len(initial_xys)
        sticks = visual.ElementArrayStim(self.win,
                                         xys=initial_xys,
                                         nElements=n_sticks,
                                         elementTex=None,
                                         elementMask="sqr",
                                         interpolate=True,
                                         texRes=128)
        self.sticks = sticks
        self.n = n_sticks

    def initial_positions(self):
        """Find positions using poisson-disc sampling."""
        # See http://bost.ocks.org/mike/algorithms/
        uniform = self.random.uniform
        randint = self.random.randint

        # Parameters of the sampling algorithm
        array_radius = self.p.array_radius
        radius = self.p.disk_radius
        candidates = self.p.disk_candidates

        # Start in the middle of the array.
        # This will get removed later, but it will ensure that
        # space around the fixation point is not crowded
        samples = [(0, 0)]
        queue = [(0, 0)]

        while queue:

            # Pick a sample to expand from
            s_idx = randint(len(queue))
            s_x, s_y = queue[s_idx]

            for i in xrange(candidates):

                # Generate a candidate from this sample
                a = uniform(0, 2 * np.pi)
                r = uniform(radius, 2 * radius)
                x, y = s_x + r * np.cos(a), s_y + r * np.sin(a)

                # Check the two conditions to accept the candidate
                in_array = np.sqrt(x ** 2 + y ** 2) < array_radius
                in_ring = np.all(cdist(samples, [(x, y)]) > radius)

                if in_array and in_ring:
                    # Accept the candidate
                    samples.append((x, y))
                    queue.append((x, y))
                    break

            if (i + 1) == candidates:
                # We've exhausted the particular sample
                queue.pop(s_idx)

        # Remove first sample to give space around the fix point
        samples = np.array(samples)[1:]

        return samples

    def lch_to_rgb(self, p):
        """Convert the color values from Lch to RGB."""
        rgbs = []
        for hue in p.hues:
            lch = LCHabColor(p.lightness, p.chroma, hue)
            rgb = convert_color(lch, sRGBColor).get_value_tuple()
            rgbs.append(rgb)
        return tuple(rgbs)

    def update(self, p_widths, p_lengths, p_colors, p_oris):

        self.update_positions()
        self.update_sizes(p_widths, p_lengths)
        self.update_colors(p_colors)
        self.update_oris(p_oris)

    def update_positions(self):

        x, y = self.sticks.xys.T
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)

        phi += self.random.uniform(0, 2 * np.pi)
        phi %= 2 * np.pi

        x, y = rho * np.cos(phi), rho * np.sin(phi)
        self.sticks.setXYs(np.c_[x, y])

    def update_sizes(self, p_w, p_l):

        thick, thin = self.p.widths
        n_thick = np.round(self.n * p_w)
        n_thin = self.n - n_thick
        widths = np.r_[np.repeat(thick, n_thick), np.repeat(thin, n_thin)]
        widths = self.random.permutation(widths)

        long, short = self.p.lengths
        n_long = np.round(self.n * p_l)
        n_short = self.n - n_long
        lengths = np.r_[np.repeat(long, n_long),
                        np.repeat(short, n_short)]
        lengths = self.random.permutation(lengths)

        sizes = np.c_[widths, lengths]
        self.sticks.setSizes(sizes)

    def update_colors(self, p):

        red, green = self.rgb_colors
        n_red = np.round(self.n * p)
        n_green = self.n - n_red
        colors = np.r_[np.tile(red, n_red).reshape(-1, 3),
                       np.tile(green, n_green).reshape(-1, 3)]

        colors = self.random.permutation(colors)
        self.sticks.setColors(colors)

    def update_oris(self, p):

        left, right = self.p.oris
        n_left = np.round(self.n * p)
        n_right = self.n - n_left
        oris = np.r_[np.repeat(left, n_left),
                     np.repeat(right, n_right)]

        oris = self.random.permutation(oris)
        self.sticks.setOris(oris)

    def draw(self):

        if self.p.debug:
            self.edge.draw()
        self.sticks.draw()
