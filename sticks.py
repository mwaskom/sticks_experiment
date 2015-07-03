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

    # Text that cues the context for each block
    cue = visual.TextStim(win, text="", color="white",
                          pos=(0, 0), height=.75)

    # Text that indicates correct or incorrect responses
    feedback = Feedback(win)

    stims = dict(

        fix=fix,
        array=array,
        cue=cue,
        feedback=feedback,

    )

    # Execute the experiment function
    globals()[mode](p, win, stims)


def prototype(p, win, stims):

    stim_event = EventEngine(win, p, stims)

    with cregg.PresentationLoop(win, p):

        while True:

            dim = np.random.choice(p.dim_names)
            dim_idx = p.dim_names.index(dim)

            stims["cue"].setText(dim)
            stims["cue"].draw()
            win.flip()
            cregg.wait_check_quit(p.cue_dur)

            for trial in range(4):

                ps = [.3, .4, .45, .55, .6, .7]
                ps = [.1, .9]
                trial_ps = np.random.choice(ps, 4)
                stims["array"].set_feature_probs(*trial_ps)
                correct_resp = trial_ps[dim_idx] > .5

                stim_event(correct_resp)
                cregg.wait_check_quit(p.feedback_dur)

            stims["fix"].draw()
            win.flip()
            cregg.wait_check_quit(2)


# =========================================================================== #
# =========================================================================== #


class EventEngine(object):

    def __init__(self, win, p, stims):

        self.win = win
        self.p = p

        self.fix = stims["fix"]
        self.array = stims["array"]
        self.feedback = stims["feedback"]

        self.break_keys = p.resp_keys + p.quit_keys
        self.resp_keys = p.resp_keys
        self.quit_keys = p.quit_keys

        self.stim_frames = int(p.stim_dur * win.refresh_hz)

        self.clock = core.Clock()
        self.resp_clock = core.Clock()

        self.draw_feedback = True

    def __call__(self, correct_response):

        self.array.reset()

        draw_stim = True

        keys = []
        event.clearEvents()
        self.resp_clock.reset()
        self.clock.reset()
        correct = False

        self.win.nDroppedFrames = 0

        for _ in xrange(self.stim_frames):

            if not keys:
                keys = event.getKeys(self.break_keys,
                                     timeStamped=self.resp_clock)
                draw_stim = not bool(keys)

            if draw_stim:
                self.array.update()
                self.array.draw()

            self.fix.draw()
            self.win.flip()

        dropped = self.win.nDroppedFrames

        for key, key_time in keys:

            if key in self.quit_keys:
                core.quit()

            if key in self.resp_keys:
                used_key = key
                response = self.resp_keys.index(key)
                correct = response == correct_response
                rt = key_time

        self.feedback.update(correct)
        self.feedback.draw()
        self.win.flip()


# =========================================================================== #
# =========================================================================== #


class Feedback(object):

    def __init__(self, win):

        vertices = [(0, 0), (0, 1), (0, 0), (1, 0),
                    (0, 0), (0, -1), (0, 0), (-1, 0), (0, 0)]
        self.shape = visual.ShapeStim(win, vertices=vertices, lineWidth=10)

    def update(self, correct):

        self.shape.setOri([45, 0][correct])
        self.shape.setLineColor(["black", "white"][correct])

    def draw(self):

        self.shape.draw()


class StickArray(object):
    """Array of "sticks" that vary on four dimensions."""
    def __init__(self, win, p, log=None):

        self.win = win
        self.p = p
        self.rgb_colors = self.lch_to_rgb(p)
        self.random = np.random.RandomState()

        # This will draw an edge around the stimulus for debugging
        self.edge = visual.Circle(win,
                                  p.array_radius + p.disk_radius / 2,
                                  edges=128,
                                  fillColor=p.window_color,
                                  lineColor="white")

        # Initialize the stick positions
        self.new_array()

        # Initialize the feature probabilities
        self.p_hue = .5
        self.p_tilt = .5
        self.p_width = .5
        self.p_length = .5

        # Initialize the log object
        self.log = StickLog() if log is None else log

        # Initialize the feature values and twinkle trackers
        self.reset()

    def new_array(self):
        """Initialize new stick positions."""
        initial_xys = self.poisson_disc_sample()
        n_sticks = len(initial_xys)
        sticks = visual.ElementArrayStim(self.win,
                                         xys=initial_xys,
                                         nElements=n_sticks,
                                         elementTex=None,
                                         elementMask="sqr",
                                         autoLog=False)

        self.sticks = sticks
        self.n = n_sticks

    def poisson_disc_sample(self):
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
        return np.array(rgbs)

    def rotate_array(self):
        """Rotate the array of sticks by a random angle."""
        # Get the corrent positions in polar coordinates
        x, y = self.sticks.xys.T
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)

        # Add a random angle of rotation
        phi += self.random.uniform(0, 2 * np.pi)
        phi %= 2 * np.pi

        # Convert back to cartesian coordinates and update
        x, y = rho * np.cos(phi), rho * np.sin(phi)
        self.sticks.setXYs(np.c_[x, y])

    def set_feature_probs(self, p_hue, p_tilt, p_width, p_length):
        """Add attributes for the probability of features on each dimension."""
        self.p_hue = p_hue
        self.p_tilt = p_tilt
        self.p_width = p_width
        self.p_length = p_length

    def reset(self):
        """Prepare the stimulus for a new trial."""
        # Randomize the stick positions
        self.rotate_array()

        # Initialize the feature values
        self.hue_idx = self.random_idx(self.p_hue, self.n)
        self.tilt_idx = self.random_idx(self.p_tilt, self.n)
        self.width_idx = self.random_idx(self.p_width, self.n)
        self.length_idx = self.random_idx(self.p_length, self.n)

        # Initialize the object to track which sticks are being shown
        self.on = np.ones(self.n, bool)
        self.off_frames = np.zeros(self.n)

        # "Burn-in" the stimulus so it starts with average properties
        for _ in xrange(self.p.twinkle_burnin):
            self.update(log=False)

        # Add a new trial to the log
        self.log.new_trial()

    def update(self, log=True):
        """Prepare the stimulus for a new frame."""
        # Determine which sticks are turning off
        turning_off = (self.random.uniform(size=self.n) <
                       self.p.twinkle_off_prob)
        self.on &= ~turning_off

        # Update the timeout counter
        self.off_frames[~self.on] += 1

        # Determine which sticks are turning back on
        turning_on = (self.random.uniform(size=self.n) <
                      self.p.twinkle_on_prob)
        turning_on &= self.off_frames > self.p.twinkle_timeout
        self.on |= turning_on
        self.off_frames[self.on] = 0

        # Find feature values for the new sticks
        n_on = turning_on.sum()
        self.hue_idx[turning_on] = self.random_idx(self.p_hue, n_on)
        self.tilt_idx[turning_on] = self.random_idx(self.p_tilt, n_on)
        self.width_idx[turning_on] = self.random_idx(self.p_width, n_on)
        self.length_idx[turning_on] = self.random_idx(self.p_length, n_on)

        # Log the values
        if log:
            self.log.add_frame(self.on,
                               self.hue_idx,
                               self.tilt_idx,
                               self.width_idx,
                               self.length_idx)

    def random_idx(self, p, n):
        """Get indices for feature values with given proportion."""
        n_pos = np.round(p * n)
        n_neg = n - n_pos
        idx = np.r_[np.ones(n_pos, int), np.zeros(n_neg, int)]
        return self.random.permutation(idx)

    @property
    def hue_vals(self):
        return np.take(self.rgb_colors, self.hue_idx, axis=0)

    @property
    def tilt_vals(self):
        return np.take(self.p.tilts, self.tilt_idx)

    @property
    def width_vals(self):
        return np.take(self.p.widths, self.width_idx)

    @property
    def length_vals(self):
        return np.take(self.p.lengths, self.length_idx)

    def draw(self):

        # Update the psychopy object
        self.sticks.setColors(self.hue_vals)
        self.sticks.setOris(self.tilt_vals)
        self.sticks.setSizes(np.c_[self.width_vals, self.length_vals])
        self.sticks.setOpacities(self.on)

        if self.p.debug:
            self.edge.draw()
        self.sticks.draw()


class StickLog(object):
    """Object to keep track of stimulus properties over the experiment.

    The log for each attribute is an array with shape (trial, frame, object).

    """
    def __init__(self):

        self.reset()

    def reset(self):
        """Reset the log for each attribute."""
        self._on = []

        for attr in ["hue", "tilt", "width", "length"]:
            setattr(self, "_" + attr, [])
            setattr(self, "_" + attr + "_on", [])

    def new_trial(self):
        """Add a list to catch frames from a new trial."""
        self._on.append([])

        for attr in ["hue", "tilt", "width", "length"]:
            getattr(self, "_" + attr).append([])
            getattr(self, "_" + attr + "_on").append([])

    def add_frame(self, on, hue, tilt, width, length):
        """Add data for each attribute."""
        self._on[-1].append(on.copy())

        self._hue[-1].append(hue.copy())
        self._hue_on[-1].append(hue[on].copy())

        self._tilt[-1].append(tilt.copy())
        self._tilt_on[-1].append(tilt[on].copy())

        self._width[-1].append(width.copy())
        self._width_on[-1].append(width[on].copy())

        self._length[-1].append(length.copy())
        self._length_on[-1].append(length[on].copy())

    @property
    def on(self):
        return np.array(self._on)

    @property
    def hue(self):
        return np.array(self._hue)

    @property
    def tilt(self):
        return np.array(self._tilt)

    @property
    def width(self):
        return np.array(self._width)

    @property
    def length(self):
        return np.array(self._length)

    @property
    def on_prop(self):
        return np.array([np.mean(t) for t in self._on])

    @property
    def hue_prop(self):
        return np.array([np.concatenate(t).mean() for t in self._hue_on])

    @property
    def tilt_prop(self):
        return np.array([np.concatenate(t).mean() for t in self._tilt_on])

    @property
    def width_prop(self):
        return np.array([np.concatenate(t).mean() for t in self._width_on])

    @property
    def length_prop(self):
        return np.array([np.concatenate(t).mean() for t in self._length_on])

    def save(self, fname):

        data = dict()
        for attr in ["on", "hue", "tilt", "width", "length"]:
            data[attr] = getattr(self, attr)
            data[attr + "_prop"] = getattr(self, attr + "_prop")

        np.savez(fname, **data)


if __name__ == "__main__":
    main(sys.argv[1:])
