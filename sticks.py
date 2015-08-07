from __future__ import division, print_function
import sys
import json
import itertools
from glob import glob
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

    # Open up the stimulus window
    win = cregg.launch_window(p)
    p.win_refresh_hz = win.refresh_hz

    # Fixation point
    fix = visual.Circle(win, interpolate=True,
                        fillColor=p.fix_stim_color,
                        lineColor=p.fix_stim_color,
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

    # Instructions
    if hasattr(p, "instruct_text"):
        instruct = cregg.WaitText(win, p.instruct_text,
                                  advance_keys=p.wait_keys,
                                  quit_keys=p.quit_keys)
        stims["instruct"] = instruct

    # Text that allows subjects to take a break between blocks
    if hasattr(p, "break_text"):
        take_break = cregg.WaitText(win, p.break_text,
                                    advance_keys=p.wait_keys,
                                    quit_keys=p.quit_keys)
        stims["break"] = take_break

    # Text that alerts subjects to the end of an experimental run
    if hasattr(p, "finish_text"):
        finish_run = cregg.WaitText(win, p.finish_text,
                                    advance_keys=p.finish_keys,
                                    quit_keys=p.quit_keys)
        stims["finish"] = finish_run

    # Execute the experiment function
    globals()[mode](p, win, stims)


def counterbalance_feature_response_mapping(p):
    """Randomize the feature/response mappings by the subject ID."""
    # Note that ori features always map left-left and right-right
    subject = p.subject + "_features"
    cbid = p.cbid + "_features" if p.cbid is not None else None
    rs = cregg.subject_specific_state(subject, cbid)
    flip_hue, flip_width, flip_length = rs.binomial(1, .5, 3)

    # Counterbalance the hue features
    if flip_hue:
        p.hues = p.hues[1], p.hues[0]
        p.hue_features = p.hue_features[1], p.hue_features[0]

    # Counterbalance the width features
    if flip_width:
        p.widths = p.widths[1], p.widths[0]
        p.width_features = p.width_features[1], p.width_features[0]

    # Counterbalance the length features
    if flip_length:
        p.lengths = p.lengths[1], p.lengths[0]
        p.length_features = p.length_features[1], p.length_features[0]

    return p




# =========================================================================== #
# =========================================================================== #


class EventEngine(object):
    """Controller object for trial events."""
    def __init__(self, win, p, stims):

        self.win = win
        self.p = p

        self.fix = stims["fix"]
        self.array = stims["array"]
        self.feedback = stims["feedback"]
        self.guide = stims.get("guide", None)

        self.break_keys = p.resp_keys + p.quit_keys
        self.resp_keys = p.resp_keys
        self.quit_keys = p.quit_keys

        self.stim_frames = int(p.stim_dur * win.refresh_hz)

        self.clock = core.Clock()
        self.resp_clock = core.Clock()

        self.draw_feedback = True

    def __call__(self, correct_response, feedback=True, guide=False):
        """Execute a stimulus event."""
        self.array.reset()

        # This will turn to False after a response is recorded
        draw_stim = True

        # Prepare to catch keypresses and record RTs
        keys = []
        event.clearEvents()
        self.resp_clock.reset()
        self.clock.reset()

        # Initialize the output variables
        correct = False
        used_key = np.nan
        response = np.nan
        rt = np.nan

        # Reset the window droped frames counter
        self.win.nDroppedFrames = 0

        # Precisely control the stimulus duration
        for _ in xrange(self.stim_frames):

            # Look for valid keys and stop drawing if we find them
            if not keys:
                keys = event.getKeys(self.break_keys,
                                     timeStamped=self.resp_clock)
                draw_stim = not bool(keys)

            # Show a new frame of the stimulus
            if draw_stim:
                self.array.update()
                self.array.draw()
                if guide:
                    self.guide.draw()

            # Flip the window and block until a screen refresh
            self.fix.draw()
            self.win.flip()

        # Count the dropped frames
        dropped = self.win.nDroppedFrames - 1

        # Go through the list of recorded keys and determine the response
        for key, key_time in keys:

            if key in self.quit_keys:
                core.quit()

            if key in self.resp_keys:
                used_key = key
                response = self.resp_keys.index(key)
                correct = response == correct_response
                rt = key_time

        # Show the feedback
        if feedback:
            self.feedback.update(correct)
            self.feedback.draw()
        self.win.flip()

        return dict(correct=correct, key=used_key, rt=rt,
                    response=response,
                    dropped_frames=dropped)


# =========================================================================== #
# =========================================================================== #


class Feedback(object):
    """Simple glyph-based object to report correct or error."""
    def __init__(self, win):

        verts = [(-1, 0), (1, 0)], [(0, -1), (0, 1)]
        self.shapes = [visual.ShapeStim(win, vertices=v, lineWidth=10)
                       for v in verts]

    def update(self, correct):

        ori = [45, 0][correct]
        color = ["black", "white"][correct]
        for shape in self.shapes:
            shape.setOri(ori)
            shape.setLineColor(color)

    def draw(self):

        for shape in self.shapes:
            shape.draw()


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
        self.p_ori = .5

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

    def set_feature_probs(self, p_hue, p_ori, p_width, p_length):
        """Add attributes for the probability of features on each dimension."""
        self.p_hue = p_hue
        self.p_ori = p_ori

    def reset(self):
        """Prepare the stimulus for a new trial."""
        # Randomize the stick positions
        self.rotate_array()

        # Initialize the feature values
        self.hue_idx = self.random_idx(self.p_hue, self.n)
        self.ori_idx = self.random_idx(self.p_ori, self.n)

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
        self.ori_idx[turning_on] = self.random_idx(self.p_ori, n_on)

        # Log the values
        if log:
            self.log.add_frame(self.on,
                               self.hue_idx,
                               self.ori_idx)

    def random_idx(self, p, n):
        """Get indices for feature values with given proportion."""
        n_pos = np.round(p * n)
        n_neg = n - n_pos
        idx = np.r_[np.ones(n_pos, int), np.zeros(n_neg, int)]
        return self.random.permutation(idx)

    @property
    def hue_vals(self):
        psychopy_rgb = self.rgb_colors * 2 - 1
        return np.take(psychopy_rgb, self.hue_idx, axis=0)

    @property
    def ori_vals(self):
        return np.take(self.p.oris, self.ori_idx)

    def draw(self):

        # Update the psychopy object
        self.sticks.setColors(self.hue_vals)
        self.sticks.setOris(self.ori_vals)
        self.sticks.setOpacities(self.on)

        if self.p.debug:
            self.edge.draw()
        self.sticks.draw()


class StickLog(object):
    """Object to keep track of stimulus properties over the experiment.

    The log for each attribute is an array with shape (trial, frame, object).

    Also offers properties that compute the proportion of each feature on
    each trial across the display and trial frames.

    """
    def __init__(self):

        self.reset()

    def reset(self):
        """Reset the log for each attribute."""
        self._on = []

        for attr in ["hue", "ori"]:
            setattr(self, "_" + attr, [])
            setattr(self, "_" + attr + "_on", [])

    def new_trial(self):
        """Add a list to catch frames from a new trial."""
        self._on.append([])

        for attr in ["hue", "ori"]:
            getattr(self, "_" + attr).append([])
            getattr(self, "_" + attr + "_on").append([])

    def add_frame(self, on, hue, ori):
        """Add data for each attribute."""
        self._on[-1].append(on.copy())

        self._hue[-1].append(hue.copy())
        self._hue_on[-1].append(hue[on].copy())

        self._ori[-1].append(ori.copy())
        self._ori_on[-1].append(ori[on].copy())

    @property
    def on(self):
        return np.array(self._on)

    @property
    def hue(self):
        return np.array(self._hue)

    @property
    def ori(self):
        return np.array(self._ori)

    @property
    def on_prop(self):
        return np.array([np.mean(t) for t in self._on])

    @property
    def hue_prop(self):
        hue_prop = [np.concatenate(t).mean() for t in self._hue_on if t]
        return np.array(hue_prop)

    @property
    def ori_prop(self):
        ori_prop = [np.concatenate(t).mean() for t in self._ori_on if t]
        return np.array(ori_prop)

    def save(self, fname):

        data = dict()
        for attr in ["on", "hue", "ori"]:
            data[attr] = getattr(self, attr)
            data[attr + "_prop"] = getattr(self, attr + "_prop")

        np.savez(fname, **data)


class LearningGuide(object):
    """Text with the feature-response mappings to show during training."""
    def __init__(self, win, p):

        offset = p.array_radius + .3
        self.texts = [visual.TextStim(win, pos=(-offset, 0),
                                      height=.5, alignHoriz="right"),
                      visual.TextStim(win, pos=(offset, 0),
                                      height=.5, alignHoriz="left")]

    def update(self, features):

        for feature, text in zip(features, self.texts):
            text.setText(feature)

    def draw(self):

        for text in self.texts:
            text.draw()


if __name__ == "__main__":
    main(sys.argv[1:])
