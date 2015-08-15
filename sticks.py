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

    visual.TextStim(win, "Generating stimuli...").draw()
    win.flip()

    # Randomize the response mappings consistently by subject
    counterbalance_feature_response_mapping(p)

    # Counterbalance the frame - cue mappings consistently by subject
    counterbalance_cues(p)

    # Fixation point
    fix = Fixation(win, p)

    # The main stimulus arrays
    array = StickArray(win, p)

    # Polygon that cues the context for each block
    cue = visual.Polygon(win,
                         radius=p.poly_radius,
                         lineColor=p.poly_color,
                         fillColor=p.poly_color,
                         lineWidth=p.poly_linewidth)

    # The guide text that helps during training and practice
    guide = LearningGuide(win, p)

    stims = dict(

        fix=fix,
        cue=cue,
        array=array,
        guide=guide,

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


# =========================================================================== #
# =========================================================================== #


def prototype(p, win, stims):

    stim_event = EventEngine(win, p, stims)

    with cregg.PresentationLoop(win, p, fix=stims["fix"]):

        while True:

            context = int(np.random.rand() > .5)
            cue = np.random.choice([[3, 4], [5, 6]][context])

            stims["cue"].setEdges(cue)
            stims["cue"].setVertices(stims["cue"].vertices)

            probs = np.random.choice([.3, .45, .55, .7], 2)
            stims["array"].set_feature_probs(*probs)
            correct = probs[context] > .5
            stim_event(correct)
            cregg.wait_check_quit(np.random.uniform(.5, 2))


def training(p, win, stims):

    # Create a design for this session
    design = training_design(p)

    behavior(p, win, stims, design)


def practice(p, win, stims):

    # Create a design for this session
    design = practice_design(p)

    behavior(p, win, stims, design)


def behavior(p, win, stims, design):

    # Initialize the stimlus controller
    stim_event = EventEngine(win, p, stims)

    # Show the instructions
    stims["instruct"].draw()

    # Initialize the data log object
    log_cols = list(design.columns)
    log_cols += ["correct", "rt", "response", "key",
                 "stim_frames", "dropped_frames"]
    log = cregg.DataLog(p, log_cols)

    # Execute the experiment
    with cregg.PresentationLoop(win, p, log=log):

        for t, t_info in design.iterrows():

            if t_info["break"]:

                # Show the break message
                stims["break"].draw()

                # Add a little delay after the break
                stims["fix"].draw()
                win.flip()
                cregg.wait_check_quit(p.after_break_dur)

            # Wait for the ITI before the stimulus
            # This helps us relate pre-stim delay to behavior later
            cregg.wait_check_quit(t_info["iti"])

            # Set the cue and stimulus attributes
            stims["cue"].setEdges(t_info["cue"])
            stims["cue"].setVertices(stims["cue"].vertices)
            button_names = p[t_info["context"] + "_features"]
            stims["guide"].update(button_names)
            stims["array"].set_feature_probs(t_info["hue_prop"],
                                             t_info["ori_prop"])

            # Execute the trial
            res = stim_event(correct_response=t_info["target_response"],
                             guide=t_info["guide"])

            # Record the result of the trial
            t_info = t_info.append(pd.Series(res))
            log.add_data(t_info)

        # Show the exit text
        stims["finish"].draw()


# =========================================================================== #
# =========================================================================== #


def counterbalance_feature_response_mapping(p):
    """Randomize the color/response mappings by the subject ID."""
    # Note that ori features always map left-left and right-right
    subject = p.subject + "_features"
    cbid = p.cbid + "_features" if p.cbid is not None else None
    rs = cregg.subject_specific_state(subject, cbid)
    flip_hue = rs.binomial(1, .5)

    # Counterbalance the hue features
    if flip_hue:
        p.stick_hues = p.stick_hues[1], p.stick_hues[0]
        p.hue_features = p.hue_features[1], p.hue_features[0]

    return p


def counterbalance_cues(p):
    """Randomize the frame/cue assignment by the subject ID."""
    subject = p.subject + "_cues"
    cbid = p.cbid + "_cues" if p.cbid is not None else None
    rs = cregg.subject_specific_state(subject, cbid)
    cues = rs.permutation([3, 4, 5, 6])
    p.cues = dict(hue=(cues[0], cues[1]),
                  ori=(cues[2], cues[3]))

    return p


# =========================================================================== #
# =========================================================================== #


class EventEngine(object):
    """Controller object for trial events."""
    def __init__(self, win, p, stims):

        self.win = win
        self.p = p

        self.fix = stims.get("fix", None)
        self.cue = stims.get("cue", None)
        self.array = stims.get("array", None)
        self.guide = stims.get("guide", None)

        self.break_keys = p.resp_keys + p.quit_keys
        self.resp_keys = p.resp_keys
        self.quit_keys = p.quit_keys

        self.stim_frames = int(p.stim_timeout * win.refresh_hz)
        self.feedback_frames = int(p.feedback_dur * win.refresh_hz)

        self.feedback_flip_every = [
            np.inf if fb_hz is None else int(win.refresh_hz / fb_hz)
            for fb_hz in p.feedback_hz]

        self.clock = core.Clock()
        self.resp_clock = core.Clock()

        self.draw_feedback = True

    def __call__(self, correct_response, stim_time=None,
                 feedback=True, guide=False):
        """Execute a stimulus event."""
        self.array.reset()

        # Show the orienting cue
        self.fix.color = self.p.fix_stim_color
        self.fix.draw()
        self.win.flip()
        cregg.wait_check_quit(self.p.orient_dur)

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
        for frame in xrange(1, self.stim_frames + 1):

            # Look for valid keys and stop drawing if we find them
            keys = event.getKeys(self.break_keys,
                                 timeStamped=self.resp_clock)
            if keys:
                break

            # Show a new frame of the stimulus
            self.array.update()
            self.array.draw()
            self.cue.draw()
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
            self.show_feedback(correct)

        result = dict(correct=correct, key=used_key, rt=rt,
                      response=response, stim_frames=frame,
                      dropped_frames=dropped)

        # Reset the fixation point to the ITI color
        self.fix.color = self.p.fix_iti_color
        self.fix.draw()
        self.win.flip()

        # Compute the amount of stimulus evidence over the trial
        for dim in ["hue", "ori"]:
            frame_data = [np.mean(f) for f in getattr(self.array.log, dim)[-1]]
            result[dim + "_prop_actual"] = np.mean(frame_data)

        return result

    def show_feedback(self, correct):

        flip_every = self.feedback_flip_every[int(correct)]
        for frame in xrange(self.feedback_frames):
            if not frame % flip_every:
                self.fix.color = -1 * self.fix.color
            self.fix.draw()
            self.win.flip()


# =========================================================================== #
# =========================================================================== #


class StickArray(object):
    """Array of "sticks" that vary on four dimensions."""
    def __init__(self, win, p, log=None):

        self.win = win
        self.p = p
        self.rgb_colors = self.lch_to_rgb(p)
        self.stick_sizes = p.stick_width, p.stick_length
        self.random = np.random.RandomState()

        # This will draw an edge around the stimulus for debugging
        self.edge = visual.Circle(win,
                                  p.array_radius,
                                  edges=128,
                                  fillColor=None,
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
                                         sizes=self.stick_sizes,
                                         autoLog=False)

        self.sticks = sticks
        self.n = n_sticks

    def poisson_disc_sample(self):
        """Find positions using poisson-disc sampling."""
        # See http://bost.ocks.org/mike/algorithms/
        uniform = self.random.uniform
        randint = self.random.randint

        # Parameters of the sampling algorithm
        array_radius = self.p.array_radius - self.p.disk_radius
        fixation_radius = self.p.fixation_radius
        radius = self.p.disk_radius
        candidates = self.p.disk_candidates

        # Start at a fixed point we know will work
        start = 0, array_radius / 2
        samples = [start]
        queue = [start]

        while queue:

            # Pick a sample to expand from
            s_idx = randint(len(queue))
            s_x, s_y = queue[s_idx]

            for i in xrange(candidates):

                # Generate a candidate from this sample
                a = uniform(0, 2 * np.pi)
                r = uniform(radius, 2 * radius)
                x, y = s_x + r * np.cos(a), s_y + r * np.sin(a)

                # Check the three conditions to accept the candidate
                in_array = np.sqrt(x ** 2 + y ** 2) < array_radius
                in_ring = np.all(cdist(samples, [(x, y)]) > radius)
                in_fixation = np.sqrt(x ** 2 + y ** 2) < fixation_radius

                if in_array and in_ring and not in_fixation:
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
        for hue in p.stick_hues:
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

    def set_feature_probs(self, p_hue, p_ori):
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
        return np.take(self.p.stick_oris, self.ori_idx)

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

        offset = p.array_radius + p.guide_offset
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


class Fixation(object):

    def __init__(self, win, p):

        self.dot = visual.Circle(win, interpolate=True,
                                 fillColor=p.fix_iti_color,
                                 lineColor=p.fix_iti_color,
                                 size=p.fix_size)

        self._color = p.fix_iti_color

    @property
    def color(self):

        return self._color

    @color.setter  # pylint: disable-msg=E0102r
    def color(self, color):

        self._color = color
        self.dot.setFillColor(color)
        self.dot.setLineColor(color)

    def draw(self):

        self.dot.draw()


# =========================================================================== #
# =========================================================================== #


def training_design(p, rs=None):

    if rs is None:
        rs = np.random.RandomState()

    # Set up the structure of the design object
    cols = [
            "cycle", "block",
            "context", "cue_idx", "cue", "guide",
            "context_switch", "cue_switch",
            "hue_switch", "ori_switch",
            "iti", "break",
            "hue_val", "ori_val", "congruent",
            "hue_prop", "ori_prop",
            "hue_strength", "ori_strength",
            "context_prop", "context_strength",
            ]
    design = []

    # Iterate over the cycle parameters
    for info in zip(p.block_lengths, p.cycles_per_length,
                    p.randomize_blocks, p.show_guides):

        block_length, n_cycles, randomize, guide = info

        # Iterate over the cycles within each level of
        # "cycle" is an appearance of each cue
        for cycle in xrange(n_cycles):

            # Posibly randomize the order of contexts/cues
            if randomize:
                block_contexts = rs.permutation(["hue", "hue", "ori", "ori"])
                block_cues = dict(hue=list(rs.permutation([0, 1])),
                                  ori=list(rs.permutation([0, 1])))
            else:
                block_contexts = ["hue", "ori", "hue", "ori"]
                block_cues = dict(hue=[0, 1], ori=[0, 1])

            # Iterate over the blocks in each cycle
            # "block" is some number of trials with the same cue
            for block, block_dim in enumerate(block_contexts):

                # Find the cue index and actual cue (polygon shape)
                block_cue_idx = block_cues[block_dim].pop()
                block_cue = p.cues[block_dim][block_cue_idx]

                # Set up a design object for this block
                block_trials = pd.Series(np.arange(block_length),
                                         name="block_trial")
                block_design = pd.DataFrame(columns=cols, index=block_trials)

                # Insert design information that is constant across the block
                block_design["cycle"] = cycle
                block_design["block"] = block
                block_design["context"] = block_dim
                block_design["cue_idx"] = block_cue_idx
                block_design["cue"] = block_cue
                block_design["guide"] = guide

                # Find ITI values for each trial
                block_design["iti"] = rs.uniform(*p.iti_params,
                                                 size=block_length)

                # Iterate over trials within the block
                for trial in xrange(block_length):

                    # Assign feature values for each dimension
                    for dim in ["hue", "ori"]:
                        dim_val_idx = rs.randint(2)
                        dim_names = p[dim + "_features"]
                        dim_val = dim_names[dim_val_idx]
                        block_design.loc[trial, dim + "_val"] = dim_val

                        # Determine the proportion of sticks
                        prop = p.targ_prop if dim_val_idx else 1 - p.targ_prop
                        block_design.loc[trial, dim + "_prop"] = prop

                design.append(block_design.reset_index())

    # Combine each chunk of the design into one file
    design = pd.concat(design, ignore_index=True)

    # Add additional columns based on the information currently in the design
    design = add_design_information(design, p)

    return design


def practice_design(p, rs=None):

    if rs is None:
        rs = np.random.RandomState()

    # Set up the structure of the design object
    cols = [
            "context", "cue_idx", "cue", "guide",
            "context_switch", "cue_switch",
            "hue_switch", "ori_switch",
            "iti", "break",
            "hue_val", "ori_val", "congruent",
            "hue_prop", "ori_prop",
            "hue_strength", "ori_strength",
            "context_prop", "context_strength",
            ]

    # Determine the major parameters for each trial
    context = rs.permutation(np.tile(["hue", "ori"], p.trials / 2))
    hue = rs.permutation(np.tile(p.hue_features, p.trials / 2))
    ori = rs.permutation(np.tile(p.ori_features, p.trials / 2))

    # Begin setting up the design object
    trials = np.arange(p.trials)
    design = pd.DataFrame(columns=cols, index=trials)
    design["context"] = context
    design["hue_val"] = hue
    design["ori_val"] = ori
    design["guide"] = p.guides
    design["iti"] = rs.uniform(*p.iti_params, size=p.trials)

    # Assign the cues
    for dim in ["hue", "ori"]:
        cue_idx = np.tile([0, 1], p.trials / 4)
        cues = np.array(p.cues[dim])[cue_idx]
        design.loc[design.context == dim, "cue_idx"] = cue_idx
        design.loc[design.context == dim, "cue"] = cues

    # Assign the feature proportions
    for dim in ["hue", "ori"]:
        names = p[dim + "_features"]
        features = design[dim + "_val"]
        design.loc[features == names[1], dim + "_prop"] = p.targ_prop
        design.loc[features == names[0], dim + "_prop"] = 1 - p.targ_prop

    # Add additional columns based on the information currently in the design
    design = add_design_information(design, p)

    return design


def add_design_information(d, p):
    """Add information that is derived from other columns in the design."""
    # Determine when there, context, cue, or evidence switches
    d["context_switch"] = d.context != d.context.shift(1)
    d["cue_switch"] = d.cue != d.cue.shift(1)
    for dim in ["hue", "ori"]:
        d[dim + "_switch"] = d[dim + "_val"] != d[dim + "_val"].shift(1)

    # Determine the strength (unsigned correspondence of proportion
    for dim in ["hue", "ori"]:
        d[dim + "_strength"] = ((d[dim + "_prop"] - .5).abs()
                                                       .astype(np.float)
                                                       .round(2))

    # Determine the proportion and strength of the relevent evidence
    for dim in ["hue", "ori"]:
        idx = d.context == dim
        d.loc[idx, "context_strength"] = d.loc[idx, dim + "_strength"]
        d.loc[idx, "context_prop"] = d.loc[idx, dim + "_prop"]

    # Determine evidence congruency
    hue_resp = d["hue_val"] == p.hue_features[1]
    ori_resp = d["ori_val"] == p.ori_features[1]
    d["congruent"] = hue_resp == ori_resp

    # Determine the correct response
    d["target_response"] = (d["context_prop"] > .5).astype(int)

    # Determine when there will be breaks
    d["break"] = ~(d.index.values % p.trials_per_break).astype(bool)
    d.loc[0, "break"] = False


    return d


if __name__ == "__main__":
    main(sys.argv[1:])
