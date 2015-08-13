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

    # Text that cues the context for each block
    frame = Frame(win, p)

    # Polygon that cues the context for each block
    cue = visual.Polygon(win,
                         radius=p.poly_radius,
                         lineColor=p.poly_color,
                         fillColor=p.poly_color,
                         lineWidth=p.poly_linewidth)

    stims = dict(

        fix=fix,
        array=array,
        cue=cue

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

    with cregg.PresentationLoop(win, p):

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


def learn(p, win, stims):

    # Initialize the stimlus controller
    stim_event = EventEngine(win, p, stims)

    # Randomizer to control stimulus delivery
    rs = np.random.RandomState()

    # Set up the data log
    log_cols = ["cycle", "block", "block_trial",
                "context", "cue", "frame", "iti",
                "hue_val", "ori_val",
                "hue_strength", "ori_strength",
                "hue_prop", "ori_prop",
                "hue_prop_actual", "ori_prop_actual",
                "context_strength", "context_prop",
                "correct", "rt", "response", "key"]
    log = cregg.DataLog(p, log_cols)

    # Set up variables that will control the flow of the session
    show_guide = False
    need_practice = True
    good_blocks = {"hue": np.zeros(2), "ori": np.zeros(2)}
    cycle = -1

    with cregg.PresentationLoop(win, p):
        while need_practice:

            # Update the cycle counter
            cycle += 1

            contexts = ["hue", "ori", "hue", "ori"]
            cues = [0, 0, 1, 1]
            block_info = zip(contexts, cues)

            for cycle_block, (block_dim, block_cue) in enumerate(block_info):

                # Take a break every few blocks
                block = (cycle * 4) + cycle_block
                if block and not block % p.blocks_per_break:
                    stims["break"].draw()
                    stims["fix"].draw()
                    win.flip()
                    cregg.wait_check_quit(p.post_break_dur)

                # Update the cue
                block_frame = p.cue_frames[block_dim][block_cue]
                stims["frame"].set_texture(block_frame)

                # Track the accuracy on each trial
                block_acc = []

                # Balance the features we see in the block
                trials_per_feature = int(p.trials_per_block / 2)
                f = [0] * trials_per_feature + [1] * trials_per_feature
                features = {dim: rs.permutation(f) for dim in ["hue", "ori"]}

                for block_trial in xrange(p.trials_per_block):

                    # ITI fixation
                    iti = rs.uniform(*p.iti_params)
                    cregg.wait_check_quit(iti)

                    # Log the trial info
                    t_info = dict(cycle=cycle, block=block,
                                  block_trial=block_trial,
                                  context=block_dim, cue=block_cue,
                                  frame=block_frame, guide=show_guide,
                                  iti=iti)

                    # Get the feature values
                    trial_strengths = []
                    for dim in ["hue", "ori"]:

                        # Log the trial feature
                        feature_idx = features[dim][block_trial]
                        names = getattr(p, dim + "_features")
                        t_info[dim + "_val"] = names[feature_idx]
                        if dim == block_dim:
                            correct_response = feature_idx

                        # Determine the stick proportions on this dimension
                        prop = p.targ_prop if feature_idx else 1 - p.targ_prop
                        trial_strengths.append(prop)
                        t_info[dim + "_strength"] = np.abs(prop - .5)
                        t_info[dim + "_prop"] = prop
                        if dim == block_dim:
                            t_info["context_strength"] = np.abs(prop - .5)
                            t_info["context_prop"] = prop

                    # Update the stimulus array
                    stims["array"].set_feature_probs(*trial_strengths)

                    # Execute the trial
                    res = stim_event(correct_response)
                    t_info.update(res)
                    block_acc.append(res["correct"])
                    log.add_data(t_info)

                # Update the object tracking learning performance
                good_block = np.mean(block_acc) >= p.trial_criterion
                if good_block:
                    good_blocks[block_dim][block_cue] += 1

                # Check if we've hit criterion
                c = p.block_criterion
                at_criterion = all([all(good_blocks["hue"] >= c),
                                    all(good_blocks["ori"] >= c)])
                if at_criterion:
                    need_practice = False

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
    frames = rs.permutation(list("ABCD"))
    p.cue_frames = dict(hue=(frames[0], frames[1]),
                        ori=(frames[2], frames[3]))

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

    def __call__(self, correct_response, feedback=True, guide=False):
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

        offset = p.array_radius + p.frame_gap + p.frame_width + .3
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


class Frame(object):

    def __init__(self, win, p):

        self.p = p
        size = 2 * (p.array_radius + p.frame_gap + p.frame_width)

        self.object = visual.GratingStim(win,
                                         mask=self.mask,
                                         contrast=p.frame_contrast,
                                         size=size,
                                         sf=1 / size)

        self.textures = {}
        self.textures["A"] = self.make_ring_tex(p.frame_ring_cycles[0])
        self.textures["B"] = self.make_ring_tex(p.frame_ring_cycles[1])
        self.textures["C"] = self.make_spoke_tex(p.frame_spoke_reversals[0])
        self.textures["D"] = self.make_spoke_tex(p.frame_spoke_reversals[1])

    def set_texture(self, which):

        self.object.setTex(self.textures[which])

    def make_ring_tex(self, cycles_per_degree):

        x, y = self.meshgrid
        r = np.sqrt(x ** 2 + y ** 2)
        return np.sin(r * 2 * np.pi * cycles_per_degree)

    def make_spoke_tex(self, reversals_per_hemifield):

        x, y = self.meshgrid
        a = np.arctan(y / x)
        n = reversals_per_hemifield
        return np.sin(a * n) * np.cos(a * n) * 2

    @property
    def meshgrid(self):

        lim = self.p.array_radius + self.p.frame_gap + self.p.frame_width
        grid = np.linspace(-lim, lim, 1024)
        return np.meshgrid(grid, grid)

    @property
    def mask(self):

        x, y = self.meshgrid
        r = np.sqrt(x ** 2 + y ** 2)
        inner = self.p.array_radius + self.p.frame_gap
        outer = self.p.array_radius + self.p.frame_gap + self.p.frame_width
        return np.where((r > inner) & (r < outer), 1, -1)

    def draw(self):

        self.object.draw()


if __name__ == "__main__":
    main(sys.argv[1:])
