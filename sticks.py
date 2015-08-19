from __future__ import division, print_function
import sys
import json
import itertools
from string import letters
from textwrap import dedent

import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd

from colormath.color_objects import LCHabColor, sRGBColor
from colormath.color_conversions import convert_color

from psychopy import core, visual, event
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

    visual.TextStim(win, "Generating stimuli...",
                    height=p.setup_text_size).draw()
    win.flip()

    # Randomize the response mappings consistently by subject
    counterbalance_feature_response_mapping(p)

    # Counterbalance the frame - cue mappings consistently by subject
    counterbalance_cues(p)

    # Load the subject specific lightness values for the second color
    subject_specific_colors(p)

    # Fixation point
    fix = Fixation(win, p)

    # The main stimulus arrays
    array = StickArray(win, p)

    # Polygon that cues the context for each block
    cue = PolygonCue(win, p)

    # The guide text that helps during training and practice
    guide = LearningGuide(win, p)

    # Progress bar to show during behavioral breaks
    progress = ProgressBar(win, p)

    stims = dict(

        fix=fix,
        cue=cue,
        array=array,
        guide=guide,
        progress=progress,

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


def instruct(p, win, stims):

    stim_event = EventEngine(win, p, stims)

    main_text = visual.TextStim(win, height=.5)
    next_text = visual.TextStim(win, "(press space to continue)",
                                height=.4,
                                pos=(0, -5))

    def slide(message, with_next_text=True):

        main_text.setText(dedent(message))
        main_text.draw()
        if with_next_text:
            next_text.draw()
        win.flip()
        cregg.wait_and_listen("space", .25)

    slide("""
          Welcome to the experiment - thank you for participating!
          """)

    slide("""
          In the experiment today, you're going to be looking at simple
          patterns and making decisions about what you see in them.
          Hit space to see the kind of pattern you'll be looking at.
          """, False)

    stims["fix"].draw()
    win.flip()
    cregg.wait_check_quit(1)
    stims["array"].set_feature_probs(.7, .7)
    stim_event(correct_response=1, feedback=False)
    stims["fix"].draw()
    win.flip()
    cregg.wait_check_quit(1)

    slide("""
          This pattern is made up of small sticks that can change in
          two different ways:

          - Each stick has a color (red or green)
          - Each stick has an orientation (left or right)
          """)

    slide("""
          For each attribute (color and orientation) there will always
          be more sticks with one of the two features.

          On each trial, you will be cued to attend to one of the two
          attributes (color or orientation) and you'll have to decide
          whether there are more sticks with one or the other feature.

          In other words, on a "color" trial, you'll have to decide
          whether there are more red or more green sticks.
          """)

    slide("""
          Try it yourself. Hit space to see the pattern again and try
          to make an "orientation" decision (left or right).

          You can just say your answer out loud for now.
          """)

    stims["fix"].draw()
    win.flip()
    cregg.wait_check_quit(1)
    stims["cue"].set_shape(p.cues["ori"][0])
    stims["array"].set_feature_probs(.7, .7)
    stim_event(correct_response=1, feedback=False)
    stims["fix"].draw()
    win.flip()
    cregg.wait_check_quit(1)

    slide("""
          If you said "right", that's correct!

          You'll get plenty of practice to learn how to make these decisions
          during the training session today.

          Before we start the practice, there are some important things you
          need to know about.
          """)

    slide("""
          The way you know what kind of decision to make is by the shape
          presented in the middle of the stimulus.

          There are four different shapes. Two of them mean you should make
          a color decision, and two of them mean you should make an
          orientation decision.

          Press space to see each of the shapes along with the rule it cues.
          """, False)

    for rule, rule_name in zip(["color", "orientation"], ["hue", "ori"]):
        for cue in [0, 1]:

            stims["cue"].set_shape(p.cues[rule_name][cue])
            stims["cue"].draw()
            slide("""
                  {}




                  """.format(rule))

    slide("""
          To respond, you'll be using your left and right hands. During
          behavioral testing, you should press the left and right shift
          buttons on the keyboard.

          During scanning, we'll ask you to respond with both your index
          and middle fingers, so you should practice using both during
          training.
          """)

    slide("""
          For orientation decisions, you should press left if you think
          more sticks are tilted to the left and right if you think more
          sticks are tilted to the right.

          For color decisions, you should press left if you think more
          sticks are {} and right if you think more sticks are {}.
          """.format(*p.hue_features))

    slide("""
          You'll get feedback on your responses.

          When you make the wrong decision, the fixation point will flicker.

          Nothing will happen when you are correct.
          """)

    slide("""
          That was a lot of information!

          Please ask the experimenter if you have any questions.
          """, False)


def prototype(p, win, stims):

    stim_event = EventEngine(win, p, stims)

    with cregg.PresentationLoop(win, p, fix=stims["fix"]):

        while True:

            context = int(np.random.rand() > .5)
            cue = np.random.choice([[3, 4], [5, 6]][context])

            stims["cue"].set_shape(cue)

            probs = np.random.choice([.3, .45, .55, .7], 2)
            stims["array"].set_feature_probs(*probs)
            correct = probs[context] > .5
            stim_event(correct)
            cregg.wait_check_quit(np.random.uniform(.5, 2))


def training(p, win, stims):

    design = training_design(p)
    behavior(p, win, stims, design)


def practice(p, win, stims):

    design = practice_design(p)
    behavior(p, win, stims, design)


def psychophys(p, win, stims):

    visual.TextStim(win, "Generating design...",
                    height=p.setup_text_size).draw()
    win.flip()
    design = psychophys_design(p)
    behavior(p, win, stims, design)


def behavior(p, win, stims, design):

    # Initialize the stimlus controller
    stim_event = EventEngine(win, p, stims)

    # Show the instructions
    stims["instruct"].draw()

    # Initialize the data log object
    log_cols = list(design.columns)
    log_cols += ["cue_onset", "stim_onset",
                 "correct", "rt", "response", "key",
                 "stim_frames", "dropped_frames"]
    log = cregg.DataLog(p, log_cols)

    # Execute the experiment
    with cregg.PresentationLoop(win, p, log=log,
                                fix=stims["fix"],
                                exit_func=behavioral_exit):

        stim_event.clock.reset()

        for t, t_info in design.iterrows():

            if t_info["break"]:

                # Show a progress bar
                stims["progress"].update_bar(t / len(design))
                stims["progress"].draw()

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
            stims["cue"].set_shape(t_info["cue"])
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


def behavioral_exit(log):
    """Report subject and computer performance."""
    df = pd.read_csv(log.fname)
    p = log.p

    pd.set_option("display.precision", 4)

    print("")
    print("Subject: {}".format(p.subject))
    print("Session: {}".format(p.exp_name))
    print("Run: {}".format(p.run))
    print("\nPerformance:")
    print("\nAccuracy:")
    print(df.pivot_table("correct", "context", "context_prop"))
    print("\nRT:")
    print(df.pivot_table("rt", "context", "context_prop"))
    print("\nDropped frames:")
    print("Max: {:d}".format(df.dropped_frames.max()))
    print("Median: {:.0f}".format(df.dropped_frames.median()))
    print("")


def scan(p, win, stims):

    # Initialize the stimlus controller
    stim_event = EventEngine(win, p, stims)

    # Generate the full design object
    design = scan_design(p)

    # Show the instructions
    stims["instruct"].draw()

    # Initialize the data log object
    log_cols = list(design.columns)
    log_cols += ["cue_onset", "stim_onset",
                 "correct", "rt", "response", "key",
                 "stim_frames", "dropped_frames"]
    log = cregg.DataLog(p, log_cols)

    # Execute the experiment
    with cregg.PresentationLoop(win, p, log=log,
                                fix=stims["fix"],
                                exit_func=scan_exit):

        stim_event.clock.reset()

        for t, t_info in design.iterrows():

            # Set the cue and stimulus attributes
            stims["cue"].set_shape(t_info["cue"])
            stims["array"].set_feature_probs(t_info["hue_prop"],
                                             t_info["ori_prop"])

            # Execute the trial
            res = stim_event(cue_time=t_info["cue_time"],
                             stim_time=t_info["stim_time"],
                             correct_response=t_info["target_response"])

            # Record the result of the trial
            t_info = t_info.append(pd.Series(res))
            log.add_data(t_info)

        # Show the fix point during leadout TRs
        stims["fix"].draw()
        finish_time = (t_info["trial_time_tr"] +
                       p.trs_per_trial +
                       t_info["iti_trs"] +
                       p.leadout_trs) * p.tr
        cregg.precise_wait(win, stim_event.clock, finish_time, stims["fix"])

        # Show the exit text
        stims["finish"].draw()


def scan_exit(log):
    """Report subject and computer performance."""
    df = pd.read_csv(log.fname)
    p = log.p

    pd.set_option("display.precision", 4)

    print("")
    print("Subject: {}".format(p.subject))
    print("Session: {}".format(p.exp_name))
    print("Run: {}".format(p.run))
    print("\nPerformance:")
    print("\nAccuracy:")
    print(df.pivot_table("correct", "context", "context_prop"))
    print("\nRT:")
    print(df.pivot_table("rt", "context", "context_prop"))
    print("\nDropped frames:")
    print("Max: {:d}".format(df.dropped_frames.max()))
    print("Median: {:.0f}".format(df.dropped_frames.median()))
    print("\nStimulus time difference:")
    time_diff = df.stim_time - df.stim_onset
    print("Mean: {:.2f}".format(time_diff.mean()))
    print("Max absolute: {:.2f}".format(time_diff.abs().max()))
    print("")



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


def counterbalance_cues(p):
    """Randomize the frame/cue assignment by the subject ID."""
    subject = p.subject + "_cues"
    cbid = p.cbid + "_cues" if p.cbid is not None else None
    rs = cregg.subject_specific_state(subject, cbid)
    cues = rs.permutation([3, 4, 5, 6])
    p.cues = dict(hue=(cues[0], cues[1]),
                  ori=(cues[2], cues[3]))


def subject_specific_colors(p):
    """Load the pre-calibrated lightness value for this subject."""
    fname = p.color_file.format(subject=p.subject, monitor=p.monitor_name)
    try:
        with open(fname) as fid:
            L = json.load(fid)["calibrated_L"]
    except IOError:
        print("Could not open {}; using defaults".format(fname))
        L = p.lightness

    p.lightness_by_hue = [p.lightness, L]


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

    def __call__(self, correct_response, cue_time=None, stim_time=None,
                 feedback=True, guide=False):
        """Execute a stimulus event."""
        self.array.reset()

        # Initialize the output variables
        correct = False
        used_key = np.nan
        response = np.nan
        rt = np.nan

        # Determine when to show the main stimulus
        if cue_time is None:
            cue_time = self.clock.getTime()
        if stim_time is None:
            stim_time = self.clock.getTime() + self.p.orient_dur

        # Wait till cue time
        cregg.precise_wait(self.win, self.clock, cue_time, self.fix)
        cue_onset = self.clock.getTime()

        # Show the orienting cue
        self.fix.color = self.p.fix_stim_color
        cregg.precise_wait(self.win, self.clock, stim_time, self.fix)

        # Prepare to catch keypresses and record RTs
        keys = []
        event.clearEvents()
        self.resp_clock.reset()

        # Reset the window droped frames counter
        self.win.nDroppedFrames = 0

        # Precisely control the stimulus duration
        for frame in xrange(self.stim_frames):

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

            # Record the time of the first flip
            if not frame:
                stim_onset = self.clock.getTime()

        # Count the dropped frames
        dropped = self.win.nDroppedFrames

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

        result = dict(cue_onset=cue_onset,
                      stim_onset=stim_onset,
                      correct=correct,
                      key=used_key,
                      rt=rt,
                      response=response,
                      stim_frames=frame,
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
        """Indicate feedback by blinking the fixation point."""
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
        for lightness, hue in zip(p.lightness_by_hue, p.stick_hues):
            lch = LCHabColor(lightness, p.chroma, hue)
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


class PolygonCue(visual.Polygon):

    def __init__(self, win, p):

        super(PolygonCue, self).__init__(win,
                                         radius=p.poly_radius,
                                         lineColor=p.poly_color,
                                         fillColor=p.poly_color,
                                         lineWidth=p.poly_linewidth)

    def set_shape(self, sides):

        self.setEdges(sides)
        self.setVertices(self.vertices)


class ProgressBar(object):

    def __init__(self, win, p):

        self.p = p

        self.width = width = p.prog_bar_width
        self.height = height = p.prog_bar_height
        self.position = position = p.prog_bar_position

        color = p.prog_bar_color
        linewidth = p.prog_bar_linewidth

        self.full_verts = np.array([(0, 0), (0, 1),
                                    (1, 1), (1, 0)], np.float)

        frame_verts = self.full_verts.copy()
        frame_verts[:, 0] *= width
        frame_verts[:, 1] *= height
        frame_verts[:, 0] -= width / 2
        frame_verts[:, 1] += position

        self.frame = visual.ShapeStim(win,
                                      fillColor=None,
                                      lineColor=color,
                                      lineWidth=linewidth,
                                      vertices=frame_verts)

        self.bar = visual.ShapeStim(win,
                                    fillColor=color,
                                    lineColor=color,
                                    lineWidth=linewidth)

    def update_bar(self, prop):

        bar_verts = self.full_verts.copy()
        bar_verts[:, 0] *= self.width * prop
        bar_verts[:, 1] *= self.height
        bar_verts[:, 0] -= self.width / 2
        bar_verts[:, 1] += self.position
        self.bar.vertices = bar_verts
        self.bar.setVertices(bar_verts)

    def draw(self):

        self.bar.draw()
        self.frame.draw()


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
            "hue", "ori", "congruent",
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
                        feat_idx = rs.randint(2)
                        feature = p[dim + "_features"][feat_idx]
                        block_design.loc[trial, dim] = feature

                        # Determine the proportion of sticks
                        prop = p.targ_prop if feat_idx else 1 - p.targ_prop
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
            "hue", "ori", "congruent",
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
    design["hue"] = hue
    design["ori"] = ori
    design["guide"] = p.guides
    design["iti"] = rs.uniform(*p.iti_params, size=p.trials)

    # Assign the cues
    for dim in ["hue", "ori"]:
        cue_idx = rs.randint(0, 2, p.trials / 2)
        cues = np.array(p.cues[dim])[cue_idx]
        design.loc[design.context == dim, "cue_idx"] = cue_idx
        design.loc[design.context == dim, "cue"] = cues

    # Assign the feature proportions
    for dim in ["hue", "ori"]:
        names = p[dim + "_features"]
        design.loc[design[dim] == names[1], dim + "_prop"] = p.targ_prop
        design.loc[design[dim] == names[0], dim + "_prop"] = 1 - p.targ_prop

    # Add additional columns based on the information currently in the design
    design = add_design_information(design, p)

    return design


def psychophys_design(p, rs=None):

    if rs is None:
        rs = np.random.RandomState()

    # Set up the structure of the design object
    cols = [
            "context", "cue_idx", "cue", "guide",
            "context_switch", "cue_switch",
            "hue_switch", "ori_switch",
            "iti", "break",
            "hue", "ori", "congruent",
            "hue_prop", "ori_prop",
            "hue_strength", "ori_strength",
            "context_prop", "context_strength",
            ]

    # Find the cartesian product of the different variables
    conditions = itertools.product(["hue", "ori"], [0, 1],
                                   p.hue_features, p.ori_features,
                                   p.targ_props, p.targ_props)
    condition_cols = ["context", "cue_idx", "hue", "ori",
                      "hue_targ_prop", "ori_targ_prop"]
    conditions = pd.DataFrame(list(conditions) * p.cycles,
                              columns=condition_cols)

    # Set up the design object
    ntrials = len(conditions)
    design = pd.DataFrame(columns=cols, index=np.arange(ntrials))

    # Add information that isn't contingenton the main variables
    design["guide"] = False
    design["iti"] = rs.uniform(*p.iti_params, size=ntrials)

    # Add a subset of the main variables
    design[condition_cols[:4]] = conditions.iloc[:, :4]

    # Add the feature proportions
    for dim in ["hue", "ori"]:
        lower_prop_feature = design[dim] == p[dim + "_features"][0]
        design[dim + "_prop"] = np.where(lower_prop_feature,
                                         1 - conditions[dim + "_targ_prop"],
                                         conditions[dim + "_targ_prop"])

    # Do the mapping from context cue index to identity
    for dim in ["hue", "ori"]:
        cue_idx = design.loc[design.context == dim, "cue_idx"]
        cues = np.array(p.cues[dim])[cue_idx.values]
        design.loc[design.context == dim, "cue"] = cues

    # Explore a large space to balance context switches with respect to
    # the coherence values on the two dimension
    best_permuter = None
    best_cost = np.inf
    for _ in xrange(p.permutation_attempts):

        # Randomize the rows of the design
        permuter = rs.permutation(design.index)
        candidate = design.iloc[permuter].copy().reset_index(drop=True)

        # Identify context switch trials
        candidate["context_switch"] = (candidate.context !=
                                       candidate.context.shift(1))

        # Find the proportion of switch trials within each bin
        switch_table = candidate.pivot_table(values="context_switch",
                                             index="hue_prop",
                                             columns="ori_prop")

        # Compute how far this deviates from ideal and save
        ideal = np.ones(switch_table.shape) * .5
        cost = np.square(switch_table - ideal).stack().sum()
        if cost < best_cost:
            best_cost = cost
            best_permuter = permuter

    # Randomize the design using the winning order
    design = design.iloc[best_permuter].reset_index(drop=True)

    # Add in other information that can be derived from what we already know
    design = add_design_information(design, p)

    return design


def scan_design(p):

    # Use a predictably random schedule for each run
    rs = cregg.subject_specific_state(p.subject, p.cbid)
    labels = list(letters[:p.n_designs].lower())
    run_label = rs.permutation(labels)[p.run - 1]
    schedule_fname = p.design_base.format(run_label)
    schedule = pd.read_csv(schedule_fname)

    # Load the subject's strength file
    fname = p.strength_file_base.format(subject=p.subject) + ".json"
    try:
        with open(fname) as fid:
            stim_strength = json.load(fid)
    except IOError:
        print("Could not open {}; using defaults".format(fname))
        stim_strength = p.strength_defaults

    # Initialize the design object
    cols = [
            "context", "cue_idx", "cue",
            "context_switch", "cue_switch",
            "hue_switch", "ori_switch",
            "hue", "ori", "congruent",
            "hue_diff", "ori_diff",
            "hue_prop", "ori_prop",
            "hue_strength", "ori_strength",
            "context_prop", "context_strength",
            "iti_trs", "iti", "break",
            "trial_time_tr",
            "cue_time", "stim_time",
            "target_response",
            ]

    trials = np.arange(len(schedule))
    design = pd.DataFrame(columns=cols, index=trials)
    design.update(schedule)

    # Do the mapping from context cue index to identity
    for dim in ["hue", "ori"]:
        cue_idx = design.loc[design.context == dim, "cue_idx"]
        cues = np.array(p.cues[dim])[cue_idx.values.astype(int)]
        design.loc[design.context == dim, "cue"] = cues

    # Assign strengths to each feature at each difficulty level
    for dim in ["hue", "ori"]:
        for diff in ["easy", "hard"]:
            idx = design[dim + "_diff"] == diff
            design.loc[idx, dim + "_strength"] = stim_strength[dim][diff]
        for mul, feat in zip([-1, 1], p[dim + "_features"]):
            idx = design[dim] == feat
            prop = (.5 + mul * design.loc[idx, dim + "_strength"])
            design.loc[idx, dim + "_prop"] = prop.astype(np.float).round(2)

    # Convert ITI duration from TR units to seconds
    design["iti"] = design.iti_trs * p.tr

    # Schedule cue and stimulus onset in seconds
    design["stim_time"] = (design.trial_time_tr + 1) * p.tr
    design["cue_time"] = design.stim_time - p.orient_dur

    # No breaks in the scan session
    design["break"] = False

    # Add columns dependent on information in the design
    design = add_design_information(design, p)

    return design


def add_design_information(d, p):
    """Add information that is derived from other columns in the design."""
    # Determine when there, context, cue, or evidence switches
    d["context_switch"] = d.context != d.context.shift(1)
    d["cue_switch"] = d.cue != d.cue.shift(1)
    for dim in ["hue", "ori"]:
        d[dim + "_switch"] = d[dim != d[dim].shift(1)]

    # Determine the strength (unsigned correspondence of proportion
    for dim in ["hue", "ori"]:
        if dim + "_prop" in d:
            d[dim + "_strength"] = ((d[dim + "_prop"] - .5).abs()
                                                           .astype(np.float)
                                                           .round(2))

    # Determine the proportion and strength of the relevent evidence
    for dim in ["hue", "ori"]:
        if dim + "_prop" in d:
            idx = d.context == dim
            d.loc[idx, "context_strength"] = d.loc[idx, dim + "_strength"]
            d.loc[idx, "context_prop"] = d.loc[idx, dim + "_prop"]

    # Determine evidence congruency
    hue_resp = d["hue"] == p.hue_features[1]
    ori_resp = d["ori"] == p.ori_features[1]
    d["congruent"] = hue_resp == ori_resp

    # Determine the correct response
    for dim in ["hue", "ori"]:
        idx = d.context == dim
        right_target = p[dim + "_features"][1]
        d.loc[idx, "target_response"] = d.loc[idx, dim] == right_target

    # Determine when there will be breaks
    d["break"] = ~(d.index.values % p.trials_per_break).astype(bool)
    d.loc[0, "break"] = False

    return d


if __name__ == "__main__":
    main(sys.argv[1:])
