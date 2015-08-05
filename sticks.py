from __future__ import division
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

    # Randomize the response mappings consistently by subject
    counterbalance_feature_response_mapping(p)

    # Randomize the dimensions by subject
    counterbalance_dimension_order(p)

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
    # Note that tilt features always map left-left and right-right
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


def counterbalance_dimension_order(p):
    """Randomize the order of dimensions for the training manipulation."""
    subject = p.subject + "_dimensions"
    cbid = p.cbid + "_dimensions" if p.cbid is not None else None
    rs = cregg.subject_specific_state(subject, cbid)
    p.dim_counterbal = list(rs.permutation(p.dim_names))


def make_staircases(p, stair_file=None):
    """Make staircases, either de novo or from previous values."""
    if stair_file is None:
        stairs = {dim: [StairHandler(startVal=p.stair_start,
                                     stepSizes=p.stair_step,
                                     nTrials=np.inf,
                                     nUp=1, nDown=4,
                                     stepType="lin",
                                     minVal=0,
                                     maxVal=.5) for _ in range(p.n_staircases)]
                  for dim in p.dim_names}
    else:

        # Open the file that has the final state of each staircase
        with open(stair_file) as fid:
            previous_stairs = json.load(fid)

        stairs = {dim: [] for dim in p.dim_names}
        for dim in p.dim_names:
            for val in previous_stairs[dim]:
                stairs[dim].append(StairHandler(startVal=val,
                                                stepSizes=p.stair_step,
                                                nTrials=np.inf,
                                                nUp=1, nDown=4,
                                                stepType="lin",
                                                minVal=0,
                                                maxVal=.5))

    # Set all stairs pointing up to avoid getting stuck in weird places
    for dim, dim_stairs in stairs.items():
        for sub_stairs in dim_stairs:
            sub_stairs.currentDirection = "up"

    return stairs


def save_staircase_values(stairs, json_fname):
    """Save final staircase values to a json file."""
    stair_vals = {}
    for dim, dim_stairs in stairs.iteritems():
        stair_vals[dim] = []
        for sub_stairs in dim_stairs:
            stair_vals[dim].append(sub_stairs.next())
    cregg.archive_old_version(json_fname)
    with open(json_fname, "w") as fid:
        json.dump(stair_vals, fid)


def prototype(p, win, stims):
    """Simple prototype loop of the stimulus."""
    stim_event = EventEngine(win, p, stims)

    with cregg.PresentationLoop(win, p):

        stims["break"].draw()

        while True:

            dim = np.random.choice(p.dim_names)
            dim_idx = p.dim_names.index(dim)

            stims["cue"].setText(dim)
            stims["cue"].draw()
            win.flip()
            cregg.wait_check_quit(p.cue_dur)

            for trial in range(4):

                ps = [.3, .4, .45, .55, .6, .7]
                trial_ps = np.random.choice(ps, 4)
                stims["array"].set_feature_probs(*trial_ps)
                correct_resp = trial_ps[dim_idx] > .5
                stim_event(correct_resp)
                cregg.wait_check_quit(p.feedback_dur)

            stims["fix"].draw()
            win.flip()
            cregg.wait_check_quit(2)


def learn(p, win, stims):
    """Initial learning session to acquaint subject with the task."""
    # Initialize the learning guide
    stims["guide"] = LearningGuide(win, p)

    # Initialize the trial controller
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
    stims["array"].set_feature_probs(.7, .7, .7, .7)
    stim_event(correct_response=1, feedback=False)

    slide("""
          This pattern is made up of small sticks that can change in
          four different ways on each trial.
          """)

    slide("""
          Here are the different features each stick can have:

          - hue: red or green
          - tilt: left or right
          - width: thick or thin
          - length: long or short
          """)

    slide("""
          The patterns will be a mix of these, and for each dimension
          (i.e. hue, tilt, width, and length), there will always be more
          sticks with one of the two features.

          On each trial, you'll only have to pay attention to one of these
          dimensions (we will tell you which one).

          Your job will be to decide whether there are more sticks with one
          feature or the other feature on that dimension. In other words, if
          the rule is "hue", you have to decide whether there are more red
          or more green sticks.
          """)

    slide("""
          Try it for yourself. Hit space to see the pattern again and try
          to make a "tilt" decision (left or right).
          """)

    stims["fix"].draw()
    win.flip()
    cregg.wait_check_quit(1)
    stims["array"].set_feature_probs(.7, .7, .7, .7)
    stim_event(correct_response=1, feedback=False)

    slide("""
          If you said "right", that's correct!

          You'll get plenty of practice to learn how to make these decisions
          before we start the main experiment.

          Before we start the practice, there are some important things you
          need to know about.
          """)

    slide("""
          The way the experiment works is that there will be blocks of trials
          where you need to make decisions using the same rule.

          Each block will start with a word (e.g., "tilt") and then you will
          see several different patterns that you have to make "tilt"
          decisions about.

          Every time you see a new cue, you'll need to change what kind of
          decision you are making.
          """)

    slide("""
          To respond, you'll be using the < and > keys on the keyboard.

          The meaning of these keys changes with the rule. So the < key might
          mean "short" in a "length" block but "thick" in a "width" block.

          At the start of the practice, there will be labels on the screen to
          help you learn which button to press for each decision, but these
          won't be present during the main experiment so you will need to learn
          what each button means.
          """)

    slide("""
          You'll get feedback on your responses to let you know if you were
          right or wrong.

          A white "+" will mean you were right, and a black "x" will mean
          you were wrong.

          You'll have a limited time to respond on each trial, and if you
          don't get a response in, it will be counted as wrong.
          """)

    slide("""
          That was a lot of information!

          Please tell the experimenter if you have any questions.

          Press space once you are ready to start the practice session.

          Remember, you're using the < and > keys to respond.
          """, False)

    stims["fix"].draw()
    win.flip()
    cregg.wait_check_quit(2)

    # -----------------------------------------------------------------------

    # Set up a randomizer for the coherences
    rs = np.random.RandomState()

    # Set up mid-session instructions
    post_guide_instruct = cregg.WaitText(win, p.post_guide_instruct_text,
                                         advance_keys=p.wait_keys,
                                         quit_keys=p.quit_keys)

    # Set up the data log
    log_cols = ["cycle", "block",
                "block_trial", "context", "guide",
                "hue_val", "tilt_val", "width_val", "length_val",
                "correct", "rt", "response", "key"]
    log = cregg.DataLog(p, log_cols)

    # Randomize the order of contexts that appear in each cycle
    block_order = rs.permutation(p.dim_names)

    # Set up variables that will control the flow of the session
    show_guide = True
    need_practice = True
    good_blocks = {dim: 0 for dim in p.dim_names}
    cycle = -1

    # Execute the experiment
    with cregg.PresentationLoop(win, p):
        while need_practice:

            # Update the cycle counter
            cycle += 1

            for cycle_block, block_dim in enumerate(block_order):

                # Take a break every few blocks
                block = (cycle * 4) + cycle_block
                if block and not block % p.blocks_per_break:
                    stims["break"].draw()
                    stims["fix"].draw()
                    win.flip()

                # Show the cue
                stims["cue"].setText(block_dim)
                stims["cue"].draw()
                win.flip()
                cregg.wait_check_quit(p.cue_dur)

                # Update the guide text
                stims["guide"].update(getattr(p, block_dim + "_features"))

                # Track the accuracy on each trial
                block_acc = []

                # Balance the features we see in the block
                features = [rs.permutation([True, True, False, False])
                            for _ in range(4)]
                features = np.array(features).T

                for block_trial in xrange(p.trials_per_block):

                    # Determine the features for each dimension
                    trial_features = features[block_trial]

                    # Log the trial info
                    t_info = dict(cycle=cycle, block=block,
                                  block_trial=block_trial,
                                  context=block_dim, guide=show_guide)

                    # Set the stimulus coherences
                    coh = p.coherence
                    trial_coherences = []
                    for dim, feature in zip(p.dim_names, trial_features):

                        # Log the trial feature
                        names = getattr(p, dim + "_features")
                        t_info[dim + "_val"] = names[int(feature)]

                        # Determine the stick coherence on this dimension
                        dim_coh = coh if feature else 1 - coh
                        if dim == block_dim:
                            rel_coh = dim_coh
                        trial_coherences.append(dim_coh)

                    # Configure the stimulus array
                    stims["array"].set_feature_probs(*trial_coherences)

                    # Execute the trial
                    correct_response = rel_coh > .5
                    res = stim_event(correct_response, guide=show_guide)
                    t_info.update(res)
                    block_acc.append(res["correct"])
                    log.add_data(t_info)

                    cregg.wait_check_quit(p.feedback_dur)

                # Update the object tracking learning performance
                good_block = np.mean(block_acc) > p.trial_criterion
                if good_block:
                    good_blocks[block_dim] += 1

                # Check if we've hit criterion
                at_criterion = all([v >= p.block_criterion
                                    for v in good_blocks.values()])
                if show_guide:
                    # Check if we can take the training wheels off
                    if at_criterion:
                        show_guide = False
                        good_blocks = {dim: 0 for dim in p.dim_names}
                        post_guide_instruct.draw()
                else:
                    # Check if we are done learning
                    if at_criterion:
                        need_practice = False

        stims["finish"].draw()


def psychophys(p, win, stims):
    """Method of constants psychophysics experiment."""
    # Initialize the trial controller
    stim_event = EventEngine(win, p, stims)

    # Create a design for this run
    design = psychophys_design(p)

    # Show the instructions
    stims["instruct"].draw()

    # Initialize the data log object
    log_cols = list(design.columns)
    log_cols += ["correct", "rt", "response", "key", "dropped_frames"]
    log = cregg.DataLog(p, log_cols)
    log.stick_log = stims["array"].log

    # Execute the experiment
    with cregg.PresentationLoop(win, p, log=log, exit_func=psychophys_exit):

        for t, t_info in design.iterrows():

            # Have a subject-controlled break between some blocks
            take_break = (t and
                          not t_info["block_trial"] and
                          not t_info["block"] % p.blocks_per_break)

            if take_break:
                stims["break"].draw()
                stims["fix"].draw()
                win.flip()

            if not t_info["block_trial"]:

                # Short timed break between blocks
                if t:
                    stims["fix"].draw()
                    win.flip()
                    cregg.wait_check_quit(p.ibi_dur)

                # Show the cue for this block
                stims["cue"].setText(t_info["context"])
                stims["cue"].draw()
                win.flip()
                cregg.wait_check_quit(p.cue_dur)

            # Determine the feature proportions for this trial
            ps = [t_info[dim + "_p"] for dim in p.dim_names]
            stims["array"].set_feature_probs(*ps)

            # Determine the correct response for this trial
            correct_resp = t_info[t_info["context"] + "_p"] > .5

            # Execute the trial
            res = stim_event(correct_resp)

            # Record the trial result and wait for the next trial
            t_info = t_info.append(pd.Series(res))
            log.add_data(t_info)
            cregg.wait_check_quit(p.feedback_dur)

        # Show the exit text
        stims["finish"].draw()


def psychophys_exit(log):
    """Save the stick stimulus log."""
    stick_log_fname = log.p.stick_log_base.format(subject=log.p.subject,
                                                  run=log.p.run)
    log.stick_log.save(stick_log_fname)


def training(p, win, stims):
    """Staircased training with structured block probabilities."""
    # Initialize the trial controller
    stim_event = EventEngine(win, p, stims)

    # Create a design for this run
    design = training_design(p)

    # Show the instructions
    stims["instruct"].draw()

    # Initialize the data log object
    log_cols = list(design.columns)
    log_cols += [dim + "_strength" for dim in p.dim_names]
    log_cols += [dim + "_p" for dim in p.dim_names]
    log_cols += ["correct", "rt", "response", "key", "dropped_frames"]
    log = cregg.DataLog(p, log_cols)

    # Initialize the staircases
    if p.run == 1:
        stair_file = None
    else:
        stair_file = p.stair_temp.format(subject=p.subject, run=p.run - 1)
    stairs = make_staircases(p, stair_file)
    log.stairs = stairs

    # Execute the experiment
    with cregg.PresentationLoop(win, p, log=log, exit_func=training_exit):

        for t, t_info in design.iterrows():

            # Have a subject-controlled break between blocks
            if t_info["block"] and not t_info["block_trial"]:
                stims["break"].draw()
                stims["fix"].draw()
                win.flip()
                cregg.wait_check_quit(p.post_break_dur)

            if not t_info["chunk_trial"]:

                # Short timed break between chunks
                if t:
                    stims["fix"].draw()
                    win.flip()
                    cregg.wait_check_quit(p.inter_chunk_dur)

                # Show the cue for this block
                stims["cue"].setText(t_info["context"])
                stims["cue"].draw()
                win.flip()
                cregg.wait_check_quit(p.cue_dur)

            # Determine the feature proportions for this trial
            dim_ps = []
            for dim in p.dim_names:
                stair = stairs[dim][t_info[dim + "_stairs"]].next()
                dim_p = .5 + (-1, 1)[t_info[dim + "_val"]] * stair
                t_info[dim + "_strength"] = stair
                t_info[dim + "_p"] = dim_p
                dim_ps.append(dim_p)
            stims["array"].set_feature_probs(*dim_ps)

            # Determine the correct response for this trial
            correct_resp = t_info[t_info["context"] + "_val"]

            # Execute the trial
            res = stim_event(correct_resp)

            # Record the trial result
            t_info = t_info.append(pd.Series(res))
            log.add_data(t_info)

            # Update the relevant staircase
            rel_dim = t_info["context"]
            rel_stairs = stairs[rel_dim][t_info[rel_dim + "_stairs"]]
            rel_stairs.addResponse(res["correct"])

            # Wait for the next trial
            cregg.wait_check_quit(p.feedback_dur)

        # Show the exit text
        stims["finish"].draw()


def training_exit(log):
    """Save the state of the staircase."""
    json_fname = log.p.stair_temp.format(subject=log.p.subject, run=log.p.run)
    save_staircase_values(log.stairs, json_fname)
    # TODO print out some information about performance


def behavior(p, win, stims):
    """Behavioral experiment, similar to what will be in the scanner."""
    # Initialize the trial controller
    stim_event = EventEngine(win, p, stims)

    # Create a design for this run
    design = behavior_design(p)

    # Show the instructions
    stims["instruct"].draw()

    # Initialize the data log object
    log_cols = list(design.columns)
    log_cols += [dim + "_strength" for dim in p.dim_names]
    log_cols += [dim + "_p" for dim in p.dim_names]
    log_cols += ["correct", "rt", "response", "key", "dropped_frames"]
    log = cregg.DataLog(p, log_cols)

    # Initialize the staircases
    if p.run == 1:
        # For the first run of behavior, use the last run of training
        training_stairs = glob(p.training_stair_temp.format(subject=p.subject))
        stair_file = sorted(training_stairs)[-1]
    else:
        stair_file = p.stair_temp.format(subject=p.subject, run=p.run - 1)
    stairs = make_staircases(p, stair_file)
    log.stairs = stairs

    # Execute the experiment
    with cregg.PresentationLoop(win, p, log=log, exit_func=behavior_exit):

        for t, t_info in design.iterrows():

            # Have a subject-controlled break between some blocks
            take_break = (t and
                          not t_info["block_trial"] and
                          not t_info["block"] % p.blocks_per_break)

            if take_break:
                stims["break"].draw()
                stims["fix"].draw()
                win.flip()

            if not t_info["block_trial"]:

                # Short timed break between blocks
                if t:
                    stims["fix"].draw()
                    win.flip()
                    cregg.wait_check_quit(p.inter_block_dur)

                # Show the cue for this block
                stims["cue"].setText(t_info["context"])
                stims["cue"].draw()
                win.flip()
                cregg.wait_check_quit(p.cue_dur)

            # Determine the feature proportions for this trial
            dim_ps = []
            for dim in p.dim_names:
                stair = stairs[dim][t_info[dim + "_stairs"]].next()
                dim_p = .5 + (-1, 1)[t_info[dim + "_val"]] * stair
                t_info[dim + "_strength"] = stair
                t_info[dim + "_p"] = dim_p
                dim_ps.append(dim_p)
            stims["array"].set_feature_probs(*dim_ps)

            # Determine the correct response for this trial
            correct_resp = t_info[t_info["context"] + "_val"]

            # Execute the trial
            res = stim_event(correct_resp)

            # Record the trial result
            t_info = t_info.append(pd.Series(res))
            log.add_data(t_info)

            # Update the relevant staircase
            rel_dim = t_info["context"]
            rel_stairs = stairs[rel_dim][t_info[rel_dim + "_stairs"]]
            rel_stairs.addResponse(res["correct"])

            # Wait for the next trial
            cregg.wait_check_quit(p.feedback_dur)

        # Show the exit text
        stims["finish"].draw()


def behavior_exit(log):
    """Save the state of the staircase."""
    json_fname = log.p.stair_temp.format(subject=log.p.subject, run=log.p.run)
    save_staircase_values(log.stairs, json_fname)


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
        psychopy_rgb = self.rgb_colors * 2 - 1
        return np.take(psychopy_rgb, self.hue_idx, axis=0)

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

    Also offers properties that compute the proportion of each feature on
    each trial across the display and trial frames.

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
        hue_prop = [np.concatenate(t).mean() for t in self._hue_on if t]
        return np.array(hue_prop)

    @property
    def tilt_prop(self):
        tilt_prop = [np.concatenate(t).mean() for t in self._tilt_on if t]
        return np.array(tilt_prop)

    @property
    def width_prop(self):
        width_prop = [np.concatenate(t).mean() for t in self._width_on if t]
        return np.array(width_prop)

    @property
    def length_prop(self):
        length_prop = [np.concatenate(t).mean() for t in self._length_on if t]
        return np.array(length_prop)

    def save(self, fname):

        data = dict()
        for attr in ["on", "hue", "tilt", "width", "length"]:
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


# =========================================================================== #
# =========================================================================== #


def psychophys_design(p, rs=None):

    if rs is None:
        rs = np.random.RandomState()

    cols = ["context", "block", "block_trial",
            "hue_p", "tilt_p", "width_p", "length_p",
            "hue_val", "tilt_val", "width_val", "length_val",
            "context_p"]

    design = []
    block_dim = None
    for cycle in xrange(p.cycles):

        # Determine the order of rules for this block
        good_block = False
        while not good_block:
            # Ensure that every cycle has a rule switch
            dimensions = rs.permutation(p.dim_names)
            if dimensions[0] != block_dim:
                good_block = True

        # Determine the trials for each block in the cycle
        for block, block_dim in enumerate(dimensions):

            # Set up the design object for this block
            block_trials = np.arange(p.trials_per_block)
            block_design = pd.DataFrame(columns=cols,
                                        index=block_trials)

            # Update with info we currently know
            block_design["context"] = block_dim
            block_design["block"] = cycle * 4 + block

            # Add trialwise info
            for trial in block_trials:
                block_design.loc[trial, "block_trial"] = trial

                # Choose the coherence values for each dimension
                for dim in p.dim_names:

                    dim_p = rs.choice(p.coherences)
                    block_design.loc[trial, dim + "_p"] = dim_p

                    dim_val = getattr(p, dim + "_features")[dim_p > .5]
                    block_design.loc[trial, dim + "_val"] = dim_val

                    # Update the field with the relevant dimension
                    if dim == block_dim:
                        block_design.loc[trial, "context_p"] = dim_p

            design.append(block_design)

    # Build the full design
    design = pd.concat(design).reset_index(drop=True)
    return design


def training_design(p, rs=None):

    if rs is None:
        rs = np.random.RandomState()

    # Initialize the count matrix that controls the design
    count_mat = np.zeros((4, 4), int)
    triu = np.triu_indices_from(count_mat, 1)
    count_mat[triu] = p.pair_counts
    count_mat.T[triu] = p.pair_counts

    # Initialize a list of block-wise designs
    block_designs = []

    # Set up the columns we will be generating
    design_cols = ["block_trial", "block_chunk", "context", "context_switch"]
    feature_val_cols = [dim + "_val" for dim in p.dim_names]
    design_cols += feature_val_cols
    staircase_cols = [dim + "_stairs" for dim in p.dim_names]
    design_cols += staircase_cols
    chunk_trial = pd.Series(range(p.trials_per_chunk), name="chunk_trial")

    # Iterate over each cell in the design matrix
    for i, j in itertools.product(range(4), range(4)):

        # Find the paired dimensions in this cell
        dims = p.dim_counterbal[i], p.dim_counterbal[j]
        count = count_mat[i, j]

        # Iterate over the number of blocks for this pairing
        for block in xrange(count):

            # Set up a list of chunk-wise designs
            chunk_designs = []

            # Iterate over thre chunks in this block
            for chunk in xrange(p.chunks_per_block):

                # Make a dataframe with design information
                chunk_design = pd.DataFrame(columns=design_cols,
                                            index=chunk_trial)

                # Determine the relevant dimension for this chunk
                chunk_dim = dims[chunk % 2]

                # Add information we currently know to the design
                chunk_design["context"] = chunk_dim
                chunk_design["context_switch"] = chunk_trial == 0
                chunk_design["block_chunk"] = chunk
                chunk_design["block_trial"] = (chunk_trial +
                                               p.trials_per_chunk * chunk)

                # Find feature values for each dimension/trial
                feature_vals = rs.rand(p.trials_per_chunk, 4) > .5
                chunk_design[feature_val_cols] = feature_vals.astype(int)

                # Assign staircases for each dimension/trial
                staircase_ids = rs.randint(0, p.n_staircases,
                                           (p.trials_per_chunk, 4))
                chunk_design[staircase_cols] = staircase_ids

                # We are done with the design for this chunk
                chunk_designs.append(chunk_design.reset_index())

            # We are done with the design for this block
            block_designs.append(pd.concat(chunk_designs))

    # Randommize the order of the blocks
    rs.shuffle(block_designs)

    # Combine the blocks into a single design dataframe
    full_design = pd.concat(block_designs)

    # Add in a column that identifies the block number
    blocks = np.repeat(range(len(block_designs)),
                       p.chunks_per_block * p.trials_per_chunk)
    full_design.insert(0, "block", blocks)

    return full_design


def behavior_design(p, rs=None):

    if rs is None:
        rs = np.random.RandomState()

    cols = ["trial", "block", "block_trial", "context", "context_switch"]
    feature_val_cols = [dim + "_val" for dim in p.dim_names]
    cols += feature_val_cols
    staircase_cols = [dim + "_stairs" for dim in p.dim_names]
    cols += staircase_cols

    def make_block_dimensions():

        block_dimensions = []
        for cycle in xrange(p.cycles):

            trans_mat = np.ones((4, 4), int)
            trans_mat[np.diag_indices_from(trans_mat)] = 0
            trans_mat = pd.DataFrame(trans_mat, p.dim_names, p.dim_names)

            if not block_dimensions:
                dim = rs.choice(p.dim_names)
                block_dimensions.append(dim)
            else:
                dim = block_dimensions[-1]

            while (trans_mat.values > 0).any():
                choices = list(trans_mat.index[trans_mat.ix[dim] > 0])
                next_dim = rs.choice(choices)
                trans_mat.loc[dim, next_dim] -= 1
                block_dimensions.append(next_dim)
                dim = next_dim

        return block_dimensions

    # Find an ideal order of relevant dimensions
    # This balances the transistions between dimensions, excluding
    # self-transitions
    good_order = False
    while not good_order:
        try:
            block_order = make_block_dimensions()
            good_order = True
        except ValueError:
            # Sometimes the above fails, but it has a success rate of
            # over 50%, so it seems easier just to keep trying until
            # we get a good order than to figure out a smarter way.
            pass

    # Compute the total number of events
    n_blocks = len(block_order)
    n_trials = n_blocks * p.trials_per_block

    # Build the design dataframe
    design = pd.DataFrame(columns=cols)
    design["trial"] = np.arange(n_trials)
    design["block"] = np.repeat(np.arange(n_blocks), p.trials_per_block)
    design["block_trial"] = np.tile(np.arange(p.trials_per_block), n_blocks)
    design["context"] = np.repeat(block_order, p.trials_per_block)
    design["context_switch"] = design.block_trial == 0
    design[feature_val_cols] = rs.randint(0, 2, (n_trials, 4))
    design[staircase_cols] = rs.randint(0, p.n_staircases, (n_trials, 4))

    return design

if __name__ == "__main__":
    main(sys.argv[1:])
