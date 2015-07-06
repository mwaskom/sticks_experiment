from __future__ import division
from copy import deepcopy

base = dict(

    experiment_name="sticks",

    # Display setup
    monitor_name="mlw-mbair",
    fmri_monitor_name="cni_47",
    screen_number=0,
    fmri_screen_number=1,
    monitor_units="deg",
    full_screen=True,
    window_color=0,

    # Fixation
    fix_size=.2,
    fix_stim_color="white",

    # Response settings
    quit_keys=["escape", "q"],
    wait_keys=["space"],
    finish_keys=["return"],
    trigger_keys=["5", "t"],
    resp_keys=["comma", "period"],

    # Feedback settings
    feedback_glyphs=("X", "+"),
    feedback_colors=("black", "white"),

    # Stick parameters
    hues=(60, 140),
    tilts=(-45, 45),
    widths=(.15, .075),
    lengths=(.4, .2),

    # Dimension info
    dim_names=["hue", "tilt", "width", "length"],

    # Feature names
    hue_features=("red", "green"),
    tilt_features=("left", "right"),
    width_features=("thick", "thin"),
    length_features=("long", "short"),

    # Fixed color parameters
    chroma=80,
    lightness=70,

    # Stick array parameters
    array_radius=4,
    array_offset=0,
    disk_radius=.45,
    disk_candidates=20,

    # Twinkle parameters
    twinkle_off_prob=.05,
    twinkle_on_prob=.5,
    twinkle_timeout=9,
    twinkle_burnin=20,

    # Timing
    stim_dur=1.5,
    feedback_dur=.5,
    cue_dur=.5,

    # Communication
    break_text=(
        "Take a quick break, if you'd like!",
        "",
        "Press space to start the next block",
    ),

    finish_text=(
        "Run Finished!",
        ""
        "Please tell the experimenter",
    )

)

prototype = deepcopy(base)

learn = deepcopy(base)
learn.update(dict(

    log_base="data/{subject}_learn",
    ibi_dur=1.5,
    coherence=.8,
    trials_per_block=4,
    blocks_per_break=5,
    trial_criterion=.7,  # accuracy thresh to count as a "good" trial
    block_criterion=3,  # blocks at criterion to move on

    post_guide_instruct_text=(
        "Now you have to remember which button to press for each response",
        "",
        "The buttons are the same as before, and they won't ever change",
        "",
        "Hit space to continue",
    )


))

psychophys = deepcopy(base)
psychophys.update(dict(

    log_base="data/{subject}_psychophys_run{run:02d}",
    stick_log_base="data/{subject}_psychophys_stim_run{run:02d}.npz",

    ibi_dur=1.5,

    trials_per_block=4,
    blocks_per_break=5,
    coherences=(.15, .25, .35, .45, .55, .65, .75, .85),

    instruct_text=(
        "Use the < and > keys to respond",
        "as soon as you make your decision",
        "",
        "Press space to begin",
    )


))
def psychophys_cmdline(parser):
    parser.add_argument("-cycles", type=int, default=30)
