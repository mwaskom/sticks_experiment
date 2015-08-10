from __future__ import division
from copy import deepcopy

base = dict(

    experiment_name="sticks",

    # Display setup
    monitor_name="waglab-mbpro",
    fmri_monitor_name="cni_47",
    screen_number=0,
    fmri_screen_number=1,
    monitor_units="deg",
    full_screen=True,
    window_color=-.5,

    # Fixation
    fix_size=.2,
    fix_iti_color=-1,
    fix_stim_color=1,

    # Response settings
    quit_keys=["escape", "q"],
    wait_keys=["space"],
    finish_keys=["return"],
    trigger_keys=["5", "t"],
    resp_keys=["lshift", "rshift"],

    # Stick parameters
    stick_width=.1,
    stick_length=.3,

    # Target features
    stick_hues=(60, 140),
    stick_oris=(-45, 45),

    # Dimension info
    dim_names=["hue", "ori"],

    # Feature names
    hue_features=("red", "green"),
    ori_features=("left", "right"),

    # Fixed color parameters
    lightness=80,
    chroma=30,

    # Stick array parameters
    array_radius=4,
    fixation_radius=1,
    disk_radius=.35,
    disk_candidates=30,

    # Twinkle parameters
    twinkle_off_prob=.05,
    twinkle_on_prob=.5,
    twinkle_timeout=9,
    twinkle_burnin=20,

    # Cue frame parameters
    frame_gap=1.1,
    frame_width=.8,
    frame_ring_cycles=(1.5, -3.5),
    frame_spoke_reversals=(7, 14),
    frame_contrast=.75,

    # Feedback settings
    feedback_dur=.5,
    feedback_hz=(10, None),

    # Timing
    orient_dur=.5,
    stim_timeout=3,

    # Communication
    instruct_text=(
        "Use the < and > keys to respond",
        "as soon as you make your decision",
        "",
        "Press space to begin",
    ),

    break_text=(
        "Take a quick break, if you'd like!",
        "",
        "Press space to start the next block",
    ),

    finish_text=(
        "Run Finished!",
        ""
        "Please tell the experimenter",
    ),

)

prototype = deepcopy(base)

learn = deepcopy(base)
learn.update(dict(

    log_base="data/{subject}_learn",

    ibi_dur=1.5,
    targ_prop=.8,
    trials_per_block=4,
    blocks_per_break=4,
    trial_criterion=.8,  # accuracy thresh to count as a "good" trial
    block_criterion=3,  # blocks at criterion to move on


))
