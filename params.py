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
    chroma=40,

    # Stick array parameters
    array_radius=4,
    fixation_radius=1,
    disk_radius=.35,
    disk_candidates=60,

    # Twinkle parameters
    twinkle_off_prob=.05,
    twinkle_on_prob=.5,
    twinkle_timeout=9,
    twinkle_burnin=20,

    # Cue polygon parameters
    poly_radius=.5,
    poly_linewidth=3,
    poly_color=-.2,

    # Feedback settings
    feedback_dur=.5,
    feedback_hz=(10, None),

    # Timing
    orient_dur=.5,
    stim_timeout=3,

    # fMRI Parameters
    equilibrium_trs=16,
    tr=.720,

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


calibrate = deepcopy(base)
calibrate.update(

    patch_size=3,
    patch_mask="circle",
    patch_sf=1.5,

    arrow_size=.5,
    arrow_width=5,
    arrow_life=3,
    arrow_offset=1,

    resp_keys=["left", "right", "space"],

    flicker_every=7,

    diff_start=(5, 5, 5),
    diff_step=(.5, .5, .5),
    average_last=2,

)

learn = deepcopy(base)
learn.update(

    log_base="data/{subject}_learn",

    ibi_dur=1.5,
    targ_prop=.8,
    trials_per_block=4,
    blocks_per_break=4,
    post_break_dur=1,
    trial_criterion=1,  # accuracy thresh to count as a "good" trial
    block_criterion=3,  # blocks at criterion to move on
    iti_params=(.5, 1.5),  # range of uniform ITI distribution


)

training = deepcopy(base)
training.update(

    log_base="data/{subject}_training_run{run:02d}",

)
def behavior_cmdline(parser):
    parser.add_argument("-trials_at_cue", type=int, default=3)
