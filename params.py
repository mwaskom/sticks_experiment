from __future__ import division
from copy import deepcopy

base = dict(

    experiment_name="sticks",

    # Display setup
    monitor_name="mlw-mbair",
    fmri_monitor_name="cni_47",
    screen_number=0,
    fmri_screen_number=0,
    monitor_units="deg",
    full_screen=True,
    window_color=-.4,

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

    # File with calibrated color values
    color_file="data/{subject}_{monitor}_colors.json",

    # Stick array parameters
    array_radius=5.5,
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
    poly_linewidth=1,
    poly_color=-.2,

    # Early training button guides
    guide_offset=.3,

    # Progress bar shown during breaks
    prog_bar_width=5,
    prog_bar_height=.25,
    prog_bar_position=-3,
    prog_bar_linewidth=2,
    prog_bar_color="white",

    # Feedback settings
    feedback_dur=.5,
    feedback_hz=(10, None),

    # Timing
    orient_dur=.72,
    stim_timeout=3,
    iti_params=(.5, 1.5),
    after_break_dur=2,
    trials_per_break=16,

    # fMRI Parameters
    equilibrium_trs=16,
    tr=.720,

    # Communication
    instruct_text=(
        "Press space to begin the experiment",
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

    log_base="data/{subject}_{monitor}_calibration",

    patch_size=5,

    start_vals=[75, 85],
    step_sizes=.25,
    trials=50,
    reversals=5,

    iti=.25,

)


training = deepcopy(base)
training.update(

    log_base="data/{subject}_training_run{run:02d}",

    block_lengths=(4, 4, 4, 2, 1),
    cycles_per_length=(4, 4, 4, 12, 12),
    randomize_blocks=(False, False, True, True, True),
    show_guides=(True, False, False, False, False),

    targ_prop=.8,

)


practice = deepcopy(base)
practice.update(

    log_base="data/{subject}_practice_run{run:02d}",
    targ_prop=.7,

)
def practice_cmdline(parser):
    parser.add_argument("-trials", type=int, default=100)
    parser.add_argument("-guides", action="store_true")


psychophys = deepcopy(base)
psychophys.update(

    log_base="data/{subject}_psychophys_run{run:02d}",
    targ_props=[.52, .58, .64, .70, .76],
    permutation_attempts=1500,

)
def psychophys_cmdline(parser):
    parser.add_argument("-cycles", type=int, default=1)


scan = deepcopy(base)
scan.update(

    log_base="data/{subject}_scan_run{run:02d}",

    n_designs=16,

    focb_seed=410,
    focb_batches=50,
    focb_batch_size=100,
    focb_cost_tol=0.01,

)
