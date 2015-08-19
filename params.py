from __future__ import division
from copy import deepcopy


# --------------------------------------------------------------------- #
# Base parameters
# --------------------------------------------------------------------- #


base = dict(

    experiment_name="sticks",

    # Display setup
    monitor_name="waglab-mbpro",
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
    trigger_keys=["t", "5"],
    resp_keys=["lshift", "rshift"],
    fmri_resp_keys=["4", "9"],

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
    orient_dur=.7,
    stim_timeout=2.8,
    iti_params=(.5, 1.5),
    after_break_dur=2,
    trials_per_break=16,

    # fMRI Parameters
    equilibrium_trs=16,
    leadout_trs=12,
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

    setup_text_size=.5,

)


instruct = deepcopy(base)
prototype = deepcopy(base)


# --------------------------------------------------------------------- #
# Color calibration
# --------------------------------------------------------------------- #


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


# --------------------------------------------------------------------- #
# Training sessions
# --------------------------------------------------------------------- #


training = deepcopy(base)
training.update(

    log_base="data/{subject}_training_run{run:02d}",

    block_lengths=(4, 4, 4, 2, 1),
    cycles_per_length=(4, 4, 4, 12, 12),
    randomize_blocks=(False, False, True, True, True),
    show_guides=(True, False, False, False, False),

    targ_prop=.8,

)


# --------------------------------------------------------------------- #
# Practice sessions
# --------------------------------------------------------------------- #


practice = deepcopy(base)
practice.update(

    log_base="data/{subject}_practice_run{run:02d}",
    targ_prop=.7,

)
def practice_cmdline(parser):
    parser.add_argument("-trials", type=int, default=100)
    parser.add_argument("-guides", action="store_true")


# --------------------------------------------------------------------- #
# Psychophyiscs sessions
# --------------------------------------------------------------------- #


psychophys = deepcopy(base)
psychophys.update(

    log_base="data/{subject}_psychophys_run{run:02d}",
    targ_props=[.52, .56, .60, .64, .68],
    permutation_attempts=1500,

)
def psychophys_cmdline(parser):
    parser.add_argument("-cycles", type=int, default=1)


# --------------------------------------------------------------------- #
# Scan sessions
# --------------------------------------------------------------------- #


scan = deepcopy(base)
scan.update(

    log_base="data/{subject}_scan_run{run:02d}",
    design_base="design/scan_design_{}.csv",

    strength_file="data/{subject}_stimulus_strength.json",
    strength_defaults=dict(hue=dict(easy=.15, hard=.05),
                           ori=dict(easy=.15, hard=.05)),

    n_designs=16,
    trs_per_trial=6,

    focb_seed=410,
    focb_batches=50,
    focb_batch_size=100,
    focb_cost_tol=0.01,

    eff_seed=1045,
    eff_n_sched=5000,
    eff_geom_p=.33,
    eff_geom_loc=-1,
    eff_geom_support=(0, 12),
    eff_fir_basis=32,
    eff_leadout_trs=12,

    finish_text=(
        "Run Finished!",
    ),

)
