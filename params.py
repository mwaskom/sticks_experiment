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
    fix_iti_color="white",
    fix_stim_color="white",

    # Response settings
    quit_keys=["escape", "q"],
    wait_keys=["space"],
    finish_keys=["return"],
    trigger_keys=["5", "t"],

    # Stick parameters
    hues=(60, 140),
    tilts=(45, -45),
    widths=(.15, .075),
    lengths=(.5, .25),

    # Feature names
    hue_features=("red", "green"),
    tilt_features=("right", "left"),
    width_features=("wide", "narrow"),
    length_features=("long", "short"),

    # Fixed color parameters
    chroma=80,
    lightness=70,

    # Stick array parameters
    array_radius=4,
    array_offset=0,
    disk_radius=.55,
    disk_candidates=20,

    # Twinkle parameters
    twinkle_off_prob=.05,
    twinkle_on_prob=.5,
    twinkle_timeout=9,
    twinkle_burnin=10,

)

prototype = deepcopy(base)
