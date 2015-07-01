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
    lengths=(.5, .25),
    widths=(.15, .075),
    oris=(35, -35),
    colors=((0.93226, 0.53991, 0.26735),
            (0., 0.74055, 0.22775)),

    # Field parameters
    field_radius=4,
    field_offset=0,
    sticks_per_field=100,

)

prototype = deepcopy(base)
