"""Make a short movie demonstrating the stimulus."""
import sys
import shutil
from tempfile import mkdtemp
from subprocess import check_output

import cregg
import sticks


def main(arglist):

    # Load the parameters and launch the window
    p = cregg.Params("practice")
    p.set_by_cmdline(arglist)
    sticks.subject_specific_colors(p)
    win = cregg.launch_window(p)

    # Increase the chroma as color is degraded by ffmpeg
    p.chroma = 45
    p.debug = False

    # Load the stimulus object
    cue = sticks.PolygonCue(win, p)
    fix = sticks.Fixation(win, p)
    array = sticks.StickArray(win, p)
    array.reset()

    # Make a movie of two trials
    for trial in range(2):

        # Pre-stim fixation
        for _ in xrange(60):
            fix.draw()
            win.flip()
            win.getMovieFrame()

        # Orienting cue
        fix.color = p.fix_stim_color
        for _ in xrange(43):
            fix.draw()
            win.flip()
            win.getMovieFrame()

        # Set the cue and stimulus features
        shape = 4 if trial else 6
        cue.set_shape(shape)

        ps = [.6, .6] if trial else [.4, .4]
        array.set_feature_probs(*ps)

        # Show the stimulus
        for _ in xrange(120):
            array.update()
            array.draw()
            cue.draw()
            fix.draw()
            win.flip()
            win.getMovieFrame()

    # Post stim fixation
    fix.color = p.fix_iti_color
    for _ in xrange(60):
        fix.draw()
        win.flip()
        win.getMovieFrame()
    win.close()

    # Write out the image frames
    dir = mkdtemp()
    win.saveMovieFrames(dir + "/frame.png")

    # Convert to mp4
    check_output(["ffmpeg", "-r", "60", "-i", dir + "/frame%03d.png",
                  "-f", "mp4", "-vcodec", "mpeg4", "-b:v", "5000k",
                  "-r", "60", "stim_movie.mp4"])

    # Clean up the temporary directory
    shutil.rmtree(dir)


if __name__ == "__main__":
    main(sys.argv[1:])
