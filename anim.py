from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numpy as np

class Timeline:
    def __init__(self, ax, frames, verbose=True):
        self.ax = ax
        self.frames = frames
        self.fraction_updaters = []
        # self.time_updaters = []
        # self.abs_time_updaters = []
        self.verbose = verbose
        
    def add_fraction_updater(self, f, start, end, *args, **kwargs):
        self.fraction_updaters.append((f, start, end, args, kwargs))
        
    def update(self, f):
        if self.verbose:
            print(f"Frame {f}")
        artists = []
        for func, start, end, args, kwargs in self.fraction_updaters:
            frac = (f - start) / (end - start)
            frac = np.clip(frac, 0, 1)
            artists += func(frac, *args, **kwargs)
        return artists
    
    def save(self, fig, filename):
        ani = FuncAnimation(fig, self.update, frames=self.frames, blit=True)
        ani.save(filename)

        
        
def clip_artists(artists, patch):
    for obj in artists:
        if obj is not None:
            obj.set_clip_path(patch)


def wipe_journey(frac, ax, direction, artists):
    # Find the extents of the line to show us
    # the overall bounding box
    line = artists[0]
    xmin = line.get_xdata().min()
    xmax = line.get_xdata().max()
    ymin = line.get_ydata().min()
    ymax = line.get_ydata().max()
    

    # If there is an arrowhead, include it in the
    # dimensions to start/end the wipe
    arr = artists[2]
    if arr is not None:
        arr_xy = arr.get_xy()
        xmin = min(xmin, arr_xy[:, 0].min())
        xmax = max(xmax, arr_xy[:, 0].max())
        ymin = min(ymin, arr_xy[:, 1].min())
        ymax = max(ymax, arr_xy[:, 1].max())
    
    # determine the box we will use to unmask
    # the journey
    if direction == 'right':
        x = xmin
        y = ymin
        wx = (xmax - xmin) * frac
        wy = (ymax - ymin)
    elif direction == 'left':
        x = xmax
        y = ymax
        wx = -(xmax - xmin) * frac
        wy = -(ymax - ymin)
    elif direction == 'up':
        x = xmin
        y = ymin
        wx = (xmax - xmin)
        wy = (ymax - ymin) * frac
    elif direction == 'down':
        x = xmax
        y = ymax
        wx = -(xmax - xmin)
        wy = -(ymax - ymin) * frac
    else:
        raise ValueError("Bad direction")

    # The rectangle has to be added to the plot
    # before using it to clip the journey, I guess so
    # that it is in the right coordinates
    rect = patches.Rectangle((x,y), wx, wy, alpha=0)
    ax.add_patch(rect)

    # Clip everything. Does not work for text - bug in matplotlib
    clip_artists(artists, rect)

    # Return everything that might be animated
    return [a for a in artists if a is not None]
