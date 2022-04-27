from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numpy as np

class Timeline:
    def __init__(self, ax, frames, verbose=True):
        self.ax = ax
        self.frames = frames
        self.fraction_updaters = []
        self.transitions = []
        # self.time_updaters = []
        # self.abs_time_updaters = []
        self.verbose = verbose
        
    def add_fraction_updater(self, f, start, end, *args, **kwargs):
        self.fraction_updaters.append([f, start, end, args, kwargs, False])

    def add_transition(self, f, when, *args, **kwargs):
        self.transitions.append([f, when, args, kwargs, False])

        
    def update(self, f):
        if self.verbose:
            print(f"Frame {f}")
        artists = []
        for i,(func, start, end, args, kwargs, is_done) in enumerate(self.fraction_updaters[:]):
            if is_done:
                continue
            if f < start:
                continue
            frac = (f - start) / (end - start)
            frac = np.clip(frac, 0, 1)
            artists += func(frac, *args, **kwargs)
            if frac == 1:
                self.fraction_updaters[i][-1] = True
        for i,(func, when, args, kwargs, is_done) in enumerate(self.transitions[:]):
            if is_done:
                continue
            if f > when:
                artists += func(*args, **kwargs)
                self.transitions[i][-1] = True

        return artists
    
    def save(self, fig, filename, interval=200, **kwargs):
        ani = FuncAnimation(fig, self.update, interval=interval, frames=self.frames, blit=True)
        ani.save(filename, **kwargs)

        
        
def clip_artists(artists, patch):
    for obj in artists:
        if obj is not None:
            try:
                obj.set_clip_path(patch)
            # This happens when the path is off the
            # axes.  In that case the object isn't visible anyway so we can skip it
            except AttributeError:
                continue


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

    buffer = 0.01
    bx = (xmax - xmin) * buffer
    by = (ymax - ymin) * buffer
    xmin -= bx
    xmax += bx
    ymin -= by
    ymax += by
    
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


def zoom_to(frac, ax, starts, ends):
    start_x = starts[0]
    start_y = starts[1]
    end_x = ends[0]
    end_y = ends[1]
    xmin = start_x[0] + frac * (end_x[0] - start_x[0])
    xmax = start_x[1] + frac * (end_x[1] - start_x[1])
    ymin = start_y[0] + frac * (end_y[0] - start_y[0])
    ymax = start_y[1] + frac * (end_y[1] - start_y[1])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    return [ax.figure]

    
def get_best_direction(j):
    x = j[0].get_xdata()
    y = j[0].get_ydata()
    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    if abs(dy) > abs(dx):
        return 'up' if dy > 0 else 'down'
    else:
        return 'right' if dx > 0 else 'left'
