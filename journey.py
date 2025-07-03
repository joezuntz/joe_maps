import numpy as np
import matplotlib.pyplot as plt
import bezier
import gpxpy
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from . import anim

class BaseJourney:
    """
    Not yet written
    """

    def __init__(self, x, y, artists):
        self.artists = artists
        # Copy these because we may modify them when
        # animating and we want to retain the original
        # so that we can reset the simulation
        self.x = x.copy()
        self.y = y.copy()

    def draw(self, ax):
        raise NotImplementedError("This method should be implemented in subclasses")

    def length(self):
        """
        Return the length of the journey
        """
        return np.sum(np.sqrt(np.diff(self.x) ** 2 + np.diff(self.y) ** 2))
    
    def hide(self):
        for artist in self.artists.values():
            artist.set_visible(False)

    def show(self):
        """
        Show the journey
        """
        for artist in self.artists.values():
            artist.set_visible(True)

    def animate(self, frac):
        if frac > 0:
            self.show()
        n = int(len(self.x) * frac)
        line = self.artists["line"]
        line.set_xdata(self.x[:n])
        line.set_ydata(self.y[:n])


        if "arrow1" in self.artists:
            arrow1 = self.artists["arrow1"]
            if frac > 0.01:
                arrow1.set_visible(True)
            else:
                arrow1.set_visible(False)
        if "arrow2" in self.artists:
            arrow2 = self.artists["arrow2"]
            if frac < 0.99:
                arrow2.set_visible(False)
            else:
                arrow2.set_visible(True)

        # Return everything that might be animated
        return list(self.artists.values())



class ArcJourney(BaseJourney):
    """
    A basic line journey
    """

    def __init__(
        self,
        lat_start,
        lon_start,
        lat_end,
        lon_end,
        N=None,
        theta=None,
        start_gap=0,
        end_gap=0,
        offset=(0, 0),
        arrow_location=None,
        arrow_headwidth=None,
        **kwargs
    ):

        x, y = get_arc_path(
            lat_start,
            lon_start,
            lat_end,
            lon_end,
            theta=theta,
            N=N,
            start_gap=start_gap,
            end_gap=end_gap,
        )

        x = x + offset[0]
        y = y + offset[1]


        line = Line2D(x, y, **kwargs)

        artists = {"line": line}

        if arrow_location == "middle":
            narrow = len(x) // 2
            dx = x[narrow] - x[narrow - 1]
            dy = y[narrow] - y[narrow - 1]
            xarrow = x[narrow] - dx
            yarrow = y[narrow] - dy

        elif arrow_location == "end" or arrow_location == "both":
            dx = x[-1] - x[-2]
            dy = y[-1] - y[-2]
            xarrow = x[-1] - dx
            yarrow = y[-1] - dy

        elif arrow_location == "start":
            dx = x[0] - x[1]
            dy = y[0] - y[1]
            xarrow = x[0] - dx
            yarrow = y[0] - dy

        if arrow_location is not None:
            arr1 = mpatches.FancyArrow(
                xarrow,
                yarrow,
                dx,
                dy,
                color=line.get_color(),
                head_width=arrow_headwidth,
                head_starts_at_zero=True,
                length_includes_head=True,
            )
            artists["arrow1"] = arr1

        if arrow_location == "both":
            dx = x[0] - x[1]
            dy = y[0] - y[1]
            xarrow = x[0] - dx
            yarrow = y[0] - dy

            arr2 = mpatches.FancyArrow(
                x[0] - dx,
                y[0] - dy,
                dx,
                dy,
                color=line.get_color(),
                head_width=arrow_headwidth,
                head_starts_at_zero=False,
                length_includes_head=True,
            )
            artists["arrow2"] = arr2

        super().__init__(x, y, artists)

    def draw(self, ax):
        """
        Draw the journey on the given axes
        """
        ax.add_line(self.artists["line"])
        if "arrow1" in self.artists:
            ax.add_patch(self.artists["arrow1"])
        if "arrow2" in self.artists:
            ax.add_patch(self.artists["arrow2"])



class GPXJourney(BaseJourney):
    def __init__(
        self,
        gpx_file,
        offset=(0, 0),
        *args,
        **kwargs
    ):
        lat, lon = read_gpx(gpx_file)
        y, x = lat, lon

        x = x + offset[0]
        y = y + offset[1]

        line = Line2D(x, y, *args, **kwargs)

        artists = {"line": line}
        super().__init__(x, y, artists)


    def draw(self, ax):
        ax.add_line(self.artists["line"])


def get_arc_path(
    lat_start, lon_start, lat_end, lon_end, theta=None, N=None, start_gap=0, end_gap=0
):
    lat_mid = 0.5 * (lat_start + lat_end)
    lon_mid = 0.5 * (lon_start + lon_end)
    dlat = lat_mid - lat_start
    dlon = lon_mid - lon_start
    if theta is None:
        theta = np.random.uniform(-45, 45)
    theta = np.radians(theta)
    c = np.cos(theta)
    s = np.sin(theta)
    lon_node = lon_start + (c * dlon) + (s * dlat)
    lat_node = lat_start + (-s * dlon) + (c * dlat)
    nodes = np.array([(lon_start, lon_node, lon_end), (lat_start, lat_node, lat_end)])
    curve = bezier.Curve(nodes, degree=2)
    if N is None:
        N = 200
    x = np.zeros(N)
    y = np.zeros(N)
    S = np.linspace(start_gap, 1 - end_gap, N)
    for i, s in enumerate(S):
        x_s, y_s = curve.evaluate(s)
        x[i] = x_s
        y[i] = y_s
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    return x, y


def read_gpx(filename):
    with open(filename, "r") as f:
        gpx = gpxpy.parse(f)
    lat = np.array([p.point.latitude for p in gpx.get_points_data()])
    lon = np.array([p.point.longitude for p in gpx.get_points_data()])
    return lat, lon
