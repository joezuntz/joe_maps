import numpy as np
import matplotlib.pyplot as plt
import bezier
import gpxpy
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from . import anim
from .styles import mmc_colors, PC

LABEL_KIND_PERMANENT = "permanent"
LABEL_KIND_TRANSIENT = "transient"
LABEL_KIND_PERSISTENT = "persistent"

class BaseJourney:
    """
    Not yet written
    """

    def __init__(self, x, y, artists, fog_clearance=None):
        self.artists = artists
        # Copy these because we may modify them when
        # animating and we want to retain the original
        # so that we can reset the simulation
        self.x = x.copy()
        self.y = y.copy()
        self.label_kind = None
        self.fog_clearance = fog_clearance
        self.initial_alphas = None


    def draw(self, ax):
        raise NotImplementedError("This method should be implemented in subclasses")

    def length(self):
        """
        Return the length of the journey
        """
        return np.sum(np.sqrt(np.diff(self.x) ** 2 + np.diff(self.y) ** 2))
    
    def get_label_point(self):
        """
        Return the point at which to place a label.
        By default, this is the middle of the journey.
        """
        n = len(self.x)
        return (self.x[n//2], self.y[n//2])
    
    def add_label(self, ax, label, facecolor=mmc_colors.wheat, textcolor=mmc_colors.dark_blue, edgecolor=mmc_colors.dark_blue,**kwargs):
        """
        Add a label to the journey at the specified coordinates.
        If x and y are not provided, the label is added at the end of the journey.
        """
        if label is None:
            return
        x, y = self.get_label_point()
        offset = kwargs.pop("offset", (0, 0))
        x += offset[0]
        y += offset[1]
        box = dict(boxstyle="round", facecolor=facecolor, edgecolor=edgecolor, alpha=kwargs.get("alpha", 1.0))
        text = ax.text(x, y, label, ha='left', va='center', bbox=box, transform=PC, **kwargs)
        self.artists["label"] = text
        self.label_kind = kwargs.get("label_kind", LABEL_KIND_TRANSIENT)

    def hide(self):
        for artist in self.artists.values():
            artist.set_visible(False)
        return self.artists.values()

    def show(self):
        """
        Show the journey
        """
        for artist in self.artists.values():
            artist.set_visible(True)
        return self.artists.values()

    def animate(self, frac):
        if frac > 0:
            self.show()
        n = int(len(self.x) * frac)
        line = self.artists["line"]
        line.set_xdata(self.x[:n])
        line.set_ydata(self.y[:n])

        if "arrow1" in self.artists:
            arrow1 = self.artists["arrow1"]
            loc = arrow1.jm_user_location
            if loc == "start":
                thresh = 0.01
            elif loc == "end":
                thresh = 0.99
            else:
                thresh = 0.5
            arrow1.set_visible(frac > thresh)
        if "arrow2" in self.artists:
            arrow2 = self.artists["arrow2"]
            loc = arrow2.jm_user_location
            if loc == "start":
                thresh = 0.01
            elif loc == "end":
                thresh = 0.99
            else:
                thresh = 0.5
            arrow2.set_visible(frac > thresh)

        self.animate_label(frac)


        # Return everything that might be animated
        return list(self.artists.values())
    
    def fade_out(self, frac):
        if self.initial_alphas is None:
            self.initial_alphas = {k: a.get_alpha() for k, a in self.artists.items()}
        for k, a in self.artists.items():
            if a.get_visible():
                alpha = self.initial_alphas[k] * (1 - frac)
                a.set_alpha(alpha)
    
    def animate_label(self, frac):
        if "label" in self.artists:
            label = self.artists["label"]
            if self.label_kind == LABEL_KIND_PERMANENT:
                visible = True
            elif self.label_kind == LABEL_KIND_TRANSIENT:
                visible = frac > 0.01 and frac < 0.99
            elif self.label_kind == LABEL_KIND_PERSISTENT:
                visible = frac > 0.01
            label.set_visible(visible)
            return [label]
        return []

    def distance_map(self, ax, nx, ny, frac=1):
        d = np.zeros((ny, nx))
        d[:] = np.inf
        npoint = int(len(self.x) * frac)
        for xi, yi in zip(self.x[:npoint], self.y[:npoint]):
            d_i = distance_map_from_point(ax, xi, yi, nx, ny)
            np.minimum(d, d_i, out=d)
        return d
        

    def add_arrows(self, line, x, y, arrow_location="end", arrow_headwidth=0.02, arrow_length=0.1):
        artists = {}
        if arrow_location is None:
            return artists

        if arrow_location == "middle":
            mid = len(x) // 2
            dx = x[mid] - x[mid - 1]
            dy = y[mid] - y[mid - 1]
            xarrow = x[mid] - dx
            yarrow = y[mid] - dy

        elif arrow_location == "end" or arrow_location == "both":
            dx = x[-1] - x[-2]
            dy = y[-1] - y[-2]
            xarrow = x[-1] - dx
            yarrow = y[-1] - dy

        elif arrow_location == "start":
            dx = x[1] - x[0]
            dy = y[1] - y[0]
            xarrow = x[1] - dx
            yarrow = y[1] - dy
        else:
            raise ValueError(f"Unknown arrow location: {arrow_location}")

        if (dx == 0) and (dy == 0):
            raise ValueError("Cannot add arrow with zero length")
        
        # set the length as a fixed value
        # Normalize the direction vector
        norm = np.sqrt(dx**2 + dy**2)
        dx *= arrow_length / norm
        dy *= arrow_length / norm

        arr1 = mpatches.FancyArrow(
            xarrow,
            yarrow,
            dx,
            dy,
            color=line.get_color(),
            head_width=arrow_headwidth,
            head_starts_at_zero=True,
            length_includes_head=True,
            transform=PC
        )
        arr1.jm_user_location = "start" if arrow_location == "both" else arrow_location
        artists["arrow1"] = arr1

        print(xarrow, yarrow, dx, dy, arrow_location, arrow_headwidth)

        if arrow_location == "both":
            dx = x[0] - x[1]
            dy = y[0] - y[1]
            xarrow = x[0] - dx
            yarrow = y[0] - dy

            if (dx == 0) and (dy == 0):
                raise ValueError("Cannot add arrow with zero length")

            arr2 = mpatches.FancyArrow(
                x[0] - dx,
                y[0] - dy,
                dx,
                dy,
                color=line.get_color(),
                head_width=arrow_headwidth,
                head_starts_at_zero=False,
                length_includes_head=True,
                transform=PC,
            )
            arr2.jm_user_location = "end" if arrow_location == "both" else arrow_location
            artists["arrow2"] = arr2


        return artists



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
        arrow_length=0.1,
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
        fog_clearance = kwargs.pop("fog_clearance", None)

        line = Line2D(x, y, transform=PC, **kwargs)

        artists = {"line": line}
        arrows = self.add_arrows(line, x, y, arrow_location, arrow_headwidth, arrow_length=arrow_length)
        artists.update(arrows)


        super().__init__(x, y, artists, fog_clearance=fog_clearance)

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
    def __init__(self, gpx_file, offset=(0, 0), arrow_location=None, arrow_headwidth=None, arrow_length=0.1, interp=0, *args, **kwargs):
        
        
        lat, lon = read_gpx(gpx_file, interp=interp)
        fog_clearance = kwargs.pop("fog_clearance", None)
        y, x = lat, lon

        x = x + offset[0]
        y = y + offset[1]

        line = Line2D(x, y, *args, transform=PC, **kwargs)
        artists = {"line": line}
        arrrows = self.add_arrows(line, x, y, arrow_location=arrow_location, arrow_headwidth=arrow_headwidth, arrow_length=arrow_length)
        artists.update(arrrows)
        super().__init__(x, y, artists, fog_clearance=fog_clearance)

    def draw(self, ax):
        ax.add_line(self.artists["line"])
        if "arrow1" in self.artists:
            ax.add_patch(self.artists["arrow1"])
        if "arrow2" in self.artists:
            ax.add_patch(self.artists["arrow2"])


class BandedArcJourney(BaseJourney):
    def __init__(
        self,
        lat_start,
        lon_start,
        lat_end,
        lon_end,
        theta_width,
        N=None,
        start_gap=0,
        end_gap=0,
        **kwargs
    ):
        x1, y1 = get_arc_path(
            lat_start,
            lon_start,
            lat_end,
            lon_end,
            theta=-theta_width / 2,
            N=N,
            start_gap=start_gap,
            end_gap=end_gap,
        )
        x2, y2 = get_arc_path(
            lat_start,
            lon_start,
            lat_end,
            lon_end,
            theta=+theta_width / 2,
            N=N,
            start_gap=start_gap,
            end_gap=end_gap,
        )

        fog_clearance = kwargs.pop("fog_clearance", None)

        x = np.concatenate((x1, x2[::-1]))
        y = np.concatenate((y1, y2[::-1]))
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.polygon = mpatches.Polygon(
            np.array([x, y]).T,
            closed=True,
            transform=PC,
            **kwargs
        )
        artists = {"polygon": self.polygon}
        super().__init__(x, y, artists, fog_clearance=fog_clearance)
    
    def length(self):
        """
        Return the length of the banded arc journey.
        This is the length of one of the arcs.
        """
        np.sum(np.sqrt(np.diff(self.x1) ** 2 + np.diff(self.y1) ** 2))

    def get_label_point(self):
        """
        Return the point at which to place a label.
        We check which of the two arcs is to the right
        """
        n = len(self.x1)
        p1 = (self.x1[n//2], self.y1[n//2])
        p2 = (self.x2[n//2], self.y2[n//2])
        if p1[0] > p2[0]:
            return p1
        else:
            return p2

    def update_polygon(self, frac):
        polygon: mpatches.Polygon = self.artists["polygon"]
        nfull = len(self.x1)
        n = int(nfull * frac)
        x1 = self.x1[:n]
        x2 = self.x2[:n]
        y1 = self.y1[:n]
        y2 = self.y2[:n]
        x = np.concatenate((x1, x2[::-1]))
        y = np.concatenate((y1, y2[::-1]))
        polygon.set_xy(np.array([x, y]).T)
        return polygon


    def animate(self, frac):
        if frac > 0:
            self.show()

        polygon = self.update_polygon(frac)
        return [polygon] + self.animate_label(frac)
    
    def draw(self, ax):
        """
        Draw the banded arc journey on the given axes
        """
        ax.add_patch(self.artists["polygon"])

    def distance_map(self, ax, nx, ny, frac=1):
        # This might have already happened in the animate method
        # but it might not
        self.update_polygon(frac)


        d = np.zeros((ny, nx))
        d[:] = np.inf
        npoint = int(len(self.x1) * frac)
        for xi, yi in zip(self.x1[:npoint], self.y1[:npoint]):
            d_i = distance_map_from_point(ax, xi, yi, nx, ny)
            np.minimum(d, d_i, out=d)
        for xi, yi in zip(self.x2[:npoint], self.y2[:npoint]):
            d_i = distance_map_from_point(ax, xi, yi, nx, ny)
            np.minimum(d, d_i, out=d)

        # Masking the area outside the band is the same as
        # the base map, so we can use the superclass method
        # first.
        # d = super().distance_map(ax, nx, ny, frac=frac)

        # But we also want to mask the area inside the band
        # So we set the distance to zero inside the band
        # pixel centre coordinates
        x = np.arange(nx) / nx
        y = np.arange(ny) / ny
        # mesh of points
        xg, yg = np.meshgrid(x, y)
        points = np.array([xg.flatten(), yg.flatten()]).T
        # transform from axis coordiantes to display coordinates
        points = ax.transAxes.transform(points)

        # contains_points requires display coordinates,
        # i.e. pixels.
        # polygon should already have been animated
        # so we don't need to use the frac parameter here.
        inside = self.polygon.contains_points(points)
        # reshape inside back to the shape of the distance map
        inside = inside.reshape((ny, nx))
        # Set the distance to zero inside the band
        d[inside] = 0
        
        return d
        

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


def read_gpx(filename, interp=100):
    with open(filename, "r") as f:
        gpx = gpxpy.parse(f)
    lat = np.array([p.point.latitude for p in gpx.get_points_data()])
    lon = np.array([p.point.longitude for p in gpx.get_points_data()])

    # strip repeated pairs of points using np.diff, remembering
    # to always keep the first point
    repeats =  (np.diff(lat) == 0) & (np.diff(lon) == 0)
    mask = np.concatenate([[True], ~repeats])
    lat = lat[mask]
    lon = lon[mask]

    if not interp:
        return lat, lon

    lat, lon = interpolate_journey(lat, lon, n=interp)

    return lat, lon




def distance_map_from_point(ax, lon, lat, nx, ny):
    # from data coordinates to display coordinates
    lon, lat = ax.projection.transform_point(lon, lat, PC)
    x0, y0 = ax.transData.transform([lon, lat])
    # from display coordinates to axes coordinates (0 to 1)
    x0, y0 = ax.transAxes.inverted().transform([x0, y0])
    dx = np.arange(nx)
    dy = np.arange(ny)
    dx, dy = np.meshgrid(dx, dy)
    dx = dx - x0 * nx
    dy = dy - y0 * ny
    d = np.sqrt(dx**2 + dy**2)
    return d



def interpolate_journey(x: np.ndarray, y: np.ndarray, n: int):
    # interpolate evenly onto the path defined by x and y
    d = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    d = np.concatenate(([0], d))
    cum_d = np.cumsum(d)
    t = np.linspace(0, cum_d[-1], n)
    x_interp = np.interp(t, cum_d, x)
    y_interp = np.interp(t, cum_d, y)
    return x_interp, y_interp

