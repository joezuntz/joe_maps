import matplotlib.colors
from matplotlib import pyplot as plt
import cartopy
import matplotlib.patheffects as PathEffects
import numpy as np
import shapely
import cartopy.io.img_tiles
from .journey import ArcJourney, GPXJourney, BandedArcJourney, interpolate_journey
from .styles import mmc_colors, basic_map, PC
from .tiles import TileSuite
from .fog import FogOfWar
from . import zorders


def add_wiggles_to_path(x, y, wiggles):
    x = x.copy()
    y = y.copy()
    wiggle_wavelength, wiggle_amplitude, wiggle_noise = wiggles

    dx = np.diff(x)
    dy = np.diff(y)
    cum_length = np.cumsum(np.sqrt(dx**2 + dy**2))
    dtheta = np.arctan2(dy, dx)
    dperp = (dtheta + np.pi / 2) % (2 * np.pi)

    phase = (cum_length / wiggle_wavelength) * 2 * np.pi + np.random.normal(
        0, wiggle_noise, len(cum_length)
    )
    wiggle_size = wiggle_amplitude * (1 + np.random.normal(0, wiggle_noise, len(phase)))
    x[1:-1] += (wiggle_size * np.cos(phase) * np.cos(dperp))[1:]
    y[1:-1] += (wiggle_size * np.cos(phase) * np.sin(dperp))[1:]
    return x, y


def jitter_point_cloud(x, y, npoint, jitter_tan_range, jitter_perp_sigma):
    x1, y1 = interpolate_journey(x, y, npoint)
    dx = np.diff(x1)
    dy = np.diff(y1)
    theta = np.arctan2(dy, dx)
    theta = np.concatenate([theta, [theta[-1]]])
    jitter_tan = np.random.uniform(-jitter_tan_range / 2, jitter_tan_range / 2, len(x1))
    x1 += jitter_tan * np.sin(theta)
    y1 += jitter_tan * np.cos(theta)
    jitter_perp = np.random.normal(0, jitter_perp_sigma, len(x1))
    x1 += jitter_perp * np.cos(theta)
    y1 += jitter_perp * np.sin(theta)
    return x1, y1


def get_border_between(gdf, region1, region2):
    d1 = gdf.query(f'iso3=="{region1}"')
    d2 = gdf.query(f'iso3=="{region2}"')
    border = d1.geometry.iloc[0].intersection(d2.geometry.iloc[0])
    return shapely.get_coordinates(border).T


class Map:
    """
    Class for making a static map.
    """

    def __init__(
        self,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        figsize,
        coast_color="grey",
        border_color="grey",
        lakes=False,
        land=True,
        ocean=True,
        rivers=False,
        coast=True,

        ocean_color=None,
        land_color=None,
        projection=None,
        tiler=None,
    ):
        self.fig, self.ax = basic_map(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            figsize=figsize,
            ocean_color=ocean_color,
            land_color=land_color,
            coast_color=coast_color,
            border_color=border_color,
            coast=coast,
            land=land,
            ocean=ocean,
            rivers=rivers,
            lakes=lakes,
            projection=projection,
        )
        self.locations = {}
        self.gpx_routes = {}
        self.labels = {}
        self.journeys = []
        self.tile_suite = TileSuite(self.ax, tiler)
        self.fog = None

    def add_fog(self, nx, alpha=0.5, animated=True, smoothing=0):
        """
        Add a fog layer to the map.
        """
        if self.fog is not None:
            raise ValueError("Fog layer already exists. Use `reset_fog` to reset it.")

        self.fog = FogOfWar(nx, self.ax, alpha=alpha, animated=animated, smoothing=smoothing)
        self.fog.draw()
        return self.fog


    def add_locations(self, places):
        self.locations.update(places)


    def add_text(
        self,
        place,
        text,
        offset=(0, 0),
        facecolor=mmc_colors.wheat,
        textcolor=mmc_colors.dark_blue,
        edgecolor=mmc_colors.dark_blue,
        zorder=zorders.LABELS,
        fontname="Arial",
        fontsize=14,
        bbox=True,
        **kwargs,
    ):
        if isinstance(place, str):
            y, x = self.locations[place]
        else:
            y, x = place

        if bbox:
            box = dict(boxstyle="round", facecolor=facecolor, alpha=1, edgecolor=edgecolor)
        else:
            box = None

        return self.ax.text(
            x + offset[0],
            y + offset[1],
            text,
            fontsize=fontsize,
            bbox=box,
            fontname=fontname,
            color=textcolor,
            zorder=zorder,
            transform=PC,
        )

    def add_label(
        self,
        place,
        offset=(0, 0),
        facecolor=mmc_colors.wheat,
        textcolor=mmc_colors.dark_blue,
        edgecolor=mmc_colors.dark_blue,
        **kwargs,
    ):
        y, x = self.locations[place]
        label = self.add_text(
            (y, x),
            place,
            offset=offset,
            facecolor=facecolor,
            textcolor=textcolor,
            edgecolor=edgecolor,
            **kwargs,
        )
        self.labels[place] = label
        return label

    def add_point(self, place, *args, **kwargs):
        y, x = self.locations[place]
        return self.ax.plot(x, y, *args, transform=PC, **kwargs)[0]

    def add_all_points(self, *args, **kwargs):
        for location in self.locations:
            self.add_point(location, *args, **kwargs)

    def add_all_labels(self, offsets=None):
        if offsets is None:
            offsets = {}
        for location in self.locations:
            offset = offsets.get(location, (0.0, 0))
            self.add_label(location, offset=offset)

    def add_journey(self, start, end, label=None, label_args={}, **kwargs):
        lat_start, lon_start = self.locations[start]
        lat_end, lon_end = self.locations[end]
        journey = ArcJourney(lat_start, lon_start, lat_end, lon_end, **kwargs)
        journey.add_label(self.ax, label, **label_args)
        journey.draw(self.ax)
        self.journeys.append(journey)
        return journey
    
    def add_gpx_journey(self, gpx_file, label=None, label_args={}, **kwargs):
        journey = GPXJourney(gpx_file, **kwargs)
        journey.add_label(self.ax, label, **label_args)
        journey.draw(self.ax)
        self.journeys.append(journey)
        return journey
    
    def add_band_journey(
        self,
        start,
        end,
        theta_width,
        label=None,
        label_args={},
        **kwargs
    ):
        lat_start, lon_start = self.locations[start]
        lat_end, lon_end = self.locations[end]
        journey = BandedArcJourney(
            lat_start,
            lon_start,
            lat_end,
            lon_end,
            theta_width=theta_width,
            **kwargs
        )
        journey.add_label(self.ax, label, **label_args)
        journey.draw(self.ax)
        self.journeys.append(journey)
        return journey            


    def set_title(self, title, **kwargs):
        self.ax.set_title(title, **kwargs)


    def add_tile_level(self, zoom_level, alpha=1.0):
        self.tile_suite.add_level(zoom_level, alpha=alpha)
        self.tile_suite.redraw()
