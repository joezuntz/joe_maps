import matplotlib.colors
from matplotlib import pyplot as plt
import cartopy
import matplotlib.patheffects as PathEffects
import numpy as np
import shapely
import cartopy.io.img_tiles
from .journey import ArcJourney, GPXJourney
from .styles import mmc_colors, basic_map


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


def interpolate_journey(x: np.ndarray, y: np.ndarray, n: int):
    # interpolate evenly onto the path defined by x and y
    d = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    d = np.concatenate(([0], d))
    cum_d = np.cumsum(d)
    t = np.linspace(0, cum_d[-1], n)
    x_interp = np.interp(t, cum_d, x)
    y_interp = np.interp(t, cum_d, y)
    return x_interp, y_interp


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
        ocean_color=None,
        land_color=None,
        noise_level=None,
        noise_mask=None,
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
            lakes=lakes,
        )
        self.locations = {}
        self.gpx_routes = {}
        self.labels = {}
        self.journeys = []
        self._tiler = None
        self.fog_mask = None
        self.fog_img = None
        self.fog_alpha_max = None
        self.fog_array = None

    @property
    def lat_min(self):
        return self.ax.get_ylim()[0]

    @property
    def lat_max(self):
        return self.ax.get_ylim()[1]

    @property
    def lon_min(self):
        return self.ax.get_xlim()[0]

    @property
    def lon_max(self):
        return self.ax.get_xlim()[1]

    def add_locations(self, places):
        self.locations.update(places)

    def distance_map_from_point(self, lon, lat, nx, ny):
        x0, y0 = self.ax.transData.transform([lon, lat])
        x0, y0 = self.ax.transAxes.inverted().transform([x0, y0])
        dx = np.arange(nx)
        dy = np.arange(ny)
        dx, dy = np.meshgrid(dx, dy)
        dx = dx - x0 * nx
        dy = dy - y0 * ny
        d = np.sqrt(dx**2 + dy**2)
        return d

    def geometry_from_extent(self, extent, crs):
        x1, x2, y1, y2 = extent
        domain_in_src_proj = shapely.Polygon(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]
        )
        boundary_poly = shapely.Polygon(self.ax.projection.boundary)
        eroded_boundary = boundary_poly.buffer(-self.ax.projection.threshold)
        geom_in_src_proj = eroded_boundary.intersection(domain_in_src_proj)
        geom_in_crs = crs.project_geometry(geom_in_src_proj, self.ax.projection)
        return geom_in_crs

    def add_tiles(self, zoom_level, extent=None, tiler=None):
        if tiler is None:
            if self._tiler is None:
                self._tiler = cartopy.io.img_tiles.OSM(cache=True)
            tiler = self._tiler
        if extent is None:
            extent = self.ax.get_extent()
        geometry = self.geometry_from_extent(extent, tiler.crs)
        img, extent, origin = tiler.image_for_domain(geometry, zoom_level)
        return self.ax.imshow(img, extent=extent, origin=origin, transform=tiler.crs)

    def add_text(
        self,
        x,
        y,
        text,
        offset=(0, 0),
        facecolor=mmc_colors.wheat,
        textcolor=mmc_colors.dark_blue,
        edgecolor=mmc_colors.dark_blue,
        **kwargs,
    ):
        box = dict(boxstyle="round", facecolor=facecolor, alpha=1, edgecolor=edgecolor)
        return self.ax.text(
            x + offset[0],
            y + offset[1],
            text,
            fontsize=14,
            bbox=box,
            fontname="Arial",
            color=textcolor,
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
            x,
            y,
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
        return self.ax.plot(x, y, *args, **kwargs)[0]

    def add_all_points(self, *args, **kwargs):
        for location in self.locations:
            self.add_point(location, *args, **kwargs)

    def add_all_labels(self, offsets=None):
        if offsets is None:
            offsets = {}
        for location in self.locations:
            offset = offsets.get(location, (0.4, 0))
            self.add_label(location, offset)

    def add_journey(self, start, end, **kwargs):
        lat_start, lon_start = self.locations[start]
        lat_end, lon_end = self.locations[end]
        journey = ArcJourney(lat_start, lon_start, lat_end, lon_end, **kwargs)
        journey.draw(self.ax)
        self.journeys.append(journey)
        return journey

    def add_gpx_journey(self, gpx_file, **kwargs):
        journey = GPXJourney(gpx_file, **kwargs)
        journey.draw(self.ax)
        self.journeys.append(journey)
        return journey

    def add_uncertain_journey(
        self,
        start,
        end,
        npoint,
        tangential_scatter_sigma,
        perpendicular_scatter_width,
        *args,
        theta=0,
        start_gap=0.03,
        end_gap=0.03,
        lw=3,
        include_line=True,
        **kwargs,
    ):
        # optional line under the points
        if include_line:
            self.add_journey(
                start,
                end,
                theta=theta,
                start_gap=start_gap,
                end_gap=end_gap,
                lw=lw,
                **kwargs,
            )
        x, y = self.get_path(
            start, end, theta=theta, end_gap=end_gap, start_gap=start_gap
        )
        x, y = jitter_point_cloud(
            x, y, npoint, tangential_scatter_sigma, perpendicular_scatter_width
        )
        # don't want to include this
        ls = kwargs.pop("linestyle", None)
        (journey,) = self.ax.plot(x, y, *args, **kwargs)
        self.journeys.append(journey)
        return journey

    # def add_return_journey(self, journey, lw=3, **kwargs):
    #     x = journey[0].get_xdata()[::-1]
    #     y = journey[0].get_ydata()[::-1]
    #     journey = add_journey(self.ax, None, None, None, None, path_x=x, path_y=y, lw=lw, **kwargs)
    #     self.journeys.append(journey)
    #     return journey

    def add_uncertain_gpx_route(
        self,
        start,
        end,
        npoint,
        tangential_scatter_sigma,
        perpendicular_scatter_width,
        *args,
        offset=(0, 0),
        include_line=True,
        **kwargs,
    ):
        if include_line:
            self.add_gpx_route(start, end, *args, offset=offset, **kwargs)
        x, y = self.gpx_routes[start, end]
        x, y = jitter_point_cloud(
            x, y, npoint, tangential_scatter_sigma, perpendicular_scatter_width
        )
        (journey,) = self.ax.plot(x, y, *args, **kwargs)
        self.journeys.append([journey])
        return journey

    def set_title(self, title, **kwargs):
        self.ax.set_title(title, **kwargs)

    def unmask_circle(self, x, y, r, mask=None):
        if mask is None:
            mask = self.fog_mask

        ny, nx = mask.shape

        mask_i = self.distance_map_from_point(x, y, nx, ny)
        mask_i = 1 - np.exp(-(mask_i**2) / 2 / r**2)
        np.minimum(mask, mask_i, out=mask)
        return mask

    def unmask_journey(self, journey, r, mask=None, frac=None):
        if mask is None:
            mask = self.fog_mask

        ny, nx = mask.shape

        line = journey[0]
        x = line.get_xdata()
        y = line.get_ydata()
        if frac is not None:
            x = x[: int(frac * len(x))]
            y = y[: int(frac * len(y))]

        for xi, yi in zip(x, y):
            mask_i = self.distance_map_from_point(xi, yi, nx, ny)
            mask_i = 1 - np.exp(-(mask_i**2) / 2 / r**2)
            np.minimum(mask, mask_i, out=mask)

        return mask

    def add_fog_of_war(self, nx, alpha=0.5, should_change_each_image=True):
        (lon_min, lon_max), (lat_min, lat_max) = self._current_zoom
        ny = int((lat_max - lat_min) / (lon_max - lon_min) * nx)
        fog = np.random.uniform(0, 1, (ny, nx))
        self.fog_is_animated = should_change_each_image
        fog_rgba = np.zeros((ny, nx, 4))
        fog_rgba[:, :, 0] = fog
        fog_rgba[:, :, 1] = fog
        fog_rgba[:, :, 2] = fog
        fog_rgba[:, :, 3] = alpha
        self.fog_array = fog_rgba
        extent = [lon_min, lon_max, lat_min, lat_max]
        self.fog_alpha_max = alpha
        self.fog_img = self.ax.imshow(
            self.fog_array,
            cmap="Greys",
            extent=extent,
            interpolation="nearest",
            origin="lower",
        )
        self.fog_mask = np.ones((ny, nx))

    def rebuild_fog_of_war(self):
        if self.fog_img is None:
            return
        ny, nx = self.fog_mask.shape

        # we might want fog that just vanishes but is otherwise
        # fixed
        if self.fog_is_animated:
            fog = np.random.uniform(0, 1, (ny, nx))
            self.fog_array[:, :, 0] = fog
            self.fog_array[:, :, 1] = fog
            self.fog_array[:, :, 2] = fog
        self.fog_array[:, :, 3] = self.fog_mask * self.fog_alpha_max
        self.fog_img.set_data(self.fog_array)
        return (self.fog_img,)

    def reset_fog_of_war(self):
        self.fog_mask[:] = 1
        return self.rebuild_fog_of_war()

    def hide_fog_of_war(self):
        self.fog_mask[:] = 0
        return self.rebuild_fog_of_war()
