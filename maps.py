import matplotlib.colors
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy
import matplotlib.patheffects as PathEffects
import bezier
import numpy as np
from . import anim
import os
import shapely
import geopandas as gpd
import cartopy.io.img_tiles
from .journey import ArcJourney, GPXJourney

def make_color(r, g, b, a=255):
    return (r/255., g/255., b/255., a/255.)


class mmc_colors:
    brand_color = make_color(0, 155, 164)
    dark_red = make_color(93, 19, 62)
    land_color = '#E9E9E9'
    wheat = make_color(240, 234, 216)
    ocean_color='#FFFFFF'
    dark_blue = make_color(0, 0, 49)
    darkish_blue = make_color(0, 61, 88)
    brand_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('branded', ['white', brand_color])


def basic_map(lat_min, lat_max, lon_min, lon_max, rivers=False, lakes=True, ocean=True, coast=True,land_color=None, coast_color='black', ocean_color=None, border_color='darkgrey', figsize=(16,16)):
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection)

    if coast:
        ax.coastlines(color=coast_color)

    ax.set_extent([lon_min, lon_max, lat_min, lat_max])

    if land_color is None:
        ax.add_feature(cartopy.feature.LAND)
    else:
        ax.add_feature(cartopy.feature.LAND, color=land_color)

    if lakes:
        ax.add_feature(cartopy.feature.LAKES)

    if rivers:
        ax.add_feature(cartopy.feature.RIVERS)
    ax.add_feature(cartopy.feature.BORDERS,  color=border_color)

    if ocean:
        if ocean_color is None:
            ax.add_feature(cartopy.feature.OCEAN)
        else:
            ax.add_feature(cartopy.feature.OCEAN, color=ocean_color)

    return fig, ax

def add_wiggles_to_path(x, y, wiggles):
    x = x.copy()
    y = y.copy()
    wiggle_wavelength, wiggle_amplitude, wiggle_noise = wiggles

    dx = np.diff(x)
    dy = np.diff(y)
    cum_length = np.cumsum(np.sqrt(dx**2 + dy**2))
    dtheta = np.arctan2(dy, dx)
    dperp = (dtheta + np.pi/2) % (2 * np.pi)


    phase = (cum_length / wiggle_wavelength) * 2 * np.pi + np.random.normal(0, wiggle_noise, len(cum_length))
    wiggle_size = wiggle_amplitude * (1 + np.random.normal(0, wiggle_noise, len(phase)))
    x[1:-1] += (wiggle_size * np.cos(phase) * np.cos(dperp))[1:]
    y[1:-1] += (wiggle_size * np.cos(phase) * np.sin(dperp))[1:]
    return x, y


def interpolate_journey(x: np.ndarray, y: np.ndarray, n: int):
    # interpolate evenly onto the path defined by x and y
    d = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
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
    jitter_tan = np.random.uniform(-jitter_tan_range/2, jitter_tan_range/2, len(x1))
    x1 += jitter_tan * np.sin(theta)
    y1 += jitter_tan * np.cos(theta)
    jitter_perp = np.random.normal(0, jitter_perp_sigma, len(x1))
    x1 += jitter_perp * np.cos(theta)
    y1 += jitter_perp * np.sin(theta)
    return x1, y1



def mmc_map(lat_min, lat_max, lon_min, lon_max, figsize, **kwargs):
    return basic_map(lat_min, lat_max, lon_min, lon_max, figsize=figsize, coast_color='grey', 
                       border_color='grey', ocean_color=mmc_colors.ocean_color, land_color=mmc_colors.land_color)


def blue_marble(ax):
    os.environ['CARTOPY_USER_BACKGROUNDS'] = "/Users/jzuntz/src/anne/base-images/"
    extent = ax.get_extent()
    ax.set_extent([-125, -66.5, 20, 50], ccrs.Geodetic())
    ax.background_img(name='BM', resolution='high')
    ax.set_extent(extent)

def explorer(ax):
    os.environ['CARTOPY_USER_BACKGROUNDS'] = "/Users/jzuntz/src/anne/base-images/"
    extent = ax.get_extent()
    ax.set_extent([-125, -66.5, 20, 50], ccrs.Geodetic())
    ax.background_img(name='explorer', resolution='high')
    ax.set_extent(extent)




def get_border_between(gdf, region1, region2):
    d1 = gdf.query(f'iso3=="{region1}"')
    d2 = gdf.query(f'iso3=="{region2}"')
    border = d1.geometry.iloc[0].intersection(d2.geometry.iloc[0])
    return shapely.get_coordinates(border).T



class Map:
    """
    Class for making a static map.
    """
    def __init__(self, lat_min, lat_max, lon_min, lon_max, figsize, coast_color='grey', border_color='grey', lakes=False, ocean_color=None, land_color=None, noise_level=None, noise_mask=None):
        self.fig, self.ax = basic_map(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
                                         figsize=figsize, ocean_color=ocean_color, land_color=land_color,
                                         coast_color=coast_color, border_color=border_color, lakes=lakes)
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
        domain_in_src_proj = shapely.Polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
        boundary_poly = shapely.Polygon(self.ax.projection.boundary)
        eroded_boundary = boundary_poly.buffer(-self.ax.projection.threshold)
        geom_in_src_proj = eroded_boundary.intersection(
            domain_in_src_proj)
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
            

    def add_text(self, x, y, text, offset=(0, 0),facecolor=mmc_colors.wheat, textcolor=mmc_colors.dark_blue, edgecolor=mmc_colors.dark_blue, **kwargs):
        box = dict(boxstyle='round', facecolor=facecolor, alpha=1, edgecolor=edgecolor)
        return self.ax.text(x+offset[0], y+offset[1], text, fontsize=14, bbox=box, fontname='Arial', color=textcolor)
    
    def add_label(self, place, offset=(0,0), facecolor=mmc_colors.wheat, textcolor=mmc_colors.dark_blue, edgecolor=mmc_colors.dark_blue, **kwargs):
        y, x = self.locations[place]
        label = self.add_text(x, y, place, offset=offset, facecolor=facecolor, textcolor=textcolor, edgecolor=edgecolor, **kwargs)
        self.labels[place] = label
        return label

    def add_point(self, place, *args, **kwargs):
        y, x = self.locations[place]
        return self.ax.plot(x, y,  *args, **kwargs)[0]

    def add_all_points(self,  *args, **kwargs):
        for location in self.locations:
            self.add_point(location,  *args, **kwargs)

    def add_all_labels(self, offsets=None):
        if offsets is None:
            offsets = {}
        for location in self.locations:
            offset = offsets.get(location, (0.4,0))
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


    def add_uncertain_journey(self, start, end, npoint, tangential_scatter_sigma, perpendicular_scatter_width, *args, theta=0, start_gap=0.03, end_gap=0.03, lw=3,  include_line=True, **kwargs):
        # optional line under the points
        if include_line:
            self.add_journey(start, end, theta=theta, start_gap=start_gap, end_gap=end_gap, lw=lw, **kwargs)
        x, y = self.get_path(start, end,  theta=theta, end_gap=end_gap, start_gap=start_gap)
        x, y = jitter_point_cloud(x, y, npoint, tangential_scatter_sigma, perpendicular_scatter_width)
        # don't want to include this
        ls = kwargs.pop('linestyle', None)
        journey, = self.ax.plot(x, y, *args, **kwargs)
        self.journeys.append(journey)
        return journey


    def add_return_journey(self, journey, lw=3, **kwargs):
        x = journey[0].get_xdata()[::-1]
        y = journey[0].get_ydata()[::-1]
        journey = add_journey(self.ax, None, None, None, None, path_x=x, path_y=y, lw=lw, **kwargs)
        self.journeys.append(journey)
        return journey     




    def add_uncertain_gpx_route(self, start, end, npoint, tangential_scatter_sigma, perpendicular_scatter_width,  *args, offset=(0,0), include_line=True, **kwargs):
        if include_line:
            self.add_gpx_route(start, end, *args, offset=offset, **kwargs)
        x, y = self.gpx_routes[start, end]
        x, y = jitter_point_cloud(x, y, npoint, tangential_scatter_sigma, perpendicular_scatter_width)
        journey,  = self.ax.plot(x, y, *args, **kwargs)
        self.journeys.append([journey])
        return journey


    def set_title(self, title, **kwargs):
        self.ax.set_title(title, **kwargs)

    def unmask_circle(self, x, y, r, mask=None):
        if mask is None:
            mask = self.fog_mask
        
        ny, nx = mask.shape

        mask_i = self.distance_map_from_point(x, y, nx, ny)
        mask_i = 1 - np.exp(-mask_i**2 / 2 / r**2)
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
            x = x[:int(frac * len(x))]
            y = y[:int(frac * len(y))]

        for xi, yi in zip(x, y):
            mask_i = self.distance_map_from_point(xi, yi, nx, ny)
            mask_i = 1 - np.exp(-mask_i**2 / 2 / r**2)
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
        self.fog_img = self.ax.imshow(self.fog_array,
                                      cmap='Greys',
                                      extent=extent,
                                      interpolation='nearest',
                                      origin='lower')
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


class AnimatedMap(Map):
    def __init__(self, *args, duration, delta, **kwargs):
        super().__init__(*args, **kwargs)
        frames = int(np.ceil(duration / delta))
        print(f"Animation will have {frames} frames")
        self.delta = delta
        self.duration = duration
        self.timeline = anim.Timeline(self.ax, frames=frames)
        self.init_rect = matplotlib.patches.Rectangle((2,28), 0.01, 0.01, alpha=0)
        self.ax.add_patch(self.init_rect)
        self.current_time = 0.0
        self._current_zoom = self.ax.get_xlim(), self.ax.get_ylim()
        self._current_label_offsets = {}
        self.fog_img = None

    def add_label(self, place, offset=(0,0), facecolor=mmc_colors.wheat, textcolor=mmc_colors.dark_blue, edgecolor=mmc_colors.dark_blue, **kwargs):
        y, x = self.locations[place]
        box = dict(boxstyle='round', facecolor=facecolor, alpha=1, edgecolor=edgecolor)
        label = self.ax.text(x+offset[0], y+offset[1], place, fontsize=14, bbox=box, fontname='Arial', color=textcolor)
        self.labels[place] = label
        self._current_label_offsets[place] = offset
        label.set_visible(False)
        print(f"Showing label {place} at", self.current_time)
        self.show(label)
        return label
    
    def add_point(self, place, *args, **kwargs):
        points = super().add_point(place, *args, **kwargs)
        points.set_visible(False)
        print(f"Showing point at {place} at", self.current_time)
        self.show(points)
        return points

    def _animate_journey(self, journey, time, speed, unmask_radius=0):
        if time is None and speed is None:
            raise ValueError("Set time or speed for animated map journeys")
        elif time is None:
            time = journey.length() / speed

        # This hides the journey to start with
        journey.hide()

        # Work out timing parameters
        end_time = self.current_time + time
        start_frame = self.current_time / self.delta
        end_frame = end_time / self.delta
        print(f"Journey will take from time {self.current_time:.2f} to {end_time:.2f} == frame {start_frame:.1f} to {end_frame:.1f}")

        # # If there is a label, make it appear and disappear.  Really need a journey class to do this properly.
        # if len(journey) > 1 and journey[1] is not None:
        #     journey[1].set_visible(False)
        #     self.timeline.add_transition(anim.make_appear, start_frame, journey[1])
        #     self.timeline.add_transition(anim.make_disappear, end_frame, journey[1])

        # if (self.fog_img is not None) and unmask_radius:

        #     def unmask_journey(f):
        #         self.unmask_journey(journey, unmask_radius, frac=f)
        #         return self.rebuild_fog_of_war()
        #     self.timeline.add_fraction_updater(unmask_journey, start_frame, end_frame)

        self.timeline.add_fraction_updater(journey.animate, start_frame, end_frame)


        # Add the animation to the timeline
        # self.timeline.add_fraction_updater(anim.wipe_journey, start_frame, end_frame, self.ax, direction, journey)
        self.current_time = end_time

    def add_journey(self, *args, time=None, speed=None, direction=None, fog_clear_radius=0, **kwargs):
        journey = super().add_journey(*args, **kwargs)
        self._animate_journey(journey, time, speed, fog_clear_radius)
        return journey

    def add_return_journey(self, *args, time=None, speed=None, direction=None, fog_clear_radius=0,**kwargs):
        journey = super().add_return_journey(*args, **kwargs)
        self._animate_journey(journey, time, speed, fog_clear_radius)
        return journey

    def add_gpx_journey(self, *args, speed=None, time=None, direction=None, fog_clear_radius=0, **kwargs):
        journey = super().add_gpx_journey(*args, **kwargs)
        self._animate_journey(journey, time, speed, fog_clear_radius)
        return journey

    def save(self, path, **kwargs):
        interval = self.delta * 1000
        self.timeline.save(self.fig, path, interval=interval, **kwargs)

    def delay_last_transition(self):
        self.timeline.transitions[-1][1] = self.current_time / self.delta
        
    def set_title(self, title, **kwargs):
        def f(t, **kwargs):
            return [self.ax.set_title(t, **kwargs)]
        self.timeline.add_transition(
            f,
            self.current_time / self.delta,
            title,
            **kwargs
        )

    def hide(self, object):
        def f(obj):
            obj.set_visible(False)
            return [obj]
        self.timeline.add_transition(f, self.current_time / self.delta, object)
    
    def show(self, object):
        def f(obj):
            obj.set_visible(True)
            return [obj]
        self.timeline.add_transition(f, self.current_time / self.delta, object)
    
    def hide_all_journeys(self):
        journeys = self.journeys[:]
        def f():
            out = []
            for journey in journeys:
                for j in journey:
                    if hasattr(j, 'set_visible'):
                        j.set_visible(False)
                        out.append(j)
            return out
        self.timeline.add_transition(f, self.current_time / self.delta)

    def set_label_offsets(self, new_offsets, time):
        frame0 = self.current_time / self.delta
        frame1 = (self.current_time + time) / self.delta
        print(f"Updating label locations from time {self.current_time} to {self.current_time + time} == frame {frame0} to {frame1}")
        locations = {name: self.locations[name] for name in new_offsets}
        old_offsets = self._current_label_offsets.copy()
        def f(frac, new_offsets):
            out = []
            for name, offset in new_offsets.items():
                x0, y0 =  locations[name]
                x0 = x0 + old_offsets[name][0]
                y0 = y0 + old_offsets[name][1]
                label = self.labels[name]
                y1, x1 = self.locations[name]
                x1 += offset[0]
                y1 += offset[1]
                x = x0 + frac * (x1 - x0)
                y = y0 + frac * (y1 - y0)
                label.set_position((x, y))
                out.append(label)
            return out
        self.timeline.add_fraction_updater(f, frame0, frame1, new_offsets)
        self.current_time += time
        self._current_label_offsets.update(new_offsets)

    def pulsing_circle(self, x, y, radius, radius_variation, time, npulse, **kwargs):
        frame0 = self.current_time / self.delta
        frame1 = (self.current_time + time) / self.delta
        print(f"Pulsing circle from time {self.current_time} to {self.current_time + time} == frame {frame0} to {frame1}")
        c = plt.Circle((x, y), radius, **kwargs)
        self.ax.add_patch(c)
        c.set_visible(False)
        def f(frac):
            dr = radius_variation * np.sin(2 * np.pi * frac * npulse)
            c.set_visible(True)
            c.set_radius(radius * (1 + dr))
            return [c]
        self.timeline.add_fraction_updater(f, frame0, frame1)
        self.current_time += time
        return c

    def wait(self, time):
        self.current_time += time

    def zoom(self, xmin, xmax, ymin, ymax, time):
        t0 = self.current_time
        start_x, start_y = self._current_zoom
        frame0 = t0 / self.delta
        t1 = t0 + time
        frame1 = t1 / self.delta
        end_x = (xmin, xmax)
        end_y = (ymin, ymax)
        print(f"Zoom to x={end_x} and y={end_y} will take from time {t0} to {t1} == frame {frame0} to {frame1}")
        self.timeline.add_fraction_updater(
            anim.zoom_to,
            frame0,
            frame1,
            self.ax, [start_x, start_y], [end_x, end_y]
        )
        if self.fog_mask is not None:
            # zoom the fog of war too?
            self.timeline.add_fraction_updater(anim.zoom_extent, 
                                               frame0, 
                                               frame1, 
                                               self.fog_img, 
                                               [start_x, start_y], 
                                               [end_x, end_y])

        self.current_time = t1
        self._current_zoom = end_x, end_y

    def add_tiles(self, zoom_level, extent=None, tiler=None):
        if extent is None:
            xlim, ylim = self._current_zoom
            print("Will add tiles at", xlim, ylim, "at", self.current_time)
            extent = [xlim[0], xlim[1], ylim[0], ylim[1]]

        frame = self.current_time / self.delta
        supe = super()

        def f():
            return [supe.add_tiles(zoom_level, extent, tiler)]
        
        # only actually add the tiles later when we are at the
        # correct zoom
        self.timeline.add_transition(f, frame)


    def set_uncertain_title(self, titles, start_time, end_time, flicker_period, **kwargs):
        """
        Flicker between possible titles
        """
        def f(t, **kwargs):
            pass


    def add_animated_fog_of_war(self, nx, alpha=0.5):
        self.add_fog_of_war(nx, alpha=alpha, should_change_each_image=True)
        def update(i):
            return self.rebuild_fog_of_war()
        self.timeline.add_every_frame_updater(update)


    def unmask_circle(self, x, y, r, time, **kwargs):
        frame0 = self.current_time / self.delta
        frame1 = (self.current_time + time) / self.delta
        print(f"Unmasking circle from time {self.current_time} to {self.current_time + time} == frame {frame0} to {frame1}")
        supe = super()
        def f(frac):
            supe.unmask_circle(x, y, r * frac)
            return self.rebuild_fog_of_war()
        self.timeline.add_fraction_updater(f, frame0, frame1)
        self.current_time += time

    def hide_fog_of_war(self):
        supe = super()
        def f():
            return supe.hide_fog_of_war()
        self.timeline.add_transition(f, self.current_time / self.delta)

    def reset_fog_of_war(self):
        supe = super()
        def f():
            return supe.reset_fog_of_war()
        self.timeline.add_transition(f, self.current_time / self.delta)



def geocode(places):
    locations = gpd.tools.geocode(places, 'Photon', timeout=30)
    out = []
    for (_, row) in locations.iterrows():
        if row.geometry is not None:
            try:
                out.append((row.geometry.x, row.geometry.y))
            except:
                out.append((0.0, 0.0))
    return out
