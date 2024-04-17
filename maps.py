import matplotlib.colors
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy
import matplotlib.patheffects as PathEffects
import bezier
import numpy as np
from .text import CurvedText
from . import anim
import os
import gpxpy
import shapely
import geopandas as gpd
import cartopy.io.img_tiles

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



def mmc_map(lat_min, lat_max, lon_min, lon_max, figsize, **kwargs):
    return basic_map(lat_min, lat_max, lon_min, lon_max, figsize=figsize, coast_color='grey', 
                       border_color='grey', ocean_color=mmc_colors.ocean_color, land_color=mmc_colors.land_color)

def add_journey(ax, lat_start, lon_start, lat_end, lon_end, theta=None, text=None, text_start=0.4,start=0.0, fin=1.0, headwidth=0.5, text_flip=False, bidirectional=False, text_properties=None, text_offset=(0,0), path_x=None, path_y=None, **kwargs):
    if path_x is None or path_y is None:
        lat_mid = 0.5 * (lat_start + lat_end)
        lon_mid = 0.5 * (lon_start + lon_end)
        dlat =  lat_mid - lat_start
        dlon = lon_mid - lon_start
        if theta is None:
            theta = np.random.uniform(-45, 45)
        theta = np.radians(theta)
        c = np.cos(theta)
        s = np.sin(theta)
        lon_node = lon_start + (c * dlon) + (s * dlat)
        lat_node = lat_start + (-s*dlon) + (c * dlat)
        nodes = np.array([(lon_start, lon_node, lon_end), (lat_start, lat_node, lat_end)])
        curve = bezier.Curve(nodes, degree=2)
        N = 30
        x = np.zeros(N)
        y = np.zeros(N)
        s0 = start
        S = np.linspace(s0, fin, N)
        for i, s in enumerate(S):
            x_s, y_s = curve.evaluate(s)
            x[i] = x_s
            y[i] = y_s
        x = np.array(x).flatten()
        y = np.array(y).flatten()
    else:
        x = path_x
        y = path_y
    
    # # Decide whether to flip the order? I think, can't remember.
    # dist1 = (x[0] - lon_start)**2 + (y[0] - lat_start)**2
    # dist2 = (x[0] - lon_end)**2 + (y[0] - lat_end)**2
    # if dist2 < dist1:
    #     x = x[::-1]
    #     y = y[::-1]
    line, = ax.plot(x, y, **kwargs)
    
    if text:
        text_properties = {} if text_properties is None else text_properties
        # ds = S[1] - S[0]
        # istart = int((text_start - s0) / ds)
        istart = int(len(x) * text_start)
        if text_flip:
            tx = x[::-1][istart:]
            ty = y[::-1][istart:]
        else:
            tx = x[istart:]
            ty = y[istart:]
        ct = CurvedText(
            x = tx + text_offset[0],
            y = ty + text_offset[1],
            text = text,
            va = 'bottom',
            axes = plt.gca(),
            **text_properties,
        )
    else:
        ct = None
    
    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]

    arr1 = ax.arrow(x[-1] - dx, y[-1] - dy, dx, dy, color=line.get_color(), 
                     head_width=headwidth, head_starts_at_zero=True, length_includes_head=True)
    
    if bidirectional:
        dx = x[0] - x[1]
        dy = y[0] - y[1]

        arr2 = ax.arrow(x[0] - dx, y[0] - dy, dx, dy, color=line.get_color(),
                         head_width=headwidth, head_starts_at_zero=False, length_includes_head=True)
    else:
        arr2 = None

    return line, ct, arr1, arr2


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


def fancy_arrow(ax, lat_start, lon_start, lat_end, lon_end, theta=None, text=None, text_start=0.4,start=0.0, text_offset=(0, 0), fin=1.0, headwidth=0.5, text_flip=False, bidirectional=False, text_properties=None, min_width=1, max_width=5, dash=False, **kwargs):
    lat_mid = 0.5 * (lat_start + lat_end)
    lon_mid = 0.5 * (lon_start + lon_end)
    dlat =  lat_mid - lat_start
    dlon = lon_mid - lon_start
    if theta is None:
        theta = np.random.uniform(-45, 45)
    theta = np.radians(theta)
    c = np.cos(theta)
    s = np.sin(theta)
    lon_node = lon_start + (c * dlon) + (s * dlat)
    lat_node = lat_start + (-s*dlon) + (c * dlat)
    nodes = np.array([(lon_start, lon_node, lon_end), (lat_start, lat_node, lat_end)])
    curve = bezier.Curve(nodes, degree=2)
    N = 30
    x = np.zeros(N)
    y = np.zeros(N)
    s0 = start
    S = np.linspace(s0, fin, N)
    for i, s in enumerate(S):
        x_s, y_s = curve.evaluate(s)
        x[i] = x_s
        y[i] = y_s
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    
    lwidths = np.linspace(min_width, max_width, x.size)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if dash:
        segments = segments[::2]
        lwidths = lwidths[::2]
    lc = plt.matplotlib.collections.LineCollection(segments, linewidths=lwidths, **kwargs)
    ax.add_collection(lc)

    
    if text:
        text_properties = {} if text_properties is None else text_properties
        ds = S[1] - S[0]
        istart = int((text_start - s0) / ds)
        if text_flip:
            tx = x[::-1][istart:]
            ty = y[::-1][istart:]
        else:
            tx = x[istart:]
            ty = y[istart:]
        ct = CurvedText(
            x = tx + text_offset[0],
            y = ty + text_offset[1],
            text = text,
            va = 'bottom',
            axes = plt.gca(),
            **text_properties,
        )
    else:
        ct = None

    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]

    arr = ax.arrow(x[-1] - dx, y[-1] - dy, dx, dy, color=lc.get_color(), 
                     head_width=headwidth, head_starts_at_zero=True, length_includes_head=True)

    return lc, ct, arr
    


def add_journey_from(ax, locations, origin, dest, offset=(0,0), **kwargs):
    x0 = locations[origin]
    y0 = locations[dest]
    return add_journey(ax, x0[1] + offset[1], x0[0] + offset[0], y0[1] + offset[1], y0[0] + offset[0], **kwargs)

def journey_length(j):
    x = j[0].get_xdata()
    y = j[0].get_ydata()
    return np.sum(np.sqrt((np.diff(x)**2 + np.diff(y)**2)))


def read_gpx(filename):
    with open(filename, 'r') as f:
        gpx = gpxpy.parse(f)    
    lat = np.array([p.point.latitude for p in gpx.get_points_data()])
    lon = np.array([p.point.longitude for p in gpx.get_points_data()])
    return lon, lat

def get_border_between(gdf, region1, region2):
    d1 = gdf.query(f'iso3=="{region1}"')
    d2 = gdf.query(f'iso3=="{region2}"')
    border = d1.geometry.iloc[0].intersection(d2.geometry.iloc[0])
    return shapely.get_coordinates(border).T


class Journey:
    """
    Not yet written
    """
    def __init__(self):
        self.x = None
        self.y = None
        self.label = None
        self.text = None
        self.start_arrow = None
        self.end_arrow = None

    def make_return_journey(self):
        pass

class Map:
    """
    Class for making a static map.
    """
    def __init__(self, lat_min, lat_max, lon_min, lon_max, figsize, coast_color='grey', border_color='grey', lakes=False):
        self.fig, self.ax = basic_map(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
                                         figsize=figsize, 
                                         coast_color=coast_color, border_color=border_color, lakes=lakes)
        self.locations = {}
        self.gpx_routes = {}
        self.labels = {}
        self.journeys = []
        self._tiler = None

    def add_locations(self, places):
        self.locations.update(places)

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
            
    
    def add_label(self, place, offset=(0,0), facecolor=mmc_colors.wheat, textcolor=mmc_colors.dark_blue, edgecolor=mmc_colors.dark_blue, **kwargs):
        x, y = self.locations[place]
        box = dict(boxstyle='round', facecolor=facecolor, alpha=1, edgecolor=edgecolor)
        label = self.ax.text(x+offset[0], y+offset[1], place, fontsize=14, bbox=box, fontname='Arial', color=textcolor)
        self.labels[place] = label
        self._current_label_offsets[place] = offset
        return label

    def add_point(self, place, *args, **kwargs):
        x, y = self.locations[place]
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
    
    def add_journey(self, start, end, theta=0, start_gap=0.03, end_gap=0.03, bidirectional=False, lw=3, **kwargs):
        x1, y1 = self.locations[start]
        x2, y2 = self.locations[end]
        journey = add_journey(self.ax, y1, x1, y2, x2, theta=theta, start=start_gap, fin=1 - end_gap, bidirectional=bidirectional, lw=lw, **kwargs)
        self.journeys.append(journey)
        return journey

    def add_return_journey(self, journey, *args, lw=3, **kwargs):
        x = journey[0].get_xdata()[::-1]
        y = journey[0].get_ydata()[::-1]
        journey = add_journey(self.ax, None, None, None, None, path_x=x, path_y=y, lw=lw, **kwargs)
        self.journeys.append(journey)
        return journey        

    def read_gpx_route(self, gpx_file, start, end):
        path_x, path_y = read_gpx(gpx_file)
        self.gpx_routes[start, end] = (path_x, path_y)
        self.gpx_routes[end, start] = (path_x[::-1], path_y[::-1])

    def add_gpx_route(self, start, end, offset=(0,0), text=None, text_properties=None, *args, **kwargs):
        path_x, path_y = self.gpx_routes[start, end]
        path_x = path_x + offset[0]
        path_y = path_y + offset[1]
        journey = self.ax.plot(path_x, path_y, *args, **kwargs)
        # add a text label for this journey
        self.journeys.append(journey)
        if text:
            text_properties=text_properties or {}
            xt = 0.5 * (path_x.max() + path_x.min())
            yt = path_y.min() - 0.2
            t = self.ax.text(xt, yt, text, **text_properties)
            journey = (journey[0], t)
        return journey

    def set_title(self, title, **kwargs):
        self.ax.set_title(title, **kwargs)

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

    def _animate_journey(self, journey, time, speed, direction):
        if time is None and speed is None:
            raise ValueError("Set time or speed for animated map journeys")
        elif time is None:
            time = journey_length(journey) / speed
        if direction is None:
            direction = anim.get_best_direction(journey)

        # This hides the journey to start with
        anim.clip_artists(journey, self.init_rect)

        # Work out timing parameters
        end_time = self.current_time + time
        start_frame = self.current_time / self.delta
        end_frame = end_time / self.delta
        print(f"Journey will take from time {self.current_time} to {end_time} == frame {start_frame} to {end_frame}")

        # If there is a label, make it appear and disappear.  Really need a journey class to do this properly.
        if len(journey) > 1 and journey[1] is not None:
            journey[1].set_visible(False)
            self.timeline.add_transition(anim.make_appear, start_frame, journey[1])
            self.timeline.add_transition(anim.make_disappear, end_frame, journey[1])
            

        # Add the animation to the timeline
        self.timeline.add_fraction_updater(anim.wipe_journey, start_frame, end_frame, self.ax, direction, journey)
        self.current_time = end_time

    def add_journey(self, *args, time=None, speed=None, direction=None, **kwargs):
        journey = super().add_journey(*args, **kwargs)
        self._animate_journey(journey, time, speed, direction)
        return journey

    def add_return_journey(self, *args, time=None, speed=None, direction=None, **kwargs):
        journey = super().add_return_journey(*args, **kwargs)
        self._animate_journey(journey, time, speed, direction)
        return journey

    def add_gpx_route(self, *args, speed=None, time=None, direction=None, **kwargs):
        journey = super().add_gpx_route(*args, **kwargs)
        self._animate_journey(journey, time, speed, direction)
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
                x1, y1 = self.locations[name]
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
