import matplotlib.colors
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy
import matplotlib.patheffects as PathEffects
import bezier
import numpy as np
from .text import CurvedText
import os

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

def add_journey(ax, lat_start, lon_start, lat_end, lon_end, theta=None, text=None, text_start=0.4,start=0.0, fin=1.0, headwidth=0.5, text_flip=False, bidirectional=False, text_properties=None, text_offset=(0,0), **kwargs):
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
    dist1 = (x[0] - lon_start)**2 + (y[0] - lat_start)**2
    dist2 = (x[0] - lon_end)**2 + (y[0] - lat_end)**2
    if dist2 < dist1:
        x = x[::-1]
        y = y[::-1]
    line, = ax.plot(x, y, **kwargs)
    
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
    return ((x[-1] - x[0])**2 + (y[-1] - y[0])**2)**0.5


