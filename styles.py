import os
import cartopy.crs as ccrs
import matplotlib
import cartopy.feature
import matplotlib.pyplot as plt


def make_color(r, g, b, a=255):
    return (r / 255.0, g / 255.0, b / 255.0, a / 255.0)


class mmc_colors:
    brand_color = make_color(0, 155, 164)
    dark_red = make_color(93, 19, 62)
    land_color = "#E9E9E9"
    wheat = make_color(240, 234, 216)
    ocean_color = "#FFFFFF"
    dark_blue = make_color(0, 0, 49)
    darkish_blue = make_color(0, 61, 88)
    brand_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "branded", ["white", brand_color]
    )


def basic_map(
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    rivers=False,
    lakes=True,
    ocean=True,
    coast=True,
    land_color=None,
    coast_color="black",
    ocean_color=None,
    border_color="darkgrey",
    figsize=(16, 16),
):
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
    ax.add_feature(cartopy.feature.BORDERS, color=border_color)

    if ocean:
        if ocean_color is None:
            ax.add_feature(cartopy.feature.OCEAN)
        else:
            ax.add_feature(cartopy.feature.OCEAN, color=ocean_color)

    return fig, ax


def mmc_map(lat_min, lat_max, lon_min, lon_max, figsize, **kwargs):
    return basic_map(
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        figsize=figsize,
        coast_color="grey",
        border_color="grey",
        ocean_color=mmc_colors.ocean_color,
        land_color=mmc_colors.land_color,
    )


def blue_marble(ax):
    os.environ["CARTOPY_USER_BACKGROUNDS"] = "/Users/jzuntz/src/anne/base-images/"
    extent = ax.get_extent()
    ax.set_extent([-125, -66.5, 20, 50], ccrs.Geodetic())
    ax.background_img(name="BM", resolution="high")
    ax.set_extent(extent)


def explorer(ax):
    os.environ["CARTOPY_USER_BACKGROUNDS"] = "/Users/jzuntz/src/anne/base-images/"
    extent = ax.get_extent()
    ax.set_extent([-125, -66.5, 20, 50], ccrs.Geodetic())
    ax.background_img(name="explorer", resolution="high")
    ax.set_extent(extent)
