import cartopy.io.img_tiles
from .utils import geometry_from_extent
from . import zorders
import os

# set cache directory for cartopy tiles
# TODO: make this configurable
cartopy.config["cache_dir"] = "/Users/jzuntz/src/anne/tile-cache"

class TileSuite:
    def __init__(self, ax, tiler=None):
        if tiler is None:
            tiler = cartopy.io.img_tiles.OSM(cache=True)
        if not tiler.cache_path:
            raise ValueError("Tiler must have cache=True or it will cost money to use some things")
        self.ax = ax
        self.tiler = tiler
        self.alphas = {}
        self.previous_draws = {}

    def add_level(self, zoom_level, alpha=1.0):
        """
        Add a tile level to the axes.
        """
        self.alphas[zoom_level] = alpha

    def set_alpha(self, zoom_level, alpha):
        self.alphas[zoom_level] = alpha
    
    def redraw(self):
        """
        Draw all the tile levels on the axes.
        """
        updated_artists = []
        new_extent = self.ax.get_extent()

        for zoom_level, new_alpha in self.alphas.items():
            if zoom_level in self.previous_draws:
                old_alpha, old_extent, old_artist = self.previous_draws[zoom_level]

                # if both old and new are zero, we don't need to redraw
                if new_alpha == 0 and old_alpha == 0:
                    continue

                # if the extent and alpha are the same as the previous one
                # then nothing has changed so there are no updates
                if (new_extent == old_extent) and (new_alpha == old_alpha):
                    continue

                # Of only the alpha has changed we can re-use the same
                # artist but just update its alpha value
                if (new_extent == old_extent):
                    # If the alpha has changed, we can just update it
                    old_artist.set_alpha(new_alpha)
                    updated_artists.append(old_artist)
                    self.previous_draws[zoom_level] = new_alpha, old_extent, old_artist
                    continue


                # Otherwise something has changed, so we need to
                # remove the old image from the canvas and re-draw the
                # new one.
                old_artist.remove()
                del self.previous_draws[zoom_level]
                updated_artists.append(old_artist)

            if new_alpha == 0:
                # If the new alpha is zero, we don't need to draw anything
                continue

            # This could be a new artist or a replacement one,
            # but either way the result is the same
            new_artist = self._show(zoom_level, new_extent, new_alpha)
            self.previous_draws[zoom_level] = new_alpha, new_extent, new_artist
            updated_artists.append(new_artist)

        return updated_artists
    
    
    def _show(self, zoom_level, extent, alpha):
        assert alpha != 0
        geometry = geometry_from_extent(self.ax, extent, self.tiler.crs)
        img, tile_extent, origin = self.tiler.image_for_domain(geometry, zoom_level)
        artist = self.ax.imshow(img, extent=tile_extent, origin=origin, transform=self.tiler.crs, alpha=alpha, zorder=zorders.TILE)
        return artist

