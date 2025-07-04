import cartopy.io.img_tiles
from .utils import geometry_from_extent
import os

# set cache directory for cartopy tiles
# TODO: make this configurable
cartopy.config["cache_dir"] = "/Users/jzuntz/src/anne/tile-cache"

class TileSuite:
    """
    A set of multiple tile images
    """

    def __init__(self, ax, tiler=None):
        if tiler is None:
            tiler = cartopy.io.img_tiles.OSM(cache=True)
        if not tiler.cache_path:
            raise ValueError("Tiler must have cache=True or it will cost money to use some things")
        self.ax = ax
        self.tiler = tiler
        self._tile_images = {}

    def has_tiles(self):
        """
        Check if there are any tiles added to the axes.
        """
        return len(self._tile_images) > 0

    
    def add_tiles(self, zoom_level, name=None):
        """
        Use "name" when you want to add multiple tile layers.
        """
        if name is None:
            n = 1
            name = f"tile_{n}"
            while name in self._tile_images:
                n += 1
                name = f"tile_{n}"

        extent = self.ax.get_extent()
        artist = self._show(extent, zoom_level)
        self._tile_images[name] = (artist, extent, zoom_level)
        
            
        return artist

    def _show(self, extent, zoom_level):
        geometry = geometry_from_extent(self.ax, extent, self.tiler.crs)
        img, tile_extent, origin = self.tiler.image_for_domain(geometry, zoom_level)
        artist = self.ax.imshow(img, extent=tile_extent, origin=origin, transform=self.tiler.crs)
        return artist

    def redraw(self):
        """
        Redraw the tiles on the axes
        """
        new_extent = self.ax.get_extent()
        new_tile_images = {}
        artists = []
        for name, (artist, extent, zoom_level) in self._tile_images.items():
            if new_extent == extent:
                continue
            # remove the old image
            artist.remove()
            # not sure if we want this:
            artists.append(artist)

            new_artist = self._show(new_extent, zoom_level)
            new_tile_images[name] = (new_artist, new_extent, zoom_level)
            artists.append(new_artist)
        
        self._tile_images.update(new_tile_images)
        return artists
