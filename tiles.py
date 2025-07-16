import cartopy.io.img_tiles
from .utils import geometry_from_extent
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
        artist = self.ax.imshow(img, extent=tile_extent, origin=origin, transform=self.tiler.crs, alpha=alpha)
        return artist


# class TileSuite:
#     """
#     A set of multiple tile images
#     """

#     def __init__(self, ax, tiler=None):
#         if tiler is None:
#             tiler = cartopy.io.img_tiles.OSM(cache=True)
#         if not tiler.cache_path:
#             raise ValueError("Tiler must have cache=True or it will cost money to use some things")
#         self.ax = ax
#         self.tiler = tiler
#         self._tile_images = {}

#     def has_tiles(self):
#         """
#         Check if there are any tiles added to the axes.
#         """
#         return len(self._tile_images) > 0
    
#     def __getitem__(self, name):
#         """
#         Get the tile image by name.
#         """
#         return self._tile_images[name]


#     def add_tiles(self, zoom_level, name=None):
#         """
#         Use "name" when you want to add multiple tile layers.
#         """
#         if name is None:
#             n = 1
#             name = f"tile_{n}"
#             while name in self._tile_images:
#                 n += 1
#                 name = f"tile_{n}"

#         extent = self.ax.get_extent()
#         artist = self._show(extent, zoom_level)
#         self._tile_images[name] = artist
#         return artist
    
#     def remove_tiles(self, name):
#         print("Removing tiles with name:", name)
#         artist = self._tile_images.pop(name)
#         artist.remove()
#         return artist


#     def _show(self, extent, zoom_level):
#         geometry = geometry_from_extent(self.ax, extent, self.tiler.crs)
#         img, tile_extent, origin = self.tiler.image_for_domain(geometry, zoom_level)
#         artist = self.ax.imshow(img, extent=tile_extent, origin=origin, transform=self.tiler.crs)
#         artist._user_zoom_level = zoom_level
#         artist._user_extent = extent
#         return artist

#     def redraw(self):
#         """
#         Redraw the tiles on the axes
#         """
#         new_extent = self.ax.get_extent()
#         new_tile_images = {}
#         artists = []
#         for name, artist in self._tile_images.items():
#             if new_extent == artist._user_extent:
#                 continue
#             if (artist.alpha == 0) or (not artist.get_visible()):
#                 # if the artist is not visible then we don't want to animate it
#                 continue

#             # remove the old image
#             artist.remove()
#             # not sure if we want this:
#             artists.append(artist)

#             new_artist = self._show(new_extent, artist._user_zoom_level)
#             new_tile_images[name] = new_artist
#             artists.append(new_artist)
        
#         self._tile_images.update(new_tile_images)
#         return artists
