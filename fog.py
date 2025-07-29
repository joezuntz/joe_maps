import numpy as np
from . import zorders
from scipy.ndimage import gaussian_filter

class FogOfWar:

    def __init__(self, nx, ax, alpha=0.5, animated=True, zorder=zorders.FOG, smoothing=0):

        lon_min, lon_max, lat_min, lat_max = ax.get_extent()
        ny = int((lat_max - lat_min) / (lon_max - lon_min) * nx)
        self.smoothing = smoothing

        self.animated = animated
        self.array = np.zeros((ny, nx, 4))
        self.artist = None
        self.ax = ax
        self.zorder = zorder

        self.alpha_max = alpha
        self.mask = np.ones((ny, nx))
        self.previous_journeys = set()
        self.last_mask_sum = -1.0
        self.nx = nx
        self.ny = ny

        self.array[:, :, 3] = alpha
        self.randomize_fog(nx, ny)

    def draw(self):
        self.artist = self.ax.imshow(
            self.array,
            cmap="Greys",
            extent=self.ax.get_extent(),
            origin="lower",
            # interpolation="nearest",
            zorder=self.zorder,
        )

    def redraw(self, reanimate=False):
        if self.artist is None:
            return []
        
        updated = False

        ny, nx = self.mask.shape

        if self.animated and reanimate:
            self.randomize_fog(nx, ny)
            updated = True

        if not np.allclose(self.ax.get_extent(), self.artist.get_extent()):
            # the extent has changed, so we need to re-do all
            # the masking
            self.mask[:] = 1
            for journey, frac in self.previous_journeys:
                self.unmask_journey(journey, frac)
            updated = True

        # If the mask has changed due to a journey being
        # unmasked, we need to update.
        if self.last_mask_sum != np.sum(self.mask):
            updated = True

        if not updated:
            # No changes to the mask, fog, or extent, so nothing to redraw
            return []

        
        self.last_mask_sum = np.sum(self.mask)
        self.array[:, :, 3] = self.mask * self.alpha_max
        self.artist.set_data(self.array)
        self.artist.set_extent(self.ax.get_extent())
        return (self.artist,)
    
    def hide(self):
        self.mask[:] = 0
        self.redraw()

    def reset(self):
        self.mask[:] = 1
        self.redraw()


    def randomize_fog(self, nx, ny):
        fog = np.random.uniform(0, 1, (ny, nx))
        if self.smoothing > 0:
            fog = gaussian_filter(fog, sigma=self.smoothing)

        self.array[:, :, 0] = fog
        self.array[:, :, 1] = fog
        self.array[:, :, 2] = fog

    def unmask_circle(self, map, x, y, r):
        ny, nx = self.mask.shape
        mask_i = map.distance_map_from_point(x, y, nx, ny)
        mask_i = 1 - np.exp(-(mask_i**2) / 2 / r**2)
        np.minimum(self.mask, mask_i, out=self.mask)
        return self.mask

    def unmask_journey(self, journey, frac=1):
        # if not journey_overlaps_extent(journey, self.ax.get_extent()):
        #     # The journey does not overlap with the current extent,
        #     # so we don't need to unmask it.
        #     return self.mask
        ny, nx = self.mask.shape
        r = journey.fog_clearance
        if r is None:
            raise ValueError("Journey must have a fog_clearance attribute set to animate")
        distance = journey.distance_map(self.ax, nx, ny, frac)
        mask = 1 - np.exp(-(distance**2) / 2 / r**2)
        np.minimum(self.mask, mask, out=self.mask)

        # We need to re-mask previous journeys when we change the
        # zoom level.
        self.previous_journeys.add((journey, frac))
        return self.mask

def journey_overlaps_extent(journey, extent):
    """
    Check if the journey overlaps with the given extent.
    """
    for artist in journey.artists.values():
        if artist_overlaps_extent(artist, extent):
            return True
    return False

def artist_overlaps_extent(artist, extent):
    """
    Check if the artist overlaps with the given extent.
    """
    artist_extent = artist.get_window_extent()
    artist_extent = artist.get_extent()
    return not (artist_extent[0] > extent[1] or
                artist_extent[1] < extent[0] or
                artist_extent[2] > extent[3] or
                artist_extent[3] < extent[2])


