import numpy as np
from . import zorders
class FogOfWar:

    def __init__(self, nx, ax, alpha=0.5, animated=True, zorder=zorders.FOG):

        lon_min, lon_max, lat_min, lat_max = ax.get_extent()
        ny = int((lat_max - lat_min) / (lon_max - lon_min) * nx)

        self.animated = animated
        self.array = np.zeros((ny, nx, 4))
        self.array[:, :, 3] = alpha
        self.randomize_fog(nx, ny)
        self.artist = None
        self.ax = ax
        self.zorder = zorder

        self.alpha_max = alpha
        self.mask = np.ones((ny, nx))

    def draw(self):
        self.artist = self.ax.imshow(
            self.array,
            cmap="Greys",
            extent=self.ax.get_extent(),
            origin="lower",
            # interpolation="nearest",
            zorder=self.zorder,
        )

    def redraw(self):
        if self.artist is None:
            return

        ny, nx = self.mask.shape

        if self.animated:
            self.randomize_fog(nx, ny)

        self.array[:, :, 3] = self.mask * self.alpha_max
        a = self.array[:, :, 3]
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
        ny, nx = self.mask.shape
        r = journey.fog_clearance
        if r is None:
            raise ValueError("Journey must have a fog_clearance attribute set to animate")
        distance = journey.distance_map(self.ax, nx, ny, frac)
        mask = 1 - np.exp(-(distance**2) / 2 / r**2)
        np.minimum(self.mask, mask, out=self.mask)
        return self.mask

