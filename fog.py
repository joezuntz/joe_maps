import numpy as np
from . import zorders
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectSphereBivariateSpline
import cartopy.crs as ccrs
from .journey import BaseJourney
import numba
import healpy


def spherical_random_noise(lats, lons, data, nside, sigma):
    # Generate a random healpix map
    npix = healpy.nside2npix(nside)
    healpix_map = np.random.uniform(size=npix).astype(np.float32)

    # smooth the healpix map
    healpix_map = healpy.smoothing(healpix_map, sigma=np.radians(sigma), verbose=False)

    # Interpolate back to the original lat/lon grid
    smoothed_data = np.zeros_like(data)
    for i in range(len(lats)):
        # use the single latitude but all longitudes
        pix_row = healpy.ang2pix(nside, lons, lats[i], lonlat=True, nest=False)
        smoothed_data[i] = healpix_map[pix_row]
    return smoothed_data

    

class FogOfWar:
    def __init__(self, ax, nx, ny, nlat, nlon, alpha, projection, sigma=1.0, nside=1024):
        self.ax = ax
        self.ny = ny
        self.nx = nx
        self.nlat = nlat
        self.nlon = nlon
        self.animated = False

        # We use these grids to compute distances when unmasking
        lon_max = 180.0 - 180.0 / nlon
        globe_lat = np.linspace(-89, 89, nlat)
        globe_lon = np.linspace(-180, lon_max, nlon)
        self.globe_lat_1d = globe_lat
        self.globe_lon_1d = globe_lon
        globe_lon, globe_lat = np.meshgrid(globe_lon, globe_lat)
        self.globe_lat = globe_lat.flatten()
        self.globe_lon = globe_lon.flatten()
        self.alpha_max = alpha

        self.globe_noise = np.random.uniform(0, 1, (nlat, nlon)).astype(np.float32)
        self.globe_noise = spherical_random_noise(
            self.globe_lat_1d,
            self.globe_lon_1d,
            self.globe_noise,
            nside=nside,
            sigma=sigma
        )

        # self.data is the noise image
        self.data = np.zeros((ny, nx, 4), dtype=np.float32)
        # d = np.random.uniform(0, 1, (ny, nx)).astype(np.float32)
        # gaussian_filter(d, sigma=1, output=d)
        # for i in range(3):
        #     self.data[:, :, i] = d

        # This is the image alpha - we interpolate from the global alpha map
        self.globe_mask = np.ones((nlat, nlon), dtype=np.float32)
        # npix = healpy.nside2npix(nside)
        # self.globe_mask = np.ones(npix, dtype=np.float32)

        # ... to the image mask which is the same shape as the noise image
        # self.image_mask = np.ones((ny, nx), dtype=np.float32)


        self.artist = None
        self.projection = projection
        self.extent = None
        self.alpha_was_updated = False

        self.last_journey_frac = {}

    def unmask_journey(self, journey: BaseJourney, frac=1.0):
        if journey.fog_clearance is None:
            raise ValueError("Journey must have a fog_clearance attribute set.")
        last_frac = self.last_journey_frac.get(journey, 0.0)
        lats, lons = journey.get_lat_lon_update(last_frac, frac)
        self.unmask(lats, lons, r_km=journey.fog_clearance)
        self.last_journey_frac[journey] = frac


    def unmask(self, lats, lons, r_km=100.0):
        if len(lats) == 0:
            return
        lats = np.array(lats)
        lons = np.array(lons)
        # print(f"Unmasking {len(lats)} points with radius {r_km} km")
        # Use only new points for lats and lons!
        # for each point in lats and lons determine the distance of the whole map
        # to that point, and multiple the alpha values by a gaussian fall-off
        distance_matrix = compute_distance_matrix2(lats, lons, self.globe_lat, self.globe_lon)
        distance_matrix = distance_matrix.reshape((-1, self.nlat, self.nlon))
        min_distance = np.min(distance_matrix, axis=0)

        # Now we have a distance matrix of shape (nlat, nlon) where each point
        # is the distance to the nearest point in lats and lons.
        # Our mask is a gaussian fall-off from that point, with a radius of r_km:
        mask = 1 - np.exp(-0.5 * (min_distance / r_km)**2)
    
        # Check if this will change anything
        self.alpha_was_updated = np.any(mask < self.globe_mask)

        if self.alpha_was_updated:
            np.minimum(self.globe_mask, mask, out=self.globe_mask)
    
    def interpolate(self):
        # Get the x and y coordinates of the current axis in its own 
        # projection coordinates
        x0, x1, y0, y1 = self.ax.get_extent()

        # Get a mesh grid of coordinates in the projection
        x = np.linspace(x0, x1, self.nx)
        y = np.linspace(y0, y1, self.ny)
        y, x = np.meshgrid(y, x)

        # Transform these coordinates to the PlateCarree projection
        # from the current projection
        points = ccrs.PlateCarree().transform_points(
            self.projection, x, y
        )
        lon = points[:, :, 0]
        lat = points[:, :, 1]

        # Interpolate the global mask to the current lat/lon grid
        interp = RectSphereBivariateSpline(
            np.radians(90 + self.globe_lat_1d), np.radians(self.globe_lon_1d), self.globe_mask
        )

        alpha = interp(np.radians(90 + lat), np.radians(lon), grid=False).T.clip(0, 1)

        # Also interpolate the noise image to the current lat/lon grid
        interp_noise = RectSphereBivariateSpline(
            np.radians(90 + self.globe_lat_1d), np.radians(self.globe_lon_1d), self.globe_noise
        )
        d = interp_noise(np.radians(90 + lat), np.radians(lon), grid=False).T
        for i in range(3):
            self.data[:, :, i] = d
        self.data[:, :, 3] = alpha * self.alpha_max  # Set alpha channel to 1.0 initially

        # Reset the flag indicating that the alpha was updated
        self.alpha_was_updated = False

    def draw(self):
        extent = self.ax.get_extent()
        self.interpolate()
        # alpha = self.image_mask * self.alpha_max
        # self.data[:, :, 3] = alpha
        self.artist = self.ax.imshow(
            self.data,
            extent=extent,
            origin='lower',
            interpolation='bilinear',
            zorder=zorders.FOG,
        )

        self.extent = extent

    def redraw(self):
        extent_changed = self.extent != self.ax.get_extent()

        # we only need to redraw anything if either the extent
        # has changed or the alpha values have been updated
        if not (self.alpha_was_updated or extent_changed):
            return []
            
        # Otherwise we need to re-interpolate the alpha values
        # to the current extent, because one or both of them must
        # have changed.
        self.interpolate()
        self.artist.set_extent(self.ax.get_extent())

        # alpha = self.image_mask * self.alpha_max
        # self.data[:, :, 3] = alpha
        self.artist.set_data(self.data)

        return [self.artist]


def compute_distance_matrix(lat1, lon1, lat2, lon2, earth_radius=6378.0):
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)

    lat_dif = lat1[:, None]/2 - lat2/2
    lon_dif = lon1[:, None]/2 - lon2/2

    lat_dif = np.sin(lat_dif) ** 2
    lon_dif = np.sin(lon_dif) ** 2

    lon_dif *= np.cos(lat1[:, None]) * np.cos(lat2)
    lon_dif += lat_dif

    lon_dif = np.arctan2(np.sqrt(lon_dif), np.sqrt(1-lon_dif))
    lon_dif *= (2 * earth_radius)

    return lon_dif


def compute_distance_matrix2(lat1, lon1, lat2, lon2, earth_radius=6378.0):
    n = len(lat1)
    if len(lat2) != len(lon2):
        raise ValueError("lat2 and lon2 must have the same length")

    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)

    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    out = np.empty((n, len(lat2)), dtype=lat1.dtype)
    for i in range(n):
        out[i] = compute_distance_matrix2_core(
            lat1[i], lon1[i], lat2, lon2, earth_radius=earth_radius
        )

    return out

@numba.jit(nopython=True, cache=True, parallel=True)
def compute_distance_matrix2_core(lat1, lon1, lat2, lon2, earth_radius=6378.0):
    # lat1 and lon1 are now floats

    lat_dif = 0.5*(lat1 - lat2)
    lon_dif = 0.5*(lon1 - lon2)

    lat_dif = np.sin(lat_dif) ** 2
    lon_dif = np.sin(lon_dif) ** 2

    lon_dif *= np.cos(lat1) * np.cos(lat2)
    lon_dif += lat_dif

    lon_dif = np.arctan2(np.sqrt(lon_dif), np.sqrt(1-lon_dif))
    lon_dif *= (2 * earth_radius)

    return lon_dif
