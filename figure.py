import matplotlib.figure
import io
import PIL.Image
import scipy.ndimage
import numpy as np
import skimage.io
from .waves import WaterWave2D
import warnings

class TransformableFigure(matplotlib.figure.Figure):
    """
    A Figure subclass that allows saving with an arbitrary transformation
    to be applied to the image before saving, including when saving animations.
    
    """
    def __init__(self, *args, post_transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_transform = post_transform

    def savefig(self, fname, *, transparent=None, **kwargs):
        # If the update function has set a frame index, use it
        # to tell the transform how to apply the transformation
        if hasattr(self, "jm_frame_index"):
            frame_index = self.jm_frame_index
        else:
            frame_index = None
            warnings.warn("TransformableFigure.savefig called without setting jm_frame_index on the figure. No transformation will be applied.", UserWarning)

        # If there is no transform, just use the default savefig method
        if self.post_transform is None or self.post_transform.is_null(frame_index):
            return super().savefig(fname, transparent=transparent, **kwargs)

        # Otherwise, save the figure to a buffer, apply the transformation,
        # and then save the transformed image to the file
        buffer = io.BytesIO()
        super().savefig(buffer, transparent=transparent, **kwargs)
        buffer.seek(0)
        image = PIL.Image.open(buffer, formats=["PNG"])

        # Apply the transformation to the image
        image = self.post_transform(image, frame_index)

        # Save the transformed image back to where it was supposed to go
        image.save(fname, format='PNG')


class Transform():
    """
    A class that applies a coordinate transformation to an image
    for use as a post-transform in a TransformableFigure.
    
    """
    def __init__(self, frame_start, frame_end):
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.frame = 0

    def is_null(self, frame):
        if (frame is None) or (frame < self.frame_start) or (frame > self.frame_end):
            return True
        return False

class MultiTransform:
    def __init__(self):
        self.transforms = []

    def add_transform(self, transform):
        self.transforms.append(transform)

    def is_null(self, frame):
        for transform in self.transforms:
            if not transform.is_null(frame):
                return False
        return True

    def __call__(self, image, frame_index):
        for transform in self.transforms:
            if not transform.is_null(frame_index):
                image = transform(image, frame_index)
        return image


class RollTransform(Transform):
    def __init__(self, frame_start, frame_end, amplitudes, amplitude_function, wavelengths, period, phase):
        super().__init__(frame_start, frame_end)
        self.amplitudes = amplitudes
        self.amplitude_function = amplitude_function
        self.wavelengths = wavelengths
        self.period = period
        self.phase = phase
        self.phases = np.random.uniform(0, 2 * np.pi, len(wavelengths))

    def transform(self, coords, time):
        row = coords[:, 0]
        col = coords[:, 1]
        dy = 0
        for lam, amp, phi in zip(self.wavelengths, self.amplitudes, self.phases):
            phase = 2 * np.pi * (row / lam + time / self.period) + self.phase + phi
            dy += np.cos(phase) * amp

        dy *= self.amplitude_function(time)

        return np.array([row, col + dy]).T

    def __call__(self, image, frame_index):
        map_args = {
            "time": frame_index - self.frame_start,
        }
        image = np.array(image)
        image = skimage.transform.warp(image, inverse_map=self.transform, map_args=map_args, mode="constant", cval=1.0) * 255
        image = PIL.Image.fromarray(image.astype(np.uint8))
        return image



class DuckTransform(Transform):
    def __init__(self, frame_start, frame_end, wave_kwargs, iterations_per_frame, path_steps_per_frame, path_x, path_y, path_sigma, scaling):
        self.scaling = scaling
        self.wave = WaterWave2D(**wave_kwargs)
        self.iterations_per_frame = iterations_per_frame
        self.path_x = path_x
        self.path_y = path_y
        self.path_sigma = path_sigma
        self.path_steps_per_frame = path_steps_per_frame
        super().__init__(frame_start, frame_end)

    def set_pattern(self, g):
        g_x, g_y = np.gradient(g)
        self.nx, self.ny = g.shape
        self.g_x = g_x
        self.g_y = g_y

        # make spline interpolators
        self.spline_x = scipy.interpolate.RectBivariateSpline(
            np.arange(g_x.shape[0]), 
            np.arange(g_x.shape[1]), 
            g_x, 
            kx=1, ky=1
        )
        self.spline_y = scipy.interpolate.RectBivariateSpline(
            np.arange(g_y.shape[0]), 
            np.arange(g_y.shape[1]), 
            g_y, 
            kx=1, ky=1
        )

    def transform(self, coords):
        # ny = g_x.shape[1]
        ny = self.ny
        x = coords[:, 0]
        y = coords[:, 1]
        dx = self.spline_x(x, ny - y, grid=False) * self.scaling
        dy = self.spline_y(x, ny - y, grid=False) * self.scaling
        out = np.empty_like(coords)
        out[:, 0] = x + dx
        out[:, 1] = y - dy
        return out
    
    def transform_image(self, image: PIL.Image.Image):
        data = np.array(image)
        # transform the image using skimage.transform.warp
        dx = self.g_x * self.scaling
        dy = -self.g_y * self.scaling
        tdata = map_coordinates_with_shift(data, dx, dy)
        return PIL.Image.fromarray(tdata.astype(np.uint8))
    
    def __call__(self, image, frame_index):
        step = (frame_index - self.frame_start) * self.path_steps_per_frame
        if step < self.path_x.shape[0]:
            self.wave.add_impulse(
                int(self.path_x[step]), 
                int(self.path_y[step]), 
                0.1 * self.path_steps_per_frame,
                self.path_sigma
            )
        # Run the wave simulation for the number of iterations specified
        for _ in range(self.iterations_per_frame):
            self.wave.iterate()

        # Get the current wave height
        eta = self.wave.eta
        self.set_pattern(eta)

        return self.transform_image(image)   

def map_coordinates_with_shift(image, dx, dy):
    ny, nx = image.shape[:2]
    y, x = np.mgrid[0.:ny, 0.:nx]

    # We need to handle the lower-origin vs higher origin difference.
    # What I should actually do is fix waves.py to match what is expected here.
    coords = np.array([ny - y.ravel(), x.ravel()])
    coords += np.array([dy.T.ravel(), dx.T.ravel()])

    # I don't really understand why I have to do the vertical flips like this,
    # but any other way seems to cause it to look wrong somehow
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    # Run each channel through map_coordinates separatrely
    r = scipy.ndimage.map_coordinates(r, coords, order=1, mode="constant", cval=0).reshape(ny, nx)
    g = scipy.ndimage.map_coordinates(g, coords, order=1, mode="constant", cval=0).reshape(ny, nx)
    b = scipy.ndimage.map_coordinates(b, coords, order=1, mode="constant", cval=0).reshape(ny, nx)

    # Put all the channels back together
    out = image.copy()
    out[::-1, :, 0] = r
    out[::-1, :, 1] = g
    out[::-1, :, 2] = b

    # Do the same for alpha channel if it exists
    if image.shape[2] == 4:
        a = image[:, :, 3]
        a = scipy.ndimage.map_coordinates(a, coords, order=1, mode="constant", cval=0).reshape(ny, nx)
        out[::-1, :, 3] = a

    return out
