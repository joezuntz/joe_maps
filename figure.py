import matplotlib.figure
import io
import PIL.Image
import scipy.ndimage
import numpy as np
import skimage.io

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
            "time": frame_index,
        }
        image = np.array(image)
        image = skimage.transform.warp(image, inverse_map=self.transform, map_args=map_args, mode="constant", cval=1.0) * 255
        image = PIL.Image.fromarray(image.astype(np.uint8))
        return image



