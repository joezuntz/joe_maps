import numba
import numpy as np
from .figure import Transform

class CoordinateTransform(Transform):
    def __init__(self, mapping, frame_start, frame_end):
        self.mapping = mapping
        super().__init__(frame_start, frame_end)

    def is_null(self, frame):
        return super().is_null(frame) or (self.mapping is None) or (self.mapping.is_null())

    def __call__(self, image, frame_index):
        data = np.array(image)

        data = scipy.ndimage.geometric_transform(data, self.mapping)
        return PIL.Image.fromarray(data)
    

class RippleMapping:
    def __init__(self, x0=250., y0=250., r0=150.0, a=0.5):
        self.x0 = x0
        self.y0 = y0
        self.r0 = r0
        self.xmin = x0 - r0
        self.xmax = x0 + r0
        self.ymin = y0 - r0
        self.ymax = y0 + r0
        self.a = a
        self.phase = 0
        self.last_x = -1
        self.last_y = -1
        self.last_out = None
    
    def is_null(self):
        return self.a == 0

    def set_phase(self, phase):
        self.phase = phase
    
    def set_amplitude(self, a):
        self.a = a

    def __call__(self, coords):
        if self.a == 0:
            return coords
        x, y, *rest = coords


        if x < self.xmin or x > self.xmax or y < self.ymin or y > self.ymax:
            return coords

        if x == self.last_x and y == self.last_y:
            return (*self.last_out, *rest)

        self.last_x = x
        self.last_y = y
        outcoords = fast_ripple_func(x, y, self.x0, self.y0, self.r0, self.a, self.phase)
        self.last_out = outcoords
        return (*outcoords, *rest)

class LineRippleMapping:
    def __init__(self, xmin, xmax, ymin, ymax, A, B, C, r, rmax, alpha):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.A = A
        self.B = B
        self.C = C
        self.r = r
        self.rmax = rmax
        self.phase = 0
        self.alpha = alpha
        self.last_x = -1
        self.last_y = -1
        self.last_out = None
        self.denom = 1 / np.sqrt(A**2 + B**2)

    def is_null(self):
        return self.alpha == 0

    def set_phase(self, phase):
        self.phase = phase

    def set_amplitude(self, alpha):
        self.alpha = alpha

    def __call__(self, coords):
        if self.alpha == 0:
            return coords

        x, y, *rest = coords

        if x < self.xmin or x > self.xmax or y < self.ymin or y > self.ymax:
            return coords

        if x == self.last_x and y == self.last_y:
            return (*self.last_out, *rest)

        r = np.abs(self.A * x + self.B * y + self.C) * self.denom

        if r > self.rmax:
            return coords
        
        outcoords = fast_line_ripple_func(x, y, self.A, self.B, self.C, self.r, self.alpha, self.phase)

        self.last_x = x
        self.last_y = y
        self.last_out = outcoords
        return (*outcoords, *rest)


@numba.jit(nopython=True, cache=True)
def fast_line_ripple_func(x, y, A, B, C, r0, alpha, phase):
    # find the point on the line Ax + By + C = 0 that is closest to (x, y)
    x_close = (B * (B * x - A * y) - A * C) / (A**2 + B**2)
    y_close = (A * (A * y - B * x) - B * C) / (A**2 + B**2)
    # calculate the distance from (x, y) to the line
    r = np.sqrt((x - x_close)**2 + (y - y_close)**2)

    if r > r0:
        return (x, y)
    
    # amplitude modulation based on distance from the line
    m = (1 - r / r0) * np.cos(2 * np.pi * r / r0 - phase)

    # apply the modulation to the coordinates, scaling from the distance to the line
    # towards the closest point on the line
    x = x_close + (x - x_close) * (1 - alpha * m)
    y = y_close + (y - y_close) * (1 - alpha * m)
    return (x, y)

@numba.jit(nopython=True, cache=True)
def fast_ripple_func(x, y, x0, y0, r0, a, phase):
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    if r > r0:
        return (x, y)
    m = (1 - r / r0) * np.cos(2 * np.pi * r / r0 + phase)
    x = x0 + (x - x0) * (1 - a * m)
    y = y0 + (y - y0) * (1 - a * m)
    return (x, y)


