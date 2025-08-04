import numpy as np

# def circular_distance(nx, ny, x0, y0, ax):
#     xs = 500
#     aspect = (lat_max - lat_min) / (lon_max - lon_min)
#     ys = int(xs * aspect)
#     x0, y0 = ax.transData.transform([x0, y0])
#     x0, y0 = ax.transAxes.inverted().transform([x0, y0])
#     dx = np.arange(xs)
#     dy = np.arange(ys)
#     dx, dy = np.meshgrid(dx, dy)
#     dx = dx - x0 * xs
#     dy = dy - y0 * ys
#     d = np.sqrt(dx**2 + dy**2)
#     return d


def make_noise(nx, ny):
    noise = np.random.uniform(0, 1, (ny, nx))
    rgba = np.zeros((ny, nx, 4))
    rgba[:, :, 0] = noise
    rgba[:, :, 1] = noise
    rgba[:, :, 2] = noise
    rgba[:, :, 3] = 1
    return rgba

