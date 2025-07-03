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


# def unmask_line(map, noise, x0, y0, x1, y1, r, n=20):
#     mask = np.ones(noise.shape[:2])
#     for i in range(n):
#         xi = x0 + (x1 - x0) * i / n
#         yi = y0 + (y1 - y0) * i / n
#         mask_i = map.distance_map_from_point(xi, yi)
#         mask_i = 1 - np.exp(-mask_i**2 / 2 / r**2)
#         np.minimum(mask, mask_i, out=mask)
#     return mask

# def unmask_path(noise, path_x, path_y, r):
#     mask = np.ones(noise.shape[:2])
#     for xi, yi in zip(path_x, path_y):
#         mask_i = map.distance_map_from_point(xi, yi)
#         mask_i = 1 - np.exp(-mask_i**2 / 2 / r**2)
#         np.minimum(mask, mask_i, out=mask)
#     return mask
