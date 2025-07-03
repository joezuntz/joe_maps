import numpy as np
import matplotlib.patches
import geopandas as gpd

def random_point_in_circle(n, center, radius):

    theta = np.random.uniform(0, 2*np.pi, size=n)
    r = radius * np.sqrt(np.random.uniform(size=n))
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.array([x, y]).T



def circle_to_cicle_patch(x1, y1, r1, x2, y2, r2, n=100, **kwargs):
    # get the angle between the two centers
    theta = np.arctan2(y2 - y1, x2 - x1)
    # generate points along the two opposite facing halves of the two circles
    t1 = theta + np.pi/2 + np.linspace(0, np.pi, n)
    p1 = np.array([x1 + r1 * np.cos(t1), y1 + r1 * np.sin(t1)]).T
    t2 = theta - np.pi/2 + np.linspace(0, np.pi, n)
    p2 = np.array([x2 + r2 * np.cos(t2), y2 + r2 * np.sin(t2)]).T
    # add points connecting the two circles
    r = np.linspace(0, 1, n)
    p3 = p1[0][:, np.newaxis] + r * (p2[-1] - p1[0])[:, np.newaxis]
    p4 = p1[-1][:, np.newaxis] + r * (p2[0] - p1[-1])[:, np.newaxis]

    # collect all the points
    p = np.concatenate([p1, p4.T, p2, p3.T])

    return matplotlib.patches.Polygon(p, closed=True, **kwargs)
    

def random_point_in_quadrilateral(n, points):
    # get the triangle points
    A1, B1, C1 = points[0:3]
    A2, B2, C2 = np.array([points[2], points[3], points[0]])

    # get the edges
    s1 = B1 - A1
    t1 = C1 - A1
    s2 = B2 - A2
    t2 = C2 - A2

    # select the number of points in each triangle
    # with the mean proportional to the area of the triangle
    area1 = 0.5 * (A1[0] * (B1[1] - C1[1]) + B1[0]*(C1[1] - A1[1]) + C1[0]*(A1[1] - B1[1]))
    area2 = 0.5 * (A2[0] * (B2[1] - C2[1]) + B2[0]*(C2[1] - A2[1]) + C2[0]*(A2[1] - B2[1]))
    triangles = np.random.choice([0, 1], size=n, p=(area1/(area1+area2), area2/(area1+area2)))

    # pick random lengths for the two sides
    u = np.random.uniform(size=n)
    v = np.random.uniform(size=n)
    in_triangle = u + v <= 1

    # select the edge vectors depending on which triangle the point is in
    s = np.zeros((n, 2))
    s[:] = s1
    t = np.zeros((n, 2))
    t[:] = t1
    s[triangles == 1] = s2
    t[triangles == 1] = t2

    # select the origin depending on which triangle the point is in
    A = np.zeros((n, 2))
    A[:] = A1
    A[triangles == 1] = A2
    # flip the points if they are not in the triangle
    u[~in_triangle] = 1 - u[~in_triangle]
    v[~in_triangle] = 1 - v[~in_triangle]

    # build the point set
    points = u * s.T + v * t.T + A.T
    return points.T


def geocode(places):
    locations = gpd.tools.geocode(places, 'Photon', timeout=30)
    out = []
    for (_, row) in locations.iterrows():
        if row.geometry is not None:
            try:
                out.append((row.geometry.x, row.geometry.y))
            except:
                out.append((0.0, 0.0))
    return out
