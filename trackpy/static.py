from scipy.spatial import cKDTree
import random
import numpy as np
from warnings import warn


def pairCorrelation2D(feat, cutoff, fraction = 1., dr = .5, p_indices = None, ndensity=None, boundary = None,
                            handle_edge=True):
    """   
    Calculate the pair correlation function in 2 dimensions.

    Parameters
    ----------
    feat : Pandas DataFrame
        DataFrame containing the x and y coordinates of particles
    cutoff : float
        Maximum distance to calculate g(r)
    fraction : float, optional
        The fraction of particles to calculate g(r) with. May be used to increase speed of function.
        Particles selected at random.
    dr : float, optional
        The bin width
    p_indices : sequence, optional
        Only consider a pair of particles if one of them is in 'p_indices'.
        Uses zero-based indexing, regardless of how 'feat' is indexed.
    ndensity : float, optional
        Density of particle packing. If not specified, density will be calculated assuming rectangular homogeneous
        arrangement.
    boundary : tuple, optional
        Tuple specifying rectangular boundary of particles (xmin, xmax, ymin, ymax). Must be floats.
        Default is to assume a rectangular packing. Boundaries are determined by edge particles.
    handle_edge : boolean, optional
        If true, compensate for reduced area around particles near the edges.

    Returns
    -------
    r_edges : array
        Return the bin edges
    g_r : array
        The values of g_r
    """

    if boundary is None:
        xmin, xmax, ymin, ymax =  feat.x.min(), feat.x.max(), feat.y.min(), feat.y.max()
    else:
        xmin, xmax, ymin, ymax = boundary
        # Disregard all particles outside the bounding box
        feat = feat[(feat.x >= xmin) & (feat.x <= xmax) & (feat.y >= ymin) & (feat.y <= ymax)]

    if ndensity is None:
        ndensity = feat.x.count() / ((xmax - xmin) * (ymax - ymin))  #  particle packing density

    if p_indices is None:
        p_indices = random.sample(range(len(feat)), int(fraction * len(feat)))  # grab random sample of particles

    r_edges = np.arange(0, cutoff + dr, dr)  # radii bins to search for particles
    g_r = np.zeros(len(r_edges) - 1) 
    max_p_count =  int(np.pi * (r_edges.max() + dr)**2 * ndensity * 10)  # upper bound for neighborhood particle count
    ckdtree = cKDTree(feat[['x', 'y']])  # initialize kdtree for fast neighbor search
    points = feat.as_matrix(['x', 'y'])  # Convert pandas dataframe to numpy array for faster indexing
        

    area = np.pi * (np.arange(dr, cutoff + dr, dr)**2 - np.arange(0, cutoff, dr)**2)
    for idx in p_indices:
        dist, idxs = ckdtree.query(points[idx], k=max_p_count, distance_upper_bound=cutoff)
        dist = dist[dist > 0] # We don't want to count the same particle

        
        surface_area_frac = np.ones(len(r_edges) - 1)
        if handle_edge:
            surface_area_frac *= arclen_2d_bounded(r_edges[1:], points[idx], ((xmin, xmax), (ymin, ymax))) / (2*np.pi*r_edges[1:])
        
        g_r +=  np.histogram(dist, bins = r_edges)[0] / (area*surface_area_frac)

    g_r /= (ndensity * len(p_indices))
    return r_edges, g_r



def pairCorrelation3D(feat, cutoff, fraction = 1., dr = .5, p_indices = None, ndensity=None, boundary = None,
                            handle_edge=True):
    """   
    Calculate the pair correlation function in 3 dimensions.

    Parameters
    ----------
    feat : Pandas DataFrame
        DataFrame containing the x, y and z coordinates of particles
    cutoff : float
        Maximum distance to calculate g(r)
    fraction : float, optional
        The fraction of particles to calculate g(r) with. May be used to increase speed of function. Particles selected at random.
    dr : float, optional
        The bin width
    p_indices : sequence, optional
        Only consider a pair of particles if one of them is in 'p_indices'.
        Uses zero-based indexing, regardless of how 'feat' is indexed.
    ndensity : float, optional
        Density of particle packing. If not specified, density will be calculated assuming rectangular homogenous
        arrangement.
    boundary : tuple, optional
        Tuple specifying rectangular prism boundary of particles (xmin, xmax, ymin, ymax, zmin, zmax). Must be floats.
        Default is to assume a rectangular packing. Boundaries are determined by edge particles.
    handle_edge : boolean, optional
        If true, compensate for reduced volume around particles near the edges.

    Returns
    -------
    r_edges : array
        Return the bin edges
    g_r : array
        The values of g_r
    """   

    if boundary is None:
        xmin, xmax, ymin, ymax, zmin, zmax = (feat.x.min(), feat.x.max(), feat.y.min(), feat.y.max(),
                                              feat.z.min(), feat.z.max())
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = boundary

        # Disregard all particles outside the bounding box
        feat = feat[(feat.x >= xmin) & (feat.x <= xmax) & (feat.y >= ymin) & (feat.y <= ymax) &
                    (feat.z >= zmin) & (feat.z <= zmax)]

    if ndensity is None:
        ndensity = feat.x.count() / ((xmax - xmin) * (ymax - ymin) * (zmax - zmin)) #  particle packing density 

    if p_indices is None:
        p_indices = random.sample(range(len(feat)), int(fraction * len(feat)))  # grab random sample of particles

    r_edges = np.arange(0, cutoff + dr, dr)  # radii bins to search for particles
    g_r = np.zeros(len(r_edges) - 1)
    # Estimate upper bound for neighborhood particle count
    max_p_count =  int((4./3.) * np.pi * (r_edges.max() + dr)**3 * ndensity * 10)
    ckdtree = cKDTree(feat[['x', 'y', 'z']])  # initialize kdtree for fast neighbor search
    points = feat.as_matrix(['x', 'y', 'z'])  # Convert pandas dataframe to numpy array for faster indexing
        
    
    shell_vol = (4./3.) * np.pi * (np.arange(dr, cutoff + dr, dr)**3 - np.arange(0, cutoff, dr)**3)
    for idx in p_indices:
        dist, idxs = ckdtree.query(points[idx], k=max_p_count, distance_upper_bound=cutoff)
        dist = dist[dist > 0] # We don't want to count the same particle

        surface_area_frac = np.ones(len(r_edges) - 1)
        if handle_edge:
            boundary = ((xmin, xmax), (ymin, ymax), (zmin, zmax))
            surface_area_frac *= area_3d_bounded(r_edges[1:], points[idx], boundary) / (4*np.pi*r_edges[1:]**2)

        g_r +=  np.histogram(dist, bins = r_edges)[0] / (shell_vol * surface_area_frac)

    g_r /= (ndensity * len(p_indices))
    return r_edges, g_r


def arclen_2d_bounded(R, pos, box):
    arclen = 2*np.pi*R

    h = np.array([pos[0] - box[0][0], box[0][1] - pos[0],
                  pos[1] - box[1][0], box[1][1] - pos[1]])

    for h0 in h:
        mask = h0 < R
        arclen[mask] -= circle_cap_arclen(h0, R[mask])

    for h1, h2 in [[0, 2], [0, 3], [1, 2], [1, 3]]:  # adjacent sides
        mask = h[h1]**2 + h[h2]**2 < R**2
        arclen[mask] += circle_corner_arclen(h[h1], h[h2], R[mask])

    arclen[arclen < 10**-5 * R] = np.nan
    return arclen

def circle_corner_arclen(h1, h2, R):
    """ Length of a circle arc of circle with radius R that is bounded by
    two perpendicular straight lines `h1` and `h2` from the origin.
    h1**2 + h2**2 < R**2
    h1 >= R
    h2 >= R
    """
    return R*(np.arccos(h2 / R) - np.arcsin(h1 / R))

def circle_cap_arclen(h, R):
    """ Length of a circle arc of circle with radius R that is bounded by
    a straight line `h` from the origin. h >= 0, h < R"""
    return 2*R*np.arccos(h / R)

def area_3d_bounded(dist, pos, box, min_z=None, min_x=None):
    """ Calculated using the surface area of a sphere equidistant
    to a certain point.
    When the sphere is truncated by the box boundaries, this distance
    is subtracted using the formula for the sphere cap surface. We
    calculate this by defining h = the distance from point to box edge.
    When for instance sphere is bounded by the top and right boundaries,
    the area in the edge may be counted double. This is the case when
    h1**2 + h2**2 < R**2. This double counted area is calculated
    and added if necessary.
    When the sphere is bounded by three adjacant boundaries,
    the area in the corner may be subtracted double. This is the case when
    h1**2 + h2**2 + h3**2 < R**2. This double counted area is calculated
    and added if necessary.
    The result is the sum of the weights of pos0 and pos1."""

    area = 4*np.pi*dist**2

    h = np.array([pos[0] - box[0][0], box[0][1] - pos[0],
                  pos[1] - box[1][0], box[1][1] - pos[1],
                  pos[2] - box[2][0], box[2][1] - pos[2]])

    for h0 in h:
        mask = h0 < dist
        area[mask] -= sphere_cap_area(h0, dist[mask])

    for h1, h2 in [[0, 2], [0, 3], [0, 4], [0, 5],
                   [1, 2], [1, 3], [1, 4], [1, 5],
                   [2, 4], [2, 5], [3, 4], [3, 5]]:  #2 adjacent sides
        mask = h[h1]**2 + h[h2]**2 < dist**2
        area[mask] += sphere_edge_area(h[h1], h[h2], dist[mask])

    for h1, h2, h3 in [[0, 2, 4], [0, 2, 5], [0, 3, 4], [0, 3, 5],
                       [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5]]:  #3 adjacent sides
        mask = h[h1]**2 + h[h2]**2 + h[h3]**2 < dist**2
        area[mask] -= sphere_corner_area(h[h1], h[h2], h[h3], dist[mask])

    area[area < 10**-7 * dist**2] = np.nan

    return area


def sphere_cap_area(h, R):
    """ Area of a sphere cap of sphere with radius R that is bounded by
    a flat plane `h` from the origin. h >= 0, h < R"""
    return 2*np.pi*R*(R-h)


def sphere_edge_area(x, y, R):
    """ Area of a sphere 'edge' of sphere with radius R that is bounded by
    two perpendicular flat planes `h0`, `h1` from the origin. h >= 0, h < R"""
    p = np.sqrt(R**2 - x**2 - y**2)
    A = (R - x - y)*np.pi - 2*R*np.arctan(x*y/(p*R)) + \
        2*x*np.arctan(y/p) + 2*y*np.arctan(x/p)
    return A*R


def sphere_corner_area(x, y, z, R):
    """ Area of a sphere 'corner' of sphere with radius R that is bounded by
    three perpendicular flat planes `h0`, `h1`, `h2` from the origin. """
    pxy = np.sqrt(R**2 - x**2 - y**2)
    pyz = np.sqrt(R**2 - y**2 - z**2)
    pxz = np.sqrt(R**2 - x**2 - z**2)
    A = np.pi*(R - x - y - z)/2 + \
        x*(np.arctan(y/pxy) + np.arctan(z/pxz)) - R*np.arctan(y*z/(R*pyz)) + \
        y*(np.arctan(x/pxy) + np.arctan(z/pyz)) - R*np.arctan(x*z/(R*pxz)) + \
        z*(np.arctan(x/pxz) + np.arctan(y/pyz)) - R*np.arctan(x*y/(R*pxy))
    return A*R

