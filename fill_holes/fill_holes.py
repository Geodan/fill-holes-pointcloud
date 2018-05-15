# -*- coding: utf-8 -*-
"""
Created on Mar 21 2018

@author: chrisl
"""

import numpy as np
from scipy.spatial import Delaunay
from matplotlib.path import Path
from numba import njit
from shapely.geometry import Polygon, Point
from shapely.ops import cascaded_union, transform
from shapely.wkt import loads


@njit()
def triangle_geometries(points, tri_simplices):
    area_list = []
    circum_radius_list = []

    for i in range(len(tri_simplices)):
        triangle = points[tri_simplices[i], :2]

        pa = triangle[0]
        pb = triangle[1]
        pc = triangle[2]
        # Lengths of sides of triangle
        a = (((pa[0]-pb[0])*(pa[0]-pb[0]))+((pa[1]-pb[1])*(pa[1]-pb[1])))**0.5
        b = (((pb[0]-pc[0])*(pb[0]-pc[0]))+((pb[1]-pc[1])*(pb[1]-pc[1])))**0.5
        c = (((pc[0]-pa[0])*(pc[0]-pa[0]))+((pc[1]-pa[1])*(pc[1]-pa[1])))**0.5
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = (s*(s-a)*(s-b)*(s-c))**0.5
        if area != 0:
            circum_radius = a*b*c/(4.0*area)
        else:
            circum_radius = 0

        circum_radius_list.append(circum_radius)
        area_list.append(area)

    return circum_radius_list, area_list


def determine_big_triangles(points, tri_simplices,
                            max_circum_radius, max_ratio_radius_area):
    """
    Determines big triangles based on the circumradius and area.

    Parameters
    ----------
    points : (Mx3) array
        The coordinates of the points.
    tri_simplices : (Mx3) array
        the indices of the simplices of the triangles
    max_circum_radius : float or int
        A triangle with a bigger circumradius than this value will be
        considered big, if the triangle also meets the max_ratio_radius_area
        requirement.
    max_ratio_radius_area : float or int:
        A triangle with a bigger ratio between the circumradius and the area
        of the triangle than this value will be considered big, if
        the triangle also meets the max_circum_radius requirement.

    Returns
    -------
    big_triangles : list
        A list of the indices of the big triangles
    """
    big_triangles = []

    circum_radius_list, area_list = triangle_geometries(points, tri_simplices)

    for i in range(len(circum_radius_list)):
        area = area_list[i]
        circum_radius = circum_radius_list[i]
        if (circum_radius > max_circum_radius and
                area/circum_radius > max_ratio_radius_area):
            big_triangles.append(i)

    return big_triangles


def generate_samples(width, height, distance):
    """
    Generate evenly spread points based on a width, height and distance.

    Parameters
    ----------
    width : float or int
        The width of the rectangle
    height : float or int
        The height of the rectangle
    distance : float or int
        The distance between the points

    Returns
    -------
    XY : (Mx2) array
        The X and Y coordinates of the generated points.
    """
    cellsize = distance/np.sqrt(2)
    rows = int(np.ceil(width/cellsize))
    cols = int(np.ceil(height/cellsize))
    x = np.linspace(0.0, width, rows)
    y = np.linspace(0.0, height, cols)
    X, Y = np.meshgrid(x, y)
    XY = np.array([X.flatten(), Y.flatten()]).T
    return XY


def generate_synthetic_points(points, shape, distance, percentile,
                              normals=None, min_norm_z=0):
    """
    Generate synthetic points within the surrounding points.

    Parameters
    ----------
    points : (Mx3) array
        The coordinates of the surrounding points.
    distance : float or int
        The distance between the points that will be added.
    percentile : int
        The percentile of the Z component of the points neighbouring a hole
        to use for as the Z of the synthetic points.
    normals : (Mx3) array
        The normals of the surrounding points. Will be used to determine
        which points should be considered when determining the Z value of
        the synthetic points.
    min_norm_z : float or int
        The minimal value the Z component of the normal vector of a point
        should be to be considered when determining the Z value of the
        synthetic points.

    Returns
    -------
    synthetic_points : (Mx3) array
        The coordinates of the synthetic points.
    """
    hole_path = Path(points[:, :2])

    # Generate points based on the height and width of the hole
    width = np.ptp(points[:, 0])
    height = np.ptp(points[:, 1])
    samples = generate_samples(width, height, distance)
    samples += np.min(points[:, :2], axis=0)

    # Filter the points that are not within the hole
    within_hole = hole_path.contains_points(samples)
    samples_in_hole = samples[within_hole]
    if len(samples_in_hole) == 0:
        samples_in_hole = np.array(shape.centroid.coords)

    # The height of the points is determined by the height of the
    # points around the hole. A percentile is used to determine the
    # height to be used.
    X = samples_in_hole[:, 0]
    Y = samples_in_hole[:, 1]

    if normals is not None:
        z_values = points[normals[:, 2] > min_norm_z][:, 2]
    else:
        z_values = points[:, 2]

    Z = np.repeat(np.percentile(z_values, percentile, axis=0),
                  len(samples_in_hole))

    return X, Y, Z


def remove_outside_holes(holes, bounding_shape):
    """
    """
    outside_shape = []
    for i, h in enumerate(holes):
        if not bounding_shape.contains(h.centroid):
            outside_shape.append(h)

    holes = [h for h in holes if h not in outside_shape]

    return holes


def fill_holes(points, max_circum_radius=0.4, max_ratio_radius_area=0.2,
               distance=0.4, percentile=50, normals=None, min_norm_z=0,
               bounding_shape=None):
    """
    Fill holes found in a point cloud with synthetic data.

    Parameters
    ----------
    points : (Mx3) array
        The coordinates of the points.
    max_circum_radius : float or int
        A triangle with a bigger circumradius than this value will be
        considered to be a hole, if the triangle also meets the
        max_ratio_radius_area requirement.
    max_ratio_radius_area : float or int:
        A triangle with a bigger ratio between the circumradius and the area
        of the triangle than this value will be considered to be a hole, if
        the triangle also meets the max_circum_radius requirement.
    distance : float or int
        The distance between the points that will be added.
    percentile : int
        The percentile of the Z component of the points neighbouring a hole
        to use for as the Z of the synthetic points.
    normals : (Mx3) array
        The normals of the points. Will be used to determine which points
        should be considered when determining the Z value of the synthetic
        points.
    min_norm_z : float or int
        The minimal value the Z component of the normal vector of a point
        should be to be considered when determining the Z value of the
        synthetic points.
    bounding_shape : str or Path or Polygon
        A shape defined by a polygon WKT string, a matplotlib Path,
        or a shapely Polygon. All points will be clipped to this shape.

    Returns
    -------
    synthetic_points : (Mx3) array
        The synthetic points
    """
    # shift points to 0,0 to increase precision
    shift = np.min(points, axis=0)
    points -= shift

    # Do a triangulation of the points and check the size of the triangles to
    # find the holes
    tri = Delaunay(points[:, :2])
    big_triangles = determine_big_triangles(points, tri.simplices,
                                            max_circum_radius,
                                            max_ratio_radius_area)

    if len(big_triangles) != 0:
        holes = list(cascaded_union([Polygon(points[tri.simplices[t]])
                                     for t in big_triangles]))

        if bounding_shape is not None:
            if type(bounding_shape) == str:
                bounding_shape = loads(bounding_shape)
            bounding_shape = transform(lambda x, y: (x-shift[0], y-shift[1]),
                                       bounding_shape)
            holes = remove_outside_holes(holes, bounding_shape)

        listX = []
        listY = []
        listZ = []
        for h in holes:
            if normals is not None:
                indices = []
                for p in np.array(h.exterior.coords):
                    pi = np.bincount(np.argwhere(points == p)[:, 0]).argmax()
                    indices.append(pi)
                hole_normals = normals[indices]
                X, Y, Z = generate_synthetic_points(np.array(h.exterior.coords),
                                                    h,
                                                    distance,
                                                    percentile,
                                                    hole_normals,
                                                    min_norm_z)
            else:
                X, Y, Z = generate_synthetic_points(np.array(h.exterior.coords),
                                                    h,
                                                    distance,
                                                    percentile)
            listX.extend(X)
            listY.extend(Y)
            listZ.extend(Z)

    synthetic_points = np.array((listX, listY, listZ)).T

    points += shift
    synthetic_points += shift

    return synthetic_points
