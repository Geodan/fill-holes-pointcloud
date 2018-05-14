# -*- coding: utf-8 -*-
"""
Created on Mar 21 2018

@author: chrisl
"""

import math
import re
import numpy as np
from scipy.spatial import Delaunay
from matplotlib.path import Path
from numba import jit


@jit
def triangle_geometry(triangle):
    """
    Compute the area and circumradius of a triangle.

    Parameters
    ----------
    triangle : (3x3) array-like
        The coordinates of the points which form the triangle.

    Returns
    -------
    area : float
        The area of the triangle
    circum_r : float
        The circumradius of the triangle
    """
    pa = triangle[0]
    pb = triangle[1]
    pc = triangle[2]
    # Lengths of sides of triangle
    a = math.hypot((pa[0]-pb[0]), (pa[1]-pb[1]))
    b = math.hypot((pb[0]-pc[0]), (pb[1]-pc[1]))
    c = math.hypot((pc[0]-pa[0]), (pc[1]-pa[1]))
    # Semiperimeter of triangle
    s = (a + b + c)/2.0
    # Area of triangle by Heron's formula
    area = math.sqrt(s*(s-a)*(s-b)*(s-c))
    if area != 0:
        circum_r = a*b*c/(4.0*area)
    else:
        circum_r = 0
    return area, circum_r


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

    for i, t in enumerate(tri_simplices):
        area, circum_r = triangle_geometry(points[t, :2])
        if (circum_r > max_circum_radius and
                area/circum_r > max_ratio_radius_area):
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


def clip_points(points, shape):
    """
    Clips an array of points to a shape.

    Parameters
    ----------
    points : (Mx3) array
        The coordinates of the points.
    shape : Path or Polygon
        A shape defined by a matplotlib Path or a shapely
        Polygon. All points will be clipped to this shape.

    Returns
    -------
    points_clip : (Mx3) array
        The coordinates of the points that fall within the shape.

    Note
    ----
    Using a matplotlib Path is significantly faster.
    """
    if type(shape) == Path:
        within_shape = shape.contains_points(points[:, :2])
        points_clip = points[within_shape]
    else:
        if type(shape) == Polygon:
            shapely_points = [Point(p) for p in points]
            within_shape = filter(shape.contains, shapely_points)
            points_clip = np.array([list(p.coords)[0] for p in within_shape])
        else:
            print("Error: bounding shape type not recognized!")

    return points_clip


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


def parsePolygonWKT(wkt):
    """
    Parses a WKT string of a polygon to a numpy array of coordinates.

    Parameters
    ----------
    wkt : str
        A WKT string of a polygon.

    Returns
    -------
    coords : array
        An array of coordinates of the points of the polygon.
    """
    if wkt.find('POLYGON ') == -1:
        raise ValueError("Error: invalid WKT string. Not a Polygon.")
    elif wkt.find('MULTI') != -1:
        raise ValueError(
            "Error: invalid WKT string. MultiPolygons not supported.")
    elif wkt.find('),') != -1:
        print("Warning: interiors will be ignored.")
        wkt = wkt.split('),')[0]

    nums = re.findall(r'\d+(?:\.\d*)?', wkt)
    nums = list(map(float, nums))

    num_dimensions = len(wkt.split(', ')[1].split(' '))
    if num_dimensions == 2:
        coords = np.reshape(nums, (-1, 2))
    elif num_dimensions == 3:
        coords = np.reshape(nums, (-1, 3))
    else:
        print("Error: invalid number of dimension or invalid WKT string")

    return coords


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
        holes = cascaded_union([Polygon(points[tri.simplices[t]])
                                for t in big_triangles])

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

    if bounding_shape is not None:
        if type(bounding_shape) == str:
            bounding_coords = parsePolygonWKT(bounding_shape)
            bounding_shape = Path(bounding_coords)
        synthetic_points = clip_points(synthetic_points, bounding_shape)

    return synthetic_points
