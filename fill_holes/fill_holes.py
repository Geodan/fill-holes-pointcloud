# -*- coding: utf-8 -*-
"""
fill_holes
----------
Generate synthetic points to fill holes in point clouds.

@author: chrisl / Geodan
"""

import numpy as np
from matplotlib.path import Path
from numba import njit
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import cascaded_union, transform
from shapely.wkt import loads
from sklearn.cluster import MeanShift


@njit()
def triangle_geometries(points, tri_simplices):
    """
    Compute the circumradius and area of a set of triangles.

    Parameters
    ----------
    points : (Mx3) array
        The coordinates of the points.
    tri_simplices : (Mx3) array
        the indices of the simplices of the triangles.

    Returns
    -------
    circumradii : list of float
        The circumradii of the triangles
    areas : list of float
        The areas of the triangles
    """
    circumradii = []
    areas = []

    for i in range(len(tri_simplices)):
        triangle = points[tri_simplices[i], :2]

        point_a = triangle[0]
        point_b = triangle[1]
        point_c = triangle[2]
        # Lengths of sides of triangle
        x_diff_ab = point_a[0]-point_b[0]
        y_diff_ab = point_a[1]-point_b[1]
        x_diff_bc = point_b[0]-point_c[0]
        y_diff_bc = point_b[1]-point_c[1]
        x_diff_ca = point_c[0]-point_a[0]
        y_diff_ca = point_c[1]-point_a[1]

        length_a = ((x_diff_ab * x_diff_ab) + (y_diff_ab * y_diff_ab))**0.5
        length_b = ((x_diff_bc * x_diff_bc) + (y_diff_bc * y_diff_bc))**0.5
        length_c = ((x_diff_ca * x_diff_ca) + (y_diff_ca * y_diff_ca))**0.5
        # Semiperimeter of triangle
        semiperimeter = (length_a + length_b + length_c) / 2.0
        # Area of triangle by Heron's formula
        area = (semiperimeter * (semiperimeter - length_a) *
                (semiperimeter - length_b) * (semiperimeter - length_c))**0.5
        if area != 0:
            circumradius = (length_a * length_b * length_c) / (4.0 * area)
        else:
            circumradius = 0

        circumradii.append(circumradius)
        areas.append(area)

    return circumradii, areas


def determine_big_triangles(points, tri_simplices,
                            max_circumradius, max_ratio_radius_area):
    """
    Determines big triangles based on the circumradius and area.

    Parameters
    ----------
    points : (Mx3) array
        The coordinates of the points.
    tri_simplices : (Mx3) array
        the indices of the simplices of the triangles
    max_circumradius : float or int
        A triangle with a bigger circumradius than this value will be
        considered big, if the triangle also meets the max_ratio_radius_area
        requirement.
    max_ratio_radius_area : float or int:
        A triangle with a bigger ratio between the circumradius and the area
        of the triangle than this value will be considered big, if
        the triangle also meets the max_circumradius requirement.

    Returns
    -------
    big_triangles : list of int
        A list of the indices of the big triangles
    """
    big_triangles = []

    circumradii, areas = triangle_geometries(points, tri_simplices)

    for i in range(len(circumradii)):
        area = areas[i]
        circumradius = circumradii[i]
        if (circumradius > max_circumradius and
                area/circumradius > max_ratio_radius_area):
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
                              normals_z=None, min_norm_z=0):
    """
    Generate synthetic points within the surrounding points.

    Parameters
    ----------
    points : (Mx3) array
        The coordinates of the surrounding points.
    distance : float or int
        The distance between the points that will be added. Default: 0.4
    percentile : int
        The percentile of the Z component of the points neighbouring a hole
        to use for as the Z of the synthetic points. Default: 50 (median)
    normals_z : array-like of float
        The  Z component of the normals of the surrounding points. Will be
        used to determine which points should be considered when determining
        the Z value of the synthetic points. Default: None
    min_norm_z : float or int
        The minimal value the Z component of the normal vector of a point
        should be to be considered when determining the Z value of the
        synthetic points. Default: 0

    Returns
    -------
    X : (Mx1) array
        The X coordinates of the synthetic points.
    Y : (Mx1) array
        The Y coordinates of the synthetic points.
    Z : (Mx1) array
        The Z coordinates of the synthetic points.
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

    if normals_z is not None:
        z_values = points[normals_z > min_norm_z][:, 2]
        if len(z_values) == 0:
            z_values = np.array([points[np.argmax(normals_z), 2]])
    else:
        z_values = points[:, 2]

    Z = np.repeat(np.percentile(z_values, percentile, axis=0),
                  len(samples_in_hole))

    return X, Y, Z


def clip_holes(holes, bounding_shape):
    """
    Clip holes outside of the bounding shape.

    Parameters
    ----------
    holes : MultiPolygon or list of Polygons
        The polygons of the holes.
    bounding_shape : Polygon
        A shape defined by a shapely Polygon.
        No sythetic points will be added outside this shape.

    Returns
    -------
    holes : list of Polygons
        The holes within the bounding shape.
    """
    if type(holes) == MultiPolygon or type(holes) == Polygon:
        holes = holes.intersection(bounding_shape)
        if type(holes) == Polygon:
            holes = [holes]
        elif type(holes) == MultiPolygon:
            holes = list(holes)
    elif type(holes) == list:
        clipped_holes = []
        for hole in holes:
            hole_clip = hole.intersection(bounding_shape)
            if type(hole_clip) == Polygon:
                clipped_holes.append(hole_clip)
            elif type(hole_clip) == MultiPolygon:
                clipped_holes.extend(hole_clip)
        holes = clipped_holes
    else:
        raise TypeError("Type of holes not recognized.")

    return holes


def triangles_to_holes(points, tri, big_triangles):
    """
    """
    triangles = points[tri.simplices[big_triangles]]
    z_means = np.mean(triangles, axis=1)[:, 2]

    ms = MeanShift().fit(z_means.reshape(-1, 1))

    holes = []
    for label in range(max(ms.labels_)):
        holes_cluster = cascaded_union(
            [Polygon(t) for t in triangles[ms.labels_ == label]])
        holes_cluster = [holes_cluster] if type(
            holes_cluster) == Polygon else list(holes_cluster)
        holes.extend(holes_cluster)

    return holes


def fill_holes(points, max_circumradius=0.4, max_ratio_radius_area=0.2,
               distance=0.4, percentile=50, normals_z=None, min_norm_z=0,
               bounding_shape=None, height_clustering=False):
    """
    Generate synthetic points to fill holes in point clouds.

    Parameters
    ----------
    points : (Mx3) array
        The coordinates of the points.
    max_circumradius : float or int
        A triangle with a bigger circumradius than this value will be
        considered to be a hole, if the triangle also meets the
        max_ratio_radius_area requirement. Default: 0.4
    max_ratio_radius_area : float or int:
        A triangle with a bigger ratio between the circumradius and the area
        of the triangle than this value will be considered to be a hole, if
        the triangle also meets the max_circumradius requirement. Default: 0.2
    distance : float or int
        The distance between the points that will be added.  Default: 0.4
    percentile : int
        The percentile of the Z component of the points neighbouring a hole
        to use for as the Z of the synthetic points. Default: 50 (median)
    normals_z : array-like of float
        The Z component of the normals of the points. Will be used to determine
        which points should be considered when determining the Z value of
        the synthetic points. Default: None
    min_norm_z : float or int
        The minimal value the Z component of the normal vector of a point
        should be to be considered when determining the Z value of the
        synthetic points. Default: 0
    bounding_shape : str or Polygon
        A shape defined by a polygon WKT string or a shapely Polygon.
        No sythetic points will be added outside this shape.  Default: None

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
                                            max_circumradius,
                                            max_ratio_radius_area)

    if len(big_triangles) != 0:
        if len(big_triangles) == 1:
            holes = [Polygon(points[tri.simplices[big_triangles[0]]])]
        elif height_clustering and len(big_triangles) <= 3:
            holes = [Polygon(points[tri.simplices[t]]) for t in big_triangles]
        elif height_clustering and len(big_triangles) > 3:
            holes = triangles_to_holes(points, tri, big_triangles)
        else:
            holes = cascaded_union([Polygon(points[tri.simplices[t]])
                                    for t in big_triangles])

        if bounding_shape is not None:
            if type(bounding_shape) == str:
                bounding_shape = loads(bounding_shape)
            bounding_shape = transform(lambda x, y: (x-shift[0], y-shift[1]),
                                       bounding_shape)
            holes = clip_holes(holes, bounding_shape)
        else:
            holes = [holes] if type(holes) == Polygon else list(holes)

        listX = []
        listY = []
        listZ = []
        for h in holes:
            if normals_z is not None:
                indices = []
                for p in np.array(h.exterior.coords):
                    pi = np.bincount(np.argwhere(points == p)[:, 0]).argmax()
                    indices.append(pi)
                hole_normals_z = normals_z[indices]
                X, Y, Z = generate_synthetic_points(np.array(h.exterior.coords),
                                                    h,
                                                    distance,
                                                    percentile,
                                                    hole_normals_z,
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
    else:
        return np.empty((0, 3), dtype=np.float64)
