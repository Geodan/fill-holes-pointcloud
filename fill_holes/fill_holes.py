# -*- coding: utf-8 -*-
"""
fill_holes
----------
Generate synthetic points to fill holes in point clouds.

@author: chrisl / Geodan
"""

import numpy as np
from matplotlib.path import Path
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
from shapely.geometry import Polygon, Point
from shapely.ops import cascaded_union, transform
from shapely.wkt import loads


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
    triangles = points[tri_simplices]
    triangles = np.moveaxis(triangles, (1, 0), (0, 2)).reshape((6, -1))
    p0x, p0y, p1x, p1y, p2x, p2y = triangles
    length_a = np.hypot((p0x-p1x), (p0y-p1y))
    length_b = np.hypot((p1x-p2x), (p1y-p2y))
    length_c = np.hypot((p0x-p2x), (p0y-p2y))
    semiperimeter = (length_a + length_b + length_c) / 2.0
    areas = (semiperimeter * (semiperimeter - length_a) *
             (semiperimeter - length_b) * (semiperimeter - length_c))**0.5
    valid_triangles = areas != 0
    circumradii = np.divide((length_a * length_b * length_c), (4.0 * areas),
                            where=valid_triangles, out=np.zeros_like(areas))

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
    circumradii, areas = triangle_geometries(points, tri_simplices)

    big_triangles = np.where(np.logical_and(
        circumradii > max_circumradius,
        areas/circumradii > max_ratio_radius_area
    ))

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
    hole_path = Path(points[:, :2], closed=True)

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


def clip_points(points, bounding_shape):
    """
    Clip points outside of the bounding shape.

    Parameters
    ----------
    points : (Mx3) array
        The coordinates of the points.
    bounding_shape : Polygon
        A bounding shape defined by a shapely Polygon.

    Returns
    -------
    points : (Mx3) array
        The coordinates of the clipped points.
    """
    if len(bounding_shape.interiors) > 0:
        mask = [bounding_shape.contains(Point(p)) for p in points]
    else:
        bounding_path = Path(np.array(bounding_shape.exterior.coords)[:, :2],
                             closed=True)
        mask = bounding_path.contains_points(points[:, :2])

    clipped_points = points[mask]

    return clipped_points


def kde_clustering(values, bandwidth=0.05):
    """
    Cluster values using Kernel Density Estimation
    by splitting values at minima.

    Parameters
    ----------
    values : list of float
        The values to cluster.
    bandwidth : float
        The bandwidth of the kernel.

    Returns
    -------
    labels : list of int
        The labels of the clusters the values belong to.
    """
    X = np.arange(min(values), max(values), bandwidth)
    kernel = gaussian_kde(values)
    estimates = kernel.evaluate(X)
    minima = argrelextrema(estimates, np.less)[0]
    splits = X[minima]

    labels = np.zeros(len(values), dtype=np.int)

    for i in range(len(splits)+1):
        if i == 0:
            labels[values < splits[i]] = i
        elif i == len(splits):
            labels[values > splits[i-1]] = i
        else:
            labels[
                np.logical_and(values > splits[i-1], values < splits[i])
            ] = i

    return labels


def triangles_to_holes(points, tri_simplices, big_triangles,
                       height_clustering=False, kde_bandwidth=0.05):
    """
    Converts the big triangles to polygons, which represent the holes.

    Parameters
    ----------
    points : (Mx3) array
        The coordinates of the points.
    tri_simplices : (Mx3) array
        The indices of the simplices of the triangles.
    big_triangles : list
        The indices of the triangles that are considered big.
    height_clustering : bool
        Option to cluster the triangles based on height using a KDE to
        prevent triangles at different heights from ending up in the same
        polygon.
    kde_bandwidth : float
        The bandwidth of the kernel during kernal density estimation for
        clustering.

    Returns
    -------
    holes : MultiPolygon or list of Polygons
        The polygons of the holes.
    """
    points_indexed = np.array((points[:, 0],
                               points[:, 1],
                               list(range(len(points))))).T
    if len(big_triangles) == 1:
        holes = [Polygon(points_indexed[tri_simplices[big_triangles[0]]])]
    elif not height_clustering:
        holes = cascaded_union([Polygon(points_indexed[tri_simplices[t]])
                                for t in big_triangles])
    elif height_clustering and len(big_triangles) <= 3:
        holes = [Polygon(points_indexed[tri_simplices[t]]) for
                 t in big_triangles]
    else:
        triangles = points[tri_simplices[big_triangles]]
        z_means = np.mean(triangles, axis=1)[:, 2]

        labels = kde_clustering(z_means, kde_bandwidth)

        holes = []
        big_triangles_points = points_indexed[tri_simplices[big_triangles]]
        for label in range(max(labels) + 1):
            big_triangles_cluster = big_triangles_points[labels == label]
            triangle_polygons = [Polygon(t) for t in big_triangles_cluster]
            holes_cluster = cascaded_union(triangle_polygons)
            holes_cluster = [holes_cluster] if type(
                holes_cluster) == Polygon else list(holes_cluster)
            holes.extend(holes_cluster)

    return holes


def find_holes(points, max_circumradius=0.4, max_ratio_radius_area=0.2,
               height_clustering=False, kde_bandwidth=0.05,
               suppress_qhull_errors=False):
    """
    Find holes in a point cloud.

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
    height_clustering : bool
        Option to cluster the triangles based on height using a KDE to
        prevent triangles at different heights from ending up in the same
        polygon. Default: False
    kde_bandwidth : float
        The bandwidth of the kernel during kernal density estimation for
        clustering. Default: 0.05
    suppress_qhull_errors : bool
        If set to true an empty array will be returned when qhull raises an
        error when creating the delaunay triangulation.

    Returns
    -------
    holes : list of Polygon
        The holes in the point cloud.
    """
    # Do a triangulation of the points and check the size of the triangles to
    # find the holes
    try:
        tri = Delaunay(points[:, :2])
    except QhullError as e:
        if suppress_qhull_errors:
            return []
        else:
            raise(e)

    big_triangles = determine_big_triangles(points, tri.simplices,
                                            max_circumradius,
                                            max_ratio_radius_area)

    if len(big_triangles) != 0:
        holes = triangles_to_holes(points, tri.simplices, big_triangles,
                                   height_clustering, kde_bandwidth)

        holes = [holes] if type(holes) == Polygon else list(holes)

        return holes
    else:
        return []


def fill_holes(points, max_circumradius=0.4, max_ratio_radius_area=0.2,
               distance=0.4, percentile=50, normals_z=None, min_norm_z=0,
               bounding_shape=None, height_clustering=False,
               kde_bandwidth=0.05, suppress_qhull_errors=False):
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
    height_clustering : bool
        Option to cluster the triangles based on height using a KDE to
        prevent triangles at different heights from ending up in the same
        polygon. Default: False
    kde_bandwidth : float
        The bandwidth of the kernel during kernal density estimation for
        clustering. Default: 0.05
    suppress_qhull_errors : bool
        If set to true an empty array will be returned when qhull raises an
        error when creating the delaunay triangulation.

    Returns
    -------
    synthetic_points : (Mx3) array
        The synthetic points
    """
    # shift points to 0,0 to increase precision
    shift = np.min(points, axis=0)
    points -= shift

    holes = find_holes(
        points,
        max_circumradius=max_circumradius,
        max_ratio_radius_area=max_ratio_radius_area,
        height_clustering=height_clustering,
        kde_bandwidth=kde_bandwidth,
        suppress_qhull_errors=suppress_qhull_errors
    )

    if len(holes) > 0:
        listX = []
        listY = []
        listZ = []
        for h in holes:
            indices = np.array(h.exterior.coords)[:, 2].astype(int)
            if normals_z is not None:
                hole_normals_z = normals_z[indices]
                X, Y, Z = generate_synthetic_points(points[indices],
                                                    h,
                                                    distance,
                                                    percentile,
                                                    hole_normals_z,
                                                    min_norm_z)
            else:
                X, Y, Z = generate_synthetic_points(points[indices],
                                                    h,
                                                    distance,
                                                    percentile)
            listX.extend(X)
            listY.extend(Y)
            listZ.extend(Z)

        synthetic_points = np.array((listX, listY, listZ)).T

        if bounding_shape is not None:
            if type(bounding_shape) == str:
                bounding_shape = loads(bounding_shape)
            bounding_shape = transform(
                lambda x, y, z=None: (x-shift[0], y-shift[1]), bounding_shape
            )
            synthetic_points = clip_points(synthetic_points, bounding_shape)

        points += shift
        synthetic_points += shift

        return synthetic_points
    else:
        points += shift
        return np.empty((0, 3), dtype=np.float64)
