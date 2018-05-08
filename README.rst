Fill holes
==========
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
points_filled : (Mx3) array
    The new points added to the original points.