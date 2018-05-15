Fill holes
==========
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