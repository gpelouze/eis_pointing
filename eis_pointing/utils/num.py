#!/usr/bin/env python3

import datetime
import dateutil.parser
import warnings

import numpy as np
import scipy.interpolate as si

def affine_transform(x, y, transform_matrix, center=(0, 0)):
    ''' Apply an affine transform to an array of coordinates.

    Parameters
    ==========
    x, y : arrays with the same shape
        x and y coordinates to be transformed
    transform_matrix : array_like with shape (2, 3)
        The matrix of the affine transform, [[A, B, C], [D, E, F]]. The new
        coordinates (x', y') are computed from the input coordinates (x, y)
        as follows:

            x' = A*x + B*y + C
            y' = D*x + E*y + F

    center : 2-tulpe of floats (default: (0, 0))
        The center of the transformation. In particular, this is useful for
        rotating arrays around their central value and not the origin.

    Returns
    =======
    transformed_x, transformed_y : arrays
        Arrays with the same shape as the input x and y, and with their values
        transformed by `transform_matrix`.
    '''

    # build array of coordinates, where the 1st axis contains (x, y, ones)
    # values
    ones = np.ones_like(x)
    coordinates = np.array((x, y, ones))

    # transform transform_matrix from (2, 3) to (3, 3)
    transform_matrix = np.vstack((transform_matrix, [0, 0, 1]))

    # add translation to and from the transform center to
    # transformation_matrix
    x_cen, y_cen = center
    translation_to_center = np.array([
        [1, 0, - x_cen],
        [0, 1, - y_cen],
        [0, 0, 1]])
    translation_from_center = np.array([
        [1, 0, x_cen],
        [0, 1, y_cen],
        [0, 0, 1]])
    transform_matrix = np.matmul(transform_matrix, translation_to_center)
    transform_matrix = np.matmul(translation_from_center, transform_matrix)

    # apply transform
    # start with coordinates of shape : (3, d1, ..., dn)
    coordinates = coordinates.reshape(3, -1) # (3, N) N = product(d1, ..., dn)
    coordinates = np.moveaxis(coordinates, 0, -1) # (N, 3)
    coordinates = coordinates.reshape(-1, 3, 1) # (N, 3, 1)
    new_coordinates = np.matmul(transform_matrix, coordinates) # (N, 3, 1)
    new_coordinates = new_coordinates.reshape(-1, 3) # (N, 3)
    new_coordinates = np.moveaxis(new_coordinates, -1, 0) # (3, N)
    new_coordinates = new_coordinates.reshape(3, *ones.shape)
    transformed_x, transformed_y, _ = new_coordinates

    return transformed_x, transformed_y

def roll_nd(a, shifts=None):
    ''' Roll a n-D array along its axes.

    (A wrapper around np.roll, for n-D array.)

    Parameters
    ==========
    array : array_like
        Input array.
    shifts : tuple of ints or None (default: None)
        Tuple containing, for each axis, the number of places by which elements
        are shifted along this axes. If a value of this tuple is None, elements
        of the corresponding axis are shifted by half the axis length.

        If None, shift all axes by half their respective lengths.

    Returns
    =======
    output : ndarray
        Array with the same shape as the input array.

    Example
    =======
    >>> a = np.arange(9).reshape((3,3))
    >>> a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> roll_2d(a, (1, 1))
    array([[8, 6, 7],
           [2, 0, 1],
           [5, 3, 4]])

    '''
    if shifts is None:
        shifts = np.array(a.shape) // 2
    for i, s in enumerate(shifts):
        if s is None:
            s = a.shape[i] // 2
        a = np.roll(a, s, axis=i)
    return a

def almost_identical(arr, threshold, **kwargs):
    ''' Reduce an array of almost identical values to a single one.

    Parameters
    ==========
    arr : np.ndarray
        An array of almost identical values.
    threshold : float
        The maximum standard deviation that is tolerated for the values in arr.
    **kwargs :
        Passed to np.std and np.average. Can be used to reduce arr across a
        choosen dimension.

    Raises
    ======
    ValueError if the standard deviation of the values in arr exceedes the
    specified threshold value

    Returns
    =======
    average : float or np.ndarray
        The average value of arr.
    '''

    irregularity = np.std(arr, **kwargs)
    if np.any(irregularity > threshold):
        msg = 'Uneven array:\n'
        irr_stats = [
            ('irregularity:', irregularity),
            ('irr. mean:', np.mean(irregularity)),
            ('irr. std:', np.std(irregularity)),
            ('irr. min:', np.min(irregularity)),
            ('irr. max:', np.max(irregularity)),
            ]
        for title, value in irr_stats:
            msg += '{} {}\n'.format(title, value)
        msg += 'array percentiles:\n'
        percentiles = [0, 1, 25, 50, 75, 99, 100]
        for p in percentiles:
            msg += '{: 5d}: {:.2f}\n'.format(p, np.percentile(arr, p))
        raise ValueError(msg)

    return np.average(arr, **kwargs)

def chunks(l, n):
    ''' Split list l in chunks of size n.

    http://stackoverflow.com/a/1751478/4352108
    '''
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def friendly_griddata(points, values, new_points, **kwargs):
    ''' A friendly wrapper around scipy.interpolate.griddata.

    Parameters
    ==========
    points : tuple
        Data point coordinates. This is a tuple of ndim arrays, each having the
        same shape as `values`, and each containing data point coordinates
        along a given axis.
    values : array
        Data values. This is an array of dimension ndim.
    new_points : tuple
        Points at which to interpolate data. This has the same structure as
        `points`, but not necessarily the same shape.
    kwargs :
        passed to scipy.interpolate.griddata
    '''
    new_shape = new_points[0].shape
    # make values griddata-friendly
    points = get_griddata_points(points)
    values = values.flatten()
    new_points = get_griddata_points(new_points)
    # projection
    new_values = si.griddata(
        points.T, values, new_points.T,
        **kwargs)
    # make values user-friendly
    new_values = new_values.reshape(*new_shape)
    return new_values

def get_griddata_points(grid):
    ''' Retrieve points in mesh grid of coordinates, that are shaped for use
    with scipy.interpolate.griddata.

    Parameters
    ==========
    grid : np.ndarray
        An array of shape (2, x_dim, y_dim) containing (x, y) coordinates.
        (This should work with more than 2D coordinates.)
    '''
    if type(grid) in [list, tuple]:
        grid = np.array(grid)
    points = np.array([grid[i].flatten()
        for i in range(grid.shape[0])])
    return points

def replace_missing_values(arr, missing, inplace=False, deg=1):
    ''' Interpolate missing elements in a 1D array using a polynomial
    interpolation from the non-missing values.

    Parameters
    ==========
    arr : np.ndarray
        The 1D array in which to replace the element.
    missing : np.ndarray
        A boolean array where the missing elements are marked as True.
    inplace : bool (default: False)
        If True, perform operations in place. If False, copy the array before
        replacing the element.
    deg : int (default: 1)
        The degree of the polynome used for the interpolation.

    Returns
    =======
    arr : np.ndarray
        Updated array.
    '''

    assert arr.ndim == 1, 'arr must be 1D'
    assert arr.shape == missing.shape, \
        'arr and missing must have the same shape'
    assert not np.all(missing), \
        'at least one element must not be missing'
    npx = len(arr)

    if not inplace:
        arr = arr.copy()

    x = np.arange(len(arr))
    c = np.polyfit(x[~missing], arr[~missing], deg)
    p = np.poly1d(c)
    arr[missing] = p(x[missing])

    return arr

def get_max_location(arr, sub_px=True):
    ''' Get the location of the max of an array.

    Parameters
    ==========
    arr : ndarray
    sub_px : bool (default: True)
        whether to perform a parabolic interpolation about the maximum to find
        the maximum with a sub-pixel resolution.

    Returns
    =======
    max_loc : 1D array
        Coordinates of the maximum of the input array.
    '''
    maxcc = np.nanmax(arr)
    if np.isnan(maxcc):
        return np.array([np.nan] * arr.ndim)
    max_px = np.where(arr == maxcc)
    if not np.all([len(m) == 1 for m in max_px]):
        warnings.warn('could not find a unique maximum', RuntimeWarning)
    max_px = np.array([m[0] for m in max_px])
    max_loc = max_px.copy()

    if sub_px:
        max_loc = max_loc.astype(float)
        for dim in range(arr.ndim):
            arr_slice = list(max_px)
            dim_max = max_px[dim]
            if dim_max == 0 or dim_max == arr.shape[dim] - 1:
                m = 'maximum is on the edge of axis {}'.format(dim)
                warnings.warn(m, RuntimeWarning)
                max_loc[dim] = dim_max
            else:
                arr_slice[dim] = [dim_max-1, dim_max, (dim_max+1)]
                interp_points = arr[tuple(arr_slice)]
                a, b, c = interp_points**2
                d = a - 2*b + c
                if d != 0 and not np.isnan(d):
                    max_loc[dim] = dim_max - (c-b)/d + 0.5

    return max_loc

def recarray_to_dict(recarray, lower=False):
    ''' Transform a recarray containing a single row to a dictionnary.

    If lower is True, apply str.lower() to all keys.
    '''
    while recarray.dtype is np.dtype('O'):
        recarray = recarray[0]
    assert len(recarray) == 1, 'structure contains more than one row'
    array = dict(zip(recarray.dtype.names, recarray[0]))
    if lower:
        array = {k.lower(): v for k, v in array.items()}
    return array

@np.vectorize
def total_seconds(timedelta):
    return timedelta.total_seconds()

@np.vectorize
def parse_date(date):
    return dateutil.parser.parse(date)

def seconds_to_timedelta(arr):
    ''' Parse an array of seconds and convert it to timedelta.
    '''
    to_timedelta = np.vectorize(lambda s: datetime.timedelta(seconds=s))
    mask = ~np.isnan(arr)
    td = arr.astype(object)
    td[mask] = to_timedelta(td[mask])
    return td

def dt_average(a, b):
    ''' Average function that is friendly with datetime formats that only
    support substraction. '''
    return a + (b - a) / 2
