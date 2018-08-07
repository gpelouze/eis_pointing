#!/usr/bin/env python3

import numpy as np
import scipy.interpolate as si

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
