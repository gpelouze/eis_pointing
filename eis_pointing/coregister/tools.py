#!/usr/bin/env python3

import warnings

import numpy as np

from ..utils import num

class OffsetSet(object):
    def __init__(self, num_range, number=None, step=None):
        ''' Manage a set of offsets used to build cross-correlation maps.

        Parameters
        ==========
        num_range : 2-tuple of floats
            The minimum and maximum values of the offset.

        **either** number or step must be provided
        number : int or None
            The number of points within num_range.
        step : float or None
            The step between the points within num_range. If the length of
            num_range is not a multiple of step, the upper bound is raised.
        '''

        if (number is None) and (step is None):
            raise ValueError('must provide either number or step')
        if not ((number is None) or (step is None)):
            raise ValueError('must provide either number or step')

        start, stop = num_range
        if number is None:
            q, r = divmod(stop - start, step)
            number = int(q + 1)
            if r:
                number += 1
            stop = start + (number - 1) * step

        self.num_range = (start, stop)
        self.number = number
        self.step = step
        self.world = np.linspace(start, stop, number)
        self.indices = np.arange(number)

        if step is not None:
            dw = num.almost_identical(self.world[1:] - self.world[:-1], 1e-10)
            assert abs(dw - step) < 1e-10

    def world_to_index(self, w):
        ''' Get the index of a corresponding world value. '''
        return np.interp(w, self.world, self.indices)

    def index_to_world(self, i):
        ''' Get the world at a given index. '''
        return np.interp(i, self.indices, self.world)

    def __repr__(self):
        name = self.__class__.__name__
        rep = '{name}({self.num_range}, {param})'
        if self.step is not None:
            param = 'step={}'.format(self.step)
        else:
            param = 'number={}'.format(self.number)
        return rep.format(**locals())

def convert_offsets(offsets, offset_sets):
    ''' Convert offset indices to world values for any number of offsets. '''
    world_offsets = []
    for offset, offset_set in zip(offsets, offset_sets):
        world_offsets.append(offset_set.index_to_world(offset))
    return world_offsets

def create_margin(arr, margin, axis):
    ''' Create a spatial margins around a cube of coordinates.

    Parameters
    ==========
    arr : np.ndarray
        3D array containing coordinate values to which the margin must be
        added. Values are assumed to be homogenely spaced.
    margin : float
        The space add keep on both sides of arr. This is in units of arr, (eg.
        arcseconds for spatial coordinates).
    axis : int
        The axis along which to add the margin.

    Returns
    =======
    new_shape : int
        shape of a dimension of that can store the input coordinates, with
        space added for the margins.
    new_slice : slice
        slice along the input axis, describing the position of the input arr in
        the new array.
    '''

    # determine the step
    s_inf = slice(None, -1)
    s_sup = slice(1, None)
    s_all = slice(None)
    cut_inf = [s_all] * arr.ndim
    cut_sup = [s_all] * arr.ndim
    cut_inf[axis] = s_inf
    cut_sup[axis] = s_sup
    d = arr[tuple(cut_sup)] - arr[tuple(cut_inf)]
    try:
        d = num.almost_identical(d, 0.05)
    except ValueError:
        warnings.warn('Rasters appear to have inconsistent steps. '
            'Using the median step found across all rasters.')
        d = np.median(d)

    margin_px = int(np.ceil(margin / d))
    new_shape = 2 * margin_px + arr.shape[axis]
    new_slice = slice(margin_px, - margin_px)

    return new_shape, new_slice

def transform_matrix(offset, mode='rotation'):
    y_shift, x_shift, ang_shift = offset
    ca = np.cos(np.deg2rad(ang_shift))
    sa = np.sin(np.deg2rad(ang_shift))
    if mode == 'rotation':
        return np.array([[ ca, -sa, x_shift],
                         [+sa,  ca, y_shift]])
    elif mode == 'column_rotation':
        return np.array([[1, sa, x_shift],
                         [0, ca, y_shift]])
    else:
        raise ValueError('invalide mode:', mode)

def transform_center(x, y, mode='raster'):
    if mode == 'sun':
        return (0, 0)
    elif mode == 'raster':
        return (x.mean(), y.mean())
    else:
        raise ValueError('invalide mode:', mode)

def prep_for_cc(img1, img2, inplace=False):
    ''' Prepare img1 and img2 for cross correlation computation:
    - set average to 0
    - fill masked values with 0 (if masked array)
    - compute the normalisation value

    Parameters
    ==========
    img1, img2 : ndarray or masked array
        The 2D arrays to prepare
    inplace : bool (default: False)
        If True, don't copy the arrays before removing the average.
        This saves time.

    Returns
    =======
    img1, img2 : ndarray or masked array
        The 2D arrays prepared
    norm : float
        The normalisation for these arrays
    '''
    if not inplace:
        a1 = img1.copy()
        a2 = img2.copy()
    else:
        a1 = img1
        a2 = img2
    if np.issubdtype(a1.dtype, np.integer):
        a1 = a1.astype(float)
    if np.issubdtype(a2.dtype, np.integer):
        a2 = a2.astype(float)
    a1 -= a1.mean()
    a2 -= a2.mean()
    try:
        a1 = a1.filled(0) # = fill with average
        a2 = a2.filled(0)
    except AttributeError:
        # not a masked array
        pass

    norm = np.sqrt(np.sum(a1**2) * np.sum(a2**2))

    return a1, a2, norm
