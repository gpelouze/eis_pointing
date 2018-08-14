#!/usr/bin/env python3

import functools
import itertools
import multiprocessing as mp

import numpy as np

from ..utils import num
from ..utils import misc

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

def progress_iterator(iterable, nitems=None, msg='{count}', **print_kwargs):
    ''' Print progress while yielding items of an iterable.

    Parameters
    ==========
    iterable :
        The iterable to be yielded.
    nitems : int or None (default: None)
        The number of items in the iterable.
        When `None`, try to use `len(iterable)`. If this fails, set it to 1.
    msg : string or function
        The format of the message to print before yielding each item of the
        iterable.
        Any string is transformed to the function `f(*) -> msg.format(*)`.
        The function is passed 3 keyword arguments:
        - 'count': a counter starting at 1 and incremented each time an element
          is yielded.
        - 'nitems': the value of `nitems`, transformed as described above.
        - 'progress': `count / nitems`.
    **print_kwargs :
        Passed to `print()` when printing the message.

    Yields
    ======
    The items of `iterable`.

    Examples
    ========
    >>> iterable = progress_iterator(range(5), msg='{progress:.1%}')
    >>> for v in iterable:
    ...     do_stuff(v)
    20.0%
    40.0%
    60.0%
    80.0%
    100.0%
    '''
    if nitems is None:
        try:
            nitems = len(iterable)
        except TypeError:
            nitems = 1
    if isinstance(msg, str):
        msg_str = msg
        msg = lambda **kwargs: msg_str.format(**kwargs)
    for count, v in zip(itertools.count(start=1), iterable):
        print(msg(count=count, progress=count / nitems, nitems=nitems), **print_kwargs)
        yield v

def transform_matrix(offset, mode):
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

def transform_center(x, y, mode):
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

def cc_synthetic_raster_step(raster, x, y, t, ref_raster_builder, align_mode,
    x_shift, y_shift, ang_shift, norm=None):

    align_transform, align_center = align_mode

    # get shifted and rotated coordinates
    rast_x, rast_y = num.affine_transform(
        x, y,
        transform_matrix([y_shift, x_shift, ang_shift], align_transform),
        center=transform_center(x, y, align_center),
        )

    # get values of im at the locations of the shifted and rotated coordinates
    im = ref_raster_builder.get_raster(rast_x, rast_y, t)

    # keep only the parts where non-nan values overlap
    try:
        mask1 = raster.mask | np.isnan(raster.data)
    except AttributeError:
        mask1 = np.isnan(raster)
    try:
        mask2 = im.mask | np.isnan(im.data)
    except AttributeError:
        mask2 = np.isnan(im)
    mask = mask1 | mask2
    raster = raster[~mask]
    im = im[~mask]

    if norm is None:
        raster, im, norm = prep_for_cc(raster, im)

    return np.sum(raster * im) / norm

def cc_synthetic_raster(raster, x, y, t, ref_raster_builder, align_mode,
        x_set=None, y_set=None, a_set=None,
        cores=None):

    nx = x_set.number
    ny = y_set.number
    na = a_set.number
    n_iter = nx * na * na

    cc_worker = functools.partial(
        cc_synthetic_raster_step,
        raster, x, y, t, ref_raster_builder, align_mode)
    cc_iter = itertools.product(x_set.world, y_set.world, a_set.world)
    if cores is None:
        cc_iter = misc.eta_iterator(
            cc_iter,
            nitems=n_iter,
            msg='{count} / {nitems} | {progress:.1%} | ETA: {eta}',
            end='\r')
        cc = itertools.starmap(cc_worker, cc_iter)
        cc = list(cc)
    else:
        p = mp.Pool(cores)
        try:
            chunksize, extra = divmod(n_iter, len(p._pool) * 2)
            if extra:
                chunksize += 1
            cc = p.starmap(cc_worker, cc_iter, chunksize=chunksize)
        finally:
            p.terminate()
    cc = np.array(cc)
    cc = cc.reshape(nx, ny, na)
    cc = cc.swapaxes(0, 1) # from (x, y, a) to (y, x, a)

    return cc

def track_synthetic_raster(raster, x, y, t, ref_raster_builder, align_mode,
        x_set=None, y_set=None, a_set=None,
        return_full_cc=False, sub_px=True,
        **kwargs):
    ''' Find the optimal position of a raster with a synthetic raster that is
    generated for each translation and rotation.

    Parameters
    ==========
    raster : 2D ndarray
    ref_raster_builder : aia_raster.SyntheticRasterBuilder
    x_set, y_set, a_set : OffsetSet (default: None)
    return_full_cc : bool (default: False)
        If True, return the full cross-correlation array.
        If False, only return the maximum cross-correlation.
    **kwargs : passed to cc_synthetic_raster.

    Returns
    =======
    offset : ndarray
        An array containing the optimal (y, x, angle) offset between the input
        array and image
    cc : float or 3D array
        Depending on the value of return_full_cc, either the value of the
        cross-correlation at the optimal offset, or the full cross-correlation
        array.
    '''

    cc = cc_synthetic_raster(
        raster, x, y, t, ref_raster_builder, align_mode,
        x_set=x_set, y_set=y_set, a_set=a_set,
        **kwargs)

    offset = num.get_max_location(cc)

    # transform offset to arcsecs and degrees
    offset = convert_offsets(offset, [y_set, x_set, a_set])

    if return_full_cc:
        return offset, cc
    else:
        return offset, np.nanmax(cc)

def align_synthetic_raster(raster, x, y, t, ref_raster_builder, align_mode,
        x_set=None, y_set=None, a_set=None,
        cores=1, save_to='io/rot_raster/cc',
        return_offset=False):
    ''' Align a raster in translation and rotation. '''

    # explore raster with rotation
    raster = np.ma.array(raster, mask=np.isnan(raster))
    offset, cc = track_synthetic_raster(
        raster, x, y, t, ref_raster_builder, align_mode,
        x_set=x_set, y_set=y_set, a_set=a_set,
        return_full_cc=True, cores=cores)
    cc = np.array(cc)
    offset = np.array(offset)

    # get the corrected coordinates
    offset = np.array(offset)
    align_transform, align_center = align_mode
    new_x, new_y = num.affine_transform(
        x, y,
        transform_matrix(offset, align_transform),
        center=transform_center(x, y, align_center),
        )

    if return_offset:
        offset = list(offset) + [cc]
        return new_x, new_y, offset
    else:
        return new_x, new_y
