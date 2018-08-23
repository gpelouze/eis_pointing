#!/usr/bin/env python3

import functools
import itertools
import multiprocessing as mp

import numpy as np

from ..utils import num
from ..utils import misc

from . import tools

def cc_step(raster, x, y, t, ref_raster_builder,
    x_shift, y_shift, ang_shift, norm=None):

    # get shifted and rotated coordinates
    rast_x, rast_y = num.affine_transform(
        x, y,
        tools.transform_matrix([y_shift, x_shift, ang_shift]),
        center=tools.transform_center(x, y),
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
        raster, im, norm = tools.prep_for_cc(raster, im)

    return np.sum(raster * im) / norm

def compute_cc(raster, x, y, t, ref_raster_builder,
        x_set=None, y_set=None, a_set=None,
        cores=None):

    nx = x_set.number
    ny = y_set.number
    na = a_set.number
    n_iter = nx * na * na

    cc_worker = functools.partial(
        cc_step,
        raster, x, y, t, ref_raster_builder)
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

def track(raster, x, y, t, ref_raster_builder,
        x_set=None, y_set=None, a_set=None,
        **kwargs):
    ''' Find the optimal position of a raster with a synthetic raster that is
    generated for each translation and rotation.

    Parameters
    ==========
    raster : 2D ndarray
    ref_raster_builder : aia_raster.SyntheticRasterBuilder
    x_set, y_set, a_set : tools.OffsetSet (default: None)
    **kwargs : passed to compute_cc.

    Returns
    =======
    offset : ndarray
        An array containing the optimal (y, x, angle) offset between the input
        array and image
    cc : float or 3D array
        The full cross-correlation array.
    '''

    cc = compute_cc(
        raster, x, y, t, ref_raster_builder,
        x_set=x_set, y_set=y_set, a_set=a_set,
        **kwargs)

    offset = num.get_max_location(cc)

    # transform offset to arcsecs and degrees
    offset = tools.convert_offsets(offset, [y_set, x_set, a_set])

    return offset, cc

def align(raster, x, y, t, ref_raster_builder,
        x_set=None, y_set=None, a_set=None,
        cores=None):
    ''' Align a raster in translation and rotation. '''

    # explore raster with rotation
    raster = np.ma.array(raster, mask=np.isnan(raster))
    offset, cc = track(
        raster, x, y, t, ref_raster_builder,
        x_set=x_set, y_set=y_set, a_set=a_set,
        cores=cores)
    cc = np.array(cc)
    offset = np.array(offset)

    # get the corrected coordinates
    offset = np.array(offset)
    new_x, new_y = num.affine_transform(
        x, y,
        tools.transform_matrix(offset),
        center=tools.transform_center(x, y),
        )

    offset = list(offset) + [cc]
    return new_x, new_y, offset
