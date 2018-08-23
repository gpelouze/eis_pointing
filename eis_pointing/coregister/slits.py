#!/usr/bin/env python3

import datetime
import warnings
import functools
import itertools
import multiprocessing as mp

import numpy as np

from ..utils import num

from . import tools

def cc_step(a, x, y, t, ref_raster_builder,
        x_shift, y_shift, ang_shift, norm=None):
    ''' Compute the explicit cross-correlation between two arrays for a given
    integer shift and rotation.

    Returns
    =======
    cc : float
        The cross-correlation of a with im for shift (i, j)
    '''

    a_x, a_y = num.affine_transform(
        x, y,
        tools.transform_matrix([y_shift, x_shift, ang_shift], 'rotation'),
        center=tools.transform_center(x, y, 'raster'),
        )

    im = ref_raster_builder.get_raster(a_x, a_y, t, extrapolate_t=True)

    # keep only the parts where non-nan values overlap
    try:
        mask1 = a.mask | np.isnan(a.data)
    except AttributeError:
        mask1 = np.isnan(a)
    try:
        mask2 = im.mask | np.isnan(im.data)
    except AttributeError:
        mask2 = np.isnan(im)
    mask = mask1 | mask2
    a = a[~mask]
    im = im[~mask]

    if norm is None:
        a, im, norm = tools.prep_for_cc(a, im)

    return np.sum(a * im) / norm

def compute_cc(arr, x, y, t, ref_raster_builder,
        x_set=None, y_set=None, a_set=None,
        cores=None):
    ''' Compute the cross-correlation with rotation of a 1D array and a 2D
    image using explicit multiplication in the real space.

    Parameters
    ==========
    arr : 1D ndarray
    img : 2D ndarray
    x, y, t : 1D ndarrays
        coordinates for the points of arr
    x_set, y_set, a_set : OffsetSet (default: None)
    cores : int or None (default: None)
        If not None, use multiprocessing to compute the steps using the
        specified number processes.
    '''

    nx = x_set.number
    ny = y_set.number
    na = a_set.number
    n_iter = nx * na * na

    cc_worker = functools.partial(
        cc_step, arr, x, y, t, ref_raster_builder)
    cc_iter = itertools.product(x_set.world, y_set.world, a_set.world)
    if cores is None:
        cc = itertools.starmap(cc_worker, cc_iter)
        cc = list(cc)
    else:
        p = mp.Pool(cores)
        try:
            chunksize, extra = divmod(n_iter, len(p._pool) * 2)
            if extra:
                chunksize += 1
            print('start', datetime.datetime.now())
            cc = p.starmap(cc_worker, cc_iter, chunksize=chunksize)
        finally:
            p.terminate()
    cc = np.array(cc)
    cc = cc.reshape(nx, ny, na)
    cc = cc.swapaxes(0, 1) # from (x, y, a) to (y, x, a)

    return cc

def track_slit(ref_raster_builder, arr, x, y, t, missing=np.nan,
        x_set=None, y_set=None, a_set=None,
        **kwargs):
    ''' Find the optimal position of a 1D array within a 2D image using
    compute_cc().

    Parameters
    ==========
    arr : 1D ndarray
    ref_raster_builder : SyntheticRasterBuilder
    x, y, t : 1D ndarrays
        the coordinates of arr.
    missing : float or None (default: None)
        The value of the pixels in the image that should be considered as
        'missing', and thus discarded before computing the cross correlation.
        If set to None, don't handle missing values.
        If your missing values are 'None', you’re out of luck.
    **kwargs : passed to compute_cc()

    Returns
    =======
    offset : ndarray
        An array containing the optimal (y, x, angle) offset between the input
        array and image
    cc : float or 3D array
        The full cross-correlation array.
    '''

    if missing is not None:
        if np.isnan(missing):
            mask1 = np.isnan(arr)
            mask2 = np.isnan(img)
        else:
            mask1 = (arr == missing)
            mask2 = (img == missing)

        arr = np.ma.array(arr, mask=mask1)
        img = np.ma.array(img, mask=mask2)

    if np.all(arr.mask):
        offset = np.zeros(3) * np.nan
        cc = np.zeros((y_set.number, x_set.number, a_set.number)) * np.nan
        return offset, cc

    cc = compute_cc(arr, x, y, t, ref_raster_builder,
        x_set=x_set, y_set=y_set, a_set=a_set, **kwargs)

    offset = num.get_max_location(cc)
    offset = tools.convert_offsets(offset, [y_set, x_set, a_set])

    return offset, cc

def track_raster(raster, x, y, t, ref_raster_builder,
        x_set=None, y_set=None, a_set=None,
        cores=None, mp_mode='track', **kwargs):
    '''
    mp_mode : 'track' or 'cc'
        Wether to parallelize track_slit() calls (ie over each slit position),
        or cc_step() calls (ie. over each point of the cross-correlation
        cube).
    '''
    cc = []
    offset = []

    track_cores, cc_cores = None, None
    if mp_mode == 'track':
        track_cores = cores
    elif mp_mode == 'cc':
        cc_cores = cores

    _, n_iter = raster.shape
    track_iter = zip(raster.T, x.T, y.T, t.T)
    track_worker = functools.partial(
        track_slit, ref_raster_builder,
        x_set=x_set, y_set=y_set, a_set=a_set,
        missing=None, cores=cc_cores,
        **kwargs)

    if track_cores is None:
        res = itertools.starmap(track_worker, track_iter)
        res = list(res)
    else:
        p = mp.Pool(track_cores)
        try:
            chunksize, extra = divmod(n_iter, len(p._pool) * 2)
            if extra:
                chunksize += 1
            res = p.starmap(track_worker, track_iter, chunksize=chunksize)
        finally:
            p.terminate()

    offset = [r[0] for r in res]
    cc = [r[1] for r in res]
    cc = np.array(cc)
    offset = np.array(offset)

    return offset, cc

def align(raster, x, y, t, ref_raster_builder,
        x_set=None, y_set=None, a_set=None,
        cores=None, mp_mode='track'):
    ''' Align raster individual slit positions using a reference image '''

    # explore raster for all slit positions, with rotation
    raster = np.ma.array(raster, mask=np.isnan(raster))
    offset, cc = track_raster(
        raster, x, y, t, ref_raster_builder,
        x_set=x_set, y_set=y_set, a_set=a_set,
        cores=cores)

    # The additionnal shift in world units wrt the ref image, as determined by
    # track()
    offset_xy = offset[:, 1::-1]
    offset_a = offset[:, 2]

    # fill nan offsets with zeros
    offset_xy = np.ma.array(offset_xy, mask=np.isnan(offset_xy)).filled(0)
    offset_a = np.ma.array(offset_a, mask=np.isnan(offset_a)).filled(0)

    ny, nx = x.shape
    # center of each slit (n_slit_pos, 2):
    xy0 = np.stack((x, y))[:, ny//2].T
    # new center of each slit (n_slit_pos, 2):
    new_xy0 = xy0 + offset_xy

    # Transformation matrices - shape (nx, 3, 3)
    # - start with identity, repeat it nx times, and reshape to (3, 3, nx)
    # - move last axis to the beginning to get shape (nx, 3, 3)
    # - set values using slices
    # Translation matrix of -x0, -y0, for each slit position
    translation_xy0 = np.repeat(np.identity(3), nx).reshape(3, 3, nx)
    translation_xy0 = np.moveaxis(translation_xy0, -1, 0)
    translation_xy0[:, :2, 2] = - xy0
    # Translation matrix of new_x0, new_y0, for each slit position
    translation_new_xy0 = np.repeat(np.identity(3), nx).reshape(3, 3, nx)
    translation_new_xy0 = np.moveaxis(translation_new_xy0, -1, 0)
    translation_new_xy0[:, :2, 2] = new_xy0
    # Rotation matrix of offset_a for each slit position
    rotation_a = np.repeat(np.identity(3), nx).reshape(3, 3, nx)
    rotation_a = np.moveaxis(rotation_a, -1, 0)
    ca = np.cos(np.deg2rad(offset_a))
    sa = np.sin(np.deg2rad(offset_a))
    rotation_a[:, 0, 0] = ca
    rotation_a[:, 0, 1] = -sa
    rotation_a[:, 1, 1] = ca
    rotation_a[:, 1, 0] = sa

    # transform_matrix = translation_new_xy0 @ rotation_a @ translation_xy0
    transform_matrix = np.matmul(rotation_a, translation_xy0)
    transform_matrix = np.matmul(translation_new_xy0, transform_matrix)

    # apply transformation to
    xy = np.stack((x, y, np.ones_like(x))) # (3, ny, nx)
    xy = np.moveaxis(xy, 0, -1) # (ny, nx, 3)
    xy = xy.reshape(ny, nx, 3, 1) # (ny, nx, 3, 1)
    new_xy = np.matmul(transform_matrix, xy) # (ny, nx, 3, 1)
    new_xy = new_xy.reshape(ny, nx, 3) # (ny, nx, 3)
    new_x = new_xy[:, :, 0] # (ny, nx)
    new_y = new_xy[:, :, 1] # (ny, nx)

    return new_x, new_y, [offset, cc]
