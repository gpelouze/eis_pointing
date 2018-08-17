#!/usr/bin/env python3

import warnings

import align_images
import numpy as np

from ..utils import num

def align(cube, x, y, ref_cube, ref_x, ref_y, cores=None,
        return_offset=False):
    ''' 2D version of align.align_cubes '''

    # get grid info
    ny, nx = cube.shape
    dy = y[1:, :] - y[:-1, :]
    dx = x[:, 1:] - x[:, :-1]
    cdelty = num.almost_identical(dy, 1e-4)
    try:
        cdeltx = num.almost_identical(dx, 0.01)
    except ValueError as e:
        warnings.warn('{} Falling back to the median'.format(e))
        cdeltx = np.median(dx)

    if cube.shape != ref_cube.shape:
        nt, ny, nx= cube.shape
        nt_ref, ny_ref, nx_ref = ref_cube.shape
        cube_new = np.ones((nt_ref, ny_ref, nx_ref)) * np.nan
        y0 = (ny_ref - ny) // 2
        x0 = (nx_ref - nx) // 2
        cube_new[:nt, y0:y0+ny, x0:x0+nx] = cube
        cube = cube_new
    else:
        y0, x0 = 0, 0

    (y_offset, x_offset), maxcc = align_images.align.track(
        ref_cube, cube,
        sub_px=True, missing=np.nan,
        cc_function='explicit', cc_boundary='drop', cores=cores,
        )
    y_offset *= cdelty
    x_offset *= cdeltx
    offset = y_offset, x_offset, maxcc

    # The original shift wrt the reference cube
    original_offset_y = ref_y[y0, x0] - y[0, 0]
    original_offset_x = ref_x[y0, x0] - x[0, 0]
    original_offset_y_rep = np.repeat(original_offset_y, ny*nx).reshape(ny, nx)
    original_offset_x_rep = np.repeat(original_offset_x, ny*nx).reshape(ny, nx)

    # The additionnal offset in px wrt the ref cube, as determined by track()
    offset_y_rep = np.repeat(offset[0], ny*nx).reshape(ny, nx)
    offset_x_rep = np.repeat(offset[1], ny*nx).reshape(ny, nx)

    # Compute and apply the absolute offset for cube
    offset_y = original_offset_y - offset_y_rep
    offset_x = original_offset_x - offset_x_rep
    new_y = y + offset_y
    new_x = x + offset_x

    if return_offset:
        return new_x, new_y, offset
    else:
        return new_x, new_y
