#!/usr/bin/env python3

import functools
import itertools
import multiprocessing as mp
import warnings

import scipy.signal as ss
import numpy as np

from ..utils import num

from . import tools

class cc2d:
    def _get_padding_slice(img):
        ''' Get the slice for padding imag in `dft(... boundary='fill')`.

        Parameters
        ==========
        img : ndarray
            The 2D image.

        Returns
        =======
        s : slice
            The slice of the new array where the old data should be inserted and
            retrieved.
        N : tuple
            The size of the new array.
        '''
        n = np.array(img.shape)
        N = 2**np.ceil(np.log2(n * 2))
        N = N.astype(int)
        im = np.zeros(N[0])
        nmin = N//2 - n//2 - n%2
        nmax = N//2 + n//2
        s = (slice(nmin[0], nmax[0]), slice(nmin[1], nmax[1]))
        return s, N

    def _pad_array(arr, s, N, pad_value=0):
        ''' Insert arr in a larger array of shape N at the position defined by s, a
        slice in the larger array. The area of the new array that don't contain
        data of arr are filled with pad_value.

        Parameters
        ==========
        arr : ndarray
            The array to insert in a larger array
        s : slice
            The slice of the larger array where the data from arr are to be
            inserted. This slice must have the same shape as arr.
        N : tuple
            The shape of the new larger array.
        pad_value : float
            The value used to fill the areas of the larger array that are outside
            of slice s.

        Return
        ======
        a : ndarray
            A larger array containing the values of arr at the positions defined by
            slice s.
        '''
        a = np.zeros(N) + pad_value
        a[s] = arr
        return a

    def _unpad_array(arr, s, roll=False):
        ''' Reverse the operation performed by `_pad_array`.

        Parameters
        ==========
        arr : ndarray
            The larger array containing the padded data.
        s : slice
            The slice of the larger array where the data from arr are to be
            inserted. This slice must have the same shape as arr.
        roll : bool (default: False)
            A roll of half the size of the array is required before using the data.
            If True, roll, retrieve the data, and roll back.
        '''
        if roll:
            arr = num.roll_nd(num.roll_nd(arr)[s])
        else:
            arr = arr[s]
        return arr

    def dft(img1, img2, boundary='wrap'):
        ''' Compute the cross-correlation of img1 and img2 using multiplication in
        the Fourier space.

        Parameters
        ==========
        img1, img2 : ndarray
        boundary : str
            How to handle the boundary conditions.
            - 'wrap': perform a dumb product in the Fourier space, resulting in
              wrapped boundary conditions.
            - 'fill': the data are inserted in an array of size 2**n filled with
              zeros, where n is chosen such that the size of the new array is at
              least twice as big as the size of the original array.
        '''

        a1, a2, norm = tools.prep_for_cc(img1, img2)

        if boundary == 'fill':
            s, N = cc2d._get_padding_slice(img1)
            a1 = cc2d._pad_array(a1, s, N, pad_value=0)
            a2 = cc2d._pad_array(a2, s, N, pad_value=0)

            cc = np.fft.ifft2(
                np.conj(np.fft.fft2(a2)) * \
                np.fft.fft2(a1)
                )

            cc = cc2d._unpad_array(cc, s, roll=True)

        elif boundary == 'wrap':
            cc = np.fft.ifft2(
                np.conj(np.fft.fft2(a2)) * \
                np.fft.fft2(a1)
                )

        else:
            msg = "unexpected value for 'boundary': {}".format(boundary)
            raise ValueError(msg)

        cc /= norm

        return cc.real

    def _explicit_step(a1, a2, i, j, norm=None):
        ''' Compute the explicit cross-correlation between two arrays for a given
        integer shift.

        Parameters
        ==========
        a1, a2 : ndarray, 2D
            Data values.
        i, j : int
            The shift between a1 and a2 for which to compute the cross-correlation.
        norm : float or None (default: None)
            The value by which to normalize the result.
            If None, subtract their respective averages from the shifted version of
            a1 and a2:
                I = s_a1 - avg(s_a1); J = s_a2 - avg(s_a2),
            and compute a local norm:
                norm = sqrt(sum(I²) × sum(J²)).
            This is used to implement boundary='drop' when computing an explicit
            DFT map.

        Returns
        =======
        cc : float
            The cross-correlation of a1 with a2 for shift (i, j)
        '''
        ni, nj = a1.shape
        s1 = (
            slice(max(i, 0), min(ni+i-1, ni-1) + 1),
            slice(max(j, 0), min(nj+j-1, nj-1) + 1)
            )
        s2 = (
            slice(max(-i, 0), min(ni-i-1, ni-1) + 1),
            slice(max(-j, 0), min(nj-j-1, nj-1) + 1)
            )
        a1 = a1[s1]
        a2 = a2[s2]

        try:
            mask1 = a1.mask
        except AttributeError:
            mask1 = np.zeros_like(a1, dtype=bool)
        try:
            mask2 = a2.mask
        except AttributeError:
            mask2 = np.zeros_like(a2, dtype=bool)
        mask = mask1 | mask2
        a1 = np.ma.array(a1, mask=mask)
        a2 = np.ma.array(a2, mask=mask)

        if norm is None:
            a1, a2, norm = tools.prep_for_cc(a1, a2)

        return np.sum(a1 * a2) / norm

    def explicit(img1, img2, simax=None, sjmax=None, boundary='fill', cores=None):
        ''' Compute the cross-correlation of img1 and img2 using explicit
        multiplication in the real space.

        Parameters
        ==========
        img1, img2 : ndarray
        simax, sjmax : int or None (default: None)
            The maximum shift on the 0 and 1 axes resp. for which to compute the
            cross-correlation.
            If None, return a cross-correlation map with the same size of the input
            images.
        boundary : 'fill' or 'drop' (default: 'fill')
            How to handle boundary conditions. 'fill' is equivalent to padding the
            images with zeros. With 'drop' the cross-correlation is computing using
            only the overlapping part of the images.
        cores : int or None (default: None)
            If not None, use multiprocessing to compute the steps using the
            specified number processes.
        '''
        ni, nj = img1.shape
        if simax is None:
            simin = - ni // 2
            simax = + ni // 2
        else:
            simin = - simax
        if sjmax is None:
            sjmin = - nj // 2
            sjmax = + nj // 2
        else:
            sjmin = - sjmax

        if boundary == 'fill':
            img1, img2, norm = tools.prep_for_cc(img1, img2)
        elif boundary == 'drop':
            norm = None
        else:
            msg = "unexpected value for 'boundary': {}".format(boundary)
            raise ValueError(msg)

        worker = functools.partial(cc2d._explicit_step, img1, img2, norm=norm)
        i_range = range(simin, simax)
        j_range = range(sjmin, sjmax)
        ni = len(i_range)
        nj = len(j_range)
        iterable = itertools.product(i_range, j_range)
        if cores is None:
            cc = itertools.starmap(worker, iterable)
            cc = list(cc)
        else:
            p = mp.Pool(cores)
            try:
                n_iter = ni * nj
                chunksize, extra = divmod(n_iter, len(p._pool))
                if extra:
                    chunksize += 1
                cc = p.starmap(worker, iterable, chunksize=chunksize)
            finally:
                p.terminate()
        cc = np.array(cc)
        cc = cc.reshape(ni, nj)
        cc = num.roll_nd(cc)

        return cc

    def scipy(img1, img2, boundary='fill'):
        ''' Compute the cross-correlation of img1 and img2 using
        scipy.signal.correlate2d.

        Parameters
        ==========
        img1, img2 : ndarray
        boundary : str (default: 'fill')
            Passed to scipy.signal.correlate2d
        '''

        a1, a2, norm = tools.prep_for_cc(img1, img2)

        cc = ss.correlate2d(
            a1,
            a2,
            mode='same',
            boundary=boundary,
            fillvalue=0,
            )

        cc /= norm
        shift_i = cc.shape[0] // 2 + 1
        shift_j = cc.shape[1] // 2 + 1
        cc = num.roll_nd(cc, shifts=(shift_i, shift_j))
        return cc

def track(img1, img2,
        sub_px=True, missing=None, cc_function='dft', cc_boundary='wrap',
        return_full_cc=False,
        **kwargs):
    ''' Return the shift between img1 and img2 by computing their cross
    correlation.

    Parameters
    ==========
    img1, img2 : 2D ndarrays
        The images to correlate
    sub_px : bool (default: True)
        Whether to determine shifts with sub-pixel accuracy. Set to False for
        faster, less accurate results.
    missing : float or None (default: None)
        The value of the pixels in the image that should be considered as
        'missing', and thus discarded before computing the cross correlation.
        If set to None, don't handle missing values.
        If your missing values are 'None', you’re out of luck.
    cc_function : str (default: 'dft')
        Name of the function to use to compute the cross-correlation between
        two frames. Accepted values are 'dft', 'explicit', and 'scipy'.
        Functions of cc2d that have the same name are used.
    cc_boundary : str or None (default: None)
        If not None, pass this value as the `boundary` keyword to cc_function.
        See cc_function documentation for accepted values.
    return_full_cc : bool (default: False)
        If True, return the full cross-correlation array.
        If False, only return the maximum cross-correlation.
    **kwargs :
        Passed to cc_function.

    Returns
    =======
    offset : ndarray
        An array containing the optimal (y, x) offset between the input images
    cc : float or 3D array
        Depending on the value of return_full_cc, either the value of the
        cross-correlation at the optimal offset, or the full cross-correlation
        array.
    '''

    assert img1.shape == img2.shape, 'Images must have the same shape.'
    ny, nx = img1.shape

    sy, sx = ny // 2, nx // 2

    if missing is not None:
        if np.isnan(missing):
            mask1 = np.isnan(img1)
            mask2 = np.isnan(img2)
        else:
            mask1 = (img1 == missing)
            mask2 = (img2 == missing)

        img1 = np.ma.array(img1, mask=mask1)
        img2 = np.ma.array(img2, mask=mask2)

    if cc_function != 'explicit':
        kwargs.pop('cores', None)

    cc_functions = {
        'dft': cc2d.dft,
        'explicit': cc2d.explicit,
        'scipy': cc2d.scipy,
        }
    cc_function = cc_functions[cc_function]
    if cc_boundary:
        cc = cc_function(img1, img2, boundary=cc_boundary, **kwargs)
    else:
        cc = cc_function(img1, img2, **kwargs)
    cc = num.roll_nd(cc, shifts=(sy, sx))

    maxcc = np.nanmax(cc)
    cy, cx = np.where(cc == maxcc)
    if not(len(cy) == 1 and len(cx) == 1):
        m = 'Could not find a unique cross correlation maximum.'
        warnings.warn(m, RuntimeWarning)
    cy = cy[0]
    cx = cx[0]

    offset = np.zeros((2))

    if (not sub_px) or (maxcc == 0):
        offset[0] = cy
        offset[1] = cx

    else:

        # parabolic interpolation about minimum:
        yi = [cy-1, cy, (cy+1) % ny]
        xi = [cx-1, cx, (cx+1) % nx]
        ccy2 = cc[yi, cx]**2
        ccx2 = cc[cy, xi]**2

        yn = ccy2[2] - ccy2[1]
        yd = ccy2[0] - 2 * ccy2[1] + ccy2[2]
        xn = ccx2[2] - ccx2[1]
        xd = ccx2[0] - 2 * ccx2[1] + ccx2[2]

        if yd != 0 and not np.isnan(yd):
            offset[0] = yi[2] - yn / yd - 0.5
        else:
            offset[0] = float(cy)

        if xd != 0 and not np.isnan(xd):
            offset[1] = xi[2] - xn / xd - 0.5
        else:
            offset[1] = float(cx)

    offset[0] = sy - offset[0]
    offset[1] = sx - offset[1]

    if return_full_cc:
        return offset, cc
    else:
        return offset, maxcc

def align(cube, x, y, ref_cube, ref_x, ref_y, cores=None,
        cc_function='explicit', cc_boundary='drop', sub_px=True):
    ''' 2D version of align.align_cubes '''

    if cube.shape != ref_cube.shape:
        raise ValueError('cube and ref_cube must have the same shape')

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

    (y_offset, x_offset), maxcc = track(
        ref_cube, cube,
        sub_px=sub_px, missing=np.nan,
        cc_function=cc_function, cc_boundary=cc_boundary, cores=cores,
        )
    y_offset *= cdelty
    x_offset *= cdeltx
    offset = y_offset, x_offset, maxcc

    # Compute and apply the absolute offset for cube
    new_y = y - y_offset
    new_x = x - x_offset

    return new_x, new_y, offset
