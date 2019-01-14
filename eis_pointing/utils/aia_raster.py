#!/usr/bin/env python3

import itertools
import multiprocessing as mp
import os
import warnings

from astropy.io import fits
import numpy as np
import scipy.interpolate as si
import scipy.ndimage as ndimage

from . import aia
from . import num

class GenericCache(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.is_empty = True

    def update(self):
        self.is_empty = False

class SimpleCache(GenericCache):
    def clear(self):
        self.data = None
        super().clear()

    def update(self, data):
        self.data = data
        super().update()

    def get(self):
        return self.data

class FileCache(GenericCache):
    def __init__(self, path):
        self.path = path
        if self.path and os.path.exists(self.path):
            self.is_empty = False
        else:
            self.is_empty = True

    def clear(self):
        if self.path and os.path.exists(self.path):
            os.remove(self.path)
        super().clear()

    def update(self, data):
        np.save(self.path, data)
        super().update()

    def get(self):
        return np.load(self.path)

    def __bool__(self):
        return not (self.path is None)

class SyntheticRasterBuilder(object):
    ''' Class to build synthetic AIA rasters. Data are retrieved from Medoc
    using SITools, and cached when necessary. '''

    def __init__(self, file_cache=None, single_frame=False, **kwargs):
        ''' Create a new synthetic raster builder.

        Parameters
        ==========
        file_cache : str or None (default: None)
            Path to the file. If None, don't use a file cache.
        single_frame : bool or datetime (default: False)
            Wheter to always data from the same AIA image.
            If True, use the image at the center of the query results.
            If datetime, use the image which is the closest to that date.

        **kwargs can contain different sets of parameters, which trigger the
        use of different init functions:
        - dates, date_ref, and channel : use of _init_from_dates()
        - qr, qr_meta, qr_coord : use of _init_from_qr()

        See the docscrings of these functions for details on the parameters.
        '''
        self.cache = SimpleCache()
        self.file_cache = FileCache(file_cache)
        if {'dates', 'date_ref', 'channel'} == kwargs.keys():
            self._init_from_dates(
                kwargs['dates'], kwargs['date_ref'], kwargs['channel'])
        elif {'qr', 'qr_meta', 'qr_coord'} == kwargs.keys():
            self._init_from_qr(
                kwargs['qr'], kwargs['qr_meta'], kwargs['qr_coord'])
        else:
            raise ValueError('Invalid **kwargs')

        self.single_frame = single_frame
        if self.single_frame:
            date_obs = self.qr_meta['date__obs']
            if self.single_frame is True:
                frame_id = len(date_obs) // 2
            else:
                frame_id = np.argmin(np.abs(date_obs - self.single_frame))
            self.qr = self.qr[[frame_id]]
            self.qr_meta = {k: v[[frame_id]] for k, v in self.qr_meta.items()}
            self.qr_coord = self.qr_coord[[frame_id]]

    def _init_from_dates(self, dates, date_ref, channel):
        ''' Create a new synthetic raster builder for a given time window.

        Parameters
        ==========
        dates : tuple of datetimes
            A tuple containing the start and end dates of the search interval.
        date_ref : datetime
            The reference date from which relative times (in hour) are
            expressed.
        channel : str
            The AIA channel to be used.
        '''
        self.qr, self.qr_meta = aia.query_aia_data(dates, channel)
        self.qr_coord = aia.aia_cube_coords_from_metadata(
            self.qr_meta, date_ref)

    def _init_from_qr(self, qr, qr_meta, qr_coord):
        ''' Create a new synthetic raster builder using existing query results.

        This method is mostly to be used internally, as qr, qr_meta, and
        qr_coord are usually defined in _init_from_dates().

        Parameters
        ==========
        qr : array
        qr_meta : dict of arrays
        qr_coord : aia.AIACubeCoords
        '''
        self.qr = qr
        self.qr_meta = qr_meta
        self.qr_coord = qr_coord

    def filter_qr(self, filter_t):
        ''' Filter query results that correspond to a series of timestamps.

        Parameters
        ==========
        filter_t : ndarray
            Times for which to filter the search results, expressed in hours
            relative to self.qr_coord.date_ref. Array can have any number of dimensions.
        '''
        # filter_aia_synthetic(all_aia_frames, aia_coordinates, dates_rel_hours) -----
        # cast to float32 (precision of 1e-6, ie 3.6 ms) to save memory
        cast_dtype = np.float32
        qr_t = self.qr_coord.t_rel_hours.astype(cast_dtype)
        filter_t = filter_t.flatten().astype(cast_dtype)

        frames = set()
        for t in filter_t:
            frames.add(np.argmin(np.abs(qr_t - t)))
        frames = tuple(sorted(frames))

        filtered_qr = self.qr[frames]
        filtered_meta = {k: v[frames] for k, v in self.qr_meta.items()}
        filtered_coord = self.qr_coord[frames]

        return SyntheticRasterBuilder(
            qr=filtered_qr, qr_meta=filtered_meta, qr_coord=filtered_coord)

    def _download_data(self, rotate, update_cache, qr=None):
        ''' Download data from the query list. Used by self.get_data(). '''
        if qr is None:
            qr = self.qr
        else:
            if rotate is 'metadata':
                raise ValueError("cannot set rotate='metadata' with custom qr")
        data = []
        for i, aia_frame in enumerate(qr):
            progress = (i + 1) / len(qr)
            print('Opening AIA frames: {:.1%}'.format(progress), end='\r')
            fits_path = aia.get_fits(aia_frame)
            primary, img = fits.open(fits_path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                img.verify('fix') # required to avoid VerifyError
            if rotate is not False:
                if rotate is 'fits':
                    angle = img.header['crota2']
                elif rotate is 'metadata':
                    angle = self.qr_meta['crota2'][i]
                else:
                    angle = rotate[i]
                img.data = ndimage.rotate(
                    img.data, - angle,
                    reshape=False, order=1, cval=np.nan)
            data.append(img.data)
        data = np.array(data)
        if update_cache:
            self.cache.update(data)
            if self.file_cache:
                self.file_cache.update(data)
        return data

    def degrade_resolution(self, fwhm, cores=None):
        ''' Degrade the resolution of the data by convolving them with a
        gaussian.

        WARNING: This modifies the cached data. Updating the cache will
        overwrite any changes made by this method.

        Parameters
        ==========
        fwhm : float or 2-tuple of floats
            The FWHM of the gaussian used to degrade the resolution. Values are
            in the same unit as `self.qr_coord.x` and `self.qr_coord.y`. If it
            is a scalar, use the same value for both axes; if it is a 2-tuple,
            use the values for axes (y, x).
        '''

        x = self.qr_coord.x
        y = self.qr_coord.y
        dx = num.almost_identical(x[:, 1:] - x[:, :-1], 1e-4)
        dy = num.almost_identical(y[:, 1:] - y[:, :-1], 1e-4)
        px_size = np.array([dy, dx])
        try:
            fwhm_y, fwhm_x = fwhm
        except TypeError:
            fwhm_y, fwhm_x = fwhm, fwhm
        fwhm = np.array([fwhm_y, fwhm_x])
        sigma = fwhm / (px_size * 2 * np.sqrt(2 * np.log(2)))

        data = self.cache.get()

        worker = ndimage.filters.gaussian_filter
        iterator = zip(data, itertools.repeat(sigma))
        if cores is None:
            data = itertools.starmap(worker, iterator)
            data = list(data)
        else:
            p = mp.Pool(cores)
            try:
                data = p.starmap(worker, iterator)
            finally:
                p.terminate()
        data = np.array(data)

        self.cache.update(data)

    def crop_data(self, xmin, xmax, ymin, ymax):
        ''' Only keep a spatial subset of the cached data.

        WARNING: This modifies the cached data AND and changes
        to other attributes of the class so that they match the cached data.
        THINGS WON'T GO WELL if you modify the cache after calling this method.
        '''

        data_x = self.qr_coord.x
        data_y = self.qr_coord.y
        ax = np.where((xmin <= data_x) & (data_x <= xmax))[1]
        ay = np.where((ymin <= data_y) & (data_y <= ymax))[1]
        if ax.size == 0 or ay.size == 0:
            print("AIA and EIS fields of view don't intersect.")
            # try to determine why
            eis_xcen = (xmax + xmin) / 2
            eis_ycen = (ymax + ymin) / 2
            eis_r = np.sqrt(eis_xcen**2 + eis_ycen**2)
            if eis_r > 1230:
                print('This is probably because of faulty EIS coordinates:')
                print('EIS FOV X: ({:.2f}, {:.2f}) arcsec'.format(xmin, xmax))
                print('EIS FOV Y: ({:.2f}, {:.2f}) arcsec'.format(ymin, ymax))
            else:
                print('This is probably because of bad AIA pointing:')
                for i, (x, y) in enumerate(zip(data_x, data_y)):
                    aia_xcen = x[len(x)//2]
                    aia_ycen = y[len(y)//2]
                    aia_r = np.sqrt(aia_xcen**2 + aia_ycen**2)
                    found_bad_aia_frame = False
                    if aia_r > 100:
                        found_bad_aia_frame = True
                        print('> Bad pointing at frame {}:'.format(i))
                        msg = '  AIA center ({:.2f}, {:.2f}) arcsec'
                        msg = msg.format(i, aia_xcen, aia_ycen)
                        print(msg)
                        print('  URL:', self.qr[i].url)
                    if not found_bad_aia_frame:
                        print('the bad frames cannot be identified.')
            raise ValueError("AIA and EIS fields of view don't intersect. "
                             "See above for more details.")
        ax = np.array(sorted(set(ax)))
        ay = np.array(sorted(set(ay)))
        assert np.all(ax[1:] - ax[:-1] == 1)
        assert np.all(ay[1:] - ay[:-1] == 1)
        self.qr_coord.x = self.qr_coord.x[:, ax]
        self.qr_coord.y = self.qr_coord.y[:, ay]
        self.cache.update(self.cache.get()[:, ay][:, :, ax])

    def get_data(self, rotate='metadata', use_cache=True, update_cache=None):
        ''' Get data from the query results.

        Parameters
        ==========
        rotate : bool or 1D array-like (default: False)
            Wheter to rotate the AIA images after opening them:
            - If False, don't do anything.
            - If 'fits', use the CROTA2 tag from the FITS header.
            - If 'metadata', use the CROTA2 tag from self.qr_meta.
            - If array-like, it must have the same length as self.qr and
              contain CROTA2 values, that are used instead of those in the FITS
              headers. This is useful when updated values are stored separately
              from the FITS files, as it is the case in Medoc.
        use_cache : bool (default: True)
            - If True, use data cached in `self.cache`. If it is empty, data
              are loaded from `self.file_cache`, or downloaded using
              `get_data()` if the file cache is empty..
            - If False, data are downloaded regardless of what's
              in the cache and the file cache.
            Note: by default, downloaded data are cached (ie. saved to
            `self.cache` and `self.file_cache`) if `use_cache` is True. If
            `use_cache` is False, the cache is not modified. This behaviour can
            be overridden by using `update_cache`, which is passed to
            `get_data()`.
        update_cache : bool or None (default: None)
            If True, update the cache and the file cache when downloading data.
            If False, don't update the caches.
            If None, use the same value as use_cache.
        '''
        if update_cache is None:
            update_cache = use_cache
        if use_cache:
            if self.cache.is_empty:
                if not self.file_cache.is_empty:
                    data = self.file_cache.get()
                    if update_cache:
                        self.cache.update(data)
                    return data
                else:
                    return self._download_data(rotate, update_cache)
            else:
                return self.cache.get()
        else:
            return self._download_data(rotate, update_cache)

    def get_raster(self, raster_x, raster_y, raster_t,
            extrapolate_t=False, **kwargs):
        ''' Get a synthetic raster

        Parameters
        ==========
        raster_x, raster_y, raster_t : np.ndarray of float
            2D arrays containing the coordinates for the pixels of the desired
            raster. Times are in hours relative to coord.date_ref.
        extrapolate_t : bool (default: False)
            If True, use a nearest neighbours interpolation to retrieve data
            for values of t that are out of the bounds.
        **kwargs : passed to get_data()
        '''

        try:
            cube = self.get_data(**kwargs) # (nt, ny, nx)
            cube_x = num.almost_identical(self.qr_coord.x, 1e-10, axis=0)
            cube_y = num.almost_identical(self.qr_coord.y, 1e-10, axis=0)
            cube_t = self.qr_coord.t_rel_hours
            regular_grid = True
            # all AIA images have the same x and y coordinates
        except ValueError:
            cube_x = self.qr_coord.x
            cube_y = self.qr_coord.y
            cube_t = self.qr_coord.t_rel_hours
            regular_grid = False
            # AIA images have different x and y coordinates

        if extrapolate_t:
            raster_t = np.clip(raster_t, cube_t.min(), cube_t.max())

        if regular_grid:
            # perform a linear interpolation in both x, y, and t

            if self.single_frame:
                # For some unknown reason, raster is filled with nan when there
                # is a single frame in the builder. Fix this with a 2D interp.
                raster = si.interpn(
                    (cube_x, cube_y),
                    cube.T[:, :, 0],
                    (raster_x.flat, raster_y.flat),
                    bounds_error=False)

            else:
                raster = si.interpn(
                    (cube_x, cube_y, cube_t),
                    cube.T,
                    (raster_x.flat, raster_y.flat, raster_t.flat),
                    bounds_error=False)

            raster = raster.reshape(raster_x.shape)

        else:
            # perform a nearest interpolation in t, and a linear one in x and y
            #
            # That's slower and a bit less accurate than the method used when
            # regular_grid is True, but still much faster than if we used
            # scipy.interpolate.griddata. In practice, the two different
            # methods (linear in t x and y, or nearest in t and linear in x and
            # y) give results so close that it's difficult to see a difference
            # between the resulting images.
            raster_ndim = raster_t.ndim
            if raster_ndim == 1:
                raster_t = raster_t.reshape(1, -1)
                raster_x = raster_x.reshape(1, -1)
                raster_y = raster_y.reshape(1, -1)
            raster_t = num.almost_identical(raster_t, 1e-10, axis=0)
            raster = []
            for r_i, r_t in enumerate(raster_t):
                # nearest interpolation in t
                c_i = np.argmin(np.abs(cube_t - r_t))
                # linear interpolation within the image
                raster_slice = si.interpn(
                    (cube_x[c_i], cube_y[c_i]),
                    cube[c_i].T,
                    (raster_x[:, r_i].flat, raster_y[:, r_i].flat),
                    bounds_error=False)
                raster_slice = raster_slice.reshape(raster_x[:, r_i].shape)
                raster.append(raster_slice)
            raster = np.array(raster).T
            if raster_ndim == 1:
                raster = raster.reshape(-1)

        return raster

    def get_rasters(self, rasters_x, rasters_y, rasters_t, **kwargs):
        ''' Get a cube of synthetic rasters

        Parameters
        ==========
        rasters_x, rasters_y, rasters_t : np.ndarray of float
            3D arrays containing the coordinates for the pixels of the desired
            rasters. Times are in hours relative to coord.date_ref.
        **kwargs : passed to get_raster(). See in particular use_cache and
            update_cache parameters.
        '''
        rasters = []
        for x, y, t in zip(rasters_x, rasters_y, rasters_t):
            rasters.append(self.get_raster(x, y, t, **kwargs))
        return np.array(rasters)

    def get_frame(self, t, x=None, y=None):
        ''' Get the closest frame found for a given time.

        Parameters
        ==========
        t : datetime.datetime
            The time for which to return a frame
        x, y : array or None (default: None)
            If not None, interpolate the frame so that new points lie at the
            coordinates specified by these arrays. They can have 1 or 2
            dimensions. If they have 1 dimensions, use them as the x and y
            coordinates of a 2D grid. If they have 2 dimensions, interpolate
            the frame at all `(x[i, j], y[i, j])` positions.

        This function bypasses any cache.
        '''

        frame_in_qr = np.argmin(
            np.abs(self.qr_coord.t_rel_hours - t))
        data = self._download_data(
            [self.qr_meta['crota2'][frame_in_qr]],
            False,
            qr=[self.qr[frame_in_qr]])
        data = data[0]

        if (x is not None) and (y is not None):
            data_y = self.qr_coord[frame_in_qr].y
            data_x = self.qr_coord[frame_in_qr].x
            y_min = np.min(y)
            y_max = np.max(y)
            x_min = np.min(x)
            x_max = np.max(x)
            di = np.where((y_min <= data_y) & (data_y <= y_max))[0]
            dj = np.where((x_min <= data_x) & (data_x <= x_max))[0]
            msg = 'Something went wrong with frame pointing at raster # {t}.'
            assert (len(di) > 0) and (len(dj) > 0), msg.format(**locals())
            di_min = np.min(di) - 1
            di_max = np.max(di) + 1
            dj_min = np.min(dj) - 1
            dj_max = np.max(dj) + 1
            data_cut = data[di_min:di_max + 1, dj_min:dj_max + 1]
            data_y_cut = data_y[di_min:di_max + 1]
            data_x_cut = data_x[dj_min:dj_max + 1]
            # new resolution grid for data
            data_y_cut_new = np.array(y)
            data_x_cut_new = np.array(x)

            # downscale cube data to the frame resolution
            # use faster interpolation if (x, y) form a grid
            try:
                threshold = 1e-3
                x_ = num.almost_identical(x, threshold, axis=0)
                y_ = num.almost_identical(y, threshold, axis=1)
                x, y = x_, y_
            except ValueError:
                pass
            if x.ndim == 1 and y.ndim == 1:
                print('interp2d')
                data_interp = si.interp2d(
                    data_x_cut, data_y_cut,
                    data_cut,
                    )
                data = np.array(data_interp(x, y))
            elif x.ndim == 2 and y.ndim == 2:
                print('griddata')
                data = num.friendly_griddata(
                    np.meshgrid(data_x_cut, data_y_cut),
                    data_cut,
                    (x, y))
            else:
                raise ValueError('Bad number of dimensions.')

        return data
