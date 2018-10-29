#!/usr/bin/env python3

from functools import reduce
import datetime
import dateutil.parser
import operator
import os
import re
import shutil
import warnings

from astropy.io import fits
import numpy as np
import requests

from . import num

class EISPointing(object):
    def __init__(self, x, y, t, t_ref, wvl=None):
        ''' Coordinates for an EIS raster or map

        Parameters
        ==========
        x, y, t : ndarrays
            arrays with 2 or 3 dimensions, depending on wheter these are
            coordinates for a map or a raster. All arrays have the same shape.
            - x and y contain absolute coordinates
            - t contains coordinates relative to `t_ref`
        t_ref : datetime.datetime
        wvl : 3D ndarray or None (default: None)
            If an array, it has the same shape as x, y, and t.
        '''
        # verify that all arrays have the same shape
        ref_shape = x.shape
        for shape in (x.shape, y.shape, t.shape):
            if shape != ref_shape:
                raise ValueError('inconsistent shapes')
        if (wvl is not None) and (wvl.shape != ref_shape):
            raise ValueError('inconsistent shapes')
        if (wvl is None) and (len(ref_shape) == 3):
            raise ValueError('received 3D coordinates, but wvl is None')
        self.shape = ref_shape
        self.x = x
        self.y = y
        self.t = t
        self.t_ref = t_ref
        self.wvl = wvl

    def from_windata(windata, use_wvl=True):
        ''' Initialize from a windata object.

        Parameters
        ==========
        windata : idl.IDLStructure
            The windata object containing the pointing.
        use_wvl : bool (default: True)
            Wheter to represent the wvl dimension.
        '''

        # find missing slit times to interpolate their coordinates
        bad_times = (windata.exposure_time == 0)
        if np.any(bad_times):
            msg = 'Interpolated {} missing slit times.'.format(bad_times.sum())
            warnings.warn(msg)

        t_ref = dateutil.parser.parse(windata.hdr['date_obs'][0])
        x = num.replace_missing_values(windata.solar_x, bad_times)
        y = windata.solar_y
        t = num.replace_missing_values(windata.time, bad_times)

        if use_wvl:
            wave_corr = windata.wave_corr
            wave_corr = wave_corr.reshape(*wave_corr.shape, 1)
            wvl = windata.wvl.reshape(1, 1, -1) - wave_corr

            # repeat arrays to form grids
            ny, nx, nw = wvl.shape
            t = np.repeat(t, ny*nw).reshape(nx, ny, nw)
            x = np.repeat(x, ny*nw).reshape(nx, ny, nw)
            y = np.repeat(y, nx*nw).reshape(ny, nx, nw)
            t = np.swapaxes(t, 0, 1)
            x = np.swapaxes(x, 0, 1)

        else:
            wvl = None
            ny, nx = windata.solar_y.size, windata.solar_x.size
            t = np.repeat(t, ny).reshape(nx, ny)
            x = np.repeat(x, ny).reshape(nx, ny)
            y = np.repeat(y, nx).reshape(ny, nx)
            t = np.swapaxes(t, 0, 1)
            x = np.swapaxes(x, 0, 1)

        return EISPointing(x, y, t, t_ref, wvl=wvl)

    def to_bintable(self):
        ''' Create a FITS BinTableHDU containing all the pointing data. '''

        arrays = {
            'x': self.x,
            'y': self.y,
            't': self.t,
            }
        if self.wvl is not None:
            arrays['wvl'] = self.wvl
        columns = []
        for title, array in arrays.items():
            column_shape = array.shape[1:][::-1]
            column = fits.Column(
                name=title,
                array=array,
                format='{:d}D'.format(reduce(operator.mul, column_shape)),
                dim=str(column_shape),
                )
            columns.append(column)
        tbhdu = fits.BinTableHDU.from_columns(columns)

        tbhdu.header.append(('t_ref', str(self.t_ref)))

        return tbhdu

    def from_bintable(hdu):
        ''' Restore from a FITS BinTableHDU. '''

        x = hdu.data.x
        y = hdu.data.y
        t = hdu.data.t
        t_ref = dateutil.parser.parse(hdu.header['t_ref'])
        try:
            wvl = hdu.data.wvl
        except AttributeError:
            wvl = None

        return EISPointing(x, y, t, t_ref, wvl=wvl)

    def to_hdulist(self):
        return fits.HDUList([fits.PrimaryHDU(), self.to_bintable()])

    def from_hdulist(hdulist):
        return EISPointing.from_bintable(hdulist[1])


class EISData(object):
    def __init__(self, data, pointing):
        if data.shape != pointing.shape:
            raise ValueError('inconsistent shapes')
        self.data = data
        self.pointing = pointing

    def to_hdulist(self):
        data_hdu = fits.PrimaryHDU(self.data)
        pointing_bintable = self.pointing.to_bintable()
        hdulist = fits.HDUList([data_hdu, pointing_bintable])
        return hdulist

    def from_hdulist(hdulist):
        data_hdu, pointing_bintable = hdulist
        data = data_hdu.data
        pointing = EISPointing.from_bintable(pointing_bintable)
        return EISData(data, pointing)


class ReFiles():
    ''' Regex patterns to be combined and compiled at instanciation '''
    patterns = {
        'mssl data': 'http://solar.ads.rl.ac.uk/MSSL-data/',
        'path/name': ('eis/level(?P<lev>\d)/(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})/'
                      'eis_(?P<lev_str>[le][0-9e])_(?P=year)(?P=month)(?P=day)_(?P<time>\d{6})(\.fits\.gz)?'),
        'name': 'eis_(?P<lev_str>[le][0-9r])_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_(?P<time>\d{6})(\.fits\.gz)?',
        }

    def __init__(self):
        ''' Regex for matching the URL of the FITS returned after a SQL query. '''
        mssl_fits_url = '{mssl data}{path/name}'.format(**self.patterns)
        self.mssl_fits_url = re.compile(mssl_fits_url)
        ''' Regex for matching the path/filename of a FITS '''
        fits_path = '{path/name}'.format(**self.patterns)
        self.fits_path = re.compile(fits_path)
        ''' Regex for matching the filename of a FITS '''
        fits_name = '{name}'.format(**self.patterns)
        self.fits_name = re.compile(fits_name)

# compile regexes at import
re_files = ReFiles()

def fits_path(eis_file, absolute=False, url=False, gz=False):
    ''' Determine the path of the FITS file (in the Hinode data directory) from
    a `prop_dict`.

    Parameters
    ==========
    eis_file : str or dict.
        The EIS filename (eg 'eis_l0_20110803_113520'), path, URL in the MSSL
        archives, or 'prop dict'.
    absolute : bool
        If true, return the absolute path, reading environment variable
        $HINODE_DATA, with fallback to $SOLARDATA/hinode.
    url : bool
        If true, return the URL to the MSSL archives instead of the path.
    gz : bool (default: False)
        If true, use the .fits.gz extension. Else, simply return .gz.
    '''
    p = ('eis/level{lev}/{year}/{month}/{day}/'
         'eis_{lev_str}_{year}{month}{day}_{time}.fits')
    if gz:
        p += '.gz'
    if url:
        p = ReFiles.patterns['mssl data'] + p
    if absolute:
        try:
            hinode_data = os.environ['HINODE_DATA']
        except KeyError:
            try:
                hinode_data = os.path.join(os.environ['SOLARDATA'], 'hinode')
            except KeyError:
                raise ValueError('Could not find $HINODE_DATA nor $SOLARDATA')
        p = os.path.join(hinode_data, p)
    prop_dict = prop_from_filename(eis_file)
    return p.format(**prop_dict)

def prop_from_filename(filename):
    ''' Parse an EIS file name, path or URL to get a 'prop_dict'.

    If passed a dict, return it if it's a 'prop_dict', raise ValueError if not.
    '''

    # check if filename could be a prop_dict
    if type(filename) is dict:
        d = filename
        prop_dict_keys = {'lev', 'lev_str', 'year', 'month', 'day', 'time'}
        if prop_dict_keys.issubset(set(d.keys())):
            # d contains at least all prop_dict keys
            return d
        else:
            raise ValueError('Got a dict that does not look like a prop_dict.')

    # handle filename as... a filename!
    regex_to_try = (
        re_files.mssl_fits_url,
        re_files.fits_path,
        re_files.fits_name,
        )
    for r in regex_to_try:
        m = r.match(filename)
        if m:
            prop_dict = m.groupdict()
            if 'lev' not in prop_dict:
                if prop_dict['lev_str'] == 'er':
                    prop_dict['lev'] = '1'
                else: # assume lev_str is 'l\d'
                    prop_dict['lev'] = prop_dict['lev_str'][-1]
            return prop_dict
    msg = "Not a valid EIS file name/path/url format: {}".format(filename)
    raise ValueError(msg)

def get_aia_channel(line_wvl):
    ''' Get the best AIA channel for a given line.  '''
    if np.issubdtype(type(line_wvl), np.number):
        line_wvl = '{:.3f}'.format(line_wvl)
    wvl_conv = {
        '195.120': '193',
        '183.937': '131',
        '184.118': '131',
        '184.410': '193',
        '184.524': '211',
        '184.537': '193',
        '185.213': '171',
        '185.230': '335',
        '186.599': '171',
        '186.610': '94',
        '186.839': '211',
        '186.854': '193',
        '186.887': '193',
        '188.216': '193',
        '188.299': '193',
        '188.497': '171',
        '188.675': '211',
        '188.687': '171',
        '192.029': '193',
        '192.394': '193',
        '192.750': '131',
        '192.797': '131',
        '192.801': '131',
        '192.814': '193',
        '192.853': '94',
        '192.904': '131',
        '192.911': '131',
        '193.715': '193',
        '193.752': '211',
        '193.866': '94',
        '193.968': '171',
        '195.119': '193',
        '195.179': '193',
        '200.972': '94',
        '201.045': '131',
        '201.113': '193',
        '201.126': '211',
        '201.140': '193',
        '202.044': '211',
        '203.728': '193',
        '203.772': '211',
        '203.796': '211',
        '203.822': '131',
        '203.827': '211',
        '203.890': '131',
        '248.460': '131',
        '254.885': '94',
        '255.110': '131',
        '255.114': '193',
        '256.317': '304',
        '256.318': '304',
        '256.378': '193',
        '256.398': '193',
        '256.400': '193',
        '256.410': '193',
        '256.685': '335',
        '262.976': '94',
        '263.766': '193',
        '264.231': '193',
        '264.773': '193',
        '264.789': '211',
        '265.001': '94',
        '274.180': '171',
        '274.204': '211',
        '275.361': '171',
        '275.550': '131',
        '284.163': '335',
        }
    return wvl_conv[line_wvl]

def get_fits(eis_file, custom_dest=None, force_download=False, silent=True):
    ''' Get a given EIS FITS. If not found locally, download it from the MSSL
    SDC.

    Parameters
    ==========
    eis_file : str or dict.
        The EIS filename (eg 'eis_l0_20110803_113520'), path, URL in the MSSL
        archives, or 'prop dict'.
    custom_dest : str (default: None)
        If set, the location where to save the FITS. If not set, the FITS is
        saved to $SOLARDATA/hinode/eis/...
    force_download : bool (default: False)
        If True, always download the FITS, overwriting any local version.

    Notes
    =====
    Some regular FITS in the MSSL archives have the wrong extension, .fits.gz.
    This function tests if it is the case, and renames the file if needed.

    **Warning:** when using `force_download=True` to retrieve, eg.
    `foo.fits.gz`, any existing `foo.fits` might be overwritten.
    '''

    # determine fits url and save path
    eis_properties = prop_from_filename(eis_file)
    fits_url = fits_path(eis_properties, url=True, gz=True)
    if custom_dest:
        fits_save_path = custom_dest
    else:
        fits_save_path = os.path.join(
            os.environ.get('SOLARDATA'),
            'hinode',
            fits_path(eis_properties, gz=True))
    # determine if .fits.gz or .fits
    fits_base, fits_ext = os.path.splitext(fits_save_path)
    if fits_ext == '.gz':
        fits_save_path_unzip = fits_base
    else:
        fits_save_path_unzip = fits_save_path

    # download path
    if not (os.path.exists(fits_save_path) or
            os.path.exists(fits_save_path_unzip)) or force_download:
        if not silent:
            print('Downloading {} to {}'.format(fits_url, fits_save_path))
        response = requests.get(fits_url, stream=True, timeout=60)
        if not response.ok:
            m = 'Could not get {}'.format(fits_url)
            raise ValueError(m)
        fits_dir, fits_filename = os.path.split(fits_save_path)
        if not os.path.exists(fits_dir):
            os.makedirs(fits_dir)
        with open(fits_save_path, 'wb') as f:
            for block in response.iter_content(1024):
                f.write(block)
    try:
        # print('Trying {}'.format(fits_save_path))
        f = fits.open(fits_save_path)
    except IOError as e:
        if fits_ext == '.gz':
            # then the error is most likely due to the fact that the file is a
            # regular FITS with a .fits.gz extension, or has already been
            # deflated.
            if not os.path.exists(fits_save_path_unzip) or force_download:
                shutil.move(fits_save_path, fits_save_path_unzip)
            # print('Opening {}'.format(fits_save_path_unzip))
            f = fits.open(fits_save_path_unzip)
        else:
            raise e
    if not silent:
        print('Opened {}'.format(f.filename()))
    return f
