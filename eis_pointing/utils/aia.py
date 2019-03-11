#!/usr/bin/env python3

from itertools import compress
import datetime
import os
import re

from astropy import time
import numpy as np
import sitools2.clients.sdo_client_medoc as md

from . import num

verb = False

class AIACubeCoords(object):
    ''' Represent coordinates for a cube of AIA data '''

    def __init__(self, x, y, t_rel_hours, date_ref, rot):
        self.x = x
        self.y = y
        self.t_rel_hours = t_rel_hours
        self.date_ref = date_ref
        self.rot = rot

    def __getitem__(self, s):
        ''' Return a new object where x, y and t_rel_hours are cut to s. '''
        return AIACubeCoords(
            self.x[s], self.y[s], self.t_rel_hours[s], self.date_ref,
            self.rot[s])

def aia_cube_coords_from_metadata(metadata, date_ref):
    ''' Build a AIACubeCoords object

    Parameters
    ==========
    metadata : dict
        A dict of metadata, as returned by get_aia_cube().
    date_ref : datetime.datetime
        If set, add a dates_rel and a dates_rel_hours attributes.

    Returns
    =======
    coordinates : AIACubeCoords
    '''

    # extract WCS data from metadata
    crval1 = np.array(metadata['crval1'])
    crval2 = np.array(metadata['crval2'])
    cdelt1 = np.array(metadata['cdelt1'])
    cdelt2 = np.array(metadata['cdelt2'])
    crpix1 = np.array(metadata['crpix1'])
    crpix2 = np.array(metadata['crpix2'])
    crota2 = np.array(metadata['crota2'])

    # build 1D x and y arrays
    sample_key = list(metadata.keys())[0]
    nt = len(metadata[sample_key])
    ny, nx = 4096, 4096
    i = np.arange(nx).repeat(nt).reshape(nx, nt) + 1
    j = np.arange(ny).repeat(nt).reshape(ny, nt) + 1
    x = cdelt1 * (i - crpix1)
    y = cdelt2 * (j - crpix2)
    x += crval1
    y += crval2

    # build t grid
    t = np.array(metadata['date__obs'])
    t_rel = t - date_ref
    t_rel_hours = num.total_seconds(t_rel) / 3600

    return AIACubeCoords(x.T, y.T, t_rel_hours, date_ref, crota2)

def sdotime_to_utc(sdo_time):
    ''' Convert SDO time as in the MEDOC database (seconds since
    1977-01-01T00:00:00TAI) to UTC datetime. '''
    t_ref = time.Time('1977-01-01T00:00:00', scale='tai')
    t_tai = t_ref + time.TimeDelta(sdo_time, format='sec', scale='tai')
    return t_tai.utc.datetime


def query_aia_data(dates, wl0, nb_res_max=-1, cadence='1 min',
                   increase_date_range=True, keywords='default'):
    ''' Get a list of AIA data within given time limits.

    Parameters
    ==========
    dates : 2-tuple of datetime.datetime
        Interval of dates within which to search for data.
    wl0 : str
        The AIA channel to use.
    nb_res_max : int (default: -1)
        Maximum number of results to return.
        If set to -1, do not limit the number of results (?). This behaviour
        depends on sitools2.clients.sdo_client_medoc.media_search().
    cadence : str (default: '1 min')
        Data cadence, passed to sitools2.clients.sdo_client_medoc.media_search.
    increase_date_range : bool (default: True)
        If True, increase interval of dates from (d1, d2) to
        (d1 - cadence, d2 + cadence)
    keywords : list of str, str, or None (default: 'default')
        The metadata keywords to return for each AIA frame.
        Either a list of str containing AIA metadata keywords (which can be
        lowercase); None to skip the keyword query; or 'default' to query the
        following keywords:

            date__obs, exptime, int_time,
            ctype1, cunit1, crpix1, crval1, cdelt1,
            ctype2, cunit2, crpix2, crval2, cdelt2,
            crota2, r_sun, x0_mp, y0_mp,
            crln_obs, crlt_obs, car_rot.

    Returns
    =======
    aia_frames : list
        A list of md.Sdo_data objects that represent the returned AIA frames.
    metadata : dict
        A dict of lists, where each key is a metadata keyword, and each list
        contains metadata retrieved from the Medoc database for each search
        result.
    '''

    if increase_date_range:
        reg = re.compile(
            '(?:(?P<days>\d+)\s*(?:d|day|days))?\s*'
            '(?:(?P<hours>\d+)\s*(?:h|hour|hours))?\s*'
            '(?:(?P<minutes>\d+)\s*(?:m|min|minute|minutes))?\s*'
            '(?:(?P<seconds>\d+)\s*(?:s|second|seconds))?\s*'
            )
        m = reg.match(cadence)
        if m:
            m = m.groupdict()
            m = {k: float(v) if v else 0 for k, v in m.items()}
            cadence_timedelta = datetime.timedelta(**m)
        else:
            raise ValueError('could not parse cadence')
        d1, d2 = dates
        dates = [d1 - cadence_timedelta, d2 + cadence_timedelta]

    # Get a list of all AIA data
    aia_frames = md.media_search(
        dates=dates,
        waves=[wl0],
        cadence=[cadence],
        nb_res_max=nb_res_max,
        )
    msg = 'Reached AIA maximum results number.'
    assert (nb_res_max == -1) or (len(aia_frames) < nb_res_max), msg

    # remove frames with exposure times that are too short or too long
    exptime = np.array([sr.exptime for sr in aia_frames])
    if np.std(exptime) > 0.1: # [s]
        min_exptime = np.median(exptime) - np.std(exptime)
        max_exptime = np.median(exptime) + np.std(exptime)
        exptime_mask = (min_exptime < exptime) & (exptime  < max_exptime)
        aia_frames = list(compress(aia_frames, exptime_mask))

    aia_frames = np.array(aia_frames)

    if len(aia_frames) == 0:
        msg = 'No AIA {} images found between {} and {}'
        msg = msg.format(wl0, dates[0], dates[1])
        raise ValueError(msg)

    # retrieve metadata
    if keywords is None:
        metadata = {}
    else:
        if keywords is 'default':
            keywords = [
                'date__obs', 'exptime', 'int_time',
                'ctype1', 'cunit1', 'crpix1', 'crval1', 'cdelt1',
                'ctype2', 'cunit2', 'crpix2', 'crval2', 'cdelt2',
                'crota2', 'r_sun', 'x0_mp', 'y0_mp',
                'crln_obs', 'crlt_obs', 'car_rot',
                ]
        # query metadata for each item in aia_frames, handling possible
        # duplicates in this list
        metadata = md.media_metadata_search(
            keywords=keywords + ['recnum'],
            media_data_list=aia_frames,
            )
        # index metadata with their recnum
        metadata = {meta['recnum']: meta for meta in metadata}
        metadata = [metadata[af.recnum] for af in aia_frames]
        # reshape metadata
        # (this drops the recnum field if it was not selected by the user)
        metadata = {kw: [meta_dict[kw] for meta_dict in metadata]
            for kw in keywords}
        if 'date__obs' in keywords:
            metadata['date__obs'] = [sdotime_to_utc(t) for t in
                metadata['date__obs']]
        metadata = {k: np.array(v) for k, v in metadata.items()}

    return aia_frames, metadata

def filename_from_medoc_result(res, aia_data_dir=''):
    ''' Determine the path an name of an AIA fits from the object returned by
    the sitools2 medoc client.
    '''

    # Extract metadata and store them to a dict
    metadata = {}
    # date:
    date = ('year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond')
    metadata.update({
            k: res.date_obs.__getattribute__(k) for k in date
            })
    # data_level
    try:
        metadata['level'] = res.series_name.strip('aia.lev')
    except AttributeError:
        # pySitools 1.0.1 doesn't provide res.series_name. Default to 1
        metadata['level'] = '1'
    # wavelength
    metadata['wave'] = res.wave
    metadata['isodate'] = res.date_obs.strftime('%Y-%m-%dT%H-%M-%S')

    # define file path and name templates
    # eg path: "sdo/aia/level1/2012/06/05/"
    path = os.path.join(
        aia_data_dir, 'level{level}',
        '{year:04d}', '{month:02d}', '{day:02d}')
    # eg format: "aia.lev1.171A_2012-06-05T11-15-36.image_lev1.fits"
    filename = 'aia.lev{level}.{wave}A_{isodate}.image_lev{level}.fits'

    path = path.format(**metadata)
    filename = filename.format(**metadata)

    return path, filename

def download_fits(medoc_result, dirname, filename):
    ''' Download the FITS for a given medoc request result, and store it to
    dirname/filename.
    '''
    if verb:
        print('AIA: downloading to {}'.format(os.path.join(dirname, filename)))
    medoc_result.get_file(target_dir=dirname, filename=filename)

def get_fits(aia_res, ias_location_prefix='/'):
    ''' Get the path to the fits file for a md.Sdo_data object object.

    If the file exists in ias_location_prefix, return the path to this file.
    Else, download it and return the path to the downloaded file.

    The files are downloaded under the path specified in environment variable
    $SOLARDATA if is set, and in the current directory if not.
    '''

    # build aia path
    sdo_data_dir = os.environ.get('SDO_DATA', './sdo')
    aia_data_dir = os.path.join(sdo_data_dir, 'aia')
    aia_dir, aia_file = filename_from_medoc_result(
        aia_res, aia_data_dir=aia_data_dir)

    # build path to fits in ias_location
    ias_location = aia_res.ias_location.strip('/')
    ias_location_fits = os.path.join(
        ias_location_prefix, ias_location, 'S00000', 'image_lev1.fits')
    ias_location_fits = os.path.expanduser(ias_location_fits)

    # check for fits in ias_location
    if os.path.exists(ias_location_fits):
        if verb:
            print('AIA: using {}'.format(ias_location_fits))
        fits_path = ias_location_fits

    # if no fits in ias_location, get it over HTTP
    else:
        if not os.path.exists(aia_dir):
            os.makedirs(aia_dir)
        if not os.path.exists(os.path.join(aia_dir, aia_file)):
            download_fits(aia_res, aia_dir, aia_file)
        if verb:
            print('AIA: using {}'.format(os.path.join(aia_dir, aia_file)))
        fits_path = os.path.join(aia_dir, aia_file)

    return fits_path
