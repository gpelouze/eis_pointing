#!/usr/bin/env python3

from itertools import compress

import numpy as np
import sitools2.clients.sdo_client_medoc as md

from . import num

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

def query_aia_data(dates, wl0, nb_res_max=-1, keywords='default'):
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

    # Get a list of all AIA data
    aia_frames = md.media_search(
        dates=dates,
        waves=[wl0],
        cadence=['12s'],
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
