#!/usr/bin/env python3

import os
import re

import numpy as np

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
