#!/usr/bin/env python3

import os
import re

from . import eis

class Files(dict):
    data_types = {
        # key: path (directory only), prefix, extension
        'windata': ('io/windata', 'windata', '.sav'),
        'eis_aia_emission': ('io/eis_aia_emission', 'eis_aia_emission', '.fits'),
        'pointing': ('io/pointing', 'pointing', '.fits'),
        'pointing_verification': ('io/pointing_verification', '', '/'),
        'synthetic_raster_cache': ('io/cache', 'synthetic_raster', '.npy'),
        'eis_name': ('', 'eis_l0', ''),
        }

    def __init__(self, eis_l0_filename, aia_band):
        self.aia_band = aia_band
        self.aia_suffix = '_{}.sav'.format(self.aia_band)

        filenames = {}

        # EIS files in the Hinode directory structure
        for levelname in ('l0', 'l1'):
            fname = self._transform_filenames(
                eis_l0_filename,
                'eis_' + levelname,
                suffix='.fits',
                )
            filenames[levelname] = eis.fits_path(fname, absolute=True)

        # local files
        for key, (path, name, extension) in Files.data_types.items():
            path = os.path.join(path, '')
            filenames[key] = self._transform_filenames(
                eis_l0_filename, name,
                prefix=path, suffix=extension)

        super().__init__(filenames)

    def mk_output_dirs(self):
        for (d, _, _) in Files.data_types.values():
            if d and not os.path.exists(d):
                os.makedirs(d)

    def _transform_filenames(self, fname_l0, datatype, prefix='', suffix=''):
        ''' Transform an EIS level 0 filename to another filename.

        This replaces 'eis_l0_yyyymmdd_hhmmss' with
        '[<prefix>]eis_<datatype>_yyyymmdd_hhmmss[<sufix>]'.
        Eg. 'eis_l0_20120609_074253' to 'eis_l1_20120609_074253'.

        Unexpected behavior will occur when input file names are not formatted
        like 'eis_l0_yyyymmdd_hhmmss'.
        '''
        if datatype:
            datatype += '_'
        fname = re.sub('eis_l0_', '{}', fname_l0)
        fname = prefix + fname.format(datatype) + suffix
        return fname
