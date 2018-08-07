#!/usr/bin/env python3

import os
import re

from . data import eis

class Files(dict):
    data_types = {
        'windata': ('io/windata', '.sav'),
        'aia_emission': ('io/aia_emission', '.fits'),
        'pointing': ('io/pointing', '.fits'),
        }

    def __init__(self, eis_l0_filename, aia_band):
        self.aia_band = aia_band
        self.aia_suffix = '_{}.sav'.format(self.aia_band)

        filenames = {}

        # EIS files in the Hinode directory structure
        for levelname in ('l0', 'l1'):
            fname = self._transform_filenames(
                eis_l0_filename,
                levelname,
                suffix='.fits',
                )
            filenames[levelname] = eis.fits_path(fname, absolute=True)

        # local files
        for name, (path, extension) in Files.data_types.items():
            path = os.path.join(path, '')
            filenames[name] = self._transform_filenames(
                eis_l0_filename, name,
                prefix=path, suffix=extension)

        super().__init__(filenames)

    def mk_output_dirs(self):
        for (d, _) in Files.data_types.values():
            if not os.path.exists(d):
                os.makedirs(d)

    def _transform_filenames(self, fname_l0, datatype, prefix='', suffix=''):
        ''' Transform an EIS level 0 filename to another filename.

        This replaces 'eis_l0_yyyymmdd_hhmmss' with
        '[<prefix>]eis_<datatype>_yyyymmdd_hhmmss[<sufix>]'.
        Eg. 'eis_l0_20120609_074253' to 'eis_l1_20120609_074253'.

        Unexpected behavior will occur when input file names are not formatted
        like 'eis_l0_yyyymmdd_hhmmss'.
        '''
        fname = re.sub('eis_l0_', 'eis_{}_', fname_l0)
        fname = prefix + fname.format(datatype) + suffix
        return fname
