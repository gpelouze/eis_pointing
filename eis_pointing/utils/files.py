#!/usr/bin/env python3

import functools
import os
import re

from . import eis

class Files(dict):
    data_types = {
        # key: local_path (directory only, relative to io_dir), prefix, extension
        # final path: {io_dir}/{local_path}/{prefix}_yyymmdd_hhmmss.{extension}
        'windata': ('windata', 'windata', '.sav'),
        'eis_aia_emission': ('eis_aia_emission', 'eis_aia_emission', '.fits'),
        'pointing': ('pointing', 'pointing', '.fits'),
        'pointing_verification': ('pointing_verification', '', '/'),
        'synthetic_raster_cache': ('cache', 'synthetic_raster', '.npy'),
        'eis_name': ('', 'eis_l0', ''),
        }

    def __init__(self, eis_l0_filename, io_dir):
        filenames = {}

        self.io_dir = os.path.realpath(io_dir)

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
            if path:
                path = os.path.join(self.io_dir, path)
            path = os.path.join(path, '')
            filenames[key] = self._transform_filenames(
                eis_l0_filename, name,
                prefix=path, suffix=extension)

        super().__init__(filenames)

    def mk_output_dirs(self):
        if not os.path.exists(self.io_dir):
            os.makedirs(self.io_dir)
        for f in self.values():
            d = os.path.dirname(f)
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

class ManyFiles(object):
    def __init__(self, eis_l0_filenames, io_dir):
        self.files = [Files(f, io_dir) for f in eis_l0_filenames]
        self.io_dir = io_dir

    def __getitem__(self, k):
        return [f[k] for f in self.files]

    def get(self, i):
        return self.files[i]

    def mk_output_dirs(self):
        for f in self.files:
            f.mk_output_dirs()

    def as_dict(self):
        keys = [list(f.keys()) for f in self.files]
        keys = functools.reduce(lambda x, y: x+y ,keys)
        return {k: [f[k] for f in self.files] for k in keys}

    def keys(self):
        return self.as_dict().keys()

    def values(self):
        return self.as_dict().values()

    def items(self):
        return self.as_dict().items()

    def __iter__(self):
        return self.files.__iter__()

    def __len__(self):
        return self.files.__len__()
