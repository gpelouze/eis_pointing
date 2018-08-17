#!/usr/bin/env python3

import os

from astropy.io import fits
import numpy as np

from eis_pointing.utils import cli
from eis_pointing.utils import eis
from eis_pointing.utils import files
from eis_pointing.utils import idl
from eis_pointing.utils import num

from eis_pointing import eis_aia_emission
from eis_pointing import eis_aia_registration

# Long IDL list inputs are split into into
# smaller chunks to avoid bizarre bugs.
IDL_CHUNKS = 25

def make(targets, sources, method, *args, **kwargs):
    ''' Make targets from sources, only if the targets don’t exist.

    Parameters
    ==========
    targets, sources : str or list of str
        The target and source files. If these are lists, they must have the
        same length.
    method : callable
        A callable that builds the targets from the sources.
        Calling sequence: `method(targets, sources, *args, **kwargs)`.
    *args, **kwargs :
        Passed to `method`.
    '''
    str_to_list = lambda s: [s] if isinstance(s, (str, np.character)) else s
    targets = str_to_list(targets)
    sources = str_to_list(sources)
    if len(targets) != len(sources):
        raise ValueError('targets and sources have different lengths')
    to_build = list(filter(
        lambda f: not os.path.exists(f[0]),
        zip(targets, sources)))
    targets_to_build = [tb[0] for tb in to_build]
    sources_to_build = [tb[1] for tb in to_build]
    if targets_to_build:
        n_targets = len(targets_to_build)
        cli.print_now(
            'running', method.__name__,
            'to make', n_targets, 'target'+'s'*(n_targets-1))
        return method(targets, sources, *args, **kwargs)
    else:
        cli.print_now( 'skipping', method.__name__)

def prepare_data(l1_files, l0_files):
    ''' Apply eis_prep to the l0 fits given in input.

    The level 0 fits are read from the locations provided in `l0_files`, and
    then are ingested into the EIS data directory [see EIS SWN #18].

    Since the files are ingested using `eis_ingest`, the paths in `l1_files`
    are ignored, but the parameter is kept for compatibility with `make()`.

    Parameters
    ==========
    l1_files : list of str
        List of absolute paths to the l1 fits to be prepared.
    l0_files : list of str
        List of absolute paths to the l0 fits from which to prepare the l1
        files.
    '''
    if not l0_files:
        return
    for fp in num.chunks(l0_files, IDL_CHUNKS):
        prep = idl.SSWFunction('prep', arguments=[fp], instruments='eis')
        out, err = prep.run()

def export_windata(wd_files, l1_files, aia_band):
    ''' Run export_windata.pro to save windata objects from a list of l1_files.

    Parameters
    ==========
    l1_files : list of str
        List of absolute paths to the l1 fits to be prepared.
    l0_files : list of str
        List of absolute paths to the l0 fits from which to prepare the l1
        files.
    '''
    if not wd_files:
        return
    for fp in num.chunks(list(zip(wd_files, l1_files)), IDL_CHUNKS):
        wd = [f[0] for f in fp]
        l1 = [f[1] for f in fp]
        prep = idl.SSWFunction('export_windata',
            arguments=[wd, l1, aia_band], instruments='eis')
        out, err = prep.run()

def compute_eis_aia_emission(eis_aia_emission_files, wd_files, aia_band):
    if not eis_aia_emission_files:
        return
    for eis_aia_emission_file, wd_file in zip(eis_aia_emission_files, wd_files):
        windata = idl.IDLStructure(wd_file)
        emission = eis_aia_emission.compute(windata, aia_band)
        hdulist = emission.to_hdulist()
        hdulist.writeto(eis_aia_emission_file)

def compute_pointing(pointing_files, emission_files,
        verif_dirs, aia_caches, **kwargs):
    if not pointing_files:
        return
    if isinstance(verif_dirs, (str, np.character)):
        verif_dirs = [verif_dirs]
    if isinstance(aia_caches, (str, np.character)):
        aia_caches = [aia_caches]
    for (pointing_file, emission_file, verif_dir, aia_cache) in \
            zip(pointing_files, emission_files, verif_dirs, aia_caches):
        eis_data = eis.EISData.from_hdulist(fits.open(emission_file))
        pointing = eis_aia_registration.optimal_pointing(
            eis_data, verif_dir, aia_cache=aia_cache, **kwargs)
        pointing.to_hdulist().writeto(pointing_file)


if __name__ == '__main__':

    # get configuration from command line
    args = cli.get_setup()
    eis_l0_name = args.filename
    aia_band = args.aia_band

    # get filenames paths
    filenames = files.Files(args.filename, args.aia_band)
    filenames.mk_output_dirs()

    # make targets
    make(filenames['l1'], filenames['l0'], prepare_data)
    make(filenames['windata'], filenames['l1'], export_windata, aia_band)
    make(filenames['eis_aia_emission'], filenames['windata'],
        compute_eis_aia_emission, aia_band)
    make(filenames['pointing'], filenames['eis_aia_emission'],
        compute_pointing,
        filenames['pointing_verification'],
        filenames['synthetic_raster_cache'],
        cores=args.cores)
