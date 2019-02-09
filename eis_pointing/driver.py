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

# Install path of Solar Soft
try:
    SSW = os.environ['SSW']
except KeyError:
    SSW = '/usr/local/ssw'

# Long IDL list inputs are split into into
# smaller chunks to avoid bizarre bugs.
IDL_CHUNKS = 25

IDL_CWD = os.path.dirname(os.path.realpath(__file__))
IDL_CWD = os.path.join(IDL_CWD)

def to_build(targets, sources, secondary_targets=None):
    str_to_list = lambda s: [s] if isinstance(s, (str, np.character)) else s
    targets = str_to_list(targets)
    sources = str_to_list(sources)
    if len(targets) != len(sources):
        raise ValueError('targets and sources have different lengths')
    to_build = [not os.path.exists(t) for t in targets]
    targets_to_build = list(np.array(targets)[to_build])
    sources_to_build = list(np.array(sources)[to_build])
    if secondary_targets is not None:
        secondary_targets_to_build = []
        for t in secondary_targets:
            if len(t) != len(targets):
                raise ValueError('targets have different lengths')
            t = str_to_list(t)
            t_to_build = list(np.array(t)[to_build])
            secondary_targets_to_build.append(t_to_build)
    else:
        secondary_targets_to_build = None
    return targets_to_build, sources_to_build, secondary_targets_to_build

def all_targets_are_built(targets):
    targets_to_build, _, _ = to_build(targets, targets)
    return not targets_to_build

def make(targets, sources, method, *args, secondary_targets=None, **kwargs):
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
    targets_to_build, sources_to_build, secondary_targets_to_build = to_build(
        targets, sources, secondary_targets=secondary_targets)
    if targets_to_build:
        n_targets = len(targets_to_build)
        cli.print_now(
            'running', method.__name__,
            'to make', n_targets, 'target'+'s'*bool(n_targets-1))
        if secondary_targets_to_build is not None:
            kwargs['secondary_targets'] = secondary_targets_to_build
        return_value = method(
            targets_to_build, sources_to_build,
            *args, **kwargs)
        for target in targets:
            if not os.path.exists(target):
                raise RuntimeError('could not build {}'.format(target))
        return return_value
    else:
        cli.print_now( 'skipping', method.__name__)

def get_fits(l0_files, eis_names):
    for l0_file, eis_name in zip(l0_files, eis_names):
        eis.get_fits(eis_name, custom_dest=l0_file)

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
    for fp in num.chunks(l0_files, IDL_CHUNKS):
        prep = idl.SSWFunction(
            'prep', arguments=[fp],
            cwd=IDL_CWD,
            instruments='eis', ssw_path=SSW)
        out, err = prep.run()

def export_windata(wd_files, l1_files, wl0):
    ''' Run export_windata.pro to save windata objects from a list of l1_files.

    Parameters
    ==========
    l1_files : list of str
        List of absolute paths to the l1 fits to be prepared.
    l0_files : list of str
        List of absolute paths to the l0 fits from which to prepare the l1
        files.
    '''
    for fp in num.chunks(list(zip(wd_files, l1_files)), IDL_CHUNKS):
        wd = [f[0] for f in fp]
        l1 = [f[1] for f in fp]
        prep = idl.SSWFunction(
            'export_windata', arguments=[wd, l1, wl0],
            cwd=IDL_CWD,
            instruments='eis', ssw_path=SSW)
        out, err = prep.run()

def compute_eis_aia_emission(eis_aia_emission_files, wd_files, *args):
    for eis_aia_emission_file, wd_file in zip(eis_aia_emission_files, wd_files):
        windata = idl.IDLStructure(wd_file)
        emission = eis_aia_emission.compute(windata, *args)
        hdulist = emission.to_hdulist()
        hdulist.writeto(eis_aia_emission_file)

def compute_pointing(pointing_files, emission_files,
                     secondary_targets, **kwargs):
    verif_dirs, aia_caches, eis_names = secondary_targets
    for (pointing_file, emission_file, verif_dir, aia_cache, eis_name) \
    in zip(pointing_files, emission_files, verif_dirs, aia_caches, eis_names):
        eis_data = eis.EISData.from_hdulist(fits.open(emission_file))
        pointing = eis_aia_registration.optimal_pointing(
            eis_data,
            verif_dir=verif_dir,
            aia_cache=aia_cache,
            eis_name=eis_name,
            **kwargs)
        pointing.to_hdulist().writeto(pointing_file)


def compute(*filename, steps_file=None, io='io', cores=4, cache_aia_data=False):
    ''' Perform all computation steps to determine the optimal EIS pointing.

    Parameters
    ==========
    filename : list
        The names of the level 0 EIS files, eg. 'eis_l0_20100815_192002'.
    steps_file : str or None (default: None)
        Path to a yaml file containing the registration steps.
    io : str (default: 'io')
        Directory where output files are written.
    cores : int (default: 4)
        Maximum number of cores used for parallelisation.
    cache_aia_data : bool (default: False)
        Cache the AIA data to a file. This uses a lot of storage, but speeds
        things up when the same raster is aligned for the second time.
    '''
    filenames = files.ManyFiles(filename, io)
    filenames.mk_output_dirs()

    aia_band = 193
    eis_wl0 = 195.119
    eis_wl_width = 0.15

    aia_cache = filenames['synthetic_raster_cache']
    if not cache_aia_data:
        aia_cache = [None] * len(aia_cache)

    # make targets
    if not all_targets_are_built(filenames['windata']):
        make(filenames['l0'], filenames['eis_name'], get_fits)
        make(filenames['l1'], filenames['l0'], prepare_data)
        make(filenames['windata'], filenames['l1'], export_windata, eis_wl0)
    make(filenames['eis_aia_emission'], filenames['windata'],
        compute_eis_aia_emission, eis_wl0, eis_wl_width)
    make(filenames['pointing'], filenames['eis_aia_emission'],
        compute_pointing,
        secondary_targets=(filenames['pointing_verification'],
                           aia_cache,
                           filenames['eis_name']),
        cores=cores,
        aia_band=aia_band,
        steps_file=steps_file)
