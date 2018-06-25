#!/usr/bin/env python3

import os

import numpy as np

from utils import cli
from utils import files
from utils import num
from utils.idl import SSWFunction

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
        prep = SSWFunction('prep', arguments=[fp], instruments='eis')
        out, err = prep.run()

def export_windata(wd_files, l1_files, aia_band):
    pass

def eis_aia_emission(aia_emission_files, wd_file):
    pass

def compute_pointing(pointing_files, aia_emission_files):
    pass


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
    make(filenames['aia_emission'], filenames['windata'], eis_aia_emission)
    make(filenames['pointing'], filenames['aia_emission'], compute_pointing)
