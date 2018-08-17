#!/usr/bin/env python3

import datetime
import os

import numpy as np

from .utils import aia_raster
from .utils import cli
from .utils import eis
from .utils import num

from . import coregister as cr

def optimal_pointing(eis_data, verif_dir, cores=None):
    ''' Determine the EIS pointing using AIA data as a reference.

    Parameters
    ==========
    eis_data : eis.EISData
        Object containing the EIS intensity and pointing.
    verif_dir : str
        Path to the directory where to save verification plots.

    Returns
    =======
    pointing : eis.EISPointing
        Optimal EIS pointing.
    '''

    wl0 = 195.119 # FIXME

    if not os.path.exists(verif_dir):
        os.makedirs(verif_dir)

    cli.print_now('> build relative and absolute date arrays') # ----------------------
    dates_rel = num.seconds_to_timedelta(eis_data.pointing.t)
    dates_rel_hours = eis_data.pointing.t / 3600
    date_ref = eis_data.pointing.t_ref
    dates_abs = date_ref + dates_rel

    cli.print_now('> get EIS grid info and add margin') # -----------------------------
    x, y = eis_data.pointing.x, eis_data.pointing.y
    x_margin = (np.max(x) - np.min(x)) / 2
    y_margin = (np.max(y) - np.min(y)) / 2
    x_margin = np.max(x_margin)
    y_margin = np.max(y_margin)
    ny, y_slice = cr.tools.create_margin(y, y_margin, 0)
    nx, x_slice = cr.tools.create_margin(x, x_margin, 1)
    new_shape = 1, ny, nx
    new_slice = slice(None), y_slice, x_slice

    eis_int = eis_data.data

    cli.print_now('> get AIA data') # -------------------------------------------------
    # (verified against the original method used in align.py)
    raster_builder = aia_raster.SyntheticRasterBuilder(
        dates=[np.min(dates_abs), np.max(dates_abs)],
        date_ref=date_ref,
        channel=eis.get_aia_channel(wl0),
        )
    raster_builder.cache.update(np.load('io/aia_raster.npy')) # FIXME
    aia_int = raster_builder.get_raster(
        x, y, dates_rel_hours,
        extrapolate_t=True)
    # np.save('io/aia_raster.npy', raster_builder.cache.get()) # FIXME

    # degrade raster_builder resolution to 3 arcsec (see DelZanna+2011)
    raster_builder.degrade_resolution(3, cores=cores)

    # crop raster_builder cached data to fix multiprocessing
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_cen = (x_min + x_max) / 2
    y_cen = (y_min + y_max) / 2
    r = np.sqrt((x_max - x_cen)**2 + (y_max - y_cen)**2)
    raster_builder.crop_data(x_cen - r, x_cen + r, y_cen - r, y_cen + r)

    # compute alignment -------------------------------------------------------
    start_time = datetime.datetime.now()

    titles = []
    offsets = []
    cross_correlations = []
    ranges = []

    cli.print_now('> correct translation')
    x, y, offset = cr.images.align(
        eis_int, x, y,
        aia_int, x, y,
        cores=cores,
        return_offset=True)
    y_offset, x_offset, cc = offset
    offsets.append([y_offset, x_offset, 0])
    cross_correlations.append(cc)
    ranges.append(None)
    titles.append('shift')

    cli.print_now('> aligning rasters')
    x_set = cr.tools.OffsetSet((-10, 10), number=11)
    y_set = cr.tools.OffsetSet((-5, 5), number=11)
    a_set = cr.tools.OffsetSet((-3, 3), step=.2)
    x, y, offset = cr.rasters.align(
        eis_int, x, y, dates_rel_hours,
        raster_builder,
        x_set=x_set, y_set=y_set, a_set=a_set,
        cores=cores,
        return_offset=True)
    y_offset, x_offset, a_offset, cc = offset
    offsets.append([y_offset, x_offset, a_offset])
    cross_correlations.append(cc)
    ranges.append((y_set, x_set, a_set))
    titles.append('rotshift')

    cli.print_now('> align slit positions')
    x_set = cr.tools.OffsetSet((-20, 20), number=21)
    y_set = cr.tools.OffsetSet((-20, 20), number=21)
    a_set = cr.tools.OffsetSet((0, 0), number=1)
    x, y, offset = cr.slits.align(
        eis_int, x, y, dates_rel_hours,
        raster_builder,
        x_set=x_set, y_set=y_set, a_set=a_set,
        cores=cores, mp_mode='track',
        return_offset=True)
    offset, cc = offset
    offsets.append(offset)
    cross_correlations.append(cc)
    ranges.append((y_set, x_set, a_set))
    titles.append('slitshift')

    cli.print_now('> aligning rasters')
    x_set = cr.tools.OffsetSet((-5, 5), number=11)
    y_set = cr.tools.OffsetSet((-5, 5), number=11)
    a_set = cr.tools.OffsetSet((-2, 2), step=.2)
    x, y, offset = cr.rasters.align(
        eis_int, x, y, dates_rel_hours,
        raster_builder,
        x_set=x_set, y_set=y_set, a_set=a_set,
        cores=cores,
        return_offset=True)
    y_offset, x_offset, a_offset, cc = offset
    offsets.append([y_offset, x_offset, a_offset])
    cross_correlations.append(cc)
    ranges.append((y_set, x_set, a_set))
    titles.append('rotshift')

    stop_time = datetime.datetime.now()

    new_pointing = eis.EISPointing(x, y, eis_data.pointing.t, date_ref)

    return new_pointing
