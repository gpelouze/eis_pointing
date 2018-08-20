#!/usr/bin/env python3

import datetime
import os

import numpy as np
import scipy.interpolate as si

import matplotlib as mpl
from matplotlib.backends import backend_pdf
import matplotlib.pyplot as plt

from .utils import aia_raster
from .utils import cli
from .utils import eis
from .utils import num
from .utils import plots

from . import coregister as cr

class OptPointingVerif(object):
    def __init__(self,
            verif_dir, eis_name, aia_band,
            old_pointing, new_pointing,
            raster_builder, eis_int,
            titles, ranges, offsets, cross_correlations,
            start_time, stop_time,
            ):

        self.verif_dir = verif_dir
        self.eis_name = eis_name
        self.aia_band = aia_band
        self.old_pointing = old_pointing
        self.new_pointing = new_pointing
        self.raster_builder = raster_builder
        self.eis_int = eis_int
        self.titles = titles
        self.ranges = ranges
        self.offsets = offsets
        self.cross_correlations = cross_correlations
        self.start_time = start_time
        self.stop_time = stop_time

        if not os.path.exists(self.verif_dir):
            os.makedirs(self.verif_dir)

    def save_all(self):
        self.save_npz()
        self.save_summary()
        self.save_figures()

    def save_npz(self):
        ''' Save cc, offset, and new coordinates '''
        np.savez(
            os.path.join(self.verif_dir, 'offsets.npz'),
            offset=np.array(self.offsets, dtype=object),
            cc=np.array(self.cross_correlations, dtype=object),
            x=self.new_pointing.x, y=self.new_pointing.y,
            )

    def save_summary(self):
        ''' Print and save yaml summary '''
        run_specs = [
            ('verif_dir', self.verif_dir),
            ('steps', self._repr_steps(
                self.titles,
                self.ranges,
                self.offsets,
                self.cross_correlations,
                indent=2)),
            ('exec_time', self.stop_time - self.start_time),
            ]
        summary = '---\n'
        for spec in run_specs:
            summary += self._repr_kv(*spec, indent=0)
        summary += '...\n'
        print('\n', summary, '\n', sep='')
        with open(os.path.join(self.verif_dir, 'summary.yml'), 'w') as f:
            f.write(summary)

    def _repr_offset(self, offset):
        offset = list(offset)
        offset[0], offset[1] = offset[1], offset[0]
        return offset

    def _repr_kv(self, name, value, indent=0, sep=': ', end='\n'):
        form = '{:#.3g}'
        if isinstance(value, (list, tuple)):
            value = [form.format(v)
                if np.issubdtype(type(v), (float, np.inexact))
                else str(v)
                for v in value]
            value = '[' + ', '.join(value) + ']'
        if value is None:
            value = 'null'
        elif np.issubdtype(type(value), (float, np.inexact)):
            value = form.format(value)
        else:
            value = str(value)
        string = ''.join([indent * ' ', name, sep, str(value), end])
        return string

    def _repr_steps(self, titles, all_ranges, offsets, ccs, indent=0):
        indent += 2
        ret = '\n'
        for name, ranges, offset, cc in zip(titles, all_ranges, offsets, ccs):
            ret += ' '*(indent-2) + '- '
            ret += self._repr_kv('name', name, indent=0)
            if ranges:
                ry, rx, ra = ranges
                ret += self._repr_kv('range_x', rx, indent=indent)
                ret += self._repr_kv('range_y', ry, indent=indent)
                ret += self._repr_kv('range_a', ra, indent=indent)
            if len(offset) <= 3:
                ret += self._repr_kv('offset', self._repr_offset(offset), indent=indent)
                ret += self._repr_kv('cc_max', np.nanmax(cc), indent=indent)
        if ret[-1] == '\n':
            ret = ret[:-1]
        return ret

    def save_figures(self):
        ''' plot alignment results '''
        self.prepare_plot()
        self.plot_intensity()
        self.plot_diff()
        self.plot_slit_align()

    def prepare_plot(self):
        ''' get maps and interpolate them on an evenly-spaced grid '''

        self.aia_int = self.raster_builder.get_raster(
            self.new_pointing.x, self.new_pointing.y,
            self.new_pointing.t / 3600,
            extrapolate_t=True)

        x, y = self.new_pointing.x, self.new_pointing.y
        self.y_interp = np.linspace(y.min(), y.max(), y.shape[0])
        self.x_interp = np.linspace(x.min(), x.max(), x.shape[1])
        xi_interp = np.moveaxis(np.array(np.meshgrid(self.x_interp, self.y_interp)), 0, -1)
        points = (x.flatten(), y.flatten())

        self.eis_int_interp = si.LinearNDInterpolator(points, self.eis_int.flatten())
        self.eis_int_interp = self.eis_int_interp(xi_interp)
        self.aia_int_interp = si.LinearNDInterpolator(points, self.aia_int.flatten())
        self.aia_int_interp = self.aia_int_interp(xi_interp)

    def plot_intensity(self):
        ''' plot intensity maps of EIS and AIA rasters '''
        pp = backend_pdf.PdfPages(os.path.join(self.verif_dir, 'intensity.pdf'))
        plt.clf() # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        plots.plot_map(
            plt.gca(),
            self.eis_int_interp, coordinates=[self.x_interp, self.y_interp],
            cmap='gray', norm=mpl.colors.LogNorm())
        plt.title('EIS')
        plt.xlabel('X [arcsec]')
        plt.ylabel('Y [arcsec]')
        plt.savefig(pp)
        plt.clf() # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        plots.plot_map(
            plt.gca(),
            self.aia_int_interp, coordinates=[self.x_interp, self.y_interp],
            cmap='gray', norm=mpl.colors.LogNorm())
        plt.title('synthetic raster from AIA {}'.format(self.aia_band))
        plt.xlabel('X [arcsec]')
        plt.ylabel('Y [arcsec]')
        plt.savefig(pp)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        pp.close()

    def plot_diff(self):
        plt.clf()
        norm = lambda arr: (arr - np.nanmean(arr)) / np.nanstd(arr)
        int_diff = norm(self.eis_int_interp) - norm(self.aia_int_interp)
        vlim = np.nanmax(np.abs(int_diff))
        im = plots.plot_map(
            plt.gca(),
            int_diff, coordinates=[self.x_interp, self.y_interp],
            cmap='gray', vmin=-vlim, vmax=+vlim, norm=mpl.colors.SymLogNorm(.1))
        plt.colorbar(im)
        plt.title('normalised EIS − AIA')
        plt.xlabel('X [arcsec]')
        plt.ylabel('Y [arcsec]')
        plt.savefig(os.path.join(self.verif_dir, 'diff.pdf'))

    def plot_slit_align(self):
        ''' plot offsets and slit coordinates '''
        pp = backend_pdf.PdfPages(os.path.join(self.verif_dir, 'slit_align.pdf'))
        x_color = '#2ca02c'
        y_color = '#1f77b4'
        old_color = '#d62728'
        new_color = '#000000'
        # offset
        plt.clf()
        slit_offset = self.offsets[2]
        plt.plot(slit_offset.T[1], '.', label='X', color=x_color)
        plt.plot(slit_offset.T[0], '.', label='Y', color=y_color)
        plt.title(self.eis_name)
        plt.xlabel('slit position')
        plt.ylabel('offset [arcsec]')
        plt.legend()
        plt.savefig(pp)
        # new coordinates
        plots = [
            ('X', self.new_pointing.x, self.old_pointing.x),
            ('Y', self.new_pointing.y, self.old_pointing.y),
            ]
        for name, aligned, original in plots:
            plt.clf()
            plt.plot(original[0], ',', label='original ' + name, color=old_color)
            plt.plot(aligned[0],  ',', label='aligned ' + name,  color=new_color)
            plt.legend()
            plt.title(self.eis_name)
            plt.xlabel('slit position')
            plt.ylabel(name + ' [arcsec]')
            plt.savefig(pp)
        pp.close()


def optimal_pointing(eis_data, cores=None, aia_band=None,
        verif_dir=None, aia_cache=None, eis_name=None):
    ''' Determine the EIS pointing using AIA data as a reference.

    Parameters
    ==========
    eis_data : eis.EISData
        Object containing the EIS intensity and pointing.
    cores : int or None
        Number of cores to use for multiprocessing, if any.
    verif_dir : str
        Path to the directory where to save verification plots.
    aia_cache : str
        Path to the synthetic AIA raster builder cache file.
    eis_name : str
        Name of the l0 EIS file eg. eis_l0_20140810_010438

    Returns
    =======
    pointing : eis.EISPointing
        Optimal EIS pointing.
    '''

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
        channel=aia_band,
        file_cache=aia_cache,
        )
    aia_int = raster_builder.get_raster(
        x, y, dates_rel_hours,
        extrapolate_t=True)

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

    if verif_dir:
        verif = OptPointingVerif(
            verif_dir, eis_name, aia_band,
            eis_data.pointing, new_pointing,
            raster_builder, eis_int,
            titles, ranges, offsets, cross_correlations,
            start_time, stop_time,
            )
        verif.save_all()

    return new_pointing
