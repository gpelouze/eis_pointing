#!/usr/bin/env python3

import datetime
import os
import warnings

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
            pointings,
            raster_builder, eis_int,
            titles, ranges, offsets, cross_correlations,
            start_time, stop_time,
            ):
        ''' Build and save pointing verification data

        Parameters
        ==========
        verif_dir : str
        eis_name : str
        aia_band : int
        pointings : list of eis.EISPointing
        raster_builder : aia_raster.SyntheticRasterBuilder
        eis_int : 2D array
        titles : list of str
        ranges : list
            Items can be either 3-tuples of cr.tools.OffsetSet, or None.
        offsets : list
            Items can be either 3-tuples of floats, or arrays of shape (n, 3).
        cross_correlations : list of arrays
        start_time : datetime.datetime
        stop_time : datetime.datetime
        '''

        self.verif_dir = verif_dir
        self.eis_name = eis_name
        self.aia_band = aia_band
        self.pointings = pointings
        self.raster_builder = raster_builder
        self.eis_int = eis_int
        self.titles = titles
        self.ranges = ranges
        self.offsets = offsets
        self.cross_correlations = cross_correlations
        self.start_time = start_time
        self.stop_time = stop_time
        self.rms = []

        if not os.path.exists(self.verif_dir):
            os.makedirs(self.verif_dir)

    def save_all(self):
        self.save_npz()
        self.save_figures()
        self.save_summary()

    def save_npz(self):
        ''' Save cc, offset, and new coordinates '''
        np.savez(
            os.path.join(self.verif_dir, 'offsets.npz'),
            offset=np.array(self.offsets, dtype=object),
            cc=np.array(self.cross_correlations, dtype=object),
            x=self.pointings[-1].x, y=self.pointings[-1].y,
            )

    def save_summary(self):
        ''' Print and save yaml summary '''
        if not self.rms:
            self.rms = [None] * (len(titles) + 1)
        run_specs = [
            ('verif_dir', self.verif_dir),
            ('initial_rms', self.rms[0]),
            ('steps', self._repr_steps(
                self.titles,
                self.ranges,
                self.offsets,
                self.cross_correlations,
                self.rms[1:],
                indent=2)),
            ('exec_time', self.stop_time - self.start_time),
            ]
        summary = ''
        for spec in run_specs:
            summary += self._repr_kv(*spec, indent=0)
        print('\n---\n', summary, '...', sep='')
        with open(os.path.join(self.verif_dir, 'summary.yml'), 'w') as f:
            f.write(summary)

    def _repr_offset(self, offset):
        offset = list(offset)
        offset[0], offset[1] = offset[1], offset[0]
        return offset

    def _repr_kv(self, name, value, indent=0, sep=': ', end='\n'):
        form = '{:#.6g}'
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

    def _repr_steps(self, titles, all_ranges, offsets, ccs, rmss, indent=0):
        indent += 2
        ret = '\n'
        for name, ranges, offset, cc, rms in \
                zip(titles, all_ranges, offsets, ccs, rmss):
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
            if rms is not None:
                ret += self._repr_kv('rms', rms, indent=indent)
        if ret[-1] == '\n':
            ret = ret[:-1]
        return ret

    def save_figures(self):
        ''' plot alignment results '''

        diff_norm = mpl.colors.Normalize(vmin=-3, vmax=+3)

        n_pointings = len(self.pointings)
        for i, pointing in enumerate(self.pointings):
            name = 'step_{}'.format(i)
            if i == 0:
                name += '_original'
            elif i == n_pointings - 1:
                name += '_optimal'
            self.plot_intensity(pointing, name=name, diff_norm=diff_norm)
        self.plot_slit_align()

    def _get_interpolated_maps(self, pointing, save_to=None):
        ''' get maps and interpolate them on an evenly-spaced grid '''

        x, y = pointing.x, pointing.y

        aia_int = self.raster_builder.get_raster(
            x, y, pointing.t / 3600,
            extrapolate_t=True)

        y_interp = np.linspace(y.min(), y.max(), y.shape[0])
        x_interp = np.linspace(x.min(), x.max(), x.shape[1])
        xi_interp = np.moveaxis(np.array(np.meshgrid(x_interp, y_interp)), 0, -1)
        points = (x.flatten(), y.flatten())

        eis_int_interp = si.LinearNDInterpolator(points, self.eis_int.flatten())
        eis_int_interp = eis_int_interp(xi_interp)
        aia_int_interp = si.LinearNDInterpolator(points, aia_int.flatten())
        aia_int_interp = aia_int_interp(xi_interp)

        if save_to:
            np.savez(
                save_to,
                x=x,
                y=y,
                eis_int=self.eis_int,
                aia_int=aia_int,
                x_interp=x_interp,
                y_interp=y_interp,
                eis_int_interp=eis_int_interp,
                aia_int_interp=aia_int_interp,
                )

        return x_interp, y_interp, eis_int_interp, aia_int_interp

    def _normalize_intensity(self, a, b, norm=mpl.colors.Normalize):
        def normalize(arr):
            arr_stat = arr[~(arr == 0)] # exclude possibly missing AIA data
            arr = (arr - np.nanmean(arr_stat)) / np.nanstd(arr_stat)
            return arr
        a = normalize(a)
        b = normalize(b)
        offset = - np.nanmin((a, b))
        offset += .1
        a += offset
        b += offset
        norm = norm(vmin=np.nanmin((a, b)), vmax=np.nanmax((a, b)))
        return a, b, norm

    def plot_intensity(self, pointing, name='', diff_norm=None):
        ''' plot intensity maps of EIS and AIA rasters '''
        if name:
            name = '_' + name
        filenames = {
            'npy': 'intensity_data{}.npz',
            'intensity': 'intensity_maps{}.pdf',
            'diff': 'intensity_diff{}.pdf',
            }
        filenames = {k: os.path.join(self.verif_dir, v.format(name))
            for k, v in filenames.items()}

        # build and save normalized intensity maps
        x, y, eis_int, aia_int = self._get_interpolated_maps(
            pointing, save_to=filenames['npy'])

        eis_int, aia_int, norm = self._normalize_intensity(
            eis_int, aia_int, mpl.colors.LogNorm)

        # plot maps
        pp = backend_pdf.PdfPages(filenames['intensity'])
        intensity_plots = (
            (eis_int, 'EIS'),
            (aia_int, 'synthetic raster from AIA {}'.format(self.aia_band)),
            )
        for int_map, title in intensity_plots:
            plt.clf()
            plots.plot_map(
                plt.gca(),
                int_map, coordinates=[x, y],
                cmap='gray', norm=norm)
            plt.title(title)
            plt.xlabel('X [arcsec]')
            plt.ylabel('Y [arcsec]')
            plt.savefig(pp)
        pp.close()

        # plot difference
        diff = eis_int - aia_int
        rms = np.sqrt(np.nanmean(diff**2))
        self.rms.append(rms)
        if not diff_norm:
            vlim = np.nanmax(np.abs(diff))
            diff_norm = mpl.colors.Normalize(vmin=-vlim, vmax=+vlim)
        plt.clf()
        im = plots.plot_map(
            plt.gca(),
            diff, coordinates=[x, y],
            cmap='gray', norm=diff_norm)
        cb = plt.colorbar(im)
        cb.set_label('normalised EIS − AIA')
        plt.title('RMS = {:.2g}'.format(rms))
        plt.xlabel('X [arcsec]')
        plt.ylabel('Y [arcsec]')
        plt.savefig(filenames['diff'])

    def _get_slit_offset(self):
        slit_offsets = []
        for offset in self.offsets:
            if np.array(offset).ndim > 1:
                slit_offsets.append(offset)
        if len(slit_offsets) == 0:
            return None
        elif len(slit_offsets) > 1:
            warnings.warn('Multiple slitshift steps. Plotting the first one')
        return slit_offsets[0]

    def plot_slit_align(self):
        ''' plot offsets and slit coordinates '''
        slit_offset = self._get_slit_offset()
        if slit_offset is None:
            return
        pp = backend_pdf.PdfPages(os.path.join(self.verif_dir, 'slit_align.pdf'))
        x_color = '#2ca02c'
        y_color = '#1f77b4'
        old_color = '#d62728'
        new_color = '#000000'
        # offset
        plt.clf()
        plt.plot(slit_offset.T[1], '.', label='X', color=x_color)
        plt.plot(slit_offset.T[0], '.', label='Y', color=y_color)
        plt.title(self.eis_name)
        plt.xlabel('slit position')
        plt.ylabel('offset [arcsec]')
        plt.legend()
        plt.savefig(pp)
        # new coordinates
        plots = [
            ('X', self.pointings[-1].x, self.pointings[0].x),
            ('Y', self.pointings[-1].y, self.pointings[0].y),
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


def shift_step(x, y, eis_int, aia_int, cores=None, **kwargs):
    cli.print_now('> correct translation')
    x, y, offset = cr.images.align(
        eis_int, x, y,
        aia_int, x, y,
        cores=cores, **kwargs)
    y_offset, x_offset, cc = offset
    offset = [y_offset, x_offset, 0]
    offset_set = None
    title = 'shift'
    return title, offset_set, offset, cc, x, y

def rotshift_step(x, y, dates_rel_hours, eis_int, raster_builder,
        cores=None, **kwargs):
    cli.print_now('> align rasters')
    x, y, offset = cr.rasters.align(
        eis_int, x, y, dates_rel_hours, raster_builder,
        cores=cores, **kwargs)
    y_offset, x_offset, a_offset, cc = offset
    offset = [y_offset, x_offset, a_offset]
    offset_set = (kwargs['y_set'], kwargs['x_set'], kwargs['a_set'])
    title = 'rotshift'
    return title, offset_set, offset, cc, x, y

def slitshift_step(x, y, dates_rel_hours, eis_int, raster_builder,
        cores=None, **kwargs):
    cli.print_now('> align slit positions')
    x, y, offset = cr.slits.align(
        eis_int, x, y, dates_rel_hours, raster_builder,
        cores=cores, **kwargs)
    offset, cc = offset
    offset_set = (kwargs['y_set'], kwargs['x_set'], kwargs['a_set'])
    title = 'slitshift'
    return title, offset_set, offset, cc, x, y

def optimal_pointing(eis_data, cores=None, aia_band=None,
        verif_dir=None, aia_cache=None, eis_name=None, steps_file=None):
    ''' Determine the EIS pointing using AIA data as a reference.

    Parameters
    ==========
    eis_data : eis.EISData
        Object containing the EIS intensity and pointing.
    cores : int or None
        Number of cores to use for multiprocessing, if any.
    aia_band : int
        The reference AIA channel. Eg. 193.
    verif_dir : str
        Path to the directory where to save verification plots.
    aia_cache : str
        Path to the synthetic AIA raster builder cache file.
    eis_name : str
        Name of the l0 EIS file eg. eis_l0_20140810_010438
    steps_file : str
        Path to a yaml file containing the registration steps.

    Returns
    =======
    pointing : eis.EISPointing
        Optimal EIS pointing.
    '''

    if steps_file:
        registration_steps = cli.load_corr_steps(steps_file)
    else:
        warnings.warn('No steps file provided, falling back to default.')
        registration_steps = {'steps': [
            {'type': 'shift',
             'cc_function': 'explicit',
             'cc_boundary': 'drop',
             'sub_px': True,
             },
            {'type': 'rotshift',
             'x_set': cr.tools.OffsetSet((-10.0, 10.0), number=11),
             'y_set': cr.tools.OffsetSet((-5.0, 5.0), number=11),
             'a_set': cr.tools.OffsetSet((-3.0, 3.0), step=0.2),
             },
            {'type': 'slitshift',
             'x_set': cr.tools.OffsetSet((-20.0, 20.0), number=21),
             'y_set': cr.tools.OffsetSet((-20.0, 20.0), number=21),
             'a_set': cr.tools.OffsetSet((0.0, 0.0), number=1),
             'mp_mode': 'track'
             },
            ]}

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
    single_aia_frame = registration_steps.get('single_aia_frame', False)
    if single_aia_frame:
        single_aia_frame = num.dt_average(np.min(dates_abs), np.max(dates_abs))
        aia_cache = None
    raster_builder = aia_raster.SyntheticRasterBuilder(
        dates=[np.min(dates_abs), np.max(dates_abs)],
        date_ref=date_ref,
        channel=aia_band,
        file_cache=aia_cache,
        single_frame=single_aia_frame,
        )
    raster_builder.get_data()

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
    titles = []
    offset_sets = []
    offsets = []
    pointings = [eis_data.pointing]
    cross_correlations = []

    start_time = datetime.datetime.now()

    for step in registration_steps['steps']:
        registration_type = step.pop('type')
        if registration_type == 'shift':
            aia_int = raster_builder.get_raster(
                x, y, dates_rel_hours,
                extrapolate_t=True)
            result = shift_step(x, y, eis_int, aia_int, cores=cores, **step)
        elif registration_type == 'rotshift':
            result = rotshift_step(x, y, dates_rel_hours,
                eis_int, raster_builder,
                cores=cores, **step)
        elif registration_type == 'slitshift':
            result = slitshift_step(x, y, dates_rel_hours,
                eis_int, raster_builder,
                cores=cores, **step)
        else:
            raise ValueError('unknown registration step')
        title, offset_set, offset, cc, x, y = result
        titles.append(title)
        offset_sets.append(offset_set)
        offsets.append(offset)
        pointings.append(eis.EISPointing(x, y, eis_data.pointing.t, date_ref))
        cross_correlations.append(cc)

    stop_time = datetime.datetime.now()

    if verif_dir:
        verif = OptPointingVerif(
            verif_dir, eis_name, aia_band,
            pointings,
            raster_builder, eis_int,
            titles, offset_sets, offsets, cross_correlations,
            start_time, stop_time,
            )
        verif.save_all()

    return pointings[-1]
