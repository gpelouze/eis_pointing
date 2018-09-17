#!/usr/bin/env python3

import warnings

import numpy as np

from .utils import eis
from .utils import num

def compute(windata, wl0, wl_width):
    ''' Compute synthetic emission for a given AIA band using EIS data.

    Parameters
    ==========
    windata : idl.IDLStructure
        Windata structure containing the wavelength windows necessary to compute
        the AIA emission.
    wl0 : float
        The central wavelength of the integration domain, in Ångström.
    wl_width : float
        The width wavelength of the integration domain, in Ångström.
    '''

    slot = windata.hdr.slit_id[0].decode() in ('40"', '266"')
    if slot:
        # slot windata don't have a .missing tag
        windata.missing = -100
        windata.exposure_time = np.array([float(et)
            for et in windata.exposure_time])
        exposure_time = windata.exposure_time
        t_ref = num.parse_date(windata.hdr['date_obs'][0])
        t_abs = num.parse_date(windata.time_ccsds)
        windata.time = num.total_seconds(t_abs - t_ref)
    else:
        exposure_time = windata.exposure_time.reshape(-1, 1)

    intensity = windata.int.copy()
    missing_places = (intensity == windata.missing)
    intensity[missing_places] = np.nan
    intensity /= exposure_time

    if not slot:
        wvl_min = wl0 - wl_width
        wvl_max = wl0 + wl_width
        i_min = np.argmin(np.abs(windata.wvl - wvl_min))
        i_max = np.argmin(np.abs(windata.wvl - wvl_max))
        intensity = intensity[:, :, i_min:i_max+1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            intensity = np.nanmean(intensity, axis=-1)

    pointing = eis.EISPointing.from_windata(windata, use_wvl=False)
    data = eis.EISData(intensity, pointing)
    return data
