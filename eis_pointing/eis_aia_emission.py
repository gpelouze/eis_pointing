#!/usr/bin/env python3

import warnings

import numpy as np

from .utils import eis

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
    raster = windata.int.copy()
    missing_places = (raster == windata.missing)
    raster[missing_places] = np.nan
    raster /= windata.exposure_time.reshape(-1, 1)
    wvl_min = wl0 - wl_width
    wvl_max = wl0 + wl_width
    i_min = np.argmin(np.abs(windata.wvl - wvl_min))
    i_max = np.argmin(np.abs(windata.wvl - wvl_max))
    raster = raster[:, :, i_min:i_max+1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        intensity = np.nanmean(raster, axis=-1)
    pointing = eis.EISPointing.from_windata(windata, use_wvl=False)
    data = eis.EISData(intensity, pointing)
    return data
