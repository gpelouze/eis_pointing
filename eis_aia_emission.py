#!/usr/bin/env python3

import warnings

import numpy as np

def compute(windata, aia_band):
    ''' Compute synthetic emission for a given AIA band using EIS data.

    Parameters
    ==========
    windata : idl.IDLStructure
        Windata structure containing the wavelength windows necessary to compute
        the AIA emission.
    aia_band : str
        The AIA band for which to compute the synthetic intensity.
        WARNING: currently, this parameter is ignored and the function sums
        the intensity over all wavelengths.
    '''
    raster = windata.int.copy()
    missing_places = (raster == windata.missing)
    raster[missing_places] = np.nan
    raster /= windata.exposure_time.reshape(-1, 1)
    # select Fe XII 195.119
    wvl_min, wvl_max = 194.961, 195.261 # FIXME
    i_min = np.argmin(np.abs(windata.wvl - wvl_min))
    i_max = np.argmin(np.abs(windata.wvl - wvl_max))
    raster = raster[:, :, i_min:i_max+1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        intensity = np.nanmean(raster, axis=-1)
    return intensity
