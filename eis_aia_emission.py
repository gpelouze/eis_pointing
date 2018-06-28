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
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        intensity = np.nanmean(raster, axis=-1)
    return intensity
