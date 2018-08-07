#!/usr/bin/env python3

import numpy as np

from . import eis

def choose_solar_radius(wvl, instrument='aia'):
    ''' Choose best solar radius based on the AIA or EIS band of the event.

    Parameters
    ==========
    wvl : int or str
        The observation wavelength. For EIS, strings must have a {.3f} format.
    instrument : str
        The name of the instrument used for the observation.
    '''

    solar_radii_aia = {
        '171': 1.0,
        '195': 1.005,
        '284': 1.005,
        '304': 1.005,
        '94' : 1.01,
        '131': 1.005,
        '193': 1.005,
        '211': 1.005,
        '335': 1.005,
        }

    try:
        instrument = instrument.lower()
        if instrument == 'aia':
            aia_wvl = str(wvl)
        elif instrument == 'eis':
            if type(wvl) is not str:
                wvl = '{:.3f}'.format(wvl)
            aia_wvl = eis.get_aia_channel(wvl)
        else:
            msg = '{} is not a valid instrument'.format(instrument)
            raise ValueError(msg)
        solar_radius = solar_radii_aia[aia_wvl]

    except KeyError:
        msg = 'Could not find wavelength {} for instrument {}.'
        msg = msg.format(wvl, instrument)
        raise ValueError(msg)

    return solar_radius
