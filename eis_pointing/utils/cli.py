#!/usr/bin/env python3

import argparse
import datetime

def print_now(*arg, **kwargs):
    ''' print('[time_stamp]', *arg, **kwarg)

    **kwargs are passed to print.
    '''
    msg_template = '[{time_stamp}]'
    timestamp = '[{}]'.format(datetime.datetime.now())
    print(timestamp, *arg, **kwargs)

def get_setup():
    parser = argparse.ArgumentParser(
        description='Coregister EIS cube.')
    parser.add_argument(
        'filename',
        type=str,
        help=("the name of the level 0 EIS file, "
            "eg. 'eis_l0_20100815_192002'"))
    parser.add_argument(
        'aia_band',
        type=str,
        help=("the AIA band to use for the coalignment, "
            "eg. '193' -- CURRENTLY IGNORED, defaults to 193"))
    parser.add_argument(
        '-c', '--cores',
        type=int,
        default=4,
        help='maximum number of cores used for parallelisation')
    args = parser.parse_args()

    return args
