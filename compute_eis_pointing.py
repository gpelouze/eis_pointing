#!/usr/bin/env python3

import argparse

import eis_pointing

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Determine the pointing of Hinode/EIS.')
    parser.add_argument(
        'filename',
        type=str,
        nargs='+',
        help=("The names of the level 0 EIS files, "
            "eg. 'eis_l0_20100815_192002'."))
    parser.add_argument(
        '-s', '--steps-file',
        type=str,
        help=('Path to a yaml file containing the registration steps.'))
    parser.add_argument(
        '--io',
        type=str,
        default='io',
        help=('Directory where output files are written, default: ./io.'))
    parser.add_argument(
        '-c', '--cores',
        type=int,
        default=4,
        help='Maximum number of cores used for parallelisation, default: 4.')
    parser.add_argument(
        '--cache-aia-data',
        action='store_true',
        help='Cache the AIA data to a file. This uses a lot of storage, '
             'but speeds things up when the same raster is aligned for '
             'the second time.')
    args = parser.parse_args()

    eis_pointing.compute(
        *args.filename,
        cores=args.cores,
        io=args.io,
        steps_file=args.steps_file,
        cache_aia_data=args.cache_aia_data)
