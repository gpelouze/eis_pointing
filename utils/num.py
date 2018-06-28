#!/usr/bin/env python3

import numpy as np

def chunks(l, n):
    ''' Split list l in chunks of size n.

    http://stackoverflow.com/a/1751478/4352108
    '''
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def recarray_to_dict(recarray, lower=False):
    ''' Transform a recarray containing a single row to a dictionnary.

    If lower is True, apply str.lower() to all keys.
    '''
    while recarray.dtype is np.dtype('O'):
        recarray = recarray[0]
    assert len(recarray) == 1, 'structure contains more than one row'
    array = dict(zip(recarray.dtype.names, recarray[0]))
    if lower:
        array = {k.lower(): v for k, v in array.items()}
    return array
