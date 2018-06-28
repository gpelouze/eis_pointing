#!/usr/bin/env python3

import numpy as np

def chunks(l, n):
    ''' Split list l in chunks of size n.

    http://stackoverflow.com/a/1751478/4352108
    '''
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))
