#!/usr/bin/env python3

import datetime

def eta_iterator(iterable, nitems=None, msg='{count}', **print_kwargs):
    start = datetime.datetime.now()
    print('start', start)
    def msg_function(**kwargs):
        now = datetime.datetime.now()
        elapsed = now - start
        p = kwargs['progress']
        eta = start + elapsed / p
        return msg.format(eta=eta, **kwargs)
    return progress_iterator(iterable, nitems=nitems, msg=msg_function, **print_kwargs)
