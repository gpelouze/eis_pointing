#!/usr/bin/env python3

import datetime
import itertools

def progress_iterator(iterable, nitems=None, msg='{count}', **print_kwargs):
    ''' Print progress while yielding items of an iterable.

    Parameters
    ==========
    iterable :
        The iterable to be yielded.
    nitems : int or None (default: None)
        The number of items in the iterable.
        When `None`, try to use `len(iterable)`. If this fails, set it to 1.
    msg : string or function
        The format of the message to print before yielding each item of the
        iterable.
        Any string is transformed to the function `f(*) -> msg.format(*)`.
        The function is passed 3 keyword arguments:
        - 'count': a counter starting at 1 and incremented each time an element
          is yielded.
        - 'nitems': the value of `nitems`, transformed as described above.
        - 'progress': `count / nitems`.
    **print_kwargs :
        Passed to `print()` when printing the message.

    Yields
    ======
    The items of `iterable`.

    Examples
    ========
    >>> iterable = progress_iterator(range(5), msg='{progress:.1%}')
    >>> for v in iterable:
    ...     do_stuff(v)
    20.0%
    40.0%
    60.0%
    80.0%
    100.0%
    '''
    if nitems is None:
        try:
            nitems = len(iterable)
        except TypeError:
            nitems = 1
    if isinstance(msg, str):
        msg_str = msg
        msg = lambda **kwargs: msg_str.format(**kwargs)
    for count, v in zip(itertools.count(start=1), iterable):
        print(msg(count=count, progress=count / nitems, nitems=nitems), **print_kwargs)
        yield v

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
