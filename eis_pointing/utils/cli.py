#!/usr/bin/env python3

import datetime
import re

import yaml

from ..coregister.tools import OffsetSet

def print_now(*arg, **kwargs):
    ''' print('[time_stamp]', *arg, **kwarg)

    **kwargs are passed to print.
    '''
    msg_template = '[{time_stamp}]'
    timestamp = '[{}]'.format(datetime.datetime.now())
    print(timestamp, *arg, **kwargs)

def load_corr_steps(filename):
    with open(filename) as f:
        y = yaml.load(f)
    offsetset_re = re.compile(
        '^OffsetSet\('
        '\s*\(\s*(?P<start>[\d+-.]+)\s*,'
             '\s*(?P<stop>[\d+-.]+)\s*\)\s*,'
        '\s*(?:(?:number\s*=\s*(?P<number>[\d+-]+)\s*)|'
              '(?:step\s*=\s*(?P<step>[\d+-.]+)\s*))'
        '\s*\)\s*$')
    for i, step in enumerate(y['steps']):
        for k, v in step.items():
            try:
                m = offsetset_re.match(v)
            except (TypeError, AttributeError):
                m = None
            if m:
                start = float(m.group('start'))
                stop = float(m.group('stop'))
                number = m.group('number')
                step = m.group('step')
                if number:
                    number = int(number)
                if step:
                    step = float(step)
                offset_set = OffsetSet((start, stop), number=number, step=step)
                y['steps'][i][k] = offset_set
    return y
