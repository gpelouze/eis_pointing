#!/usr/bin/env python3

from hashlib import md5
import os
import random
import subprocess
import sys

import numpy as np
import scipy.io as sio

from . import num

class IDLFunction(object):
    ''' Run an IDL function with arguments from Python, thus avoiding
    headaches.

    Parameters
    ==========

    function : str
        The name of the IDL function to run.
    arguments : list (default: [])
        A list of arguments in text (or at least convertible to str) format.
    cwd : str (default: '.')
        The directory in which to execute the function.
    '''

    _files_templates = {
        'csh': [
            '#!/bin/csh -f',
            'idl {self.filename_base}.bat'
            ],
        'bat': [
            '{self.function}{self.arguments_str}',
            'exit',
            ],
        }

    def __init__(self, function, arguments=[], cwd='.'):
        self.function = function
        self.arguments = arguments
        self.cwd = cwd
        # transform list of arguments into a string
        if arguments == []:
            self.arguments_str = ''
        else:
            self.arguments_str = [arg.__repr__() for arg in arguments]
            self.arguments_str = ', ' + ', '.join(self.arguments_str)
        # generate unique name for command files
        unique_key = self.arguments_str + str(random.random())
        unique_key = unique_key.encode('utf-8')
        unique_key = md5(unique_key).hexdigest()
        self.filename_base = '.{function}_{unique_key}'.format(**locals())
        # interpolate templates
        self.files = {ext: (
                self.filename_base + '.' + ext,
                '\n'.join(template).format(**locals()),
                )
            for ext, template in self._files_templates.items()}

    def _write_files(self):
        ''' Write temporary run files to the current directory.
        '''
        for (filename, commands) in self.files.values():
            with open(os.path.join(self.cwd, filename), 'w') as f:
                f.write(commands)

    def clean(self):
        ''' Clean the temporary run files that were written to the current
        directory.
        '''
        for (filename, _) in self.files.values():
            os.remove(os.path.join(self.cwd, filename))

    def run(self):
        ''' Run IDL function, returning stdout and stderr.
        '''

        # write temporary run files
        self._write_files()

        try:
            # run function
            p = subprocess.Popen(
                ['csh', self.files['csh'][0]],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.cwd,
                )
            # save and print stdout
            stdout = []
            for line in iter(p.stdout.readline, b''):
                line = line.decode()
                sys.stdout.write(line)
                stdout.append(line)
            stdout = ''.join(stdout)
            # save stderr
            _, stderr = p.communicate()
            print(79 * '-')
            print(stderr.decode())

        finally:
            # Probably useless since IDL always exits with status 0. Even on
            # errors. Yes.
            if p.returncode != 0:
                m = 'IDLFunction {0} returned with status {1}.\n'
                m += 'Leaving {2} and {3}.'
                m = m.format(
                    self.function,
                    p.returncode,
                    self.files['csh'][0],
                    self.files['bat'][0],
                    )
                raise RuntimeError(m)
            else:
                # delete temporary run files
                self.clean()

        print(p.returncode)

        return stdout, stderr

class SSWFunction(IDLFunction):
    ''' Run a Solar Software (SSW) function.

    Parameters
    ==========

    function : str
        The name of the IDL function to run.
    arguments : list (default: [])
        A list of arguments in text (or at least convertible to str) format.
    instruments : str (default: nox)
        The SSW instruments to load.
    '''

    IDLFunction._files_templates['csh'] = [
        '#!/bin/csh -f',
        'setenv IDL_PATH \+$IDL_DIR/lib:\+$IDL_PATH',
        'setenv SSW {self.ssw_path}',
        'setenv SSW_INSTR {self.instruments}',
        'source $SSW/gen/setup/setup.ssw',
        'sswidl {self.filename_base}.bat',
        ]

    def __init__(self, *args,
            instruments='nox', ssw_path='/usr/local/ssw',
            **kwargs):
        self.instruments = instruments
        self.ssw_path = ssw_path
        super().__init__(*args, **kwargs)

class IDLStructure(dict):
    def __init__(self, data, var='structure'):
        ''' Reproduce an IDL structure.

        Input
        =====
        data : str, scipy.io.idl.AttrDict or np.recarray
            Either the path to a .sav, an AttrDict returned by
            scipy.io.readsav, or a record array.
        key : str (default: 'structure')
            The name of the variable to extract from the .sav. This is only
            required when the .sav contains multiple variables.

        Raises
        ======
        ValueError
            When data contains more than one variable no `var` argument is
            given.
        '''
        if isinstance(data, str):
            data = sio.readsav(data)
        if isinstance(data, sio.idl.AttrDict):
            if len(data.keys()) == 1:
                data = list(data.values())[0]
            else:
                try:
                    data = data[var]
                except KeyError:
                    m = ".sav contains multiple variables but you didn not" \
                        + "specify the name of the one you want to extract."
                    raise ValueError(m)
            data = num.recarray_to_dict(data, lower=True)
        if isinstance(data, np.recarray):
            data = num.recarray_to_dict(data, lower=True)
            pass
        if isinstance(data, dict):
            pass
        else:
            raise ValueError('Input type is not supported')
        # self.data = data
        super().__init__(data)

    def __repr__(self):
        ''' Display a list with one key per line. Columns are:

        - key
        - type(dimension), where dimension is either len (dict, list, tuple),
          or shape (np.ndarray)
        - sample, usually the 1st element of the list, tuple, array, etc.
          Nothing for dictionnaries.
        - dtype, when applicable, ie for np.ndarrays
        '''
        ret = self.__class__.__name__ + '({\n'
        for k, v in self.items():
            k = "'{}'".format(k)
            dt = ''
            sample = ''
            if isinstance(v, np.ndarray):
                disp = '{}{}'.format(type(v).__name__, v.shape)
                dt = v.dtype.name
                try:
                    v = v.flat[0]
                except AttributeError:
                    pass
                if isinstance(v, str):
                    sample = v
                elif np.issubdtype(type(v), np.floating):
                    sample = '{:.3f}'.format(v)
                elif np.issubdtype(type(v), np.integer):
                    sample = str(v)
                elif v is np.ma.masked:
                    sample = str(v)
            elif isinstance(v, list or tuple):
                disp = '{}{}'.format(type(v).__name__, len(v))
                disp = type(v)().__repr__()
                l, r = tuple(disp)
                disp = '{}{}{}'.format(l, len(v), r)
                sample = v[0]
                while isinstance(v, list or tuple):
                    v = v[0]
                sample = v
            elif isinstance(v, dict):
                disp = "dict({})".format(len(v.keys()))
                sample = v[0]
            else:
                disp = v.__repr__()
            if sample:
                sample = '({:>17.17}…)'.format(sample)
            else:
                sample = ' ' * 20
            ret += "  {k:20}: {disp:30} {sample} {dt:10}\n".format(**locals())
        ret += '})'
        return ret

    def __getattr__(self, a):
        return self[a]

    def __setattr__(self, a, v):
        self[a] = v
