#!/usr/bin/env python3

from hashlib import md5
import os
import random
import subprocess
import sys

class IDLFunction(object):
    ''' Run an IDL function with arguments from Python, thus avoiding
    headaches.

    Parameters
    ==========

    function : str
        The name of the IDL function to run.
    arguments : list (default: [])
        A list of arguments in text (or at least convertible to str) format.
    '''

    _files_templates = {
        'csh': '''
            #!/bin/csh -f
            idl {self.filename_base}.bat
            ''',
        'bat': '''
            {self.function}{self.arguments_str}
            exit
            ''',
        }

    def __init__(self, function, arguments=[]):
        self.function = function
        self.arguments = arguments
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
                template.format(**locals()),
                )
            for ext, template in self._files_templates.items()}

    def _write_files(self):
        ''' Write temporary run files to the current directory.
        '''
        for (filename, commands) in self.files.values():
            with open(filename, 'w') as f:
                f.write(commands)

    def clean(self):
        ''' Clean the temporary run files that were written to the current
        directory.
        '''
        for (filename, _) in self.files.values():
            os.remove(filename)

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

    _files_templates = {
        'csh': '''
            #!/bin/csh -f
            setenv SSW /usr/local/ssw
            setenv SSW_INSTR {self.instruments}
            source $SSW/gen/setup/setup.ssw
            sswidl {self.filename_base}.bat
            ''',
        'bat': '''
            {self.function}{self.arguments_str}
            exit
            ''',
        }

    def __init__(self, *args, instruments='nox', **kwargs):
        self.instruments = instruments
        super().__init__(*args, **kwargs)
