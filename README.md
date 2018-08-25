# EIS pointing

Tools to correct the pointing of Hinode/EIS. 🛰

## Usage

### From the command line

This tool can be run from the command line by calling
`compute_eis_pointing.py`:

~~~
usage: compute_eis_pointing.py [-h] [-s STEPS_FILE] [--io IO] [-c CORES]
                               filename [filename ...]

Determine the pointing of Hinode/EIS.

positional arguments:
  filename              The names of the level 0 EIS files, eg.
                        'eis_l0_20100815_192002'.

optional arguments:
  -h, --help            show this help message and exit
  -s STEPS_FILE, --steps-file STEPS_FILE
                        Path to a yaml file containing the registration steps.
  --io IO               Directory where output files are written,
						default: ./io.
  -c CORES, --cores CORES
                        Maximum number of cores used for parallelisation,
                        default: 4.
  --cache-aia-data      Cache the AIA data to a file. This uses a lot of
                        storage, but speeds things up when the same raster is
                        aligned for the second time.
~~~

**Examples:**

~~~bash
./compute_eis_pointing.py -c16 eis_l0_20140810_042212
./compute_eis_pointing.py --steps-file steps/shift_only.yml eis_l0_20140810_042212
~~~

### As a Python module

The tool can also be used from within a Python script, using
`eis_pointing.compute()`.

~~~
compute(*filename, cores=4, io='io', steps_file=None, cache_aia_data=False)
    Perform all computation steps to determine the optimal EIS pointing.

    Parameters
    ==========
    filename : list
        The names of the level 0 EIS files, eg. 'eis_l0_20100815_192002'.
    cores : int (default: 4)
        Maximum number of cores used for parallelisation.
    io : str (default: 'io')
        Directory where output files are written.
    steps_file : str or None (default: None)
        Path to a yaml file containing the registration steps.
    cache_aia_data : bool (default: False)
        Cache the AIA data to a file. This uses a lot of storage, but speeds
        things up when the same raster is aligned for the second time.
~~~

**Examples:**

~~~python
import eis_pointing
eis_pointing.compute('eis_l0_20140810_042212', cores=16)
eis_pointing.compute('eis_l0_20140810_042212', steps_file='steps/shift_only.yml')
~~~

## Installation

1. Clone this repository.
2. Satisfy the following python dependencies: astropy, numpy, scipy,
   matplotlib, dateutil, pyyaml, [pySitools2], and [align_images].
3. If needed, install Solar Soft making sure that EIS is in the instrument list. If SSW
   is not installed in `/usr/local/ssw`, set the environment variable `$SSW` to
   the appropriate path.
4. (Optional) Place `compute_eis_pointing.py` in your `$PATH`, and
   `eis_pointing` in your `$PYTHONPATH`.

[pySitools2]: http://medocias.github.io/pySitools2_1.0/
[align_images]: https://git.ias.u-psud.fr/gpelouze/align_images

## Customisation

The registration steps used to find the optimal pointing can be customised in a
yaml file, and passed to `eis_pointing` using the “step file” parameter (see
examples above). The file should have a top-level key named `step`, containing
a list of registration steps. Each step must specify at least a `type`, chosen
between `shift`, `rotshift`, and `slitshift`.

By default, EIS data are coaligned with synthetic AIA raster. To coalign with a
single AIA image, add the top-level key `single_aia_frame: True`. In this case,
the reference AIA image chosen in the middle of the EIS raster.

See files in `steps/` for examples.

When no file is specified, the default behaviour is the same as using
`steps/full_registration.yml`.

## Code structure

### Pipeline

All the steps required to determine the optimal pointing data from EIS level 0
files are defined in `driver.py`. The appropriate functions are called by
`compute_eis_pointing.py` when using the tool from the CLI, and by
`eis_pointing.compute()` when using it as a Python module.

1. **Download data** Download the required EIS level 0 FITS, and place them in
   the EIS data files and directory structure described in EIS Software
   Note #18.

2. **Prepare data** Generate EIS level 1 FITS from level 0. Both files are
   found in the EIS data files and directory structure. Performed by
   `eis_pointing/prep.pro`.

3. **Export windata** Generate `{io}/windata/eis_windata_<date>.sav` from EIS
   level 1 FITS. This file contains a `windata` structure generated by the SSW
   function `eis_getwindata` (see EIS Software Note #21). Performed by
   `eis_pointing_/export_windata.pro` 

4. **Compute the EIS emission** Sum the EIS windata in wavelength to generate
   an intensity map of line Fe XII 195.119 Å. Data are saved to
   `{io}/eis_aia_emission/eis_aia_emission_<date>.fits`. Performed by
   `eis_pointing.eis_aia_emission.compute()`.

5. **Determine the optimal pointing** Determine the optimal pointing for EIS
   using the intensity map generated at the previous step, and AIA 193 data
   retrieved from Medoc as a reference.  Results from the alignment (ie. new
   EIS coordinates) are saved to `io/pointing/eis_pointing_<date>.fits`.
   Diagnostics plots, correlation cubes, as well as a YAML file containing the
   results from the coregistration are also saved to
   `{io}/pointing_verification/<date>_*`.  Performed by
   `eis_aia_registration.py`.

### Coregistration functions: `eis_pointing.coregister`

- `images` contains functions to register images in translation, relatively to
  another image.
- `rasters` contains functions to register images in translation and rotation,
  relatively to a synthetic raster.
- `slits` functions to register slit positions (ie. vertical columns in an
  image) separately, relatively to a synthetic raster.
- `tools` functions shared among the previous submodules.

### Utility functions shared by different components `eis_pointing.utils`

- `aia_raster`: defines `AIARasterGenerator` that builds synthetic rasters
  from AIA data. Also contains `SimpleCache` and `FileCache`.
- `cli`: argument parsing and output display.
- `eis`, `aia.py`: functions to handle native EIS and AIA data, filenames,
  and data queries. This does not take care of transformed data such as
  `AIARasterGenerator`.
- `files`: manage local filenames (ie. those in `io/`); canonical EIS or AIA
  filenames are handled in `eis.py` or `aia.py`.
- `idl`: run IDL or SSW code from Python, load and format data returned by
  IDL. Contains `IDLFunction`, `SSWFunction` and `IDLStructure`.
- `num`: tools that extend numpy or scipy.
- `plots`: help generate plots at step 4.
- `sun`: generic solar computations.
