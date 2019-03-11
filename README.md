# EIS pointing

Tools to correct the pointing of Hinode/EIS.

## Usage

### From the command line

This tool can be run from the command line by calling
`compute_eis_pointing`:

~~~
usage: compute_eis_pointing [-h] [-s STEPS_FILE] [--io IO] [-c CORES]
                            [--cache-aia-data]
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

**Examples (command line):**

~~~bash
compute_eis_pointing -c16 eis_l0_20140810_042212
compute_eis_pointing --steps-file steps/shift_only.yml eis_l0_20140810_042212
~~~

### As a Python module

The tool can also be used from within a Python script, using
`eis_pointing.compute()`.

~~~
compute(*filename, steps_file=None, io='io', cores=4, cache_aia_data=False)
    Perform all computation steps to determine the optimal EIS pointing.

    Parameters
    ==========
    filename : list
        The names of the level 0 EIS files, eg. 'eis_l0_20100815_192002'.
    steps_file : str or None (default: None)
        Path to a yaml file containing the registration steps.
    io : str (default: 'io')
        Directory where output files are written.
    cores : int (default: 4)
        Maximum number of cores used for parallelisation.
    cache_aia_data : bool (default: False)
        Cache the AIA data to a file. This uses a lot of storage, but speeds
        things up when the same raster is aligned for the second time.
~~~

**Examples (Python):**

~~~python
import eis_pointing
eis_pointing.compute('eis_l0_20140810_042212', cores=16)
eis_pointing.compute('eis_l0_20140810_042212', steps_file='steps/shift_only.yml')
~~~

## Installation

Install the latest release by running: `pip install eis_pointing`.

Alternatively, the latest version can be installed from GitHub by cloning this
repository with `git clone https://github.com/gpelouze/eis_pointing`, then
running `cd eis_pointing`, and `pip install .`.

### Optional: install SolarSoft

Before computing the optimal pointing, this tool can download, prepare, and
export the EIS data by calling external IDL routines from [SolarSoft]. For
these features to be available, a functioning installation of SolarSoft
containing the EIS instrument is required. [Install SolarSoft][install-ssw],
and set the environment variable `$SSW` to your installation path (by default,
SolarSoft is assumed to be installed installed into `/usr/local/ssw`).

It is perfectly fine not to install or configure SolarSoft to run with this
tool. In this case, you will need to manually download the EIS level0 FITS,
prepare them into level1 FITS, and save a windata structure containing the
Fe XII 195.119 Å line to a `.sav` file placed in
`<io directory>/windata/eis_windata_<date>.sav`.
See [pipeline](#pipeline) for details on how to do this.

[SolarSoft]: http://www.ascl.net/1208.013
[install-ssw]: http://www.lmsal.com/solarsoft/sswdoc/solarsoft/ssw_install_howto.html

## Customisation

The registration steps used to find the optimal pointing can be customised in a
YAML file, and passed to `eis_pointing` using the `--steps-file` parameter (see
examples above). The file should have a top-level key named `steps` that
contains a list of registration steps. Each step must specify at least a
`type`, chosen between `shift`, `rotshift`, and `slitshift`.

By default, EIS data are coaligned with synthetic AIA raster. To coalign with a
single AIA image, add the top-level key `single_aia_frame: True`. In this case,
the reference AIA image chosen at the middle of the EIS raster.

See files in [`steps/`](steps/) for examples.

When no file is specified, the default behaviour is the same as using
[`steps/full_registration.yml`](steps/full_registration.yml).

## Code structure

### Pipeline

All the steps required to determine the optimal pointing data from EIS level 0
files are defined in `driver.py`. The appropriate functions are called by the
executable `compute_eis_pointing` when using the tool from the CLI, or by
`eis_pointing.compute()` when using it as a Python module.

1. **Download data** Download the required EIS level 0 FITS,
   and place them in the EIS data files and directory structure
   described in [EIS Software Note #18][swn18]
   (eg.  `$HINODE_DATA/eis/level0/2014/08/10/eis_l0_20140810_042212.fits`).

2. **Prepare data** Generate EIS level 1 FITS from level 0, and save it to the
   EIS data files and directory structure
   (eg. `$HINODE_DATA/eis/level1/2014/08/10/eis_l1_20140810_042212.fits`).
   Performed by `eis_pointing/prep.pro`, which calls the SolarSoft routine
   `eis_prep.pro`.

3. **Export windata** Save a `windata` structure containing the
   Fe XII 195.119 Å line, obtained using the SolarSoft function
   `eis_getwindata` (see [EIS Software Note #21][swn21]).
   The structure is saved to `<io>/windata/eis_windata_<date>.sav`
   (eg. `./io/windata/windata_20140810_042212.sav`).
   Performed by `eis_pointing/export_windata.pro`.

---

**Alternative to steps 1-3 without SolarSoft** If SolarSoft is not installed or
configured, you will need to separately generate a windata structure containing
the Fe XII 195.119 Å line, and save it to
`<io>/windata/eis_windata_<date>.sav`.

Example (SSW):

~~~IDL
wd = eis_getwindata('eis_l1_20140810_042212.fits', 195.119, /refill)
save, wd, filename='./io/windata/windata_20140810_042212.sav'
~~~

Once this is done, run the tool normally, either from the [command
line](#from-the-command-line), or as a [Python module](#as-a-python-module). It
will detect the existing `.sav` file, and skip steps 1-3.

---

4. **Compute the EIS emission**
   Generate an intensity map of the Fe XII 195.119 Å line by summing
   the spectra between 194.969 and 195.269 Å. Data are saved to
   `<io>/eis_aia_emission/eis_aia_emission_<date>.fits` (eg.
   `./io/eis_aia_emission/eis_aia_emission_20140810_042212.fits`).
   Performed by `eis_pointing.eis_aia_emission.compute()`.

5. **Determine the optimal pointing** Determine the optimal pointing for EIS
   using the intensity map generated at the previous step, and AIA 193 data
   retrieved from [Medoc][medoc-homepage] as a reference.  (The AIA FITS are
   downloaded to `./sdo/aia`, or to `$SDO_DATA/aia/` if the environment
   variable `$SDO_DATA` is set set.)
   Results from the alignment (ie. new EIS coordinates) are saved to
   `<io>/pointing/eis_pointing_<date>.fits`. Diagnostics plots, correlation
   cubes, as well as a YAML file containing the results from the coregistration
   are also saved to `<io>/pointing_verification/<date>/`.  Performed by
   `eis_pointing.eis_aia_registration.optimal_pointing()`.

[swn18]: ftp://sohoftp.nascom.nasa.gov/solarsoft/hinode/eis/doc/eis_notes/18_FILES/eis_swnote_18.pdf
[swn21]: ftp://sohoftp.nascom.nasa.gov/solarsoft/hinode/eis/doc/eis_notes/21_WINDATA/eis_swnote_21.pdf
[medoc-homepage]: https://idoc.ias.u-psud.fr/MEDOC


### Coregistration functions: `eis_pointing.coregister`

- `images` contains functions to register images in translation, relatively to
  another image.
- `rasters` contains functions to register images in translation and rotation,
  relatively to a synthetic raster.
- `slits` functions to register slit positions (ie. vertical columns in an
  image) separately, relatively to a synthetic raster.
- `tools` functions shared among the previous submodules.

### Functions shared by different components `eis_pointing.utils`

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

## License

This package is released under a MIT open source licence. See `LICENSE.txt`.
