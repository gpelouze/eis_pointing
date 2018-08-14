# EIS pointing

Tools to correct the pointing of Hinode/EIS. 🛰

## Code structure

### Pipeline

`eis_pointing.py` performs all the following steps to determine the optimal
pointing data from EIS level 0 files. 

1. `prep.pro` generates `eis_l1_<date>.fits` from `eis_l0_<date>.fits`. Both
   files are found in the EIS data files and directory structure described in
   EIS Software Note #18.

2. `export_windata.pro` generates `io/windata/eis_windata_<date>.sav` from the
   previously generated `eis_l1_<date>.fits`. This file contains a `windata`
   structure generated by `eis_getwindata`. (See EIS Software Note #21.)

3. `eis_aia_emission.py` sums the previously saved EIS windata in wavelength to
   generate an intensity map that can be compared to AIA images. Currently, the
   approach is to only sum the FE XII 195.119 Å. In the future, this step
   should use the AIA effective area to more accurately match the AIA emission.
   Data are saved to `io/aia_emission/eis_aia_emission_<date>.fits`.

4. `eis_aia_registration.py` uses the AIA emission map generated at step 3, as
   well as AIA data retrieved from Medoc, to find the shift between EIS and
   AIA. Results from the alignment (ie. new EIS coordinates, and correlation
   cubes), are saved to `io/pointing/eis_pointing_<date>.fits`. At this step,
   diagnostics plots and a YAML file containing the results from the
   coregistration may also be saved to `io/pointing_verification/<date>_*`.

### Coregistration functions: `coregister/`

- `rasters.py` contains functions to register images in translation and
  rotation. 
- `slits.py` functions to register slit positions (ie. vertical columns in an
  image) separately.

### Utility functions shared by different components `utils/`

- `aia_raster.py`: defines `AIARasterGenerator` that builds synthetic rasters
  from AIA data. Also contains `SimpleCache` and `FileCache` (**TODO**).
- `cli.py`: argument parsing and output display.
- `eis.py` (**TODO**), `aia.py`: functions to handle native EIS and AIA data, filenames,
  and data queries. This does not take care of transformed data such as
  `AIARasterGenerator`.
  **TODO:** some functions from [`sol.data`] go here.
- `files.py`: manage local filenames (ie. those in `io/`); canonical EIS or AIA
  filenames are handled in `eis.py` or `aia.py`.
- `idl.py`: run IDL or SSW code from Python, load and format data returned by
  IDL. Contains `IDLFunction`, `SSWFunction` and `IDLStructure`.
- `num.py`: tools that extend numpy or scipy.
- `plots.py` (**TODO**): generate all plots from step 4.
- `sun.py`: generic solar computations.


[`align_images`]: https://git.ias.u-psud.fr/gpelouze/align_images
[`sol`]: https://git.ias.u-psud.fr/gpelouze/sol
[`sol.data`]: https://git.ias.u-psud.fr/gpelouze/sol/tree/master/data

## TODO

### Mandatory

- ~~refactor `utils.aia_raster`~~
- ~~refactor `utils.aia`~~
- ~~refactor `utils.eis`~~
- ~~implement functions to register images in rotation and translation in
  `coregister.rasters`, using components from [`align_images`].~~
- ~~implement functions to register slit positions separately in
  `coregister.slits`, using components from [`align_images`].~~
- refactor `utils.plots`
- refactor `__main__` of `to_integrate/coregister_eis_aia.py` into
  `eis_pointing.compute_pointing`, using `coregister` submodules.

### Optional

- implement `utils.aia_raster.FileCache`
- download EIS file from the MSSL archive if it is not found; this would
  require using `sol.data.eis.get_fits`.
