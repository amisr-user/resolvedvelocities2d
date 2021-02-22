# resolvedvelocities2d

This code is a python translation of a matlab codebase. See "translation_documentation.txt" for details.

This code is very slow.

## Example usage
This codebase requires Python 2.7 to run and all of the packages listed in `requirements.txt`.

1) Update the paths in `example/config_20170302.002_lp_1min-cal.ini` and `convert_mats_to_hdf/mats2hdf.py`.

2) Process the data file in `data/` directory:
    mkdir -p run_results/20170302.002_lp_1min-cal_2dVEF_001001
    python -u derive_from_fitted.py example/config_20170302.002_lp_1min-cal.ini data/20170302.002_lp_1min-cal.h5

  This will take a long time to complete.

3) Convert the output into HDF5 format:
    python convert_mats_to_hdf/mats2hdf.py

This will produce a file called `20170302.002_lp_1min-cal_2dVEF_001001.h5`.