# TPC Extraction Efficiency

This directory contains an analysis to calculate the electron extraction efficiency in the LUX-ZEPLIN experiment.

In this analysis, Kr83m events from a calibration dataset are used, and the number of electrons as a function of radial position are determined.

## Additional AGC features
This analysis makes use of a `Store` object (which inherits from a `dict`). 
This allows for the resultant dictionaries to be easily added together, where the dictionaries contain `awkward` arrays, histograms, and other objects.


## Notes
This AGC will only work on NERSC and has not been setup for Local or Dirac yet