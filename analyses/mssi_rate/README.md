# MSSI Rate

This directory contains an analysis to determine the rate of Multiple-Scatter-Single-Ionisation (MSSI) events from gamma-rays originating from detector components in the LUX-ZEPLIN experiment.
MSSI events are critical to understand because they are electron-recoil (ER) events which can appear nuclear-recoil (NR) like as not all of the charge is collected.
This typically occurs if the gamma-ray scatters more than once, and one of the scatters is in a region of the detector that is charge insensitive.

In this analysis, simulations are probed to determine the how many MSSI events there are from gamma simulations.
Each simulation file is a radioactive decay originating from a detector component: for example, U-238 CathodeGridWires are all decays from the U-238 decay chain originating from the Cathode Grid Wires.
The rate in the simulations is from [Eur. Phys. J. C 80, 1044](https://link.springer.com/article/10.1140/epjc/s10052-020-8420-x).

## Additional AGC features
All of the analysis cuts can be run in three ways:
1. array operations (using numpy)
2. using numba.njit
3. using nuba.cuda

The analysis cuts follow those detailed in [Phys. Rev. Lett. 135, 011802](https://journals.aps.org/prl/abstract/10.1103/4dyc-z8zf).

This analysis has been presented at [GridPP53 & SWIFT-HEP09](https://indico.cern.ch/event/1476120/contributions/6442362/).

