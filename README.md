# SimOfEmbryoCluster
Python version simulation code of starfish embryo cluster (ref. [doi.org/10.1038/s41586-022-04889-6](https://doi.org/10.1038/s41586-022-04889-6)

# How to use
1. Install package *numba, numpy, scipy, matplotlib*
2. Run *CalibratedDiskModel.py* and you will get a folder Data/**simID** containing ode solution and parameters
3. Run *OmegaCalculate.py* at the same path and you will get the animation and rorating rates of all embryos at all times
