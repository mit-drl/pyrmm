# PYRMM Experiments

Scripts for conducting benchmarking experiments between LRMM algorithms, control barrier functions, and HJ-reachability

## Conda environment

We define yet another conda environment for running the pyrmm experiments. See `./environment.yml`. Create with:

```bash
# create odp conda environment to build upon 
#(start with odp because the package has over-specified, hard-to-boil-down dependencies)
cd ../optimized_dp  # navigate to optimized_dp submodule
conda env create -f environment.yml     # create the optimized_dp conda env as starting point due to optimized_dp over-specified dependencies
conda activate odp
pip install -e .    # install the odp package in the odp conda env

# clone the odp conda environment to new conda pyrmm_experiments env 
# and add addtional dependencies
conda deactivate
conda create --name pyrmm_experiments --clone odp   # clone the odp conda environment to start the pyrmm_experiments environment
conda activate pyrmm_experiments
cd ../experiments   # navigate to experiments directory
pip install -r requirements.txt    # install additional pyrmm-experiments dependencies on top of the odp dependencies

# Add ompl python bindings to pyrmm_experiments conda env 
#(your exact command will look different depending on where ompl is installed)
echo "/home/ross/ompl-1.5.2/py-bindings" >> ~/miniconda3/envs/pyrmm_experiments/lib/python3.8/site-packages/ompl.pth

# optional
conda remove --name odp --all       # remove the stand-alone odp conda environment to clean-up potentially unused envs
conda install ipython               # convience of using ipython command line instead of python
```

This addition conda environment was created because we need an environment with _some_ of pyrmm's dependecies (e.g. hydra-zen), all of pyrmm-envs's dependencies, as well as dependencies for running control barrier functions (e.g. cvxopt) and HJ-reachability (e.g. optimized_dp). Instead of putting all of these new dependencies in the pyrmm conda environment defined in the top level of this repo (which is already growing into an ungainly dependency rat's nest with all of the procedural generation and machine learning tools), we thought it cleaner to just create a separate conda environment for running experiments that compare pyrmm, CBF, and HJ-reach algorithms.

Note: unfortunately `optimized_dp`, which is currently used for HJ-reach algorithms, has a poorly defined dependency structure consisting of an entire pip-freeze. Therefor we have to start with this over-specified set of dependencies


