# risk_metric_maps

## Installation

### 3rd Party Library Installations

__OMPL__
TODO

__Blender__
TODO

__mazegenerator__

We are using [mazegenerator](https://github.com/razimantv/mazegenerator) as a submodule which needs to be initialized
```
git submodule update --init --recursive
```

Then mazegenerator must be built
```
cd mazegenerator/src
make
```


To use the pyrmm library, clone this repo and install with:
```
pip install -e .
```

To develop the pyrmm library it is recommend to use the conda environment defined here

```
conda env create -f environment.yml
conda activate pyrmm
```

## Example Usage: Dubins Vehicle in PPM Mazes

### Procedural Generation of Obstacle Space

### Approximate Risk Metric Data Generation

### Risk Metric Model Training

