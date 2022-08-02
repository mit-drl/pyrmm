# risk_metric_maps

## Installation

### 3rd Party Library Installations

__OMPL__
TODO

__Blender__

We use [blender](https://www.blender.org/) in order to convert `.stl` meshes produced by [`pcg-gazebo`](https://github.com/boschresearch/pcg_gazebo/) into `.obj` files that can be loaded by [pybullet](https://pybullet.org/wordpress/)

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

__v-hacd__

We use the [V-HACD](https://github.com/kmammou/v-hacd) library in order to create convex decompositions of procedurally generated rooms from [pcg-gazebo](https://github.com/boschresearch/pcg_gazebo/)

Once the submodule is initialized, build the package with
```
cd v-hacd/app/
cmake -DCMAKE_BUILD_TYPE=Release CMakeLists.txt
cmake --build .
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

