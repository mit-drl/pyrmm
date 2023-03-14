# Change Log

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

## [v0.0.5] - 2023-03-13

**Major Refactor**

+ Added first implementation and tests of Inferred Risk Barrier Functions (IRBF, although currently referred to in code as "CBFLRMM" and "RiskCBF")
+ Abstracted key SystemSetup functions out of child-class parent class, e.g. `sample_control_numpy`, `propagate_path`, `path_ompl_to_numpy`, and `path_numpy_to_ompl`
+ Refactoring all system setups to have conversion functions `state_ompl_to_numpy`, `state_numpy_to_ompl`, `control_ompl_to_numpy`, `control_numpy_to_ompl`
+ Completely eliminated `StatePropagator` classes in favor of a single `propagate_path` function

## [v0.0.4] - 2022-09-14

Major update. complete implementation of parallel autonomy case study with dubins-4d vehicle compared to CBF and HJ-reach agents