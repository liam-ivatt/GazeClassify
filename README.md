# IvattGaze

> Eye-gaze data collection, calibration, and classification toolkit.

## Overview

IvattGaze provides utilities and example models for collecting eye-gaze data, calibrating a landmark model, running experiments, and training or evaluating classifiers.

## Features

- Data collection and calibration scripts
- GUI for live data capture and inspection
- Example machine learning models for gaze classification

## Repository Structure

- [main.py](main.py) — entry point / example runner
- [gui.py](gui.py) — graphical interface for live capture and visualization
- [calibration.py](calibration.py) — calibration routines for landmarks/gaze mapping
- [dataset.py](dataset.py) — dataset utilities and loaders
- [experiment.py](experiment.py) — experiment runner / evaluation harness
- [GazeClassify.py](GazeClassify.py) — high-level classification workflow
- [landmark_model/face_landmarker.task](landmark_model/face_landmarker.task) — pre-trained or task file for landmark detection
- [models/](models/) — example model implementations:
  - `CNN.py`, `mlp.py`, `random_forest.py`, `decision_tree.py`, `k_nearest_neighbours.py`, `k_means_clustering.py`, `hierarchical_clustering.py`

## Requirements

- Python 3.8+ recommended
- Typical dependencies: `numpy`, `pandas`, `opencv-python`, `scikit-learn`, and a deep-learning framework (PyTorch or TensorFlow) depending on the model implementations. These can be installed from `requirements.txt`

```bash
python -m pip install -r requirements.txt
```

## Quick Start

1. Run gui.py to use the pre-trained models, and the model training routines.
```bash
python gui.py
```  

## Notes

- The `landmark_model/face_landmarker.task` file is used by the project for facial landmark detection.
- Model implementations in `models/` are illustrative; review each file to see expected inputs, saved model formats, and training code.
