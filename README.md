# typological-analysis-of-venetian-floor-plans-through-cv

## Basic Information
* Author : Jaeyi Jeong
* Professor : Prof. Frédéric Kaplan
* Supervisor : Paul Guhennec
* Academic Year : 2024–2025 
* Affiliation : EPFL DHLab – Bachelor Semester Project

## About
This project aims to analyze the typological features of Venetian architectural floor plans through computer vision techniques. Over 280 historical plan pages were manually scanned, preprocessed, and semantically segmented using deep learning models. The goal is to enable large-scale, quantitative architectural analysis through a hybrid approach that combines classical image processing and modern machine learning.

Following segmentation, architectural elements such as walls, windows, and stairs were vectorized into GeoJSON polygons. These vector outputs were then georeferenced by aligning them with existing building footprint datasets, enabling spatial analysis and integration into GIS workflows. This pipeline supports the automated extraction of architectural patterns at scale and contributes to digital heritage research.

## Research Summary
We used a multi-stage pipeline combining OpenCV-based preprocessing and SegFormer-based semantic segmentation to extract architectural features such as walls, windows, and stairs.
Steps include :

* Manual scanning and annotation of architectural plans

* Preprocessing using thresholding, color masking, and distortion correction

* Ground truth labeling using PenUP

* Patch-based training of SegFormer with class-weighted loss

* Inference on full-size images and hybrid visualization

* Vectorization and geographic integration of results using QGIS and GeoPandas

## Dependencies
Platform : Ubuntu 20.04 or Google Colab

Python version : ≥ 3.8

Required libraries :

* numpy

* opencv-python

* matplotlib

* shapely

* geopandas

* transformers

* torch

* torchvision

* albumentations

Install with :

pip install -r requirements.txt

## Report
Find the full report and LaTeX sources in the report/ folder.

## License
typological-analysis-of-venetian-floor-plans-through-cv - Jaeyi Jeong  
Copyright (c) 2025 EPFL  
This program is licensed under the terms of the MIT license.
