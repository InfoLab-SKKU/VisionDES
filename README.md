# VisionDES
> Robust and Explainable Dynamic Vision Ensemble

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyPI](https://img.shields.io/pypi/v/vision-des.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

**VisionDES** is a vision-oriented Dynamic Ensemble Selection (DES) framework that adapts classical DES techniques to modern deep learning pipelines.

Instead of relying solely on classifier confidence, VisionDES combines:

- 🎯 Local classifier competence
- 🔍 DINO visual embeddings
- ⚡ FAISS nearest-neighbor retrieval
- 🧠 Feature-space similarity estimation
- 📊 Dynamic competence weighting
- 🔬 Explainable ensemble decisions

For every incoming image, the framework dynamically identifies the most competent classifiers within a local Region of Competence (RoC) and generates an adaptive prediction.
