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

## Framework

![VisionDES Framework](https://github.com/InfoLab-SKKU/VisionDES/blob/main/images/vision_des_framework.png?raw=true)

## Installation

```bash
pip install vision-des
```

## Quick Start

### Create a Pool of Models

```python
pool = [
    resnet50,
    efficientnet,
    convnext
]
```

### Initialize VisionDES

```python
from vision_des import VisionDES

des = VisionDES(
    dsel_dataset=dsel_dataset,
    pool=pool,
    device="cuda"
)
```

### Build the Retrieval Index

```python
des.fit()
```

### Predict

```python
prediction = des.predict(
    test_image,
    k=7
)
```

---

## Explainable Inference

```python
prediction = des.predict(
    test_image,
    k=7,
    explain=True
)
```

## Sample Output

![VisionDES Output](https://github.com/InfoLab-SKKU/VisionDES/blob/main/images/vision_des_output.PNG?raw=true)
