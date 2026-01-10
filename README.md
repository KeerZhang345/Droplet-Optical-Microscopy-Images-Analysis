# DropletCorLab  
*(Droplet Optical Microscopy Images Analysis)*

**DropletCorLab** is a research-oriented Python package for the analysis of
optical microscopy images of droplets, with a focus on data-driven monitoring
of corrosion processes occurring under droplets.

This repository accompanies the scientific article:

> **Development of a data-driven framework for monitoring corrosion under droplets**  
> Keer Zhang¹*, Arjan Mol¹, Yaiza Gonzalez-Garcia¹  
> ¹ Delft University of Technology, Department of Materials Science and Engineering,  
> Mekelweg 2, 2628CD, Delft, The Netherlands

The package implements the image processing, segmentation, feature extraction,
and analysis workflow described in the paper. It is intended to support
**reproducible research workflows**, rather than turnkey industrial deployment.

---

## Overview

DropletCorLab provides tools for:

- droplet segmentation in optical microscopy images  
- region-of-interest (ROI)–based feature extraction  
- generation of intermediate metadata (segmentation masks, features, dataframes)  
- exploratory and quantitative analysis of corrosion evolution under droplets  

The codebase is modular: users may run only selected parts of the pipeline
depending on data availability and research needs.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/KeerZhang345/Droplet-Optical-Microscopy-Images-Analysis.git
cd Droplet-Optical-Microscopy-Images-Analysis```

Install the package in editable mode:

```bash
pip install -e .```

This installs the core dependencies required for image processing, feature
extraction, and analysis.
