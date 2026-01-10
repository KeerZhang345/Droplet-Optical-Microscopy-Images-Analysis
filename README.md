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
cd Droplet-Optical-Microscopy-Images-Analysis
```

Install the package in editable mode:

```bash
pip install -e .
```

This installs the core dependencies required for image processing, feature
extraction, and analysis.

## Segmentation with SAM (optional)

Droplet segmentation can optionally be performed using Meta’s
Segment Anything Model (SAM).

Due to the size of the model and its dependencies, SAM support is optional
and not included in the default installation.

### Install optional dependencies:
```bash
pip install droplet-corr-lab[segmentation]
```
This installs:
- torch
- torchvision
- segment-anything

### Download SAM model weights

SAM model weights must be downloaded separately from the official repository:

https://github.com/facebookresearch/segment-anything

The path to the downloaded checkpoint should be provided via configuration
or environment variables as required by the segmentation code.

## Example workflow

A minimal, self-contained example demonstrating the analysis pipeline is
provided in:
```bash
examples/demo_pipeline.ipynb
```
The example illustrates:
- droplet segmentation
- ROI-based feature extraction
- generation of intermediate metadata
- data visualization

Only a small subset of raw images is used to keep the example lightweight
and self-contained.

## Data availability

This repository includes only a minimal subset of raw microscopy images
for demonstration purposes.

Full experimental datasets and derived metadata (e.g. complete segmentation
results, feature tables, time-resolved dataframes) are only available upon reasonable requests.

All intermediate metadata used in the example notebook can be regenerated
from the provided raw images using the supplied code.

## Intended use and limitations

This package is designed for research and methodological exploration.
It prioritizes transparency, reproducibility, and flexibility over
computational efficiency or production-level robustness.

Users are expected to adapt and extend the code for their specific
experimental setups and research questions.

## Citation

If you use this code in your research, please cite the associated publication
and this repository. Citation metadata is provided in the CITATION.cff file.

## License

This project is released under the MIT License.
