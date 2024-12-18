![QQ_1734528930159](https://github.com/user-attachments/assets/c5377b81-1e7c-4915-a723-dd99a4cee08c)# A2SAM: Flaws can be Applause

[NeurIPS 2024] Flaws can be Applause: Unleashing Potential of Segmenting Ambiguous Objects in SAM

## Introduction

This repository contains the official implementation of our paper "Flaws can be Applause: Unleashing Potential of Segmenting Ambiguous Objects in SAM" (NeurIPS 2024). A2SAM is a novel approach that leverages the inherent ambiguities in object segmentation to improve the performance of the Segment Anything Model (SAM).

## Features

- Enhanced segmentation for ambiguous and overlapping objects
- Improved handling of complex scenes with multiple similar objects
- Built upon Meta's Segment Anything Model (SAM)
- Easy integration with existing SAM-based pipelines

## Installation

```bash
git clone https://github.com/username/a2sam.git
cd a2sam
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
# Train on LIDC dataset
python train_lidc.py --batch-size 16 --epochs 100 --lr 1e-4

# Train on Sim10k dataset
python train_sim10k.py --batch-size 16 --epochs 100 --lr 1e-4
```

### Evaluation

```bash
# Evaluate on LIDC test set
python test_lidc.py --checkpoint path/to/checkpoint.pth

# Evaluate on Sim10k test set
python test_sim10k.py --checkpoint path/to/checkpoint.pth
```

### Main Arguments

- `--batch-size`: Batch size for training (default: 16)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--checkpoint`: Path to model checkpoint for testing
- `--data-dir`: Path to dataset directory
- `--output-dir`: Directory to save results

For more detailed configurations, please check the argument parser in each script.

## Pre-trained Models and Datasets

The pre-trained weights for the LIDC dataset and the processed Sim10k dataset are available on Hugging Face. You can access them at the following link:

- [Hugging Face Repository](https://huggingface.co/yu2hi13/asam/tree/main)

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{a2sam2024,
    title={Flaws can be Applause: Unleashing Potential of Segmenting Ambiguous Objects in SAM},
    author={},
    booktitle={Advances in Neural Information Processing Systems},
    year={2024}
}
```
