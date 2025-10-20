# Deep Learning-Based Detection of Foodborne Pathogens Using Hyperspectral Imaging and Transfer Learning

This repository provides a complete PyTorch pipeline for classifying foodborne pathogen contamination from hyperspectral imaging (HSI). It supports:
- 3D CNNs (ResNet-3D) for spatio-spectral learning
- Transfer learning from a source HSI checkpoint (or layer freezing)
- Patch-level classification (or per-pixel if masks provided)
- Real-time data augmentation (spectral jitter, mixup, CutMix, flips)
- K-fold cross validation
- Grad-CAM 3D explainability and per-band importance

## Quick Start
```bash
# 1) Create environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Update config
nano config.yaml

# 3) Train
python -m src.train --config config.yaml

# 4) Evaluate on hold-out test set
python -m src.evaluate --config config.yaml --ckpt runs/exp1/best.pt

# 5) Inference on a new cube
python -m src.infer --config config.yaml --ckpt runs/exp1/best.pt --input data/test/cubes/sample.npy

# 6) K-fold CV
python -m src.kfold_cv --config config.yaml --folds 5

# 7) Grad-CAM 3D visualization
python -m src.explain.gradcam3d --config config.yaml --ckpt runs/exp1/best.pt --input data/val/cubes/sample.npy
```

## Data
Place cubes under `data/train/cubes`, `data/val/cubes`, `data/test/cubes` and create a `labels.csv` per split:
```
filename,label
sample1.npy,1
sample2.npy,0
```

Per-pixel segmentation (optional):
```
filename,mask_path
sample1.npy,data/train/masks/sample1_mask.npy
```
Set `task: segmentation` in `config.yaml` to enable 3D U-Net head.

## Configuration
Adjust hyperparameters in `config.yaml` (paths, model, training). CLI flags override yaml keys, e.g., `--trainer.max_epochs 150`.

## Checkpoints & Transfer Learning
- To fine-tune from a source model: set `model.pretrained_ckpt` to a `.pt` path.
- To freeze early blocks: `model.freeze_layers: [stem,layer1]`.

## Citations
If this helps your research, please cite the repository in your methods section:
> "We implemented a 3D CNN pipeline for hyperspectral pathogen detection based on an open-source PyTorch repository (link)."
