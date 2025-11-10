# ECG Research — Tensor-based ML for Cardiac Disease Detection

This repository contains code and notebooks for training tensor-driven deep learning models on 12-lead ECG datasets (PTBXL, Chapman-Shaoxing, CinC 2017 AFDB, etc.).

## Project Layout
```
ecg-research/
├─ dataset/
├─ artifacts/
│  ├─ figures/
│  ├─ models/
│  ├─ saliency/
│  └─ mlflow/
├─ notebooks/
│  └─ ECG_Tensor_Training_and_Validation.ipynb
├─ preprocessing/
│  ├─ __init__.py
│  └─ ecg_preprocessing.py
├─ scripts/
│  └─ train.py
└─ requirements.txt
```

## Environment Setup (Windows PowerShell)
```powershell
# Create & activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install deps
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch for RTX 4050, use the official selector: https://pytorch.org/get-started/locally/

## MLflow Tracking
```powershell
# Start MLflow UI (stores runs locally under artifacts/mlflow)
mlflow ui --backend-store-uri file:./artifacts/mlflow
```
Open http://127.0.0.1:5000 in your browser.

## Jupyter
```powershell
# Launch Jupyter Lab/Notebook
python -m pip install jupyterlab
jupyter lab
```

## Quick Smoke Test
```powershell
# Run a tiny dummy training loop (synthetic data)
python .\scripts\train.py --model cnn --epochs 1 --batch-size 8
```
A checkpoint will be saved under `artifacts/models`.

## Next Steps
- Implement real dataset loaders (PTBXL CSV, WFDB signals, Chapman .mat + .hea).
- Connect preprocessing pipeline (denoise, resample, normalize, segment) into Dataset.
- Build tensor constructions and try CP/Tucker/HOSVD with TensorLy.
- Wire MLflow logging (params, metrics, artifacts) and Grad-CAM/Captum for interpretability.
- Add evaluation notebook cells for AUC/F1, reliability diagrams, and LaTeX tables.
