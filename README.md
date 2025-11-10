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

## Unified Label Mapping Integration
The preprocessing pipeline now consumes a consolidated label mapping at `logs/unified_label_mapping.csv`, producing five standardized classes:

Index mapping:
- 0 = MI
- 1 = AF
- 2 = BBB
- 3 = NORM
- 4 = OTHER

Artifacts generated during preprocessing:
- `artifacts/processed/train.npz`, `val.npz`, `test.npz` with keys: `x`, `y`, `signal`, `label`
- `artifacts/processed/labels.npy` — ordered label names
- `artifacts/processed/label_map.json` — forward and reverse mappings
- `artifacts/processed/splits.json` — counts and metadata

To regenerate:
```powershell
# Run notebook preprocessing cell OR convert to script:
python - <<'PY'
from pathlib import Path
import runpy
# (Optionally extract preprocessing logic into a script for CI)
runpy.run_path('ecg_tensor_pipeline.ipynb')  # if executed with a tool that supports .ipynb
PY
```

## FastAPI Serving
After training, launch an inference API:
```powershell
uvicorn ecg_tensor_pipeline:app --reload --port 8000
```
POST a file to `/predict` (.csv or .npy) containing a single ECG signal (length variable; auto-resampled to 5000 @ 500 Hz).

## Git Usage
```powershell
# Initial commit (already performed by automation)
git add .
git commit -m "Initial commit: unified ECG tensor pipeline"
# Push to remote (set your remote URL first)
git remote add origin https://github.com/vivekjindal24/ecg-tensor.git
git push -u origin master
```
