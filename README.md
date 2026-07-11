# Chest X-Ray Pneumonia Detection with Grad-CAM

Detect pneumonia from chest X-ray images using transfer learning (ResNet50) and explain predictions with Grad-CAM.

**Data:** [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) ,real JPEG X-rays labeled Normal / Pneumonia. Images are downloaded locally; Run python scripts/download_data.py to download the data.

## Setup

```bash
cd medical-image-analysis
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
pip install -r requirements.txt
```

### Kaggle credentials

1. Create a free [Kaggle](https://www.kaggle.com) account.
2. Account → Settings → API → Create New Token (`kaggle.json`).
3. Place it at `~/.kaggle/kaggle.json` (chmod 600), **or** set `KAGGLE_USERNAME` / `KAGGLE_KEY`.

## Download data

```bash
python scripts/download_data.py
```

Images land in:

```
data/chest_xray/
├── train/{NORMAL,PNEUMONIA}/
├── val/{NORMAL,PNEUMONIA}/
└── test/{NORMAL,PNEUMONIA}/
```

The official Kaggle `val` split is only 16 images, so the download script rebuilds validation as ~10% stratified from `train`. The `test` split is left unchanged for final evaluation.

## Development (Jupyter)

```bash
jupyter notebook notebooks/
```

1. `01_eda.ipynb` — explore class counts and sample images  
2. `02_train.ipynb` — train the model  
3. `03_gradcam.ipynb` — Grad-CAM explanations  

## Final project (CLI)

```bash
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --checkpoint artifacts/best.pt
python scripts/explain.py --checkpoint artifacts/best.pt --split test --n 16
```

Outputs (weights, metrics, heatmaps) go to `artifacts/`.

## Web app (Streamlit)

Upload a chest X-ray, get Normal / Pneumonia prediction, and see Grad-CAM:

```bash
pip install streamlit
streamlit run app.py
```

Requires `artifacts/best.pt` (train first).

## Project layout

```
app.py               # Streamlit upload → predict → Grad-CAM
src/cxr_pneumonia/   # reusable library
notebooks/           # development notebooks
scripts/             # download / train / evaluate / explain CLIs
configs/             # YAML configs
data/                # downloaded X-rays (gitignored)
artifacts/           # checkpoints and figures
```
