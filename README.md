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

The official Kaggle `val` split is only 16 images, so the download script rebuilds validation as ~10% of
`train`. The `test` split is left unchanged for final evaluation.

The hold-out is **grouped by patient**. Several X-rays can belong to one child (`person1003_bacteria_2934`,
`person1003_bacteria_2935`, …), so splitting by image put 272 children on both sides of the train/val
boundary — early stopping was then selecting on images the model had already memorised. `check_no_leakage()`
fails the run if any patient still straddles the split.

If you have data from before this fix, re-split it in place:

```bash
python scripts/download_data.py --resplit    # no re-download
```

Any model trained before that re-split has an optimistic validation score and should be retrained.

## Multi-class: Normal vs Bacterial vs Viral

Kermany's filenames already encode the pneumonia sub-type (`person1003_bacteria_2934.jpeg`,
`person1003_virus_1685.jpeg`), so the 3-class task needs no extra download:

```bash
python scripts/build_subtype_dataset.py
python scripts/train.py    --config configs/subtype.yaml
python scripts/evaluate.py --config configs/subtype.yaml --checkpoint artifacts/subtype/best.pt
python scripts/explain.py  --config configs/subtype.yaml --checkpoint artifacts/subtype/best.pt --n 18
```

`build_subtype_dataset.py` hardlinks images into `data/chest_xray_subtype/{train,val,test}/{BACTERIAL,NORMAL,VIRAL}`
(pass `--copy` for independent files). It uses the same patient-grouped hold-out as above, and carries the
official `test` folder over untouched so results stay comparable with published numbers.

Hardlinks cost no extra disk and, unlike symlinks, survive `download_data.py --resplit` moving files between
the source `train/` and `val/` folders.

| split | BACTERIAL | NORMAL | VIRAL | patients |
|---|---|---|---|---|
| train | 2278 | 1214 | 1212 | 2559 |
| val | 260 | 135 | 133 | 302 |
| test | 242 | 234 | 148 | 427 |

Multi-class runs report **macro** precision/recall/F1 and one-vs-rest AUC (per class and macro-averaged),
so the rare VIRAL class counts as much as BACTERIAL. Artifacts go to `artifacts/subtype/`, leaving the
binary baseline in `artifacts/` intact.

### Scope note

These images are all from pediatric patients aged 1–5 at one hospital in Guangzhou. Extending to COVID-19
or tuberculosis means pulling from adult datasets ([COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database),
[TB Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)),
which introduces a source/age confound a CNN can exploit instead of learning pathology — see
[DeGrave et al. 2021](https://www.nature.com/articles/s42256-021-00338-7). Any 5-class extension should ship
with a source-classifier control and lung-masked Grad-CAM to show the model is reading lungs, not demographics.

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

Upload a chest X-ray, get a prediction, and see Grad-CAM:

```bash
pip install streamlit
streamlit run app.py                              # binary model, artifacts/best.pt
CXR_CONFIG=configs/subtype.yaml streamlit run app.py   # 3-class, artifacts/subtype/best.pt
```

The app reads its class list from the config, so it adapts to whichever task you point it at.

## Project layout

```
app.py               # Streamlit upload → predict → Grad-CAM
src/cxr_pneumonia/   # reusable library
notebooks/           # development notebooks
scripts/             # download / train / evaluate / explain CLIs
configs/             # YAML configs
data/                # downloaded X-rays (gitignored)
artifacts/           # binary checkpoints and figures
artifacts/subtype/   # 3-class checkpoints and figures
```
