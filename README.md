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

## 5-class extension, and the confound it creates

The images above are all pediatric patients aged 1–5 at one hospital in Guangzhou. Adding COVID-19 and
tuberculosis means pulling from adult datasets ([COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database),
[TB Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)),
so disease label and data source become almost perfectly correlated — a CNN can score well by telling
hospitals apart and never read a lung ([DeGrave et al. 2021](https://www.nature.com/articles/s42256-021-00338-7)).

`build_confounding_dataset.py` emits **two datasets over the identical images**: one labelled by disease,
one by source. Training both is the control — if the source classifier scores as high as the disease one,
the disease number is not evidence of pathology detection.

```bash
python scripts/build_confounding_dataset.py
python scripts/train.py    --config configs/confounding.yaml          # 5-class disease
python scripts/train.py    --config configs/confounding_source.yaml   # 3-class source control
python scripts/evaluate.py --config configs/confounding.yaml          --checkpoint artifacts/confounding/best.pt
python scripts/evaluate.py --config configs/confounding_source.yaml   --checkpoint artifacts/confounding_source/best.pt
python scripts/lung_mask_attention.py --class-dir COVID --split test
```

Results on the 844-image test set:

| model | labels | accuracy |
|---|---|---|
| disease | BACTERIAL / COVID / NORMAL / TUBERCULOSIS / VIRAL | 0.777 |
| source control | KERMANY_PEDIATRIC / QATAR_COVID / QATAR_TB | **1.000** |

Hospital source is perfectly separable from the pixels alone. In the disease model, COVID and TB recall
are ~1.00 (each is the only class from its hospital) while NORMAL recall collapses to 0.35 — the classes
that *share* a source are the ones it cannot tell apart. `compute_source_oracle()` prints the accuracy
floor reachable from source alone, which any honest 5-class result must clear.

### Reading the lung-mask numbers

`lung_mask_attention.py` scores what fraction of Grad-CAM mass lands outside the lung fields, using the
segmentation masks shipped with the COVID-19 Radiography Database. **Its baseline is not zero.** Lungs
cover ~25% of the frame, so random attention already scores ~0.75 outside; only `concentration` (attention
density inside vs. outside) is directly interpretable.

On the COVID test split: outside fraction **0.593** against a **0.752** chance baseline, concentration
**2.22** — attention is more than twice as dense inside the lungs as outside.

That does **not** clear the model. A source confound lives inside the lung field too — scanner response,
body habitus and age all differ there — so a plausible-looking heatmap cannot rule out shortcut learning
when the source control still scores 1.000. Saliency maps are necessary but not sufficient evidence, which
is the same caution DeGrave et al. raise.

## Development (Jupyter)

```bash
jupyter notebook notebooks/
```

`01_eda.ipynb` — explore class counts and sample images. Training and Grad-CAM
run from the CLI below rather than notebooks, so results are reproducible.

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
app.py                          # Streamlit upload → predict → Grad-CAM
src/cxr_pneumonia/              # reusable library
notebooks/                      # development notebooks
scripts/                        # download / build / train / evaluate / explain CLIs
configs/                        # YAML configs, one per task
references/                     # published results for comparison
data/                           # downloaded X-rays (gitignored)
artifacts/                      # binary checkpoints and figures
artifacts/subtype/              # 3-class checkpoints and figures
artifacts/confounding/          # 5-class disease model + lung-mask attention
artifacts/confounding_source/   # source-control model
artifacts/external/             # external-hospital validation
```

## Reproducibility notes

* Splits are **grouped by patient**, never by image (`hold_out_patients`), and the build scripts fail
  loudly if any patient straddles train and val.
* Patient IDs restart between Kermany's train and test folders, so they are never pooled across the two —
  a train/test overlap check by ID would produce false matches, not real ones.
* Test metrics ship with **bootstrap 95% CIs** (percentile method, 1000 resamples) so a single point
  estimate is not mistaken for precision.
* `evaluate()` always builds its own eval-mode loader — full split, in order, no augmentation, no
  resampling — so `--split train` reports true training-set performance rather than augmented duplicates.
* `scripts/compare_published.py` lines results up against published numbers on this dataset and marks
  which comparisons are actually valid (most published work uses a random, non-patient-grouped split).
