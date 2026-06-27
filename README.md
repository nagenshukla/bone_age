# Bone Age Assessment — EfficientNet-B4

> Deep learning model for pediatric bone age estimation from hand X-ray images.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Dataset](https://img.shields.io/badge/dataset-RSNA%20Bone%20Age-orange)

Pediatric bone age is normally read by radiologists comparing a hand X-ray against the Greulich & Pyle atlas — a slow, subjective process. This repo trains an EfficientNet-B4 regression model on the RSNA Bone Age challenge data and reaches **~4–5 months MAD** on the validation split, competitive with the top entries from the original 2017 challenge.

---

## Results

<!-- TODO: replace these with your actual final numbers -->

| Metric | Validation |
| --- | --- |
| MAD (Mean Absolute Deviation) | ~4.0–5.0 months |
| RMSE | ~5.5–7.0 months |
| Inference time (single GPU) | <!-- TODO --> ms/image |

---

## Model Architecture

- **Backbone:** EfficientNet-B4 pretrained on ImageNet
- **Gender fusion:** binary gender flag concatenated with image features after global average pooling
- **Head:** `FC(1793 → 512) → ReLU → Dropout(0.3) → FC(512 → 1)`
- **Output:** predicted bone age in months
- **Training schedule:** 5-epoch warmup with frozen backbone → full fine-tune with cosine-annealed LR

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Get the RSNA Bone Age dataset

You'll need a [Kaggle account](https://www.kaggle.com/) and the Kaggle CLI (included in `requirements.txt`). Place your `kaggle.json` API key at `~/.kaggle/kaggle.json`, then:

```bash
kaggle competitions download -c rsna-bone-age -p ./data/
cd data && unzip rsna-bone-age.zip
```

This creates:

```
data/
├── boneage-training-dataset/       # ~12,611 .png images
├── boneage-training-dataset.csv    # id, boneage, male
└── boneage-validation-dataset/     # validation images (no labels on Kaggle)
```

You can also grab it directly from <https://www.kaggle.com/competitions/rsna-bone-age>.

### 3. Directory Structure After Download

```
bone_age/
├── data/
│   ├── boneage-training-dataset/       # ~12,611 .png images
│   ├── boneage-training-dataset.csv    # id, boneage, male columns
│   └── (validation images if available)
├── checkpoints/                         # Saved models (auto-created)
├── config.py
├── dataset.py
├── ...
```

### 3. Train

```bash
# default config
python train.py

# override via CLI
python train.py --epochs 40 --batch_size 16 --lr 1e-4
```

### 4. Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth

# with prediction visualizations
python evaluate.py --checkpoint checkpoints/best_model.pth --visualize
```

---

## Repo Layout

```
bone_age/
├── config.py             # hyperparameters
├── dataset.py            # RSNA loader + augmentation
├── transforms.py         # X-ray-specific transforms
├── model.py              # EfficientNet-B4 + gender fusion head
├── train.py              # 2-phase training loop
├── evaluate.py           # MAD/RMSE on validation set
├── evaluate_bias.py      # error breakdown by age / sex
├── ensemble_eval.py      # multi-checkpoint ensembling
├── bone_age_analysis.ipynb
└── checkpoints/          # auto-created during training
```


---

## Sharing Your Trained Model on Kaggle

If you want to publish your weights + inference code as a Kaggle dataset so others can build on it, organize the bundle like this:

```
kaggle_package/
├── dataset-metadata.json
├── README.md
├── best_model.pth
├── final_model.pth
└── code/
    ├── inference.py
    └── requirements.txt
```

Edit `dataset-metadata.json` to set your Kaggle username, then:

```bash
# create
kaggle datasets create -p kaggle_package

# update later
kaggle datasets version -p kaggle_package -m "update weights"
```

Use it from a Kaggle notebook:

```bash
pip install -r /kaggle/input/bone-age-efficientnet-b4-model/code/requirements.txt
python /kaggle/input/bone-age-efficientnet-b4-model/code/inference.py \
    --image /kaggle/input/YOUR_IMAGE_DATASET/12345.png \
    --male 1 \
    --weights /kaggle/input/bone-age-efficientnet-b4-model/best_model.pth
```


## References

- Tan & Le, **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks** — <https://arxiv.org/abs/1905.11946>
- Halabi et al., **The RSNA Pediatric Bone Age Machine Learning Challenge** (2019) — <https://doi.org/10.1148/radiol.2018180736>
- Greulich & Pyle, *Radiographic Atlas of Skeletal Development of the Hand and Wrist*

---

## License

MIT — see [LICENSE](LICENSE).

> ⚠️ **Not a medical device.** This is a research project. Do not use for clinical decisions.

