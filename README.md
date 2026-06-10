# Bone Age Assessment — EfficientNet-B4

Deep learning model for pediatric bone age estimation from hand X-ray images.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the RSNA Bone Age Dataset

You need a [Kaggle account](https://www.kaggle.com/) and the Kaggle CLI.

```bash
# Install Kaggle CLI (included in requirements.txt)
# Place your kaggle.json API key in ~/.kaggle/kaggle.json

# Download the dataset
kaggle competitions download -c rsna-bone-age -p ./data/

# Extract
cd data
unzip rsna-bone-age.zip
# This creates:
#   data/boneage-training-dataset/       (training images)
#   data/boneage-validation-dataset/     (validation images -- no labels on Kaggle)
#   data/boneage-training-dataset.csv    (training labels)
```

**Alternative**: Download directly from https://www.kaggle.com/competitions/rsna-bone-age

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

## Training

```bash
# Train with default settings
python train.py

# Override config via CLI
python train.py --epochs 40 --batch_size 16 --lr 1e-4
```

Training runs in two phases:
1. **Warmup** (5 epochs): backbone frozen, only the regression head trains
2. **Fine-tune** (remaining epochs): full model with cosine annealing LR

## Evaluation

```bash
# Evaluate a trained model on the validation set
python evaluate.py --checkpoint checkpoints/best_model.pth

# Generate prediction visualizations
python evaluate.py --checkpoint checkpoints/best_model.pth --visualize
```

## Minimum Expected Results

| Metric | Value |
|--------|-------|
| MAD (Mean Absolute Deviation) | ~4.0–5.0 months |
| RMSE | ~5.5–7.0 months |

## Model Architecture

- **Backbone**: EfficientNet-B4 pretrained on ImageNet
- **Gender fusion**: gender (binary) concatenated with image features after global average pooling
- **Head**: FC(1793→512) → ReLU → Dropout(0.3) → FC(512→1)
- **Output**: Predicted bone age in months

## Kaggle Packaging (Model Upload)

If you wish to share your model and inference code and keep the training related code separate, you can organize it as below. This is an example for kaggle but can apply anywhere you're democratizing access to your model weights + inference code and helping the community start from the best model you trained.

### Bundle Contents

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

### Publish to Kaggle

1. Edit `kaggle_package/dataset-metadata.json` and replace `YOUR_KAGGLE_USERNAME`.
2. Create dataset:

```bash
kaggle datasets create -p kaggle_package
```

3. Publish updates later:

```bash
kaggle datasets version -p kaggle_package -m "update model weights or docs"
```

### Use in a Kaggle Notebook

```bash
pip install -r /kaggle/input/bone-age-efficientnet-b4-model/code/requirements.txt
python /kaggle/input/bone-age-efficientnet-b4-model/code/inference.py \
	--image /kaggle/input/YOUR_IMAGE_DATASET/12345.png \
	--male 1 \
	--weights /kaggle/input/bone-age-efficientnet-b4-model/best_model.pth
```

## References


- EfficientNet: https://arxiv.org/abs/1905.11946
- Greulich & Pyle bone age atlas
