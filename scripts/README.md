# Scripts Directory

This directory contains utility scripts for training, evaluation, and analysis of the Weld Defect Detection model.

## Training & Pipeline Scripts

### `train_pipeline.py`
Complete end-to-end training pipeline that runs all 8 stages:
1. Text removal
2. DINOv2 pre-training (optional)
3. LoRA training
4. Synthetic data generation
5. Dataset merging
6. Classifier training
7. Validation with TTA
8. Model export

**Usage:**
```bash
python scripts/train_pipeline.py
python scripts/train_pipeline.py --skip-synthetic
python scripts/train_pipeline.py --skip-text-removal --skip-export
```

### `train_lora.py`
Train LoRA weights on clean real images for synthetic data generation.

### `generate_synthetic.py`
Generate synthetic weld images using SDXL-Turbo + ControlNet + LoRA.

## Overfitting Analysis Scripts

### `overfitting_analysis.py` ⭐ **New**
Comprehensive overfitting analysis using K-Fold cross-validation.

**Purpose:** Validate the 97% accuracy claim and ensure model generalizes well.

**Features:**
- K-Fold stratified cross-validation (default: 5 folds)
- Learning curve visualization (train vs validation)
- Overfitting gap analysis with severity levels
- Statistical significance testing
- Automated recommendations
- Detailed JSON and text reports

**Usage:**
```bash
# Basic usage with default settings (5 folds, 20 epochs per fold)
python scripts/overfitting_analysis.py --data-path data/merged

# Custom settings
python scripts/overfitting_analysis.py \
    --data-path data/merged \
    --k-folds 5 \
    --epochs 20 \
    --output-dir outputs/overfitting_analysis

# With specific config
python scripts/overfitting_analysis.py \
    --data-path data/merged \
    --config-path conf/train.yaml
```

**Output Files:**
- `overfitting_analysis.json` - Complete results in JSON format
- `overfitting_report.txt` - Human-readable summary report
- `learning_curves.png` - Train vs validation curves for each fold
- `cv_summary.png` - Cross-validation summary visualization

**Interpretation:**
- **Gap 0-5%**: 🟢 Model generalizes well - PASS
- **Gap 5-10%**: 🟡 Moderate overfitting - CAUTION
- **Gap 10-15%**: 🟠 High overfitting - CONCERN
- **Gap >15%**: 🔴 Severe overfitting - FAIL

### `quick_overfitting_check.py` ⭐ **New**
Fast overfitting validation for quick testing during development.

**Purpose:** Quick validation without full cross-validation.

**Features:**
- Single 80/20 train/val split
- Fast 15-epoch training
- Immediate pass/fail verdict
- Console output only

**Usage:**
```bash
# Basic usage
python scripts/quick_overfitting_check.py --data-path data/merged

# Custom epochs and learning rate
python scripts/quick_overfitting_check.py \
    --data-path data/merged \
    --epochs 15 \
    --lr 1e-4
```

**When to Use Each:**
- Use `quick_overfitting_check.py` for:
  - Rapid iteration during development
  - Quick sanity checks
  - Limited computational resources
  - Small datasets (<100 images)

- Use `overfitting_analysis.py` for:
  - Final validation before deployment
  - Verifying claimed model performance
  - Publication/production readiness
  - Comprehensive evaluation with reports

## Testing Utilities

### `create_test_dataset.py`
Create synthetic test dataset for validating analysis tools.

**Usage:**
```bash
# Create 50 test images (25 per class)
python scripts/create_test_dataset.py \
    --output-dir data/test_dataset \
    --num-per-class 25

# Create larger test set
python scripts/create_test_dataset.py \
    --output-dir data/test_dataset \
    --num-per-class 50
```

**Note:** This creates simple synthetic images for testing the pipeline. Not suitable for actual model training.

## Workflow Examples

### Complete Training with Overfitting Analysis

```bash
# 1. Run full training pipeline
python scripts/train_pipeline.py

# 2. Quick overfitting check during development
python scripts/quick_overfitting_check.py --data-path data/merged

# 3. Comprehensive analysis before deployment
python scripts/overfitting_analysis.py \
    --data-path data/merged \
    --k-folds 5 \
    --epochs 20

# 4. Review results
cat outputs/overfitting_analysis/overfitting_report.txt
```

### Testing Pipeline with Sample Data

```bash
# 1. Create test dataset
python scripts/create_test_dataset.py \
    --output-dir data/test_dataset \
    --num-per-class 30

# 2. Quick validation check
python scripts/quick_overfitting_check.py \
    --data-path data/test_dataset \
    --epochs 10
```

### Validating 97% Accuracy Claim

```bash
# Run comprehensive cross-validation
python scripts/overfitting_analysis.py \
    --data-path data/merged \
    --k-folds 5 \
    --epochs 25 \
    --output-dir outputs/accuracy_validation

# Check results
cat outputs/accuracy_validation/overfitting_report.txt

# Look for:
# - Mean validation accuracy ≈ 95-97%
# - Low standard deviation (<3%)
# - Overfitting gap <5%
# - Consistent performance across folds
```

## Common Issues

### "Insufficient training data" Error
**Problem:** Dataset has fewer than 20 images
**Solution:** 
- Collect more data, or
- Use synthetic data generation pipeline, or
- Create test dataset with `create_test_dataset.py`

### CUDA Out of Memory
**Problem:** GPU memory exhausted during training
**Solution:**
- Reduce batch size in the script
- Use smaller number of folds (3 instead of 5)
- Use `quick_overfitting_check.py` which uses smaller batch size

### Long Training Time
**Problem:** Cross-validation takes too long
**Solution:**
- Reduce number of epochs per fold
- Reduce number of folds (use 3 instead of 5)
- Use `quick_overfitting_check.py` for faster results

## Dependencies

All scripts require:
- Python 3.8+
- PyTorch 2.0+
- scikit-learn
- matplotlib
- seaborn
- loguru
- tqdm
- albumentations

Install with:
```bash
pip install -r requirements.txt
```

## Documentation

For detailed documentation on overfitting analysis:
- See [../docs/OVERFITTING_ANALYSIS.md](../docs/OVERFITTING_ANALYSIS.md)

For general project documentation:
- See [../README.md](../README.md)

## Contributing

When adding new scripts:
1. Add comprehensive docstrings
2. Include usage examples
3. Add error handling
4. Update this README
5. Test on sample data
