# Inference Pipeline - Quick Reference

## Overview
Complete inference pipeline for generating Kaggle submission from trained model.

## Changes Made

### 1. Updated `src/dataset.py`
- Added `test` parameter to constructor
- Test mode: loads images without labels
- Returns `(image, image_id)` tuples instead of `(image, label)`
- Automatically detects and uses first column as image_id

### 2. Created `src/inference.py`
Two main functions:

#### `generate_predictions()`
Full control inference function:
```python
from src.inference import generate_predictions

submission_df = generate_predictions(
    model=model,
    test_csv_path="/path/to/test.csv",
    test_img_dir="/path/to/test_images",
    checkpoint_path="/path/to/best_model.ckpt",
    output_csv_path="submission.csv",
    batch_size=64,
    model_name="efficientnetv2",
    device="cuda"
)
```

#### `create_submission()` (Recommended)
Convenience function with auto-detection:
```python
from src.inference import create_submission

submission_df = create_submission(
    checkpoint_path=checkpoint_callback.best_model_path,
    output_csv_path="submission.csv",
    model_config=train_config,
    batch_size=64
)
```

Auto-detects:
- Kaggle vs local environment
- Test CSV and image directory paths
- Model configuration
- Device (GPU/CPU)

### 3. Updated Notebooks

#### `kaggle_train.ipynb` - Step 8
```python
from src.inference import create_submission

submission_df = create_submission(
    checkpoint_path=checkpoint_callback.best_model_path,
    output_csv_path="/kaggle/working/submission.csv",
    model_config=train_config,
    batch_size=64
)
```

Output: `/kaggle/working/submission.csv`

#### `local_train.ipynb` - Step 9
```python
# Optional inference test
submission_df = create_submission(
    checkpoint_path=checkpoint_callback.best_model_path,
    test_csv_path="../data/kenyan-food-13/test.csv",
    test_img_dir="../data/kenyan-food-13/test_images",
    output_csv_path="./submission_local.csv",
    model_config=train_config,
    batch_size=16
)
```

## Usage Workflow

### On Kaggle (Full Pipeline)

1. **Train the model** (Steps 1-6)
2. **Generate predictions** (Step 8):
   ```python
   from src.inference import create_submission

   submission_df = create_submission(
       checkpoint_path=checkpoint_callback.best_model_path,
       output_csv_path="/kaggle/working/submission.csv",
       model_config=train_config,
       batch_size=64
   )
   ```

3. **Download submission.csv** from Output tab
4. **Submit to Kaggle**:
   - Go to competition page
   - Click "Submit Predictions"
   - Upload `submission.csv`

### Local Testing

1. **Train locally** (2 epochs for testing)
2. **Generate predictions** (Step 9):
   ```python
   submission_df = create_submission(
       checkpoint_path=checkpoint_callback.best_model_path,
       test_csv_path="../data/kenyan-food-13/test.csv",
       test_img_dir="../data/kenyan-food-13/test_images",
       output_csv_path="./submission_local.csv",
       model_config=train_config,
       batch_size=16
   )
   ```

3. **Verify submission.csv** format

## Submission File Format

The inference pipeline creates a CSV file with this format:

```csv
image_id,label
img_0001.jpg,chapati
img_0002.jpg,ugali
img_0003.jpg,pilau
...
```

**Columns:**
- `image_id`: Original image filename from test.csv
- `label`: Predicted class name (e.g., "chapati", "ugali", "pilau")

## Features

✅ **Automatic class mapping**: Loads train.csv to get class names from indices
✅ **Model-specific preprocessing**: Uses correct image size and normalization
✅ **Progress bar**: Shows inference progress with tqdm
✅ **Prediction distribution**: Shows how many images predicted for each class
✅ **Error handling**: Graceful fallbacks if class mapping not available
✅ **Environment detection**: Works on both Kaggle and local

## Troubleshooting

### Issue: "FileNotFoundError: test.csv not found"
**Solution**: Check test data paths:
- Kaggle: `/kaggle/input/opencv-pytorch-project-2-classification-round-3/test.csv`
- Local: `../data/kenyan-food-13/test.csv`

### Issue: "Could not load class mapping"
**Solution**: Ensure train.csv is accessible. Pipeline will use numeric indices as fallback.

### Issue: "CUDA out of memory during inference"
**Solution**: Reduce batch_size:
```python
create_submission(..., batch_size=32)  # or 16
```

### Issue: Predictions are all same class
**Possible causes:**
1. Model not trained properly (check validation accuracy)
2. Wrong preprocessing (should match training)
3. Corrupted checkpoint

**Debug:**
```python
# Check model predictions before submission
print(submission_df['label'].value_counts())
# Should show reasonable distribution, not 100% one class
```

## Advanced: Custom Inference

For more control, use `generate_predictions()` directly:

```python
from src.inference import generate_predictions
from src.model import KenyanFood13Classifier

# Load your trained model
model = KenyanFood13Classifier.load_from_checkpoint(
    "path/to/checkpoint.ckpt"
)

# Generate predictions with custom settings
submission_df = generate_predictions(
    model=model,
    test_csv_path="custom_test.csv",
    test_img_dir="custom_images/",
    checkpoint_path="path/to/checkpoint.ckpt",
    output_csv_path="custom_submission.csv",
    batch_size=128,
    num_workers=4,
    model_name="efficientnetv2",
    device="cuda"
)
```

## Next Steps

1. **Commit and push**:
   ```bash
   git add src/inference.py src/dataset.py src/__init__.py
   git add kaggle_train.ipynb local_train.ipynb
   git commit -m "Add inference pipeline for test predictions"
   git push origin main
   ```

2. **Run full training on Kaggle**:
   - 50-100 epochs
   - Expected accuracy: 75-85%
   - Generate submission in Step 8

3. **Submit to Kaggle competition**

4. **Iterate based on leaderboard score**:
   - If score < 85%, try:
     - More epochs
     - Different model (try efficientnetv2_m)
     - Adjust freeze_pct
     - Test-time augmentation (TTA)
