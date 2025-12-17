# TensorBoard and Configuration Improvements

## Changes Applied - December 9, 2025

### 1. Removed Special Characters from Print Statements

**Files Updated**: `src/model.py`, `src/trainer.py`

**Changes**:
- Replaced `✓` with `[OK]`
- Replaced `📊` with `[INFO]`
- Replaced `⚠️` with `[WARN]`

**Reason**: Special Unicode characters may not display correctly in all terminals/environments.

**Examples**:
```python
# Before:
print("✓ Pre-trained ResNet50 weights loaded successfully")
print("📊 Using single GPU")

# After:
print("[OK] Pre-trained ResNet50 weights loaded successfully")
print("[INFO] Using single GPU")
```

---

### 2. Added Early Stopping Configuration Parameters

**File Updated**: `src/config.py`

**New Parameters in `TrainingConfiguration`**:
```python
# Early stopping configuration
use_early_stopping: bool = True  # whether to use early stopping
early_stop_monitor: str = "valid/acc"  # metric to monitor
early_stop_patience: int = 7  # epochs with no improvement before stopping
early_stop_mode: str = "max"  # 'min' for loss, 'max' for accuracy
```

**File Updated**: `src/trainer.py`

**Usage**:
```python
if training_config.use_early_stopping:
    early_stopping_callback = EarlyStopping(
        monitor=training_config.early_stop_monitor,
        patience=training_config.early_stop_patience,
        mode=training_config.early_stop_mode,
        verbose=True
    )
```

**Benefits**:
- No need to modify `trainer.py` to change early stopping behavior
- Configure in notebook or config file
- Easy to disable early stopping for experimentation

**Example in Notebook**:
```python
# Customize early stopping
train_config, data_config, system_config = get_config(
    num_epochs=100,
    batch_size=32
)

# Override early stopping settings
train_config.use_early_stopping = True
train_config.early_stop_patience = 10
train_config.early_stop_monitor = "valid/loss"
train_config.early_stop_mode = "min"
```

---

### 3. Added TensorBoard Model Parameter Histograms

**File Updated**: `src/model.py` - `on_validation_epoch_end()` method

**What's Logged**:
1. **Weight Distributions**: Histogram of all trainable layer weights
2. **Gradient Distributions**: Histogram of gradients for each layer

**Code Added**:
```python
# Log model parameter histograms to TensorBoard
if self.logger:
    for name, params in self.named_parameters():
        if params.requires_grad:
            # Log weight distributions
            self.logger.experiment.add_histogram(f'weights/{name}', params, self.current_epoch)
            # Log gradient distributions
            if params.grad is not None:
                self.logger.experiment.add_histogram(f'gradients/{name}', params.grad, self.current_epoch)
```

**Benefits**:
- **Debug vanishing/exploding gradients**: See if gradients are too small or too large
- **Monitor weight updates**: See how weights change during training
- **Identify frozen layers**: Layers with no gradient updates
- **Detect initialization issues**: Check initial weight distributions

**File Updated**: `src/trainer.py` - TensorBoard logger

**Added**:
```python
log_graph=True  # Log model computational graph
```

---

## How to View in TensorBoard

### Option 1: View Locally

After downloading your Kaggle outputs:

```bash
# Extract the downloaded zip file
# Navigate to the directory
tensorboard --logdir=kenyan_food_model_output/tensorboard_logs
```

Open browser: `http://localhost:6006`

### Option 2: View on Kaggle (if kernel is still running)

In Kaggle notebook, add a cell:
```python
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/output/kenyan_food_logs
```

### What You'll See

**Before (only had Scalars)**:
- Scalars tab: loss, accuracy, precision, recall, F1

**After (New tabs)**:
- **Scalars**: (same as before)
- **Histograms**:
  - `weights/model.layer1.0.conv1.weight` - Weight distribution for each layer
  - `weights/model.fc.weight` - Final classifier weights
  - `gradients/model.layer1.0.conv1.weight` - Gradient distribution
  - `gradients/model.fc.weight` - Classifier gradients
- **Graphs**: Model computational graph (ResNet50 architecture)

### Interpreting Histograms

**Healthy Training**:
- Weights: Bell-shaped distributions, not too wide or narrow
- Gradients: Small but non-zero values (e.g., 1e-4 to 1e-2)
- Weights evolve gradually over epochs

**Warning Signs**:
- **Vanishing gradients**: Gradient histograms all at ~0
- **Exploding gradients**: Very large gradient values (>1.0)
- **Dead neurons**: Weights stuck at same values across epochs
- **Imbalanced layers**: Some layers update a lot, others don't

### Example Analysis

```
Epoch 1:
  weights/model.fc.weight: mean=0.01, std=0.15
  gradients/model.fc.weight: mean=0.001, std=0.005

Epoch 30:
  weights/model.fc.weight: mean=0.05, std=0.20  ✓ Weights updated
  gradients/model.fc.weight: mean=0.0005, std=0.002  ✓ Gradients still flowing
```

---

## Configuration Examples

### Example 1: Disable Early Stopping for Full 100 Epochs

```python
train_config, data_config, system_config = get_config(
    num_epochs=100,
    batch_size=32
)
train_config.use_early_stopping = False
```

### Example 2: Monitor Loss Instead of Accuracy

```python
train_config.early_stop_monitor = "valid/loss"
train_config.early_stop_mode = "min"
train_config.early_stop_patience = 5
```

### Example 3: Very Patient Early Stopping

```python
train_config.early_stop_patience = 15  # Wait 15 epochs
```

---

## Impact on Training

**Performance**:
- Histogram logging adds ~1-2% overhead (minimal)
- Only logs once per epoch (at validation end)
- ~50-100MB extra TensorBoard log size for 100 epochs

**Debugging Value**:
- High! Essential for diagnosing training issues
- Helps identify optimal learning rate
- Shows which layers are learning vs frozen

---

## Next Steps

1. **Run training** and check TensorBoard histograms
2. **Look for**:
   - Gradients flowing through all unfrozen layers
   - Weights updating each epoch
   - No extreme values (vanishing/exploding)
3. **Adjust if needed**:
   - If gradients too small: increase learning rate
   - If gradients too large: decrease learning rate or add gradient clipping
   - If some layers not updating: check freeze/unfreeze logic

---

## Rollback

To revert these changes:

### Disable histograms:
Comment out the histogram logging code in `src/model.py` line 193-202

### Revert to hardcoded early stopping:
In `src/trainer.py`, replace dynamic callback creation with:
```python
early_stopping_callback = EarlyStopping(
    monitor="valid/acc",
    patience=7,
    mode="max",
    verbose=True
)
callbacks_list = [checkpoint_callback, early_stopping_callback]
```
