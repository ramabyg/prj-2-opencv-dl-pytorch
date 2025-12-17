# Breaking Through 82-83% Accuracy Plateau

## Situation
- Achieved 82% validation accuracy in 20 epochs ✅
- Epochs 20-50: Only 0.01 (1%) improvement
- **Diagnosis:** Model has learned most patterns with current setup, needs more capacity or better data representation

## Most Effective Solutions (Ranked by Impact)

### 🥇 Strategy 1: Unfreeze All Layers (Highest Impact)
**Expected gain: +3-5% accuracy**

Your model is currently training only 80% of parameters (20% frozen). Unfreezing everything allows full adaptation to the Kenyan Food dataset.

**Implementation:**
```python
# In config.py or when calling get_config()
freeze_pct: 0.0  # Train all layers
learning_rate: 0.0001  # Lower LR when training all parameters (was 0.0003)
```

**Why it works:**
- Early layers (currently frozen) learn basic features that may need adjustment
- Full model can specialize to your specific food categories
- Most powerful approach for transfer learning

**Trade-off:** Slower training, needs lower LR to avoid destabilizing pretrained weights

---

### 🥈 Strategy 2: Increase Image Resolution (High Impact)
**Expected gain: +2-4% accuracy**

EfficientNetV2 was trained on larger images. Current: 384×384, but it supports up to 480×480.

**Implementation:**
```python
# In datamodule.py, line ~65 where get_model_preprocessing is called
# Or override in config:
input_size: 480  # From 384
```

**Why it works:**
- Preserves more fine-grained details (important for food textures)
- Closer to original training resolution
- EfficientNetV2-S is designed for this

**Trade-off:** 1.6× more GPU memory, ~30% slower training

---

### 🥉 Strategy 3: Upgrade to EfficientNetV2-M (Medium Impact)
**Expected gain: +2-3% accuracy**

Larger model with more capacity to learn complex patterns.

**Implementation:**
```python
# In config.py
model_name: "efficientnetv2_m"  # From "efficientnetv2"
batch_size: 24  # Reduce from 32 (model is larger)
```

**Why it works:**
- 54M parameters vs 21M (2.5× more capacity)
- Better at distinguishing similar-looking foods
- Still efficient compared to older architectures

**Trade-off:** ~50% more memory, slower training

---

### Strategy 4: Add Strong Data Augmentation (Medium Impact)
**Expected gain: +1-3% accuracy**

Current augmentation is basic. Add modern techniques to improve generalization.

**Implementation in `src/datamodule.py`:**

```python
# Around line 76-85, in train transforms, add BEFORE RandomHorizontalFlip:
from torchvision.transforms import RandAugment

train_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.RandAugment(num_ops=2, magnitude=9),  # ADD THIS
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
```

**Why it works:**
- RandAugment: random color, rotation, sharpness changes
- Helps model generalize to variations in lighting, angle, plating
- Proven to improve accuracy on food datasets

**Trade-off:** Slightly slower data loading

---

### Strategy 5: Label Smoothing (Low-Medium Impact)
**Expected gain: +0.5-2% accuracy**

Prevents overconfidence, improves generalization.

**Implementation in `src/model.py`:**

```python
# Around line 150, replace:
self.criterion = nn.CrossEntropyLoss()

# With:
self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Why it works:**
- Softens hard targets (e.g., [0,0,1,0] → [0.008, 0.008, 0.92, 0.008])
- Reduces overfitting when validation accuracy plateaus
- Standard technique for image classification

**Trade-off:** None (just one parameter change)

---

### Strategy 6: Longer Cosine Annealing (Low Impact)
**Expected gain: +0.5-1% accuracy**

Let learning rate decay more gradually over more epochs.

**Implementation:**
```python
# When calling get_config():
num_epochs: 100  # From 50
# Cosine scheduler will decay more slowly
```

**Why it works:**
- More time for fine-tuning at lower learning rates
- Can discover small improvements missed with faster decay

**Trade-off:** 2× training time

---

## 🎯 Recommended Action Plan

### Phase 1: Quick Wins (Try First)
1. **Unfreeze all layers** (Strategy 1) - biggest impact, no code changes
2. **Add label smoothing** (Strategy 5) - one line change
3. Train for 50 epochs with `freeze_pct=0.0, learning_rate=0.0001`

**Expected result: 85-88% accuracy**

### Phase 2: If Still < 85%
4. **Increase resolution to 480** (Strategy 2)
5. **Add RandAugment** (Strategy 4)
6. Train for 70 epochs

**Expected result: 87-90% accuracy**

### Phase 3: Final Push (If targeting > 90%)
7. **Upgrade to EfficientNetV2-M** (Strategy 3)
8. Combine all above strategies
9. Train for 100 epochs

**Expected result: 90-92% accuracy**

---

## Implementation: Quick Config Updates

### For Phase 1 (Immediate Try):

```python
# In kaggle_train.ipynb or when calling get_config():
train_config, data_config, system_config = get_config(
    num_epochs=50,
    freeze_pct=0.0,           # CHANGE: Unfreeze all (was 0.2)
    learning_rate=0.0001,     # CHANGE: Lower LR (was 0.0003)
    batch_size=32,
    optimizer="adamw",
    scheduler="cosine",
    weight_decay=0.01
)
```

**AND** in `src/model.py` around line 150:
```python
self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # ADD label_smoothing
```

---

## Monitoring Your Changes

After implementing, watch TensorBoard for:

✅ **Good signs:**
- Validation accuracy increases 0.5-1% per epoch initially
- Training/validation gap stays < 5%
- Loss continues decreasing smoothly

⚠️ **Warning signs:**
- Validation loss increases while training decreases → reduce learning_rate
- No improvement after 10 epochs → try next strategy
- Training loss oscillates wildly → reduce learning_rate

---

## Expected Timeline

| Strategy | Training Time | Expected Accuracy |
|----------|--------------|-------------------|
| Current (20% frozen) | 50 epochs | 82-83% ✓ (done) |
| Phase 1 (0% frozen + smoothing) | 50 epochs | 85-88% |
| Phase 2 (+ resolution + augment) | 70 epochs | 87-90% |
| Phase 3 (+ larger model) | 100 epochs | 90-92% |

---

## Code Changes Required

### Minimal (Phase 1):
1. Update `get_config()` call with `freeze_pct=0.0, learning_rate=0.0001`
2. Add `label_smoothing=0.1` in model.py

### Moderate (Phase 2):
3. Add RandAugment import and transform in datamodule.py
4. Override input_size to 480

### Extensive (Phase 3):
5. Change model_name to "efficientnetv2_m"
6. Adjust batch_size if needed

---

## Quick Reference Commands

### Check current config:
```python
print(f"Freeze %: {train_config.freeze_pct}")
print(f"LR: {train_config.learning_rate}")
print(f"Model: {train_config.model_name}")
```

### See model parameter breakdown:
```python
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Training {trainable/total*100:.1f}% of parameters")
```
