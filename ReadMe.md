# Kenyan Food 13 Classification - Kaggle Competition

Project 2 in Deeplearning with OpenCV course week 7.

## Project Structure

```
prj-2-opencv-dl-pytorch/
├── src/                       # Modular source code (Phase 2)
│   ├── __init__.py           # Package initialization with exports
│   ├── config.py             # Configuration classes
│   ├── dataset.py            # Custom PyTorch Dataset
│   ├── datamodule.py         # Lightning DataModule
│   ├── model.py              # Lightning model (GoogleNet transfer learning)
│   ├── trainer.py            # Training orchestration
│   └── utils.py              # Utility functions
├── Project2_Kaggle_Competition_Classification-Round3.ipynb  # Original full notebook
├── kaggle_train.ipynb        # Simplified Kaggle notebook (uses src/)
├── requirements.txt          # Python dependencies
├── MIGRATION_GUIDE.md        # Migration documentation
└── ReadMe.md                 # This file
```

## Installation

### Local Environment (VS Code)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ramabyg/prj-2-opencv-dl-pytorch.git
   cd prj-2-opencv-dl-pytorch
   ```

2. **Install dependencies:**
   ```bash
   python -m pip install -r requirements.txt
   ```

   Or install individually:
   ```bash
   python -m pip install matplotlib torch torchvision lightning pandas
   ```

3. **Open in VS Code:**
   - Open the project folder in VS Code
   - Select Python kernel for Jupyter notebook
   - Run cells sequentially

### Kaggle Environment

1. **Upload to Kaggle:**
   - Push changes to GitHub
   - In Kaggle notebook, run:
     ```python
     !git clone https://github.com/ramabyg/prj-2-opencv-dl-pytorch.git
     import sys
     sys.path.insert(0, '/kaggle/working/prj-2-opencv-dl-pytorch')
     ```

2. **Or use as dataset:**
   - Zip the `src/` folder
   - Upload to Kaggle as dataset
   - Add dataset to your notebook

## Usage

### Quick Start (Simplified - Phase 2)

```python
# Import everything you need from src
from src import get_config, KenyanFood13DataModule, KenyanFood13Classifier, train_model

# Get configurations (auto-detects Kaggle/local)
train_cfg, data_cfg, sys_cfg = get_config(num_epochs=20, batch_size=64)

# Create and setup data
data_module = KenyanFood13DataModule(data_cfg)
data_module.setup()

# Create model
model = KenyanFood13Classifier(train_cfg, data_module.num_classes)

# Train!
train_model(train_cfg, data_cfg, sys_cfg, model, data_module)
```

### Detailed Usage (Original Approach)

```python
# Import modules
from src.config import TrainingConfiguration, DataConfiguration, SystemConfiguration
from src.utils import calculate_dataset_mean_std

# Create configurations (auto-detects environment)
train_config = TrainingConfiguration()
data_config = DataConfiguration()
system_config = SystemConfiguration()

# Calculate dataset statistics (optional)
mean, std = calculate_dataset_mean_std(
    annotations_file=data_config.annotations_file,
    img_dir=data_config.img_dir,
    sample_size=1000
)
```

### Environment Detection

The code automatically detects whether it's running locally or on Kaggle:
- **Local:** Uses `../data/kenyan-food-13/` paths
- **Kaggle:** Uses `/kaggle/input/kenyan-food-13/` paths

You can override this behavior in configuration classes.

## Development Workflow

1. **Develop locally in VS Code:**
   - Edit `.py` files in `src/`
   - Test with small dataset locally
   - Use full VS Code features (debugging, IntelliSense)

2. **Version control:**
   ```bash
   git add .
   git commit -m "Update model architecture"
   git push origin main
   ```

3. **Execute on Kaggle:**
   - Clone repo in Kaggle notebook (browser)
   - Run training with full dataset and GPUs
   - Download results

## Features

- ✅ **Modular architecture** - Reusable components in `src/`
- ✅ **Environment auto-detection** - Works locally and on Kaggle
- ✅ **PyTorch Lightning** - Simplified training with multi-GPU support
- ✅ **Transfer learning** - GoogleNet pre-trained model
- ✅ **Comprehensive metrics** - Accuracy, F1, Precision, Recall
- ✅ **TensorBoard logging** - Experiment tracking
- ✅ **Configurable** - Easy hyperparameter tuning

## Dependencies

- torch>=2.0.0
- torchvision>=0.15.0
- lightning>=2.0.0
- pandas>=1.5.0
- matplotlib>=3.7.0
- torchmetrics>=1.0.0
