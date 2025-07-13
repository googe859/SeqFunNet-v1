# MixerEncoding: Audio Classification with MLP-Mixer and Transformer

A comprehensive audio classification framework that combines MLP-Mixer and Transformer architectures for efficient and accurate sound recognition.

## 📁 Project Structure

```
mixerencoding/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── configs/                     # Configuration files
│   ├── default_config.py        # Default configurations
│   └── model_configs.py         # Model-specific configs
├── models/                      # Model implementations
│   ├── base_model.py            # Base model class
│   ├── seq_fun_net.py           # Main hybrid model
│   ├── mlp_mixer.py             # MLP-Mixer only
│   ├── transformer.py           # Transformer only
│   ├── attention.py             # Attention mechanisms
│   └── ablation_models.py       # Ablation study models
├── data/                        # Data processing
│   └── dataset.py               # Audio dataset class
├── utils/                       # Utility functions
│   ├── logger.py                # Logging utilities
│   ├── device.py                # Device management
│   └── helpers.py               # Helper functions
├── training/                    # Training utilities
│   └── trainer.py               # Training loop
├── scripts/                     # Convenience scripts
│   ├── train_seq_fun_net.py     # SeqFunNet training
│   ├── run_ablation_study.py    # Ablation experiments
│   └── test_models.py           # Model testing
├── examples/                    # Usage examples
│   └── basic_usage.py           # Basic usage demo
├── tests/                       # Unit tests
│   └── test_models.py           # Model tests
└── results/                     # Results directory
```

## 🚀 Features

- **Multiple Model Architectures**: MLP-Mixer, Transformer-only, and hybrid SeqFunNet
- **Comprehensive Ablation Studies**: Systematic component analysis
- **Flexible Data Processing**: Support for various audio features (MFCC, Mel-spectrogram, etc.)
- **Easy Experimentation**: Configurable training and ablation studies
- **Modular Design**: Clean architecture with comprehensive testing

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/your-username/mixerencoding.git
cd mixerencoding

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## 🚀 Quick Start

### Basic Usage

```python
from models import SeqFunNet, MLPMixerModel, TransformerOnly
from data import AudioDataModule
from training import Trainer

# Load dataset
data_module = AudioDataModule("dataset_npy/bird_6", feature_type="mfcc")

# Create model
model = SeqFunNet(num_classes=6, depth=4, patch_size=15)

# Train model
trainer = Trainer(model, data_module.train_loader, data_module.val_loader)
trainer.train(epochs=100)
```

### Command Line Interface

```bash
# Train SeqFunNet
python -m training.train --model SeqFunNet --dataset bird_6 --epochs 100

# Train MLP-Mixer only
python -m training.train --model MLPMixer --dataset bird_6 --epochs 100

# Train Transformer only
python -m training.train --model TransformerOnly --dataset bird_6 --epochs 100

# Run ablation study
python -m training.train --model ablation_no_transformer --dataset bird_6 --epochs 100
```

### Quick Scripts

```bash
# Train SeqFunNet with optimized parameters
python scripts/train_seq_fun_net.py --dataset bird_6 --epochs 100

# Run complete ablation study
python scripts/run_ablation_study.py --dataset bird_6 --epochs 50

# Test all models
python scripts/test_models.py

# Run basic usage examples
python examples/basic_usage.py
```

## 📊 Models

### SeqFunNet
Our main proposed model that combines:
- **Transformer Encoder**: For temporal feature extraction
- **MLP-Mixer**: For spatial feature mixing
- **GAM Attention**: For channel and spatial attention
- **Feature Fusion**: Multi-branch feature combination

### MLP-Mixer Only
Pure MLP-Mixer implementation for spatial feature processing:
- **Patch Embedding**: Divides input into patches
- **Token Mixing**: MLPs across spatial dimensions
- **Channel Mixing**: MLPs across feature dimensions
- **Global Pooling**: Final feature aggregation

### Transformer Only
Pure Transformer implementation for temporal feature processing:
- **Multi-Head Attention**: Captures temporal dependencies
- **Feed-Forward Networks**: Non-linear transformations
- **Position Encoding**: Temporal position information
- **Global Pooling**: Final feature aggregation

### Ablation Studies
Systematic component analysis:
- **Full Model**: Complete SeqFunNet
- **No Transformer**: Removes transformer branch
- **No Mixer**: Removes MLP-Mixer branch
- **No GAM**: Removes attention mechanism
- **No MLP**: Removes MLP branch
- **Different Fusion**: Add, concatenate, or weighted fusion

## 📁 Datasets

### Supported Data Formats

The framework supports pre-processed audio features stored as numpy arrays. Your dataset should be organized as follows:

```
dataset_npy/
├── your_dataset/
│   └── feature_type/           # mfcc, mel, spectrogram, etc.
│       ├── train.npy           # Training features (N, H, W) or (N, C, H, W)
│       ├── train_label.npy     # Training labels (N,)
│       ├── val.npy             # Validation features
│       ├── val_label.npy       # Validation labels
│       ├── test.npy            # Test features
│       └── test_label.npy      # Test labels
```

**Feature Format:**
- Shape: `(N, H, W)` or `(N, C, H, W)`
- N: Number of samples
- H: Feature height (time dimension)
- W: Feature width (frequency dimension)
- C: Number of channels (optional, default=1)

**Label Format:**
- Shape: `(N,)`
- Values: Integer labels starting from 0
- Example: For 6-class classification, use labels 0, 1, 2, 3, 4, 5

### Training with Custom Data

```bash
# Basic training
python train.py --dataset your_dataset --epochs 100

# With custom parameters
python train.py --dataset your_dataset --model SeqFunNet --depth 6 --patch_size 10 --batch_size 16
```

## 🔬 Experiments

### Ablation Studies
We provide comprehensive ablation studies on:
- MLP-Mixer depth and patch size
- Transformer architecture variations
- Attention mechanism effectiveness
- Feature fusion strategies

### Batch Testing
Automated batch testing for hyperparameter optimization:
```bash
python scripts/run_ablation_study.py --dataset bird_6 --config full
```

## 🧪 Usage Examples

### Custom Dataset
```python
from data import AudioDataModule
from models import SeqFunNet

# Create custom dataset
data_module = AudioDataModule(
    data_path="path/to/audio/files",
    feature_type="mfcc",
    batch_size=32
)

# Train custom model
model = SeqFunNet(num_classes=5, depth=6, patch_size=10)
```

### Ablation Study
```python
from models import create_ablation_model

# Create ablation model
model = create_ablation_model("no_transformer", num_classes=6)

# Get ablation information
info = model.get_ablation_info()
print(f"Model: {info['model_name']}")
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 models/ data/ utils/ training/
black models/ data/ utils/ training/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

