"""
Utility script to pre-download model weights.
Run this once to cache the weights locally.
"""

import torch
import torchvision
from torchvision.models import GoogLeNet_Weights
import ssl
import os

# Fix SSL certificate verification issues
print("Configuring SSL settings...")
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    print("✓ SSL verification disabled (for downloading only)")
except Exception as e:
    print(f"Warning: Could not disable SSL verification: {e}")

print("\nDownloading GoogleNet pre-trained weights...")
print("This may take a few minutes depending on your connection.")
cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints/")
print(f"Weights will be cached in: {cache_dir}")

try:
    # This will download and cache the weights
    model = torchvision.models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    print("\n✓ Download successful!")
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Verify the model works
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"✓ Model test successful! Output shape: {output.shape}")

except Exception as e:
    print(f"\n✗ Download failed: {e}")
    print("\nTroubleshooting:")
    print("1. Check your internet connection")
    print("2. Try using a VPN if torchvision downloads are blocked")
    print("3. Use pretrained=False in TrainingConfiguration for testing")
    print("4. Manually download from: https://download.pytorch.org/models/googlenet-1378be20.pth")
