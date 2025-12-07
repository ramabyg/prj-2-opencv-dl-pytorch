"""
Test script to verify Phase 1 migration.
Run this to ensure all modules import correctly.
"""

import sys
import os

# Add src to path (simulates Kaggle import)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_import():
    """Test configuration module imports."""
    print("Testing config imports...")
    from src.config import (
        TrainingConfiguration,
        DataConfiguration,
        SystemConfiguration,
        get_config,
        is_kaggle_environment,
        get_environment_name
    )

    # Test environment detection
    env = get_environment_name()
    print(f"  ✓ Detected environment: {env}")

    # Test configuration creation
    train_config = TrainingConfiguration()
    data_config = DataConfiguration()
    sys_config = SystemConfiguration()
    print(f"  ✓ Configurations created successfully")
    print(f"    - Train config: {train_config.num_epochs} epochs")
    print(f"    - Data config: {data_config.batch_size} batch size")
    print(f"    - System config: {sys_config.device} device")

    # Test get_config convenience function
    t, d, s = get_config(num_epochs=20)
    assert t.num_epochs == 20, "Override not applied"
    print(f"  ✓ get_config() with overrides works")

    print("✅ Config tests passed!\n")


def test_utils_import():
    """Test utility module imports."""
    print("Testing utils imports...")
    from src.utils import calculate_dataset_mean_std, get_imagenet_stats

    # Test ImageNet stats
    mean, std = get_imagenet_stats()
    assert len(mean) == 3 and len(std) == 3, "Invalid ImageNet stats"
    print(f"  ✓ ImageNet stats: mean={mean}, std={std}")

    print("✅ Utils tests passed!\n")


def test_package_structure():
    """Test package structure."""
    print("Testing package structure...")
    import src

    assert hasattr(src, '__version__'), "Missing version"
    print(f"  ✓ Package version: {src.__version__}")

    print("✅ Package structure tests passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1 Migration Test Suite")
    print("=" * 60 + "\n")

    try:
        test_package_structure()
        test_config_import()
        test_utils_import()

        print("=" * 60)
        print("🎉 All tests passed! Phase 1 migration successful!")
        print("=" * 60)
        print("\nYou can now:")
        print("  1. Commit changes to git")
        print("  2. Push to GitHub")
        print("  3. Use in Kaggle by cloning the repo")
        print("  4. Proceed to Phase 2 (optional)")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
