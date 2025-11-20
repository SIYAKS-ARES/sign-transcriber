#!/usr/bin/env python3
"""
Setup Validation Script
-----------------------
Proje yapısını ve dependency'leri kontrol eder.

Kullanım:
    python validate_setup.py
"""

import os
import sys
import importlib
from pathlib import Path

# Initialize colorama
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_COLORAMA = True
except:
    HAS_COLORAMA = False
    Fore = None
    Style = None

def print_success(msg):
    if HAS_COLORAMA:
        print(f"{Fore.GREEN}✓ {msg}{Style.RESET_ALL}")
    else:
        print(f"✓ {msg}")

def print_error(msg):
    if HAS_COLORAMA:
        print(f"{Fore.RED}✗ {msg}{Style.RESET_ALL}")
    else:
        print(f"✗ {msg}")

def print_warning(msg):
    if HAS_COLORAMA:
        print(f"{Fore.YELLOW}⚠ {msg}{Style.RESET_ALL}")
    else:
        print(f"⚠ {msg}")

def print_info(msg):
    if HAS_COLORAMA:
        print(f"{Fore.CYAN}ℹ {msg}{Style.RESET_ALL}")
    else:
        print(f"ℹ {msg}")

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n" + "="*80)
    print("🔍 CHECKING DEPENDENCIES")
    print("="*80)
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('cv2', 'OpenCV'),
        ('mediapipe', 'MediaPipe'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('tqdm', 'tqdm'),
    ]
    
    optional_packages = [
        ('torchinfo', 'TorchInfo'),
        ('yaml', 'PyYAML'),
        ('joblib', 'Joblib'),
        ('wandb', 'Weights & Biases'),
        ('tensorboard', 'TensorBoard'),
    ]
    
    all_ok = True
    
    print("\n📦 Required Packages:")
    for package, name in required_packages:
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print_success(f"{name:20s} {version}")
        except ImportError:
            print_error(f"{name:20s} NOT INSTALLED")
            all_ok = False
    
    print("\n📦 Optional Packages:")
    for package, name in optional_packages:
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print_success(f"{name:20s} {version}")
        except ImportError:
            print_warning(f"{name:20s} not installed (optional)")
    
    return all_ok

def check_project_structure():
    """Check if project directory structure is correct"""
    print("\n" + "="*80)
    print("📁 CHECKING PROJECT STRUCTURE")
    print("="*80)
    
    required_dirs = [
        'data',
        'scripts',
        'models',
        'checkpoints',
        'results',
        'logs',
    ]
    
    required_files = [
        'config.py',
        'train.py',
        'evaluate.py',
        'visualize_attention.py',
        'requirements.txt',
        'README.md',
        'RUN_PIPELINE.md',
        'scripts/01_select_videos.py',
        'scripts/02_extract_keypoints.py',
        'scripts/03_normalize_data.py',
        'models/__init__.py',
        'models/transformer_model.py',
    ]
    
    all_ok = True
    
    print("\n📂 Required Directories:")
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print_success(f"{dir_path:30s} exists")
        else:
            print_error(f"{dir_path:30s} MISSING")
            all_ok = False
    
    print("\n📄 Required Files:")
    for file_path in required_files:
        if os.path.isfile(file_path):
            print_success(f"{file_path:50s} exists")
        else:
            print_error(f"{file_path:50s} MISSING")
            all_ok = False
    
    return all_ok

def check_data_availability():
    """Check if data directories exist"""
    print("\n" + "="*80)
    print("💾 CHECKING DATA AVAILABILITY")
    print("="*80)
    
    data_dirs = [
        '../Data/Train Data/train',
        '../Data/Validation Data/val',
        '../Data/Test Data & Valid, Labels/test',
        '../Data/Class ID',
    ]
    
    print("\n📊 Data Directories:")
    for data_dir in data_dirs:
        if os.path.isdir(data_dir):
            num_items = len(os.listdir(data_dir))
            print_success(f"{data_dir:50s} exists ({num_items} items)")
        else:
            print_warning(f"{data_dir:50s} NOT FOUND (may need to update path)")
    
    # Check CSV file
    csv_path = '../Data/Class ID/SignList_ClassId_TR_EN.csv'
    if os.path.isfile(csv_path):
        print_success(f"Class ID CSV: {csv_path}")
    else:
        print_warning(f"Class ID CSV not found: {csv_path}")
    
    return True

def check_config():
    """Check if config.py is properly configured"""
    print("\n" + "="*80)
    print("⚙️  CHECKING CONFIGURATION")
    print("="*80)
    
    try:
        from config import TransformerConfig
        config = TransformerConfig()
        
        print("\n📋 Configuration Summary:")
        print_info(f"Number of classes: {config.NUM_CLASSES}")
        print_info(f"Class names: {', '.join(config.CLASS_NAMES)}")
        print_info(f"Target class IDs: {config.TARGET_CLASS_IDS}")
        print_info(f"Batch size: {config.BATCH_SIZE}")
        print_info(f"Learning rate: {config.LEARNING_RATE}")
        print_info(f"Max epochs: {config.NUM_EPOCHS}")
        print_info(f"d_model: {config.D_MODEL}")
        print_info(f"Encoder layers: {config.NUM_ENCODER_LAYERS}")
        print_info(f"Attention heads: {config.NHEAD}")
        
        print_success("Configuration loaded successfully")
        return True
    except Exception as e:
        print_error(f"Configuration error: {str(e)}")
        return False

def check_class_mapping():
    """Check class ID mapping in processed data"""
    print("\n" + "="*80)
    print("🔢 CHECKING CLASS MAPPING")
    print("="*80)
    
    processed_dir = Path("data/processed")
    
    if not processed_dir.exists():
        print_warning("Processed data not found - skipping class mapping check")
        print_info("Run scripts/03_normalize_data.py first")
        return True
    
    try:
        # Check if processed files exist
        y_train_path = processed_dir / "y_train.npy"
        if not y_train_path.exists():
            print_warning("y_train.npy not found - skipping")
            return True
        
        # Load labels
        import numpy as np
        y_train = np.load(y_train_path)
        
        # Check config
        from config import TransformerConfig
        config = TransformerConfig()
        
        # Validate labels
        unique_labels = np.unique(y_train)
        
        print_info(f"Found labels: {unique_labels.tolist()}")
        print_info(f"Expected: [0, 1, 2, ...] (0-indexed)")
        print_info(f"Target class IDs: {config.TARGET_CLASS_IDS}")
        print_info(f"Class names: {config.CLASS_NAMES}")
        
        # Check if labels are 0-indexed
        if unique_labels.min() != 0:
            print_error(f"❌ Labels should start from 0, but found min={unique_labels.min()}")
            print_error("   Hint: Re-run scripts/03_normalize_data.py with updated code")
            return False
        
        # Check if labels are consecutive
        if unique_labels.max() != len(unique_labels) - 1:
            print_error(f"❌ Labels should be consecutive [0,1,2,...], found: {unique_labels}")
            return False
        
        # Check number of classes
        if len(unique_labels) != config.NUM_CLASSES:
            print_error(f"❌ Expected {config.NUM_CLASSES} classes, found {len(unique_labels)}")
            return False
        
        print_success("✅ Class mapping is correct:")
        for idx in unique_labels:
            orig_id = config.TARGET_CLASS_IDS[idx]
            name = config.CLASS_NAMES[idx]
            print_info(f"   Label {idx} → ClassId {orig_id} ({name})")
        
        return True
        
    except Exception as e:
        print_error(f"Error checking class mapping: {e}")
        return False

def check_cuda():
    """Check GPU/Device availability (CUDA, MPS, CPU)"""
    print("\n" + "="*80)
    print("🎮 CHECKING DEVICE COMPATIBILITY")
    print("="*80)
    
    try:
        import torch
        
        device_found = False
        
        # Check CUDA
        if torch.cuda.is_available():
            print_success("✅ CUDA (NVIDIA GPU) is available")
            print_info(f"   GPU: {torch.cuda.get_device_name(0)}")
            print_info(f"   CUDA Version: {torch.version.cuda}")
            print_info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            device_found = True
        else:
            print_info("❌ CUDA not available")
        
        # Check MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            print_success("✅ MPS (Apple Silicon GPU) is available")
            print_info(f"   Device: M1/M2/M3 GPU detected")
            print_info(f"   ⚡ Metal Performance Shaders enabled")
            device_found = True
        else:
            if not torch.backends.mps.is_built():
                print_info("❌ MPS not available (PyTorch not compiled with MPS)")
            else:
                print_info("❌ MPS not available")
        
        # CPU fallback
        print_info("✅ CPU is always available (fallback)")
        
        if device_found:
            print_success("\n🎯 GPU acceleration available!")
            return True
        else:
            print_warning("\n⚠️  No GPU found - Training will use CPU")
            print_info("   Tip: CPU training is much slower")
            print_info("   Consider using Google Colab for free GPU")
            return False
        
        return True
    except Exception as e:
        print_error(f"Error checking CUDA: {str(e)}")
        return False

def check_python_version():
    """Check Python version"""
    print("\n" + "="*80)
    print("🐍 CHECKING PYTHON VERSION")
    print("="*80)
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print_info(f"Python version: {version_str}")
    
    if version.major == 3 and version.minor >= 8:
        print_success("Python version is compatible (3.8+)")
        return True
    else:
        print_error("Python version should be 3.8 or higher")
        return False

def main():
    """Main validation function"""
    print("="*80)
    print("🔍 TRANSFORMER SIGN LANGUAGE - SETUP VALIDATION")
    print("="*80)
    
    results = []
    
    # Check Python version
    results.append(("Python Version", check_python_version()))
    
    # Check dependencies
    results.append(("Dependencies", check_dependencies()))
    
    # Check project structure
    results.append(("Project Structure", check_project_structure()))
    
    # Check configuration
    results.append(("Configuration", check_config()))
    
    # Check CUDA
    results.append(("CUDA/GPU", check_cuda()))
    
    # Check data availability
    results.append(("Data Availability", check_data_availability()))
    
    # Check class mapping (if processed data exists)
    results.append(("Class Mapping", check_class_mapping()))
    
    # Summary
    print("\n" + "="*80)
    print("📊 VALIDATION SUMMARY")
    print("="*80)
    
    all_passed = True
    for check_name, passed in results:
        if passed:
            print_success(f"{check_name:25s} PASSED")
        else:
            print_error(f"{check_name:25s} FAILED")
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print_success("✅ ALL CHECKS PASSED - Ready to run pipeline!")
        print_info("\nNext steps:")
        print_info("  1. python scripts/01_select_videos.py")
        print_info("  2. python scripts/02_extract_keypoints.py")
        print_info("  3. python scripts/03_normalize_data.py")
        print_info("  4. python train.py")
        print_info("  5. python evaluate.py")
        print_info("  6. python visualize_attention.py")
        print_info("\nSee RUN_PIPELINE.md for detailed instructions.")
    else:
        print_error("❌ SOME CHECKS FAILED - Please fix issues before running pipeline")
        print_info("\nRecommendations:")
        print_info("  1. Install missing dependencies: pip install -r requirements.txt")
        print_info("  2. Check data paths in config.py")
        print_info("  3. Ensure all required files exist")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())

