#!/usr/bin/env python3
"""
Device Utilities - GPU/CPU Selection
-------------------------------------
Merkezi device selection ve info fonksiyonları.
Apple Silicon (MPS), CUDA ve CPU desteği.
"""

import torch


def get_device(verbose=True):
    """
    En iyi mevcut device'ı seçer: CUDA > MPS > CPU
    
    Args:
        verbose (bool): Device bilgilerini yazdır
        
    Returns:
        torch.device: Seçilen device
        str: Device açıklaması
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
        cuda_version = torch.version.cuda
        
        if verbose:
            print(f"🖥️  Device: {device_name}")
            print(f"   CUDA Version: {cuda_version}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        return device, device_name
    
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = "MPS (Apple Silicon GPU)"
        
        if verbose:
            print(f"🖥️  Device: {device_name}")
            print(f"   ⚡ Using Metal Performance Shaders")
            print(f"   💡 Tip: MPS is optimized for M1/M2/M3 chips")
        
        return device, device_name
    
    else:
        device = torch.device('cpu')
        device_name = "CPU"
        
        if verbose:
            print(f"🖥️  Device: {device_name}")
            print(f"   ⚠️  Training will be slow. Consider using GPU.")
            print(f"   💡 Tip: Use Google Colab for free GPU access")
        
        return device, device_name


def print_device_info():
    """Device bilgilerini detaylı yazdırır"""
    print("\n" + "=" * 70)
    print("🔍 DEVICE COMPATIBILITY CHECK")
    print("=" * 70)
    
    # PyTorch version
    print(f"\n📦 PyTorch Version: {torch.__version__}")
    
    # CUDA check
    print(f"\n🎮 CUDA:")
    if torch.cuda.is_available():
        print(f"   ✅ Available")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Device Count: {torch.cuda.device_count()}")
    else:
        print(f"   ❌ Not available")
    
    # MPS check (Apple Silicon)
    print(f"\n🍎 MPS (Apple Silicon):")
    if torch.backends.mps.is_available():
        print(f"   ✅ Available")
        print(f"   MPS is built and ready to use")
    else:
        print(f"   ❌ Not available")
        if not torch.backends.mps.is_built():
            print(f"   Info: PyTorch not compiled with MPS")
    
    # CPU info
    print(f"\n💻 CPU:")
    print(f"   ✅ Always available (fallback)")
    
    # Recommended device
    device, device_name = get_device(verbose=False)
    print(f"\n🎯 Recommended Device: {device_name}")
    
    print("=" * 70)


def check_device_compatibility(config):
    """
    Config ile device uyumluluğunu kontrol eder
    
    Args:
        config: TransformerConfig objesi
        
    Returns:
        bool: Uyumluysa True
    """
    device, device_name = get_device(verbose=False)
    
    # MPS için özel kontroller
    if device.type == 'mps':
        # Pin memory MPS'te desteklenmiyor
        if hasattr(config, 'PIN_MEMORY') and config.PIN_MEMORY:
            print("⚠️  WARNING: pin_memory=True is not supported on MPS")
            print("   Automatically disabling pin_memory")
            config.PIN_MEMORY = False
    
    return True

