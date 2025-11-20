"""
Transformer Sign Language Recognition - Configuration File

Bu dosya projenin tüm hiperparametrelerini ve konfigürasyonlarını içerir.

GÜNCEL: 226 Kelime (Tüm AUTSL Dataset)
- 10 kelime → 226 kelime genişletildi (7 Ekim 2025)
- Toplam 36,302 video (train: 28,142, val: 4,418, test: 3,742)
"""

import os
from utils.load_class_names import load_all_class_names


class TransformerConfig:
    """
    Transformer tabanlı işaret dili tanıma modeli için konfigürasyon sınıfı
    
    Kullanım:
        config = TransformerConfig()
        model = TransformerSignLanguageClassifier(
            input_dim=config.INPUT_DIM,
            d_model=config.D_MODEL,
            ...
        )
    """
    
    # ==================== DATA PARAMETERS ====================
    INPUT_DIM = 258           # MediaPipe keypoint dimension (pose:99 + face:33 + hands:126)
    MAX_SEQ_LENGTH = 200      # Maximum video length (frames)
    NUM_CLASSES = 226         # Number of sign classes (TÜM AUTSL DATASET - 226 kelime)
    
    # ==================== MODEL ARCHITECTURE ====================
    D_MODEL = 256             # Transformer embedding dimension (226 sınıf için yeterli)
    NHEAD = 8                 # Number of attention heads (d_model % nhead == 0 olmalı)
    NUM_ENCODER_LAYERS = 6    # Number of Transformer encoder blocks
    DIM_FEEDFORWARD = 1024    # Feedforward network hidden dimension
    DROPOUT = 0.2             # Dropout rate (0.1→0.2: daha güçlü regularization)
    
    # Pooling strategy: 'gap' (Global Average Pooling), 'cls' (CLS token), 'last' (Last hidden state)
    POOLING_TYPE = 'gap'
    
    # ==================== TRAINING PARAMETERS ====================
    BATCH_SIZE = 16           # Batch size for training (32→16: 226 sınıf için memory optimizasyonu)
    LEARNING_RATE = 1e-4      # Initial learning rate
    WEIGHT_DECAY = 1e-5       # AdamW weight decay (L2 regularization)
    NUM_EPOCHS = 100          # Maximum number of training epochs
    WARMUP_EPOCHS = 15        # Number of warmup epochs for learning rate scheduler (10→15: daha yavaş warmup)
    
    # ==================== OPTIMIZATION ====================
    OPTIMIZER = 'adamw'       # Optimizer: 'adam', 'adamw', 'sgd'
    SCHEDULER = 'cosine'      # LR scheduler: 'cosine', 'step', 'plateau'
    LABEL_SMOOTHING = 0.15    # Label smoothing epsilon for cross-entropy loss (0.1→0.15: 226 sınıf için)
    
    # ==================== REGULARIZATION ====================
    GRADIENT_CLIP = 1.0       # Gradient clipping max norm
    EARLY_STOPPING_PATIENCE = 20  # Early stopping patience (epochs) (10→20: 226 sınıf daha sabırlı)
    
    # ==================== DATA PATHS ====================
    # Ana veri dizini (proje root'tan göreceli)
    BASE_DATA_DIR = '../Data'
    TRAIN_VIDEO_DIR = os.path.join(BASE_DATA_DIR, 'Train Data/train')
    TRAIN_LABELS_CSV = os.path.join(BASE_DATA_DIR, 'Train Data/train_labels.csv')
    CLASS_MAPPING_CSV = os.path.join(BASE_DATA_DIR, 'Class ID/SignList_ClassId_TR_EN.csv')
    
    # Transformer projesi veri dizinleri
    DATA_DIR = 'data'
    PROCESSED_DATA_DIR = 'data/processed'
    KEYPOINTS_DIR = 'data/keypoints'
    
    # Model ve sonuç dizinleri
    CHECKPOINT_DIR = 'checkpoints'
    RESULTS_DIR = 'results'
    LOG_DIR = 'logs'
    
    # ==================== DATA SPLIT RATIOS ====================
    TRAIN_RATIO = 0.8         # Training set ratio
    VAL_RATIO = 0.1           # Validation set ratio
    TEST_RATIO = 0.1          # Test set ratio
    RANDOM_STATE = 42         # Random seed for reproducibility
    
    # ==================== MEDIAPIPE PARAMETERS ====================
    MP_MIN_DETECTION_CONFIDENCE = 0.5   # MediaPipe detection confidence
    MP_MIN_TRACKING_CONFIDENCE = 0.5    # MediaPipe tracking confidence
    MP_MODEL_COMPLEXITY = 1             # MediaPipe model complexity (0, 1, or 2)
    
    # ==================== DATA AUGMENTATION ====================
    USE_AUGMENTATION = False            # Enable data augmentation
    AUGMENTATION_PROBABILITY = 0.5      # Probability of applying augmentation
    NOISE_STD = 0.01                    # Gaussian noise standard deviation
    
    # ==================== DEVICE SETTINGS ====================
    NUM_WORKERS = 4           # DataLoader num_workers
    PIN_MEMORY = True         # DataLoader pin_memory (GPU için True)
    
    # ==================== LOGGING ====================
    SAVE_FREQUENCY = 5        # Save checkpoint every N epochs
    LOG_FREQUENCY = 10        # Log metrics every N batches
    
    # ==================== CLASS NAMES ====================
    # 226 kelime (TÜM AUTSL DATASET)
    CLASS_NAMES = load_all_class_names()  # SignList_ClassId_TR_EN.csv'den otomatik yüklenir (226 kelime)
    TARGET_CLASS_IDS = list(range(0, 226))  # Tüm sınıflar: [0, 1, 2, ..., 225]
    
    # ==================== ALTERNATIVE CONFIGURATIONS ====================
    # Farklı model boyutları için alternatif konfigürasyonlar
    
    @classmethod
    def get_tiny_config(cls):
        """Tiny model configuration (~1M params)"""
        config = cls()
        config.D_MODEL = 128
        config.NHEAD = 4
        config.NUM_ENCODER_LAYERS = 3
        config.DIM_FEEDFORWARD = 512
        return config
    
    @classmethod
    def get_small_config(cls):
        """Small model configuration (~5M params)"""
        config = cls()
        config.D_MODEL = 256
        config.NHEAD = 8
        config.NUM_ENCODER_LAYERS = 4
        config.DIM_FEEDFORWARD = 1024
        return config
    
    @classmethod
    def get_base_config(cls):
        """Base model configuration (~8M params) - DEFAULT"""
        return cls()
    
    @classmethod
    def get_large_config(cls):
        """Large model configuration (~40M params)"""
        config = cls()
        config.D_MODEL = 512
        config.NHEAD = 16
        config.NUM_ENCODER_LAYERS = 12
        config.DIM_FEEDFORWARD = 2048
        config.BATCH_SIZE = 16  # Daha büyük model için küçük batch
        return config
    
    def __repr__(self):
        """Pretty print configuration"""
        config_str = "TransformerConfig(\n"
        for key, value in self.__class__.__dict__.items():
            if not key.startswith('_') and not callable(value) and key.isupper():
                config_str += f"  {key}: {getattr(self, key)}\n"
        config_str += ")"
        return config_str
    
    def to_dict(self):
        """Convert configuration to dictionary"""
        return {
            key: getattr(self, key)
            for key in dir(self)
            if not key.startswith('_') and not callable(getattr(self, key)) and key.isupper()
        }
    
    def save(self, filepath):
        """Save configuration to YAML file"""
        import yaml
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        print(f"✅ Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load configuration from YAML file"""
        import yaml
        config = cls()
        with open(filepath, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        for key, value in loaded_config.items():
            setattr(config, key, value)
        
        print(f"✅ Configuration loaded from {filepath}")
        return config


# ==================== HELPER FUNCTIONS ====================

def get_config(model_size='base'):
    """
    Get configuration by model size
    
    Args:
        model_size (str): 'tiny', 'small', 'base', or 'large'
    
    Returns:
        TransformerConfig: Configuration object
    """
    if model_size == 'tiny':
        return TransformerConfig.get_tiny_config()
    elif model_size == 'small':
        return TransformerConfig.get_small_config()
    elif model_size == 'base':
        return TransformerConfig.get_base_config()
    elif model_size == 'large':
        return TransformerConfig.get_large_config()
    else:
        raise ValueError(f"Unknown model size: {model_size}. Choose from ['tiny', 'small', 'base', 'large']")


def print_config_comparison():
    """Print comparison of different model configurations"""
    configs = {
        'Tiny': TransformerConfig.get_tiny_config(),
        'Small': TransformerConfig.get_small_config(),
        'Base': TransformerConfig.get_base_config(),
        'Large': TransformerConfig.get_large_config()
    }
    
    print("\n" + "="*80)
    print("MODEL CONFIGURATION COMPARISON")
    print("="*80)
    print(f"{'Config':<10} {'d_model':<10} {'heads':<8} {'layers':<8} {'FFN':<10} {'Batch':<8}")
    print("-"*80)
    
    for name, config in configs.items():
        print(f"{name:<10} {config.D_MODEL:<10} {config.NHEAD:<8} "
              f"{config.NUM_ENCODER_LAYERS:<8} {config.DIM_FEEDFORWARD:<10} {config.BATCH_SIZE:<8}")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    # Test configuration
    print("\n🔧 Testing TransformerConfig...\n")
    
    # Default config
    config = TransformerConfig()
    print("Default (Base) Configuration:")
    print("-" * 50)
    print(config)
    
    # Print comparison
    print_config_comparison()
    
    # Test save/load
    config.save('config_test.yaml')
    loaded_config = TransformerConfig.load('config_test.yaml')
    
    # Clean up test file
    import os
    if os.path.exists('config_test.yaml'):
        os.remove('config_test.yaml')
    
    print("\n✅ Configuration test completed successfully!")

