"""
Transformer Sign Language Model Handler
========================================
PyTorch Transformer modelini yükler ve video frame'lerinden tahmin yapar.

Model: transformer-signlang projesinden eğitilmiş 226 sınıf Transformer modeli
- Input: 258 boyutlu MediaPipe keypoints (Pose:99 + Face:33 + Hands:126)
- Sequence Length: 200 frames
- Output: 226 sınıf üzerinde tahmin

Author: AI Assistant
Date: November 2024
"""

import os
import sys
from typing import List, Optional
from pathlib import Path
import numpy as np
import pickle

# PyTorch imports
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("❌ PyTorch yüklenemedi. Lütfen 'pip install torch' ile yükleyin.")
    torch = None
    nn = None

# MediaPipe imports
try:
    import mediapipe as mp
except ImportError:
    print("❌ MediaPipe yüklenemedi. Lütfen 'pip install mediapipe' ile yükleyin.")
    mp = None

# OpenCV import
try:
    import cv2
except ImportError:
    print("❌ OpenCV yüklenemedi. Lütfen 'pip install opencv-python' ile yükleyin.")
    cv2 = None

# Transformer model import
# transformer-signlang klasörünü path'e ekle
# Bu proje için dizin yapısı:
#   sign-transcriber/
#     ├── anlamlandirma-sistemi/
#     └── transformer-signlang/
# Dolayısıyla proje kökü, bu dosyanın bir üst dizinidir.
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent          # sign-transcriber/
TRANSFORMER_DIR = PROJECT_ROOT / "transformer-signlang"

# config.py ve models/ içinde import edebilmek için bu dizini sys.path'e ekliyoruz
if str(TRANSFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSFORMER_DIR))

try:
    from config import TransformerConfig
    from models.transformer_model import TransformerSignLanguageClassifier
except ImportError as e:
    print(f"❌ Transformer modelleri yüklenemedi: {e}")
    TransformerConfig = None
    TransformerSignLanguageClassifier = None


# ==================== CONSTANTS ====================
SEQ_LEN = 200  # Transformer model için 200 frame
FEATURE_DIM = 258  # MediaPipe keypoints: Pose(99) + Face(33) + Hands(126)
NUM_CLASSES = 226  # AUTSL dataset - tüm sınıflar

# Model paths
CHECKPOINT_PATH = TRANSFORMER_DIR / "checkpoints" / "best_model.pth"
SCALER_PATH = TRANSFORMER_DIR / "data" / "scaler.pkl"
CLASS_NAMES_CSV = PROJECT_ROOT / "Data" / "Class ID" / "SignList_ClassId_TR_EN.csv"


# ==================== MEDIAPIPE KEYPOINT EXTRACTION ====================

def extract_keypoints_from_frame(results):
    """
    Bir frame'den 258 boyutlu keypoint vektörü çıkarır
    
    Format: Pose(33×3=99) + Face(11×3=33) + LeftHand(21×3=63) + RightHand(21×3=63) = 258
    
    Args:
        results: MediaPipe Holistic sonuçları
        
    Returns:
        np.array: (258,) boyutunda keypoint vektörü
    """
    
    # Pose keypoints (33 × 3 = 99)
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] 
                        for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 3)
    
    # Yüz keypoints (sadece key noktalar: 11 × 3 = 33)
    face_key_indices = [33, 133, 362, 263, 61, 291, 78, 308, 13, 14, 17]
    
    if results.face_landmarks:
        face = np.array([[results.face_landmarks.landmark[i].x,
                         results.face_landmarks.landmark[i].y,
                         results.face_landmarks.landmark[i].z]
                        for i in face_key_indices]).flatten()
    else:
        face = np.zeros(11 * 3)
    
    # Sol el keypoints (21 × 3 = 63)
    if results.left_hand_landmarks:
        left_hand = np.array([[lm.x, lm.y, lm.z]
                             for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        left_hand = np.zeros(21 * 3)
    
    # Sağ el keypoints (21 × 3 = 63)
    if results.right_hand_landmarks:
        right_hand = np.array([[lm.x, lm.y, lm.z]
                              for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        right_hand = np.zeros(21 * 3)
    
    # Birleştir: 99 + 33 + 63 + 63 = 258
    keypoints = np.concatenate([pose, face, left_hand, right_hand])
    
    return keypoints


def frames_to_keypoint_sequence(frames: List[np.ndarray]) -> np.ndarray:
    """
    BGR frame listesinden (T, 258) keypoint sequence oluşturur
    
    Args:
        frames: BGR formatında OpenCV frame listesi
        
    Returns:
        np.array: (T, 258) keypoint sequence
    """
    if mp is None:
        raise RuntimeError("MediaPipe kurulmamış.")
    
    print(f"📹 MediaPipe işleme başlıyor: {len(frames)} frame")
    # Debug: MediaPipe modül yapısını yazdır
    try:
        mp_file = getattr(mp, "__file__", None)
        mp_attrs = [a for a in dir(mp) if "solution" in a.lower()]
        print(f"   🔍 MediaPipe module: {mp}")
        print(f"   🔍 MediaPipe file: {mp_file}")
        print(f"   🔍 MediaPipe attrs (solution*): {mp_attrs}")
    except Exception as debug_exc:
        print(f"   ⚠️ MediaPipe debug bilgisi okunamadı: {debug_exc}")
    
    # Farklı MediaPipe sürümleriyle uyumlu holistic erişimi
    if hasattr(mp, "solutions"):
        mp_holistic = mp.solutions.holistic
    else:
        try:
            from mediapipe import solutions as mp_solutions  # type: ignore
            mp_holistic = mp_solutions.holistic
            print("   ✅ MediaPipe holistic, 'from mediapipe import solutions' ile yüklendi.")
        except Exception as e:
            raise RuntimeError(
                "MediaPipe 'holistic' API'sine erişilemedi. "
                "Lütfen 'pip install mediapipe==0.10.14' benzeri bir sürüm kurulu olduğundan emin olun."
            ) from e
    sequence = []
    
    pose_detected = 0
    face_detected = 0
    left_hand_detected = 0
    right_hand_detected = 0
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        
        for i, frame in enumerate(frames):
            # BGR -> RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            # MediaPipe işleme
            results = holistic.process(image_rgb)
            
            # Keypoint çıkar
            keypoints = extract_keypoints_from_frame(results)
            sequence.append(keypoints)
            
            # İstatistik
            if results.pose_landmarks:
                pose_detected += 1
            if results.face_landmarks:
                face_detected += 1
            if results.left_hand_landmarks:
                left_hand_detected += 1
            if results.right_hand_landmarks:
                right_hand_detected += 1
    
    print(f"   ✅ Tespit oranları:")
    print(f"      Pose: {pose_detected}/{len(frames)} ({pose_detected/len(frames)*100:.1f}%)")
    print(f"      Face: {face_detected}/{len(frames)} ({face_detected/len(frames)*100:.1f}%)")
    print(f"      L.Hand: {left_hand_detected}/{len(frames)} ({left_hand_detected/len(frames)*100:.1f}%)")
    print(f"      R.Hand: {right_hand_detected}/{len(frames)} ({right_hand_detected/len(frames)*100:.1f}%)")
    
    sequence = np.array(sequence, dtype=np.float32)
    print(f"   📊 Sequence shape: {sequence.shape}")
    
    return sequence


# ==================== DATA PREPROCESSING ====================

def normalize_sequence(sequence: np.ndarray, scaler) -> np.ndarray:
    """
    Keypoint sekansını StandardScaler ile normalize eder
    
    Args:
        sequence: (num_frames, 258) numpy array
        scaler: StandardScaler objesi
        
    Returns:
        normalized_sequence: (num_frames, 258) normalized array
    """
    return scaler.transform(sequence)


def pad_or_truncate_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """
    Sekansı hedef uzunluğa getirir (padding veya truncation)
    
    Args:
        sequence: (num_frames, 258) numpy array
        target_length: Hedef frame sayısı (200)
        
    Returns:
        processed_sequence: (target_length, 258) numpy array
    """
    num_frames, feature_dim = sequence.shape
    
    if num_frames == target_length:
        return sequence
    elif num_frames > target_length:
        # Truncate: Son kısmı al
        return sequence[-target_length:]
    else:
        # Pad: Başa sıfır ekle
        pad_length = target_length - num_frames
        padding = np.zeros((pad_length, feature_dim), dtype=sequence.dtype)
        return np.concatenate([padding, sequence], axis=0)


def create_padding_mask(sequence: np.ndarray) -> np.ndarray:
    """
    Padding pozisyonları için mask oluşturur
    
    Args:
        sequence: (seq_len, 258) numpy array
        
    Returns:
        mask: (seq_len,) boolean array - True for padding positions
    """
    # Eğer tüm feature'lar 0 ise padding
    mask = (sequence.sum(axis=-1) == 0)
    return mask


# ==================== MODEL LOADING ====================

class LocalSignModel:
    """Transformer Sign Language Model Wrapper"""
    
    def __init__(self, model, scaler, class_names: List[str], device, config):
        self.model = model
        self.scaler = scaler
        self.class_names = class_names
        self.device = device
        self.config = config
        self.loaded = model is not None
        
        if self.loaded:
            print(f"✅ Model yüklendi:")
            print(f"   - Device: {device}")
            print(f"   - Sınıf sayısı: {len(class_names)}")
            print(f"   - Model type: Transformer (PyTorch)")


def load_class_names() -> List[str]:
    """
    226 sınıf ismini CSV'den yükler
    
    Returns:
        list: 226 elemanlı sınıf isimleri (Türkçe)
    """
    if not CLASS_NAMES_CSV.exists():
        print(f"⚠️  Class names CSV bulunamadı: {CLASS_NAMES_CSV}")
        return [f"CLASS_{i}" for i in range(NUM_CLASSES)]
    
    try:
        import pandas as pd
        df = pd.read_csv(CLASS_NAMES_CSV)
        df = df.sort_values('ClassId')
        class_names = df['TR'].tolist()
        
        if len(class_names) != NUM_CLASSES:
            raise ValueError(f"Beklenen {NUM_CLASSES} sınıf, bulunan {len(class_names)}")
        
        print(f"✅ Sınıf isimleri yüklendi: {len(class_names)} sınıf")
        return class_names
    
    except Exception as e:
        print(f"⚠️  Sınıf isimleri yüklenirken hata: {e}")
        return [f"CLASS_{i}" for i in range(NUM_CLASSES)]


def load_model() -> Optional[LocalSignModel]:
    """
    Transformer modelini, scaler'ı ve config'i yükler
    
    Returns:
        LocalSignModel: Yüklenmiş model wrapper veya None
    """
    
    print("\n" + "="*70)
    print("🔧 TRANSFORMER MODEL YÜKLENİYOR")
    print("="*70)
    
    # Gerekli modülleri kontrol et
    if torch is None:
        print("❌ PyTorch yüklü değil!")
        return LocalSignModel(None, None, [f"CLASS_{i}" for i in range(NUM_CLASSES)], None, None)
    
    if TransformerConfig is None or TransformerSignLanguageClassifier is None:
        print("❌ Transformer model modülleri yüklenemedi!")
        return LocalSignModel(None, None, [f"CLASS_{i}" for i in range(NUM_CLASSES)], None, None)
    
    # Checkpoint kontrolü
    if not CHECKPOINT_PATH.exists():
        print(f"❌ Model checkpoint bulunamadı: {CHECKPOINT_PATH}")
        return LocalSignModel(None, None, [f"CLASS_{i}" for i in range(NUM_CLASSES)], None, None)
    
    # Scaler kontrolü
    if not SCALER_PATH.exists():
        print(f"❌ Scaler bulunamadı: {SCALER_PATH}")
        return LocalSignModel(None, None, [f"CLASS_{i}" for i in range(NUM_CLASSES)], None, None)
    
    try:
        # Config yükle
        config = TransformerConfig()
        print(f"✅ Config yüklendi")
        print(f"   - Input dim: {config.INPUT_DIM}")
        print(f"   - Sequence length: {config.MAX_SEQ_LENGTH}")
        print(f"   - Num classes: {config.NUM_CLASSES}")
        
        # Device seç
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✅ Device: CUDA ({torch.cuda.get_device_name(0)})")
        elif torch.backends.mps.is_available():
            # MPS transformer mask issue için CPU kullan
            device = torch.device('cpu')
            print(f"✅ Device: CPU (MPS mask issue nedeniyle)")
        else:
            device = torch.device('cpu')
            print(f"✅ Device: CPU")
        
        # Model oluştur
        model = TransformerSignLanguageClassifier(
            input_dim=config.INPUT_DIM,
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT,
            num_classes=config.NUM_CLASSES,
            max_seq_length=config.MAX_SEQ_LENGTH,
            pooling_type=config.POOLING_TYPE
        ).to(device)
        
        # Checkpoint yükle
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Model checkpoint yüklendi:")
        print(f"   - Epoch: {checkpoint['epoch']}")
        print(f"   - Val Accuracy: {checkpoint['val_acc']:.4f}")
        print(f"   - Val F1: {checkpoint['val_f1']:.4f}")
        
        # Scaler yükle
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ Scaler yüklendi")
        
        # Class names yükle
        class_names = load_class_names()
        
        print("="*70 + "\n")
        
        return LocalSignModel(model, scaler, class_names, device, config)
    
    except Exception as e:
        print(f"\n❌ Model yüklenirken hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        print("="*70 + "\n")
        return LocalSignModel(None, None, [f"CLASS_{i}" for i in range(NUM_CLASSES)], None, None)


# ==================== PREDICTION ====================

@torch.no_grad()
def predict_from_frames(local_model: LocalSignModel, frames: List[np.ndarray], 
                       confidence_threshold: float = 0.3) -> dict:
    """
    Frame listesi üzerinden tahmin yapar
    
    Args:
        local_model: LocalSignModel instance
        frames: BGR formatında OpenCV frame listesi
        confidence_threshold: Minimum güven eşiği
        
    Returns:
        dict: Tahmin sonuçları
            - pred_id: Tahmin edilen sınıf ID
            - pred_name: Tahmin edilen sınıf ismi
            - confidence: Güven skoru (0-1)
            - top5: Top-5 tahminler
            - threshold_met: Eşik karşılandı mı?
    """
    
    # Model kontrolü
    if local_model is None or not local_model.loaded or local_model.model is None:
        print("❌ Model yüklü değil!")
        return {
            "pred_id": -1,
            "pred_name": "",
            "confidence": 0.0,
            "top5": [],
            "threshold_met": False
        }
    
    print(f"\n{'='*70}")
    print(f"🎯 TAHMİN BAŞLIYOR")
    print(f"{'='*70}")
    print(f"📹 Frame sayısı: {len(frames)}")
    
    # Minimum frame kontrolü
    if len(frames) < 5:
        print(f"⚠️  Yetersiz frame sayısı: {len(frames)} < 5")
        return {
            "pred_id": -1,
            "pred_name": "",
            "confidence": 0.0,
            "top5": [],
            "threshold_met": False
        }
    
    try:
        # 1. MediaPipe ile keypoint extraction
        sequence = frames_to_keypoint_sequence(frames)
        
        if sequence.size == 0:
            print("❌ Keypoint çıkarılamadı!")
            return {
                "pred_id": -1,
                "pred_name": "",
                "confidence": 0.0,
                "top5": [],
                "threshold_met": False
            }
        
        # 2. Normalize
        print(f"🔧 Normalizasyon yapılıyor...")
        sequence_normalized = normalize_sequence(sequence, local_model.scaler)
        
        # 3. Pad/Truncate to 200 frames
        print(f"🔧 Sequence {SEQ_LEN} frame'e ayarlanıyor...")
        sequence_processed = pad_or_truncate_sequence(sequence_normalized, SEQ_LEN)
        print(f"   ✅ Final sequence shape: {sequence_processed.shape}")
        
        # 4. Padding mask oluştur
        mask = create_padding_mask(sequence_processed)
        
        # 5. PyTorch tensöre çevir
        sequence_tensor = torch.FloatTensor(sequence_processed).unsqueeze(0).to(local_model.device)
        mask_tensor = torch.BoolTensor(mask).unsqueeze(0).to(local_model.device)
        
        print(f"🔧 Model inference...")
        print(f"   Input shape: {sequence_tensor.shape}")
        print(f"   Mask shape: {mask_tensor.shape}")
        
        # 6. Model forward
        logits = local_model.model(sequence_tensor, mask=mask_tensor)
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # 7. Tahmin
        pred_id = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_id])
        pred_name = local_model.class_names[pred_id]
        
        # 8. Top-5
        top5_ids = np.argsort(probabilities)[-5:][::-1]
        top5 = [
            {
                "id": int(i),
                "name": local_model.class_names[int(i)],
                "confidence": float(probabilities[int(i)])
            }
            for i in top5_ids
        ]
        
        threshold_met = confidence >= confidence_threshold
        
        # Sonuçları yazdır
        print(f"\n{'='*70}")
        print(f"📊 TAHMİN SONUÇLARI")
        print(f"{'='*70}")
        print(f"🏆 Top-1: {pred_name} (ID: {pred_id})")
        print(f"   Güven: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"   Eşik: {confidence_threshold:.4f}")
        print(f"   Eşik karşılandı: {'✅ EVET' if threshold_met else '❌ HAYIR'}")
        print(f"\n📋 Top-5 Tahminler:")
        for i, item in enumerate(top5):
            print(f"   {i+1}. {item['name']:20s} - {item['confidence']:.4f} ({item['confidence']*100:.2f}%)")
        print(f"{'='*70}\n")
        
        return {
            "pred_id": pred_id,
            "pred_name": pred_name,
            "confidence": confidence,
            "top5": top5,
            "threshold_met": threshold_met
        }
    
    except Exception as e:
        print(f"\n❌ Tahmin sırasında hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        return {
            "pred_id": -1,
            "pred_name": "",
            "confidence": 0.0,
            "top5": [],
            "threshold_met": False
        }


# ==================== LEGACY COMPATIBILITY ====================

def get_transcription_from_local_model(input_data):
    """
    Legacy fonksiyon - eski API uyumluluğu için
    Gerçek kullanımda predict_from_frames() kullanılmalı
    """
    print(f"⚠️  Legacy fonksiyon çağrıldı: {input_data}")
    sample_transcription = (
        "BEN ARABA SÜRMEK ÖĞRENMEK BEN ŞİMDİ ANLAMAK"
    )
    return sample_transcription


# ==================== MAIN ====================

if __name__ == "__main__":
    """Test the model loading"""
    
    # OpenCV import (test için)
    import cv2
    
    print("\n🧪 TRANSFORMER MODEL HANDLER TEST\n")
    
    # Model yükle
    model = load_model()
    
    if model.loaded:
        print("✅ Model başarıyla yüklendi!")
        print(f"   - Sınıf sayısı: {len(model.class_names)}")
        print(f"   - İlk 5 sınıf: {model.class_names[:5]}")
        print(f"   - Son 5 sınıf: {model.class_names[-5:]}")
    else:
        print("❌ Model yüklenemedi!")
