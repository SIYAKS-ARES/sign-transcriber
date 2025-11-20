#!/usr/bin/env python3
"""
Script 02: MediaPipe Keypoint Çıkarımı
---------------------------------------
Seçilmiş videolardan MediaPipe Holistic kullanarak 258 boyutlu keypoint vektörleri çıkarır.

Keypoint Yapısı:
- Pose: 33 nokta × 3 (x,y,z) = 99 boyut
- Face (key points): 11 nokta × 3 = 33 boyut  
- Left Hand: 21 nokta × 3 = 63 boyut
- Right Hand: 21 nokta × 3 = 63 boyut
TOPLAM: 258 boyut

Kullanım:
    python scripts/02_extract_keypoints.py

Giriş:
    data/selected_videos_train.csv
    data/selected_videos_val.csv
    data/selected_videos_test.csv

Çıktı:
    data/keypoints/{video_id}.npy - Her video için keypoint dizisi (shape: num_frames × 258)
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

# Proje root'unu path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import TransformerConfig


# MediaPipe ayarları
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints_from_frame(results):
    """
    Bir frame'den 258 boyutlu keypoint vektörü çıkarır
    
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
    # Göz çevreleri, kaş, burun, ağız köşeleri
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


def process_video(video_path, config, max_frames=None):
    """
    Video dosyasından keypoint sekansı çıkarır
    
    Args:
        video_path (str): Video dosyası yolu
        config: TransformerConfig objesi
        max_frames (int): Maksimum işlenecek frame sayısı (None = tümü)
        
    Returns:
        np.array: (num_frames, 258) boyutunda keypoint sekansı
        None: Hata durumunda
    """
    
    if not os.path.exists(video_path):
        print(f"   ❌ Video bulunamadı: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"   ❌ Video açılamadı: {video_path}")
        return None
    
    keypoint_sequence = []
    
    try:
        with mp_holistic.Holistic(
            min_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MP_MIN_TRACKING_CONFIDENCE,
            model_complexity=config.MP_MODEL_COMPLEXITY
        ) as holistic:
            
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # BGR -> RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # MediaPipe işleme
                results = holistic.process(image)
                
                # Keypoint çıkarımı
                keypoints = extract_keypoints_from_frame(results)
                keypoint_sequence.append(keypoints)
                
                frame_count += 1
                
                if max_frames and frame_count >= max_frames:
                    break
        
    except Exception as e:
        print(f"   ❌ Hata: {e}")
        return None
    
    finally:
        cap.release()
    
    if len(keypoint_sequence) == 0:
        print(f"   ⚠️  Hiç frame işlenemedi: {video_path}")
        return None
    
    return np.array(keypoint_sequence)  # Shape: (num_frames, 258)


def main():
    """Ana fonksiyon"""
    config = TransformerConfig()
    
    print("=" * 80)
    print("🎬 MEDİAPİPE KEYPOINT ÇIKARIMI")
    print("=" * 80)
    
    # Train/Val/Test CSV'lerini yükle
    train_csv = os.path.join(config.DATA_DIR, 'selected_videos_train.csv')
    val_csv = os.path.join(config.DATA_DIR, 'selected_videos_val.csv')
    test_csv = os.path.join(config.DATA_DIR, 'selected_videos_test.csv')
    
    # Dosya kontrolü
    csv_files = [train_csv, val_csv, test_csv]
    missing_files = [f for f in csv_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"\n❌ HATA: Aşağıdaki dosyalar bulunamadı:")
        for f in missing_files:
            print(f"   - {f}")
        print(f"\nÖnce 01_select_videos.py scriptini çalıştırın.")
        return
    
    # CSV'leri yükle ve birleştir
    print(f"\n📂 CSV dosyaları yükleniyor...")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    print(f"   ✅ Train: {len(train_df)} video")
    print(f"   ✅ Val:   {len(val_df)} video")
    print(f"   ✅ Test:  {len(test_df)} video")
    
    # Tüm setleri birleştir
    selected_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    print(f"\n📊 Toplam: {len(selected_df)} video")
    print(f"   ✅ Sınıflar: {sorted(selected_df['class_id'].unique())}")
    
    # Sınıf dağılımı
    print(f"\n📊 Sınıf Dağılımı (Tüm Setler):")
    class_counts = selected_df['class_id'].value_counts().sort_index()
    for class_id, count in class_counts.items():
        class_name = config.CLASS_NAMES[config.TARGET_CLASS_IDS.index(class_id)]
        print(f"   ClassId {class_id} ({class_name}): {count} video")
    
    # Keypoint dizini oluştur
    keypoints_dir = config.KEYPOINTS_DIR
    os.makedirs(keypoints_dir, exist_ok=True)
    
    print(f"\n🎯 Keypoint çıkarımı başlıyor...")
    print(f"   📁 Çıktı dizini: {keypoints_dir}")
    print(f"   🧠 MediaPipe Holistic ayarları:")
    print(f"      - Detection confidence: {config.MP_MIN_DETECTION_CONFIDENCE}")
    print(f"      - Tracking confidence: {config.MP_MIN_TRACKING_CONFIDENCE}")
    print(f"      - Model complexity: {config.MP_MODEL_COMPLEXITY}")
    
    # İstatistikler
    success_count = 0
    failed_count = 0
    total_frames = 0
    frame_counts = []
    
    # Her videoyu işle
    print(f"\n" + "=" * 80)
    
    for idx, row in tqdm(selected_df.iterrows(), 
                         total=len(selected_df),
                         desc="Processing videos",
                         unit="video"):
        
        video_id = row['video_id']
        video_path = row['path']
        class_id = row['class_id']
        
        # Keypoint çıkarımı
        keypoints = process_video(video_path, config)
        
        if keypoints is not None:
            # Kaydet: .npy formatında
            save_path = os.path.join(keypoints_dir, f'{video_id}.npy')
            np.save(save_path, keypoints)
            
            success_count += 1
            num_frames = len(keypoints)
            total_frames += num_frames
            frame_counts.append(num_frames)
            
        else:
            failed_count += 1
    
    # Sonuç özeti
    print("\n" + "=" * 80)
    print("📊 KEYPOINT ÇIKARIM SONUÇLARI")
    print("=" * 80)
    
    print(f"\n✅ Başarılı: {success_count} video")
    print(f"❌ Başarısız: {failed_count} video")
    print(f"📈 Toplam işlenen frame: {total_frames:,}")
    
    if frame_counts:
        print(f"\n📏 Frame İstatistikleri:")
        print(f"   - Minimum: {np.min(frame_counts)} frame")
        print(f"   - Maksimum: {np.max(frame_counts)} frame")
        print(f"   - Ortalama: {np.mean(frame_counts):.1f} frame")
        print(f"   - Medyan: {np.median(frame_counts):.1f} frame")
        print(f"   - Std Dev: {np.std(frame_counts):.1f} frame")
    
    # Keypoint boyutu doğrulama
    if success_count > 0:
        # İlk başarılı keypoint'i yükle ve doğrula
        first_keypoint_file = os.path.join(keypoints_dir, f'{selected_df.iloc[0]["video_id"]}.npy')
        if os.path.exists(first_keypoint_file):
            sample_kp = np.load(first_keypoint_file)
            print(f"\n🔍 Keypoint Doğrulama:")
            print(f"   - Shape: {sample_kp.shape}")
            print(f"   - Expected: (num_frames, 258)")
            
            if sample_kp.shape[1] == 258:
                print(f"   ✅ Keypoint boyutu doğru!")
            else:
                print(f"   ❌ UYARI: Keypoint boyutu yanlış!")
    
    print("\n" + "=" * 80)
    print(f"✅ Keypoint çıkarımı tamamlandı!")
    print(f"📁 Kaydedilen dosyalar: {keypoints_dir}/")
    print(f"📌 Sıradaki adım: Veri normalizasyonu (03_normalize_data.py)")
    print("=" * 80)


if __name__ == '__main__':
    main()

