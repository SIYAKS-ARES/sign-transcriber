#!/usr/bin/env python3
"""
None Face modelini test etmek için basit script
OpenCV penceresi açmadan sadece tahmin sonuçlarını yazdırır
"""

import os
import glob
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras import layers, models, regularizers
import pandas as pd


def create_enhanced_model(seq_len, feature_dim, num_classes):
    inputs = layers.Input(shape=(seq_len, feature_dim))

    # Masking + normalization
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.LayerNormalization()(x)

    # Feature extraction (hafifletilmiş dense boyutları)
    x = layers.TimeDistributed(layers.Dense(256, activation="relu"))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.Dropout(0.3)(x)

    x = layers.TimeDistributed(layers.Dense(128, activation="relu"))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.Dropout(0.3)(x)

    # Attention
    attention = layers.MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.1)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)

    # LSTM blokları (daha küçük boyutlar, dropout yükseltildi)
    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.4,
            recurrent_dropout=0.3,
            recurrent_regularizer=regularizers.l2(1e-4),
        )
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(
            64,
            return_sequences=True,
            dropout=0.4,
            recurrent_dropout=0.3,
            recurrent_regularizer=regularizers.l2(1e-4),
        )
    )(x)

    # Global pooling
    max_pool = layers.GlobalMaxPooling1D()(x)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    x = layers.Concatenate()([max_pool, avg_pool])

    # Classifier head
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model


def uniform_sample_or_pad(seq, target_len):
    T, F = seq.shape
    if T == target_len:
        return seq
    elif T > target_len:
        idx = np.linspace(0, T - 1, target_len).astype(int)
        return seq[idx]
    else:
        pad = np.zeros((target_len - T, F), dtype=seq.dtype)
        return np.concatenate([seq, pad], axis=0)


def extract_keypoints_from_video(video_path, seq_len=60):
    """Videodan keypoint'leri çıkar"""
    cap = cv2.VideoCapture(video_path)
    sequence = []
    
    mp_holistic = mp.solutions.holistic
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            # POSE features (33 landmarks * 4 = 132)
            pose = (
                np.array([
                    [r.x, r.y, r.z, r.visibility]
                    for r in results.pose_landmarks.landmark
                ]).flatten()
                if results.pose_landmarks
                else np.zeros(33 * 4, dtype=np.float32)
            )
            
            # FACE features (reduced to 8 key points * 3 = 24)
            if results.face_landmarks:
                lms = results.face_landmarks.landmark
                face_points = []
                for idx in [33, 263, 13, 14, 61, 291, 105, 334]:
                    res = lms[idx]
                    face_points.extend([res.x, res.y, res.z])
                face = np.array(face_points, dtype=np.float32)
            else:
                face = np.zeros(8 * 3, dtype=np.float32)
            
            # LEFT HAND features (21 landmarks * 3 = 63)
            lh = (
                np.array([
                    [r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark
                ]).flatten()
                if results.left_hand_landmarks
                else np.zeros(21 * 3, dtype=np.float32)
            )
            
            # RIGHT HAND features (21 landmarks * 3 = 63)
            rh = (
                np.array([
                    [r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark
                ]).flatten()
                if results.right_hand_landmarks
                else np.zeros(21 * 3, dtype=np.float32)
            )
            
            # Combine all features: 132 + 24 + 63 + 63 = 282
            keypoints = np.concatenate([pose, face, lh, rh]).astype(np.float32)
            sequence.append(keypoints)
    
    cap.release()
    
    if len(sequence) == 0:
        return None
        
    # Sequence'i hedef uzunluğa getir
    seq_array = np.array(sequence, dtype=np.float32)
    seq_fixed = uniform_sample_or_pad(seq_array, seq_len)
    
    return seq_fixed


def main():
    # Ayarlar
    LABELS_CSV = "SignList_ClassId_TR_EN.csv"
    TEST_DIR = "test_videos"
    MODEL_PATH = "models/sign_classifier_NoneFace_best_89225.h5"
    SEQ_LEN = 60
    
    # Label'ları yükle
    df_labels = pd.read_csv(LABELS_CSV)
    CLASS_NAMES = df_labels["TR"].tolist()
    
    print(f"Toplam sınıf sayısı: {len(CLASS_NAMES)}")
    print(f"İlk 10 sınıf: {CLASS_NAMES[:10]}")
    
    # Model boyutları
    feature_dim = 132 + 24 + 126  # POSE + reduced FACE + HANDS = 282
    num_classes = len(CLASS_NAMES)
    
    # Modeli oluştur ve yükle
    print(f"\nModel oluşturuluyor: seq_len={SEQ_LEN}, feature_dim={feature_dim}, num_classes={num_classes}")
    model = create_enhanced_model(SEQ_LEN, feature_dim, num_classes)
    
    if not os.path.isfile(MODEL_PATH):
        print(f"HATA: Model dosyası bulunamadı: {MODEL_PATH}")
        return
        
    model.load_weights(MODEL_PATH)
    print("Model yüklendi!")
    
    # Test videolarını bul
    video_exts = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    video_list = []
    for ext in video_exts:
        video_list.extend(glob.glob(os.path.join(TEST_DIR, ext)))
    
    if not video_list:
        print(f"HATA: {TEST_DIR} dizininde video bulunamadı.")
        return
        
    print(f"\nBulunan video sayısı: {len(video_list)}")
    
    # Her video için test et
    correct_predictions = 0
    total_predictions = 0
    
    for video_path in video_list:
        video_name = os.path.basename(video_path)
        # Video adından gerçek label'ı çıkar
        true_label = video_name.split("_")[0] if "_" in video_name else video_name.split(".")[0]
        
        print(f"\n{'='*50}")
        print(f"Video: {video_name}")
        print(f"Gerçek label: {true_label}")
        
        # Keypoint'leri çıkar
        print("Keypoint'ler çıkarılıyor...")
        keypoints_seq = extract_keypoints_from_video(video_path, SEQ_LEN)
        
        if keypoints_seq is None:
            print("HATA: Keypoint çıkarılamadı!")
            continue
            
        # Tahmin yap
        seq_np = np.expand_dims(keypoints_seq, axis=0)
        print(f"Input shape: {seq_np.shape}")
        
        probs = model.predict(seq_np, verbose=0)[0]
        
        # En yüksek 5 tahmini göster
        top5_ids = np.argsort(probs)[-5:][::-1]
        
        print("\nTop-5 Tahminler:")
        for rank, i in enumerate(top5_ids, start=1):
            class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"UNK_{i}"
            confidence = probs[i]
            marker = "✓" if class_name == true_label else " "
            print(f"{marker} {rank}. {class_name:<20} ({confidence:.4f})")
        
        # En yüksek tahmin
        pred_id = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred_id]
        confidence = float(probs[pred_id])
        
        # Doğruluk kontrolü
        is_correct = pred_name == true_label
        if is_correct:
            correct_predictions += 1
        total_predictions += 1
        
        print(f"\nEn yüksek tahmin: {pred_name} ({confidence:.4f})")
        print(f"Doğru mu: {'✓ EVET' if is_correct else '✗ HAYIR'}")
    
    # Genel sonuçlar
    print(f"\n{'='*60}")
    print(f"TEST SONUÇLARI")
    print(f"{'='*60}")
    print(f"Toplam video: {total_predictions}")
    print(f"Doğru tahmin: {correct_predictions}")
    print(f"Doğruluk oranı: {correct_predictions/total_predictions*100:.2f}%" if total_predictions > 0 else "Hesaplanamadı")


if __name__ == "__main__":
    main()
