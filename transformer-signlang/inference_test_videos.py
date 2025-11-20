#!/usr/bin/env python3
"""
Transformer Sign Language Classifier - Test Video Inference
-----------------------------------------------------------
Test videolarını oynatarak model tahminlerini görselleştirir ve sonuçları kaydeder.

Kullanım:
    python inference_test_videos.py

Giriş:
    - data/selected_videos_test.csv: Test video listesi
    - checkpoints/best_model.pth: Eğitilmiş model
    - data/scaler.pkl: Normalizasyon scaler
    
Çıktı:
    - results/test_predictions.csv: Tahmin sonuçları
    - results/test_predictions.json: Detaylı tahmin raporu
    - Video oynatma ile gerçek zamanlı tahmin gösterimi
"""

import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import json
import pickle
from datetime import datetime

# Proje root'unu path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import TransformerConfig
from models.transformer_model import TransformerSignLanguageClassifier


# ==================== MEDİAPİPE SETUP ====================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


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


def draw_styled_landmarks(image, results):
    """
    MediaPipe landmark'larını görselleştirir
    
    Args:
        image: BGR image
        results: MediaPipe Holistic results
    """
    # Yüz mesh
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )
    
    # Pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    
    # El - Sol
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
        )
    
    # El - Sağ
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
        )


# ==================== DATA PROCESSING ====================

def normalize_sequence(sequence, scaler):
    """
    Keypoint sekansını normalize eder
    
    Args:
        sequence: (num_frames, 258) numpy array
        scaler: StandardScaler objesi
        
    Returns:
        normalized_sequence: (num_frames, 258) normalized array
    """
    return scaler.transform(sequence)


def pad_or_truncate_sequence(sequence, target_length):
    """
    Sekansı hedef uzunluğa getirir (padding veya truncation)
    
    Args:
        sequence: (num_frames, 258) numpy array
        target_length: Hedef frame sayısı
        
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


def create_padding_mask(sequence):
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


# ==================== MODEL INFERENCE ====================

def load_model_and_scaler(config, device):
    """
    Model ve scaler'ı yükler
    
    Args:
        config: TransformerConfig
        device: torch.device
        
    Returns:
        model: Yüklenmiş model
        scaler: StandardScaler
        checkpoint: Checkpoint dictionary
    """
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
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint bulunamadı: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Scaler yükle
    scaler_path = os.path.join(config.DATA_DIR, 'scaler.pkl')
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler bulunamadı: {scaler_path}")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler, checkpoint


@torch.no_grad()
def predict_sequence(model, sequence, device, config):
    """
    Keypoint sekansı üzerinde tahmin yapar
    
    Args:
        model: TransformerSignLanguageClassifier
        sequence: (seq_len, 258) numpy array (normalized)
        device: torch.device
        config: TransformerConfig
        
    Returns:
        pred_class: Tahmin edilen sınıf (0, 1, 2)
        confidence: Tahmin güveni (0-1)
        probabilities: Tüm sınıflar için olasılıklar (3,)
    """
    # Tensor'e çevir ve batch dimension ekle
    sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)  # (1, seq_len, 258)
    
    # Padding mask oluştur
    mask_np = create_padding_mask(sequence)
    mask = torch.BoolTensor(mask_np).unsqueeze(0).to(device)  # (1, seq_len)
    
    # Forward pass
    logits = model(sequence_tensor, mask=mask)  # (1, num_classes)
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # (num_classes,)
    
    # En yüksek tahmin
    pred_class = np.argmax(probabilities)
    confidence = probabilities[pred_class]
    
    return pred_class, confidence, probabilities


# ==================== VIDEO PROCESSING ====================

def process_and_display_video(video_path, true_class_id, model, scaler, config, device, 
                               video_id, show_video=True):
    """
    Videoyu işler, tahmin yapar ve görselleştirir
    
    Args:
        video_path: Video dosya yolu
        true_class_id: Gerçek sınıf ID (1, 2, 5)
        model: TransformerSignLanguageClassifier
        scaler: StandardScaler
        config: TransformerConfig
        device: torch.device
        video_id: Video ID (görüntüleme için)
        show_video: Video gösterimi açık mı?
        
    Returns:
        result_dict: Tahmin sonucu dictionary
    """
    
    if not os.path.exists(video_path):
        print(f"   ❌ Video bulunamadı: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"   ❌ Video açılamadı: {video_path}")
        return None
    
    # Video bilgileri
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    keypoint_sequence = []
    frames = []  # Videoyu tekrar oynatmak için
    
    # 1. AŞAMA: Tüm frame'lerden keypoint çıkar
    with mp_holistic.Holistic(
        min_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MP_MIN_TRACKING_CONFIDENCE,
        model_complexity=config.MP_MODEL_COMPLEXITY
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Keypoint çıkar
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            
            keypoints = extract_keypoints_from_frame(results)
            keypoint_sequence.append(keypoints)
            
            # Frame'i sakla (görselleştirme için)
            frames.append(frame.copy())
    
    cap.release()
    
    if len(keypoint_sequence) == 0:
        print(f"   ⚠️  Hiç keypoint çıkarılamadı")
        return None
    
    # 2. AŞAMA: Keypoint'leri işle ve tahmin yap
    sequence = np.array(keypoint_sequence)  # (num_frames, 258)
    
    # Normalize
    sequence_normalized = normalize_sequence(sequence, scaler)
    
    # Pad/Truncate
    sequence_processed = pad_or_truncate_sequence(sequence_normalized, config.MAX_SEQ_LENGTH)
    
    # Tahmin
    pred_class, confidence, probabilities = predict_sequence(
        model, sequence_processed, device, config
    )
    
    # Sınıf ID'leri ve isimleri
    # Model output: 0, 1, 2 → Config sınıfları: acele(1), acikmak(2), agac(5)
    pred_class_id = config.TARGET_CLASS_IDS[pred_class]
    pred_class_name = config.CLASS_NAMES[pred_class]
    
    # Gerçek sınıf
    true_class_idx = config.TARGET_CLASS_IDS.index(true_class_id)
    true_class_name = config.CLASS_NAMES[true_class_idx]
    
    # Doğru mu?
    is_correct = (pred_class == true_class_idx)
    
    # 3. AŞAMA: Video gösterimi (opsiyonel)
    if show_video and len(frames) > 0:
        # Tahmin bilgisini hazırla
        prediction_text = f"Tahmin: {pred_class_name} ({confidence:.2%})"
        ground_truth_text = f"Gercek: {true_class_name}"
        status_text = "DOGRU" if is_correct else "YANLIS"
        status_color = (0, 255, 0) if is_correct else (0, 0, 255)
        
        # Her frame'i göster
        for frame_idx, frame in enumerate(frames):
            # Bilgileri ekle
            h, w = frame.shape[:2]
            
            # Arka plan kutusu
            cv2.rectangle(frame, (10, 10), (w - 10, 150), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (w - 10, 150), (255, 255, 255), 2)
            
            # Video ID
            cv2.putText(frame, f"Video: {video_id}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Tahmin
            cv2.putText(frame, prediction_text, 
                       (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            
            # Gerçek
            cv2.putText(frame, ground_truth_text, 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Durum
            cv2.putText(frame, status_text, 
                       (w - 200, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
            
            # Progress bar
            progress = (frame_idx + 1) / len(frames)
            cv2.rectangle(frame, (10, h - 30), (int((w - 20) * progress), h - 10), 
                         (0, 255, 0), -1)
            cv2.rectangle(frame, (10, h - 30), (w - 10, h - 10), (255, 255, 255), 2)
            
            # Göster
            cv2.imshow(f'Test Video Inference - {video_id}', frame)
            
            # FPS'e göre bekleme (gerçek hız)
            wait_time = max(1, int(1000 / fps)) if fps > 0 else 25
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                return None
            elif key == ord('n'):  # Next video
                break
            elif key == ord('p'):  # Pause
                cv2.waitKey(0)
        
        cv2.destroyAllWindows()
    
    # 4. Sonuç dictionary
    result = {
        'video_id': video_id,
        'video_path': video_path,
        'num_frames': len(sequence),
        'true_class_id': int(true_class_id),
        'true_class_name': true_class_name,
        'pred_class_id': int(pred_class_id),
        'pred_class_name': pred_class_name,
        'confidence': float(confidence),
        'is_correct': bool(is_correct),
        'probabilities': {
            config.CLASS_NAMES[i]: float(probabilities[i]) 
            for i in range(len(config.CLASS_NAMES))
        }
    }
    
    return result


# ==================== MAIN ====================

def main():
    """Ana fonksiyon"""
    
    # Config
    config = TransformerConfig()
    
    print("=" * 80)
    print("🎬 TRANSFORMER TEST VIDEO INFERENCE")
    print("=" * 80)
    
    # Device
    # MPS'de mask operasyonları desteklenmediği için CPU kullanıyoruz
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n🖥️  Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        # MPS yerine CPU kullan (transformer mask issue)
        device = torch.device('cpu')
        print(f"\n🖥️  Device: CPU (MPS mask operasyonları desteklenmediği için)")
    
    # Model ve scaler yükle
    print(f"\n📂 Model ve scaler yükleniyor...")
    try:
        model, scaler, checkpoint = load_model_and_scaler(config, device)
        print(f"   ✅ Model yüklendi!")
        print(f"      - Epoch: {checkpoint['epoch']}")
        print(f"      - Val Acc: {checkpoint['val_acc']:.4f}")
        print(f"      - Val F1: {checkpoint['val_f1']:.4f}")
    except FileNotFoundError as e:
        print(f"\n❌ HATA: {e}")
        print("Model veya scaler bulunamadı. Lütfen eğitimi tamamlayın.")
        return
    
    # Test CSV yükle
    test_csv = os.path.join(config.DATA_DIR, 'selected_videos_test.csv')
    
    if not os.path.exists(test_csv):
        print(f"\n❌ HATA: Test CSV bulunamadı: {test_csv}")
        return
    
    test_df = pd.read_csv(test_csv)
    # Remove empty rows
    test_df = test_df.dropna(subset=['video_id'])
    
    print(f"\n📊 Test Seti:")
    print(f"   - Toplam video: {len(test_df)}")
    
    # Sınıf dağılımı
    print(f"\n📊 Sınıf Dağılımı:")
    for class_id in config.TARGET_CLASS_IDS:
        count = (test_df['class_id'] == class_id).sum()
        class_name = config.CLASS_NAMES[config.TARGET_CLASS_IDS.index(class_id)]
        print(f"   - {class_name} (ClassId {class_id}): {count} video")
    
    # Kullanıcıdan onay
    print(f"\n" + "=" * 80)
    print("⌨️  KONTROLLER:")
    print("   - 'q': Çıkış")
    print("   - 'n': Sonraki video")
    print("   - 'p': Duraklat/Devam")
    print("=" * 80)
    
    response = input("\n▶️  Videoları göstermek ister misiniz? (y/n) [y]: ").strip().lower()
    show_video = (response != 'n')
    
    # Sonuçları sakla
    results = []
    
    # Her test videosunu işle
    print(f"\n{'='*80}")
    print("🎯 TEST BAŞLIYOR")
    print(f"{'='*80}\n")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing videos"):
        video_id = row['video_id']
        class_id = row['class_id']
        video_path = row['path']
        
        print(f"\n[{idx+1}/{len(test_df)}] {video_id} (ClassId: {class_id})")
        
        # Video işle
        result = process_and_display_video(
            video_path=video_path,
            true_class_id=class_id,
            model=model,
            scaler=scaler,
            config=config,
            device=device,
            video_id=video_id,
            show_video=show_video
        )
        
        if result is None:
            print(f"   ⚠️  Video atlandı")
            continue
        
        results.append(result)
        
        # Sonucu yazdır
        status = "✅ DOĞRU" if result['is_correct'] else "❌ YANLIŞ"
        print(f"   {status}: {result['pred_class_name']} ({result['confidence']:.2%}) | "
              f"Gerçek: {result['true_class_name']}")
    
    # Sonuçları kaydet
    if len(results) == 0:
        print(f"\n⚠️  Hiç sonuç elde edilemedi!")
        return
    
    print(f"\n{'='*80}")
    print("💾 SONUÇLAR KAYDEDİLİYOR")
    print(f"{'='*80}\n")
    
    # CSV formatı
    results_df = pd.DataFrame([
        {
            'video_id': r['video_id'],
            'num_frames': r['num_frames'],
            'true_class_id': r['true_class_id'],
            'true_class_name': r['true_class_name'],
            'pred_class_id': r['pred_class_id'],
            'pred_class_name': r['pred_class_name'],
            'confidence': r['confidence'],
            'is_correct': r['is_correct']
        }
        for r in results
    ])
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    csv_path = os.path.join(config.RESULTS_DIR, 'test_predictions.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"   ✅ CSV kaydedildi: {csv_path}")
    
    # JSON formatı (detaylı)
    json_path = os.path.join(config.RESULTS_DIR, 'test_predictions.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"   ✅ JSON kaydedildi: {json_path}")
    
    # Özet istatistikler
    print(f"\n{'='*80}")
    print("📊 ÖZET İSTATİSTİKLER")
    print(f"{'='*80}\n")
    
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = correct / total if total > 0 else 0
    
    print(f"📈 Genel Performans:")
    print(f"   - Toplam video: {total}")
    print(f"   - Doğru tahmin: {correct}")
    print(f"   - Yanlış tahmin: {total - correct}")
    print(f"   - Accuracy: {accuracy:.2%}")
    
    # Sınıf bazlı accuracy
    print(f"\n📊 Sınıf Bazlı Performans:")
    for class_name in config.CLASS_NAMES:
        class_results = [r for r in results if r['true_class_name'] == class_name]
        if len(class_results) > 0:
            class_correct = sum(1 for r in class_results if r['is_correct'])
            class_acc = class_correct / len(class_results)
            print(f"   - {class_name:10s}: {class_correct}/{len(class_results)} "
                  f"({class_acc:.2%})")
    
    # Ortalama confidence
    avg_confidence = np.mean([r['confidence'] for r in results])
    correct_avg_conf = np.mean([r['confidence'] for r in results if r['is_correct']]) \
                       if correct > 0 else 0
    incorrect_avg_conf = np.mean([r['confidence'] for r in results if not r['is_correct']]) \
                         if (total - correct) > 0 else 0
    
    print(f"\n🎯 Confidence İstatistikleri:")
    print(f"   - Ortalama: {avg_confidence:.2%}")
    print(f"   - Doğru tahminler: {correct_avg_conf:.2%}")
    print(f"   - Yanlış tahminler: {incorrect_avg_conf:.2%}")
    
    # Karışıklık matrisi (basit)
    print(f"\n📋 Karışıklık Özeti (Yanlış Tahminler):")
    wrong_preds = [r for r in results if not r['is_correct']]
    if len(wrong_preds) > 0:
        for r in wrong_preds:
            print(f"   - {r['true_class_name']:10s} → {r['pred_class_name']:10s} "
                  f"({r['confidence']:.2%}) [{r['video_id']}]")
    else:
        print(f"   🎉 Tüm tahminler doğru!")
    
    print(f"\n{'='*80}")
    print("✅ TEST TAMAMLANDI")
    print(f"{'='*80}\n")
    
    print(f"📁 Sonuçlar kaydedildi:")
    print(f"   - {csv_path}")
    print(f"   - {json_path}")
    print()


if __name__ == '__main__':
    main()

