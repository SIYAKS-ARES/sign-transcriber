#!/usr/bin/env python3
"""
Test Transformer Model Integration
===================================
Anlamlandırma sistemindeki Transformer model entegrasyonunu test eder.

Usage:
    python test_transformer_model.py

Author: AI Assistant
Date: November 2024
"""

import os
import sys
import cv2
from pathlib import Path

# local_model_handler'ı import et
from local_model_handler import load_model, predict_from_frames

# Test video yolu
CURRENT_DIR = Path(__file__).parent
TEST_VIDEOS_DIR = CURRENT_DIR / "test_videos"

# Test videoları listesi
if TEST_VIDEOS_DIR.exists():
    TEST_VIDEOS = list(TEST_VIDEOS_DIR.glob("*.mp4"))
else:
    TEST_VIDEOS = []


def extract_frames_from_video(video_path, max_frames=120):
    """
    Video dosyasından frame'leri çıkarır
    
    Args:
        video_path: Video dosya yolu
        max_frames: Maximum frame sayısı
        
    Returns:
        list: BGR formatında OpenCV frame listesi
    """
    if not os.path.exists(video_path):
        print(f"❌ Video bulunamadı: {video_path}")
        return []
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Video açılamadı: {video_path}")
        return []
    
    # Video bilgileri
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📹 Video bilgileri:")
    print(f"   - Dosya: {video_path.name}")
    print(f"   - FPS: {fps:.2f}")
    print(f"   - Toplam frame: {total_frames}")
    print(f"   - Süre: {duration:.2f} saniye")
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        frame_count += 1
        
        # Maximum frame limit
        if frame_count >= max_frames:
            break
    
    cap.release()
    print(f"   ✅ {len(frames)} frame çıkarıldı")
    
    return frames


def visualize_prediction(frames, pred_name, confidence):
    """
    Tahmin sonucunu video üzerine yazar ve gösterir
    
    Args:
        frames: Frame listesi
        pred_name: Tahmin edilen sınıf ismi
        confidence: Güven skoru
    """
    print(f"\n🎬 Video gösterimi başlıyor...")
    print(f"   - 'q' tuşuna basarak çıkabilirsiniz")
    print(f"   - 'p' tuşuna basarak duraklatabilirsiniz")
    
    for idx, frame in enumerate(frames):
        # Frame'i kopyala
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Tahmin bilgisini ekle
        # Arka plan kutusu
        cv2.rectangle(display_frame, (10, 10), (w - 10, 120), (0, 0, 0), -1)
        cv2.rectangle(display_frame, (10, 10), (w - 10, 120), (255, 255, 255), 2)
        
        # Tahmin
        pred_text = f"Tahmin: {pred_name}"
        cv2.putText(display_frame, pred_text, 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Güven
        conf_text = f"Guven: {confidence:.2%}"
        cv2.putText(display_frame, conf_text, 
                   (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Progress bar
        progress = (idx + 1) / len(frames)
        cv2.rectangle(display_frame, (10, h - 30), (int((w - 20) * progress), h - 10), 
                     (0, 255, 0), -1)
        cv2.rectangle(display_frame, (10, h - 30), (w - 10, h - 10), (255, 255, 255), 2)
        
        # Göster
        cv2.imshow('Transformer Model - Demo', display_frame)
        
        # FPS'e göre bekleme
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q'):  # Quit
            break
        elif key == ord('p'):  # Pause
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    print(f"   ✅ Video gösterimi tamamlandı")


def main():
    """Ana test fonksiyonu"""
    
    print("\n" + "="*80)
    print("🧪 TRANSFORMER MODEL ENTEGRASYON TESTİ")
    print("="*80)
    
    # 1. Model yükle
    print(f"\n📦 1. Model yükleniyor...")
    model = load_model()
    
    if not model.loaded:
        print(f"\n❌ Model yüklenemedi! Test durduruluyor.")
        return
    
    print(f"\n✅ Model başarıyla yüklendi!")
    
    # 2. Test videosu bul
    print(f"\n📹 2. Test videosu aranıyor...")
    
    if not TEST_VIDEOS:
        print(f"\n❌ test_videos/ klasöründe video bulunamadı! Test durduruluyor.")
        return
    
    print(f"   ✅ {len(TEST_VIDEOS)} test videosu bulundu:")
    for i, video in enumerate(TEST_VIDEOS[:5], 1):
        print(f"      {i}. {video.name}")
    
    # İlk videoyu kullan
    test_video = TEST_VIDEOS[0]
    print(f"\n   🎯 Test için seçilen video: {test_video.name}")
    
    # 3. Frame'leri çıkar
    print(f"\n🎬 3. Video frame'leri çıkarılıyor...")
    frames = extract_frames_from_video(test_video, max_frames=120)
    
    if not frames:
        print(f"\n❌ Frame çıkarılamadı! Test durduruluyor.")
        return
    
    # 4. Tahmin yap
    print(f"\n🎯 4. Model tahmini yapılıyor...")
    result = predict_from_frames(model, frames, confidence_threshold=0.1)
    
    if result['pred_id'] == -1:
        print(f"\n❌ Tahmin yapılamadı!")
        return
    
    print(f"\n✅ Tahmin başarılı!")
    
    # 5. Sonuçları göster
    print(f"\n" + "="*80)
    print(f"📊 SONUÇLAR")
    print(f"="*80)
    print(f"🏆 Tahmin: {result['pred_name']}")
    print(f"📈 Güven: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    print(f"✅ Eşik karşılandı: {'EVET' if result['threshold_met'] else 'HAYIR'}")
    print(f"\n📋 Top-5:")
    for i, item in enumerate(result['top5']):
        print(f"   {i+1}. {item['name']:20s} - {item['confidence']:.4f}")
    print(f"="*80)
    
    # 6. Video gösterimi
    response = input(f"\n▶️  Videoyu tahmin ile birlikte göstermek ister misiniz? (y/n) [y]: ").strip().lower()
    
    if response != 'n':
        visualize_prediction(frames, result['pred_name'], result['confidence'])
    
    print(f"\n✅ TEST TAMAMLANDI!")
    print(f"="*80 + "\n")


if __name__ == "__main__":
    main()

