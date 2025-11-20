#!/usr/bin/env python3
"""
Script 03: Veri Normalizasyonu ve Padding (Train/Val/Test)
-----------------------------------------------------------
Train, Validation ve Test setlerini ayrı ayrı normalize eder ve padding uygular.

ÖNEMLİ: Scaler sadece TRAIN setinde fit edilir, Val ve Test'e transform uygulanır!

İşlemler:
1. Train/Val/Test keypoint dosyalarını yükleme
2. StandardScaler'ı sadece TRAIN'de fit etme
3. Val ve Test'e scaler transform uygulama
4. Sekans uzunluklarını analiz etme (sadece train'den)
5. Padding uygulama (aynı max_length tümü için)
6. Scaler ve normalize edilmiş verileri kaydetme

Kullanım:
    python scripts/03_normalize_data.py

Giriş:
    data/selected_videos_train.csv
    data/selected_videos_val.csv
    data/selected_videos_test.csv
    data/keypoints/{video_id}.npy

Çıktı:
    data/processed/X_train.npy, y_train.npy, train_ids.npy
    data/processed/X_val.npy, y_val.npy, val_ids.npy
    data/processed/X_test.npy, y_test.npy, test_ids.npy
    data/scaler.pkl - Fitted StandardScaler objesi (sadece train'den)
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Proje root'unu path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import TransformerConfig


def load_keypoints_for_split(csv_path, keypoints_dir, split_name, config):
    """Bir split için keypoint'leri yükler"""
    
    print(f"\n{'='*70}")
    print(f"📦 {split_name.upper()} KEYPOINT'LER YÜKLENİYOR")
    print(f"{'='*70}")
    
    if not os.path.exists(csv_path):
        print(f"   ❌ CSV bulunamadı: {csv_path}")
        return None, None, None
    
    selected_df = pd.read_csv(csv_path)
    print(f"   ✅ {len(selected_df)} video bulundu")
    
    # Class ID mapping: {1: 0, 2: 1, 5: 2}
    class_id_to_idx = {cid: idx for idx, cid in enumerate(config.TARGET_CLASS_IDS)}
    print(f"   📋 Class ID mapping: {class_id_to_idx}")
    
    keypoint_sequences = []
    labels = []
    video_ids = []
    missing_files = []
    
    for idx, row in tqdm(selected_df.iterrows(), 
                         total=len(selected_df),
                         desc=f"Loading {split_name}",
                         unit="file"):
        video_id = row['video_id']
        class_id = row['class_id']
        
        keypoint_path = os.path.join(keypoints_dir, f'{video_id}.npy')
        
        if not os.path.exists(keypoint_path):
            missing_files.append(video_id)
            continue
        
        try:
            keypoints = np.load(keypoint_path)
            
            # Shape kontrolü
            if keypoints.ndim != 2 or keypoints.shape[1] != 258:
                print(f"\n   ⚠️  Geçersiz shape: {video_id} - {keypoints.shape}")
                continue
            
            keypoint_sequences.append(keypoints)
            # Remap class_id to 0-indexed: {1: 0, 2: 1, 5: 2}
            labels.append(class_id_to_idx[class_id])
            video_ids.append(video_id)
            
        except Exception as e:
            print(f"\n   ❌ Yükleme hatası: {video_id} - {e}")
            continue
    
    print(f"   ✅ Yükleme tamamlandı: {len(keypoint_sequences)} dosya")
    
    if missing_files:
        print(f"   ⚠️  Eksik: {len(missing_files)} dosya")
    
    return keypoint_sequences, labels, video_ids


def fit_scaler_on_train(train_sequences):
    """Scaler'ı sadece train verisi üzerinde fit eder"""
    
    print(f"\n🔧 SCALER FİT EDİLİYOR (sadece TRAIN verisi)...")
    
    # Tüm train frame'lerini birleştir
    all_frames = np.vstack([seq for seq in train_sequences])
    print(f"   📊 Toplam train frame: {all_frames.shape[0]:,}")
    print(f"   📊 Feature boyutu: {all_frames.shape[1]}")
    
    # StandardScaler fit et
    scaler = StandardScaler()
    scaler.fit(all_frames)
    
    print(f"   ✅ Scaler fit edildi")
    print(f"      - Mean shape: {scaler.mean_.shape}")
    print(f"      - Std shape: {scaler.scale_.shape}")
    
    return scaler


def normalize_sequences(sequences, scaler, split_name):
    """Sekansları scaler ile normalize eder"""
    
    print(f"\n🔄 {split_name.upper()} normalize ediliyor...")
    
    normalized_sequences = []
    for seq in tqdm(sequences, desc=f"Normalizing {split_name}", unit="video"):
        normalized_seq = scaler.transform(seq)
        normalized_sequences.append(normalized_seq)
    
    print(f"   ✅ {len(normalized_sequences)} sekans normalize edildi")
    
    return normalized_sequences


def analyze_sequence_lengths(sequences, split_name=""):
    """Sekans uzunluklarını analiz eder"""
    lengths = [len(seq) for seq in sequences]
    
    print(f"\n📊 {split_name} Sekans Uzunluk Analizi:")
    print(f"   - Minimum: {np.min(lengths)} frame")
    print(f"   - Maksimum: {np.max(lengths)} frame")
    print(f"   - Ortalama: {np.mean(lengths):.1f} frame")
    print(f"   - Medyan: {np.median(lengths):.1f} frame")
    print(f"   - 95th percentile: {np.percentile(lengths, 95):.1f} frame")
    
    return lengths


def pad_sequences(sequences, max_length, padding='post', truncating='post', value=0.0, split_name=""):
    """Sekansları aynı uzunluğa getirir"""
    
    print(f"\n📏 {split_name} PADDING uygulanıyor (max_length={max_length})...")
    
    if len(sequences) == 0:
        return np.array([])
    
    feature_dim = sequences[0].shape[1]
    padded_array = np.full((len(sequences), max_length, feature_dim), value, dtype=np.float32)
    
    truncated_count = 0
    padded_count = 0
    
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        
        if seq_len > max_length:
            truncated_count += 1
            if truncating == 'post':
                padded_array[i] = seq[:max_length]
            else:
                padded_array[i] = seq[-max_length:]
        else:
            if seq_len < max_length:
                padded_count += 1
            
            if padding == 'post':
                padded_array[i, :seq_len] = seq
            else:
                padded_array[i, -seq_len:] = seq
    
    print(f"   ✅ Padding tamamlandı")
    print(f"      - Padded sequences: {padded_count}")
    print(f"      - Truncated sequences: {truncated_count}")
    print(f"      - Final shape: {padded_array.shape}")
    
    return padded_array


def main():
    """Ana fonksiyon"""
    config = TransformerConfig()
    
    print("=" * 80)
    print("📐 VERİ NORMALİZASYONU VE PADDING (TRAIN/VAL/TEST)")
    print("=" * 80)
    
    keypoints_dir = config.KEYPOINTS_DIR
    
    if not os.path.exists(keypoints_dir):
        print(f"\n❌ HATA: {keypoints_dir} bulunamadı!")
        print(f"Önce 02_extract_keypoints.py scriptini çalıştırın.")
        return
    
    # TRAIN setini yükle
    train_csv = os.path.join(config.DATA_DIR, 'selected_videos_train.csv')
    train_sequences, train_labels, train_ids = load_keypoints_for_split(
        train_csv, keypoints_dir, 'train', config
    )
    
    if train_sequences is None or len(train_sequences) == 0:
        print(f"\n❌ HATA: Train keypoint'leri yüklenemedi!")
        return
    
    # VAL setini yükle
    val_csv = os.path.join(config.DATA_DIR, 'selected_videos_val.csv')
    val_sequences, val_labels, val_ids = load_keypoints_for_split(
        val_csv, keypoints_dir, 'val', config
    )
    
    # TEST setini yükle
    test_csv = os.path.join(config.DATA_DIR, 'selected_videos_test.csv')
    test_sequences, test_labels, test_ids = load_keypoints_for_split(
        test_csv, keypoints_dir, 'test', config
    )
    
    # Sekans uzunluklarını analiz et (sadece train'den max_length belirle)
    train_lengths = analyze_sequence_lengths(train_sequences, "TRAIN")
    
    # Max length hesapla (95th percentile kullan)
    max_length = int(np.percentile(train_lengths, 95))
    print(f"\n✅ Max length (95th percentile from TRAIN): {max_length} frame")
    
    # Scaler'ı sadece TRAIN'de fit et
    scaler = fit_scaler_on_train(train_sequences)
    
    # Scaler'ı kaydet
    scaler_path = os.path.join(config.DATA_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\n💾 Scaler kaydedildi: {scaler_path}")
    
    # TRAIN'i normalize et
    train_normalized = normalize_sequences(train_sequences, scaler, 'TRAIN')
    
    # VAL'i normalize et (aynı scaler ile)
    val_normalized = None
    if val_sequences:
        val_normalized = normalize_sequences(val_sequences, scaler, 'VAL')
    
    # TEST'i normalize et (aynı scaler ile)
    test_normalized = None
    if test_sequences:
        test_normalized = normalize_sequences(test_sequences, scaler, 'TEST')
    
    # Padding uygula (hepsi aynı max_length)
    X_train = pad_sequences(train_normalized, max_length, split_name="TRAIN")
    
    X_val = None
    if val_normalized:
        X_val = pad_sequences(val_normalized, max_length, split_name="VAL")
    
    X_test = None
    if test_normalized:
        X_test = pad_sequences(test_normalized, max_length, split_name="TEST")
    
    # Çıktı dizinini oluştur
    processed_dir = config.PROCESSED_DATA_DIR
    os.makedirs(processed_dir, exist_ok=True)
    
    # Kaydet
    print(f"\n{'='*80}")
    print("💾 VERİLER KAYDEDİLİYOR")
    print(f"{'='*80}")
    print(f"   📁 Dizin: {processed_dir}/")
    
    # TRAIN
    np.save(os.path.join(processed_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(processed_dir, 'y_train.npy'), np.array(train_labels))
    np.save(os.path.join(processed_dir, 'train_ids.npy'), np.array(train_ids))
    print(f"\n   ✅ TRAIN:")
    print(f"      - X_train.npy: {X_train.shape}")
    print(f"      - y_train.npy: {np.array(train_labels).shape}")
    print(f"      - train_ids.npy: {len(train_ids)} IDs")
    
    # VAL
    if X_val is not None:
        np.save(os.path.join(processed_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(processed_dir, 'y_val.npy'), np.array(val_labels))
        np.save(os.path.join(processed_dir, 'val_ids.npy'), np.array(val_ids))
        print(f"\n   ✅ VAL:")
        print(f"      - X_val.npy: {X_val.shape}")
        print(f"      - y_val.npy: {np.array(val_labels).shape}")
        print(f"      - val_ids.npy: {len(val_ids)} IDs")
    
    # TEST
    if X_test is not None:
        np.save(os.path.join(processed_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(processed_dir, 'y_test.npy'), np.array(test_labels))
        np.save(os.path.join(processed_dir, 'test_ids.npy'), np.array(test_ids))
        print(f"\n   ✅ TEST:")
        print(f"      - X_test.npy: {X_test.shape}")
        print(f"      - y_test.npy: {np.array(test_labels).shape}")
        print(f"      - test_ids.npy: {len(test_ids)} IDs")
    
    # Sınıf dağılımı
    print(f"\n{'='*80}")
    print("📊 SINIF DAĞILIMI")
    print(f"{'='*80}")
    
    for split_name, labels in [('TRAIN', train_labels), ('VAL', val_labels), ('TEST', test_labels)]:
        if labels:
            print(f"\n   {split_name}:")
            unique, counts = np.unique(labels, return_counts=True)
            for idx, count in zip(unique, counts):
                # idx is now 0-indexed, map back to original class_id for display
                original_class_id = config.TARGET_CLASS_IDS[idx]
                class_name = config.CLASS_NAMES[idx]
                percentage = (count / len(labels) * 100)
                print(f"      Label {idx} (ClassId {original_class_id}, {class_name:10s}): {count:3d} ({percentage:5.1f}%)")
    
    # Özet
    print(f"\n{'='*80}")
    print("✅ VERİ NORMALİZASYONU TAMAMLANDI")
    print(f"{'='*80}")
    print(f"📁 Kaydedilen dosyalar: {processed_dir}/")
    print(f"📊 Scaler: SADECE train setinde fit edildi ✓")
    print(f"📊 Max length: {max_length} (train'in 95th percentile)")
    print(f"\n📌 Sıradaki adım: Model eğitimi (train.py)")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
