import os
from typing import List, Optional

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers
except Exception as _tf_exc:  # TensorFlow Apple Silicon kurulumu sonraki adımda eklenecek
    tf = None
    layers = models = regularizers = None

try:
    import mediapipe as mp
except Exception:
    mp = None


# === Sabitler ===
SEQ_LEN: int = 60
POSE_FEATURE_DIM: int = 33 * 4  # 132
HAND_FEATURE_DIM: int = 21 * 3  # 63
REDUCED_FACE_LANDMARK_INDICES = [33, 263, 13, 14, 61, 291, 105, 334]
REDUCED_FACE_FEATURE_DIM: int = len(REDUCED_FACE_LANDMARK_INDICES) * 3  # 24
FEATURE_DIM: int = POSE_FEATURE_DIM + (2 * HAND_FEATURE_DIM) + REDUCED_FACE_FEATURE_DIM  # 282
ACTUAL_FEATURE_DIM: int = FEATURE_DIM  # NoneFace modeli: azaltılmış yüz + eller + pose
NUM_CLASSES: int = 226
DROP_FACE: bool = False  # NoneFace modeli: sınırlı yüz verisi tutulur

BASE_DIR = os.path.dirname(__file__)
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "sign_classifier_NoneFace_best_89225.h5")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.csv")


def create_enhanced_model(seq_len: int, feature_dim: int, num_classes: int):
    if tf is None:
        raise RuntimeError("TensorFlow yüklenemedi. Apple Silicon için tensorflow-macos + tensorflow-metal kurulmalı.")

    inputs = layers.Input(shape=(seq_len, feature_dim))

    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.LayerNormalization()(x)

    x = layers.TimeDistributed(layers.Dense(256, activation="relu"))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.Dropout(0.3)(x)

    x = layers.TimeDistributed(layers.Dense(128, activation="relu"))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.Dropout(0.3)(x)

    attention = layers.MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.1)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)

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

    max_pool = layers.GlobalMaxPooling1D()(x)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    x = layers.Concatenate()([max_pool, avg_pool])

    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    return model


def load_labels(labels_path: str, num_classes: int) -> List[str]:
    if os.path.isfile(labels_path):
        try:
            # CSV'yi başlıkla birlikte oku; TR kolonu varsa onu kullan
            import csv
            with open(labels_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                names = []
                for row in reader:
                    name = row.get("TR") or row.get("label") or row.get("name")
                    if not name:
                        # İlk sütunu bul
                        if len(row) > 0:
                            name = list(row.values())[0]
                    names.append(str(name).strip().upper().replace("^", "_").replace(" ", "_"))
            if len(names) >= num_classes:
                return names
            # Eksik ise pad et
            names += [f"CLASS_{i}" for i in range(len(names), num_classes)]
            return names
        except Exception:
            pass
    # Fallback: CLASS_0..CLASS_{N-1}
    return [f"CLASS_{i}" for i in range(num_classes)]


class LocalSignModel:
    def __init__(self, model, class_names: List[str]):
        self.model = model
        self.class_names = class_names
        self.loaded = model is not None


def load_model() -> Optional[LocalSignModel]:
    # Yerel TİD modeli yükleme
    if not os.path.isfile(MODEL_WEIGHTS_PATH):
        print(f"Model ağırlık dosyası bulunamadı: {MODEL_WEIGHTS_PATH}")
        return LocalSignModel(None, [f"CLASS_{i}" for i in range(NUM_CLASSES)])

    try:
        model = create_enhanced_model(SEQ_LEN, ACTUAL_FEATURE_DIM, NUM_CLASSES)
        model.load_weights(MODEL_WEIGHTS_PATH)
        class_names = load_labels(LABELS_PATH, NUM_CLASSES)
        print("NoneFace model yüklendi (azaltılmış yüz, 282 boyut).")
        return LocalSignModel(model, class_names)
    except Exception as exc:
        print(f"Model yüklenirken hata: {exc}")
        return LocalSignModel(None, [f"CLASS_{i}" for i in range(NUM_CLASSES)])


def get_transcription_from_local_model(input_data):
    # Entegrasyon tamamlanana kadar placeholder; API tarafı gerçek akışı kullanacak
    print(f"Girdi verisi alındı: {input_data}")
    sample_transcription = (
        "BEN ÖNCE ARABA^SÜRMEK BİLMEK^DEĞİL BEN CAHİL BEN SONRA ARKADAŞ "
        "ÖĞRETMEK ÖĞRETMEK BEN ŞİMDİ ANLAMAK ARABA^SÜRMEK SÜRMEK"
    )
    return sample_transcription


# === Özellik çıkarımı ve sekans hazırlama ===
def frames_to_keypoint_sequence(frames: List[np.ndarray]) -> np.ndarray:
    """BGR frame listesi -> (T, 282) anahtar nokta dizisi.
    Pose(33*4) + Reduced Face(8*3) + Hands(2*21*3) sırayla birleştirilir.
    """
    if mp is None:
        raise RuntimeError("MediaPipe kurulmamış. 'mediapipe' bağımlılığını ekleyin.")

    print(f"MediaPipe işleme başlıyor: {len(frames)} frame")
    
    mp_holistic = mp.solutions.holistic
    seq: List[np.ndarray] = []
    pose_detected = 0
    face_detected = 0
    left_hand_detected = 0
    right_hand_detected = 0
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for i, frame in enumerate(frames):
            # Frame bilgisi
            if i == 0:
                print(f"Frame boyutu: {frame.shape}, dtype: {frame.dtype}")
            
            # BGR -> RGB
            image_rgb = frame[..., ::-1]
            results = holistic.process(image_rgb)

            # Pose 33*4
            if results.pose_landmarks:
                pose = np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark], dtype=np.float32).flatten()
                pose_detected += 1
            else:
                pose = np.zeros(33 * 4, dtype=np.float32)

            # Reduced face 8*3
            if results.face_landmarks:
                lms = results.face_landmarks.landmark
                face_points = []
                for idx in REDUCED_FACE_LANDMARK_INDICES:
                    if idx < len(lms):
                        res = lms[idx]
                        face_points.extend([res.x, res.y, res.z])
                    else:
                        face_points.extend([0.0, 0.0, 0.0])
                face = np.array(face_points, dtype=np.float32)
                face_detected += 1
            else:
                face = np.zeros(REDUCED_FACE_FEATURE_DIM, dtype=np.float32)

            # Hands 21*3 each
            if results.left_hand_landmarks:
                lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
                left_hand_detected += 1
            else:
                lh = np.zeros(21 * 3, dtype=np.float32)

            if results.right_hand_landmarks:
                rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
                right_hand_detected += 1
            else:
                rh = np.zeros(21 * 3, dtype=np.float32)

            keypoints = np.concatenate([pose, face, lh, rh]).astype(np.float32)
            seq.append(keypoints)
    
    print(
        "MediaPipe tespit oranları - Pose: "
        f"{pose_detected}/{len(frames)}, Face: {face_detected}/{len(frames)}, "
        f"Left Hand: {left_hand_detected}/{len(frames)}, Right Hand: {right_hand_detected}/{len(frames)}"
    )
    
    result = np.array(seq, dtype=np.float32)
    print(f"MediaPipe çıktısı: {result.shape}")
    return result


def uniform_sample_or_pad(sequence: np.ndarray, target_len: int) -> np.ndarray:
    """Uzunluk hedefe eşitse aynen, büyükse uniform örnekle, küçükse sıfırla pad et."""
    if sequence.shape[0] == target_len:
        return sequence
    if sequence.shape[0] > target_len:
        idx = np.linspace(0, sequence.shape[0] - 1, target_len).astype(int)
        return sequence[idx]
    pad = np.zeros((target_len - sequence.shape[0], sequence.shape[1]), dtype=sequence.dtype)
    return np.concatenate([sequence, pad], axis=0)


def maybe_drop_face(seq_282: np.ndarray) -> np.ndarray:
    """NoneFace modelinde azaltılmış yüz özellikleri tutulur."""
    print(f"Azaltılmış yüz dahil tutuldu: {seq_282.shape}")
    return seq_282


def prepare_sequence_for_model(seq_array: np.ndarray, target_len: int = SEQ_LEN) -> np.ndarray:
    """(T, 282) -> (1, target_len, 282) NoneFace model girişi hazırlar."""
    print(f"prepare_sequence_for_model girişi: {seq_array.shape}")
    
    if seq_array.ndim != 2 or seq_array.shape[1] != FEATURE_DIM:
        raise ValueError(f"Beklenen şekil (T,{FEATURE_DIM}), gelen: {seq_array.shape}")
    
    if seq_array.shape[0] < target_len:
        seq_fixed = uniform_sample_or_pad(seq_array, target_len)
        print(f"Padding uygulandı: {seq_array.shape[0]} -> {seq_fixed.shape[0]} frame")
    else:
        # Son 60 kareyi al
        seq_fixed = seq_array[-target_len:]
        print(f"Son {target_len} frame alındı: {seq_array.shape[0]} -> {seq_fixed.shape[0]} frame")
    
    # Yüz çıkarımı (NoneFace model için)
    seq_fixed = maybe_drop_face(seq_fixed)
    
    # Final boyut kontrolü
    expected_final_dim = ACTUAL_FEATURE_DIM  # NoneFace: 282
    if seq_fixed.shape[1] != expected_final_dim:
        raise ValueError(f"Model girişi boyut hatası. Beklenen: {expected_final_dim}, Gelen: {seq_fixed.shape[1]}")
    
    result = np.expand_dims(seq_fixed, axis=0)
    print(f"Model girişi hazır: {result.shape}")
    return result


def predict_from_frames(local_model: LocalSignModel, frames: List[np.ndarray], confidence_threshold: float = 0.3) -> dict:
    """Frame listesi üzerinden tek-pencere tahmin döndürür."""
    if local_model is None or not local_model.loaded or local_model.model is None:
        print("Model yüklü değil veya None")
        return {
            "pred_id": -1,
            "pred_name": "",
            "confidence": 0.0,
            "top5": [],
        }

    print(f"Gelen frame sayısı: {len(frames)}")
    if len(frames) < 5:
        print(f"Yetersiz frame sayısı: {len(frames)} < 5")
        return {"pred_id": -1, "pred_name": "", "confidence": 0.0, "top5": []}

    seq = frames_to_keypoint_sequence(frames)
    if seq.size == 0:
        print("Sequence boş - MediaPipe keypoint bulunamadı")
        return {"pred_id": -1, "pred_name": "", "confidence": 0.0, "top5": []}

    print(f"MediaPipe sequence şekli: {seq.shape}")
    
    # NaN kontrolü
    if np.any(np.isnan(seq)):
        print("Sequence'te NaN değer var, temizleniyor...")
        seq = np.nan_to_num(seq, nan=0.0)

    seq_np = prepare_sequence_for_model(seq, SEQ_LEN)
    
    # Model tahmininden önce son kontrol
    if np.any(np.isnan(seq_np)):
        print("Model girişinde NaN var, temizleniyor...")
        seq_np = np.nan_to_num(seq_np, nan=0.0)
    
    probs = local_model.model.predict(seq_np, verbose=0)[0]
    
    # NaN kontrol ve debug çıktısı
    if np.any(np.isnan(probs)):
        print("Model çıktısında NaN var!")
        return {"pred_id": -1, "pred_name": "", "confidence": 0.0, "top5": []}
    
    # Detaylı debug çıktısı
    max_conf = float(np.max(probs))
    min_conf = float(np.min(probs))
    mean_conf = float(np.mean(probs))
    
    print(f"=== MODEL TAHMİN SONUÇLARI ===")
    print(f"Frame sayısı: {len(frames)}")
    print(f"Sequence şekli: {seq.shape}")
    print(f"Model girişi: {seq_np.shape}")
    print(f"Model çıktısı şekli: {probs.shape}")
    print(f"Güven istatistikleri - Max: {max_conf:.4f}, Min: {min_conf:.4f}, Mean: {mean_conf:.4f}")
    print(f"Eşik: {confidence_threshold}")

    top5_ids = np.argsort(probs)[-5:][::-1]
    top5 = [
        {
            "id": int(i),
            "name": local_model.class_names[int(i)] if int(i) < len(local_model.class_names) else f"CLASS_{int(i)}",
            "confidence": float(probs[int(i)]),
        }
        for i in top5_ids
    ]

    pred_id = int(np.argmax(probs))
    pred_name = local_model.class_names[pred_id] if pred_id < len(local_model.class_names) else f"CLASS_{pred_id}"
    conf = float(probs[pred_id])

    # Etiket formatını TİD token uyumlu hale getir
    pred_name = str(pred_name).strip().upper().replace("^", "_").replace(" ", "_")
    
    threshold_met = conf >= confidence_threshold
    
    print(f"Top-1 Tahmin: {pred_name} (ID: {pred_id}, Güven: {conf:.4f})")
    print(f"Eşik karşılandı: {threshold_met}")
    print(f"Top-5 Tahminler:")
    for i, item in enumerate(top5):
        print(f"  {i+1}. {item['name']} - {item['confidence']:.4f}")
    print("=" * 40)

    return {
        "pred_id": pred_id,
        "pred_name": pred_name,
        "confidence": conf,
        "top5": top5,
        "threshold_met": threshold_met,
    }


