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
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(
        x
    )
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model


LABELS_CSV = "SignList_ClassId_TR_EN.csv"

# ==== KULLANICI AYARLARI ====
TEST_DIR = "test_videos"
MODEL_PATH = "models/sign_classifier_NoneFace_best_89225.h5"

df_labels = pd.read_csv(LABELS_CSV)
CLASS_NAMES = df_labels["TR"].tolist()
SEQ_LEN = 60
DROP_FACE = True  # None Face modeli için True olmalı
print("Clas Names: ", CLASS_NAMES)
# ==== Modeli yükle ====
if not os.path.isfile(MODEL_PATH):
    raise RuntimeError(f"Model dosyası bulunamadı: {MODEL_PATH}")

# None Face modeli için feature dimension hesaplama:
# POSE: 33 * 4 = 132
# HANDS: 2 * 21 * 3 = 126  
# FACE: 8 * 3 = 24 (reduced face points - model hala bu boyutu bekliyor)
# Total: 132 + 126 + 24 = 282 (model 282 feature bekliyor)
feature_dim = 132 + 126 + 24  # POSE + HANDS + reduced FACE
model = create_enhanced_model(SEQ_LEN, feature_dim, 226)
model.load_weights(MODEL_PATH)

# ==== MediaPipe kurulum ====
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def draw_styled_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_face_mesh.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )


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


def maybe_drop_face(seq_282):
    # Bu fonksiyon artık face'i drop etmiyor, sadece reduced face kullanıyor
    # Çünkü model 282 boyut bekliyor (pose + hands + reduced_face)
    return seq_282


# ==== Video listesi ====
video_exts = ("*.mp4", "*.avi", "*.mov", "*.mkv")
video_list = []
for ext in video_exts:
    video_list.extend(glob.glob(os.path.join(TEST_DIR, ext)))
if not video_list:
    raise RuntimeError("TEST_DIR içinde video bulunamadı.")

print(f"Bulunan video sayısı: {len(video_list)}")

# ==== Her video için sırayla çalıştır ====
for vp in video_list:
    print("\nVideo:", os.path.basename(vp))
    vidoname = os.path.basename(vp)
    # Video adından label çıkar (örn: acikmak_1.mp4 -> acikmak)
    label = vidoname.split("_")[0] if "_" in vidoname else vidoname.split(".")[0]
    print("Label: ", label)
    cap = cv2.VideoCapture(vp)
    sequence = list()
    cap.set(3, 1024)
    cap.set(4, 1024)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            print("İmage Shape: ", frame.shape)
            print("İmageRGy Shape: ", image_rgb.shape)
            pose = (
                np.array(
                    [
                        [r.x, r.y, r.z, r.visibility]
                        for r in results.pose_landmarks.landmark
                    ]
                ).flatten()
                if results.pose_landmarks
                else np.zeros(33 * 4, dtype=np.float32)
            )
            if results.face_landmarks:
                lms = results.face_landmarks.landmark
                face_points = []
                for idx in [33, 263, 13, 14, 61, 291, 105, 334]:
                    res = lms[idx]
                    face_points.extend([res.x, res.y, res.z])
                face = np.array(face_points, dtype=np.float32)
            else:
                face = np.zeros(8 * 3, dtype=np.float32)  # 8 nokta x 3 koordinat
            lh = (
                np.array(
                    [[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]
                ).flatten()
                if results.left_hand_landmarks
                else np.zeros(21 * 3, dtype=np.float32)
            )
            rh = (
                np.array(
                    [[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]
                ).flatten()
                if results.right_hand_landmarks
                else np.zeros(21 * 3, dtype=np.float32)
            )

            keypoints = np.concatenate([pose, lh, rh, face]).astype(np.float32)
            sequence.append(keypoints)

            draw_styled_landmarks(frame, results)

            if len(sequence) >= 45:
                seq_array = np.array(sequence, dtype=np.float32)
                if len(seq_array) < SEQ_LEN:
                    seq_fixed = uniform_sample_or_pad(seq_array, SEQ_LEN)
                else:
                    seq_fixed = seq_array[-SEQ_LEN:]

                seq_np = np.expand_dims(maybe_drop_face(seq_fixed), axis=0)
                probs = model.predict(seq_np, verbose=0)[0]

                # top-5 indexleri al
                top5_ids = np.argsort(probs)[-5:][::-1]

                print("Top-5 Tahminler:")
                for rank, i in enumerate(top5_ids, start=1):
                    cname = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"UNK_{i}"
                    print(f"{rank}. {cname} (conf={probs[i]:.4f})")

                pred_id = int(np.argmax(probs))
                pred_name = CLASS_NAMES[pred_id]
                conf = float(probs[pred_id])
                print("Pred:", pred_name, "CONF:", conf)

                if conf > 0.7:
                    cv2.putText(
                        frame,
                        f"Tahmin: {pred_name}",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Label: {label}",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"({conf:.2f})",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 255),
                        3,
                        cv2.LINE_AA,
                    )

            cv2.imshow("Video Tahmin", frame)
            key = cv2.waitKey(25) & 0xFF
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                exit()
            elif key == ord("n"):
                break

    cap.release()

cv2.destroyAllWindows()
