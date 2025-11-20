import cv2
import mediapipe as mp
import numpy as np
from google.protobuf.json_format import MessageToDict # Handedness label'ını almak için

# Hareketleri tanımlayan global bir sözlük
# Parmak durumları: (Başparmak, İşaret, Orta, Yüzük, Serçe) - 1: Açık, 0: Kapalı
GESTURE_MAP = {
    (1, 0, 0, 0, 0): "Begen",
    (0, 1, 0, 0, 0): "Isaret",
    (0, 1, 1, 0, 0): "Iki",
    (0, 1, 1, 1, 0): "Uc",
    (0, 1, 1, 1, 1): "Dort",
    (1, 1, 1, 1, 1): "Bes",
    (0, 0, 0, 0, 0): "Yumruk",
    (1, 1, 0, 0, 0): "L Harfi"
    # Daha fazla özel hareket buraya eklenebilir
}

class HandTracker:
    """
    MediaPipe kullanarak el takibi ve temel hareket tanıma işlemlerini gerçekleştiren sınıf.
    """
    def __init__(self, static_image_mode=False, max_num_hands=2,
                 min_detection_confidence=0.7, min_tracking_confidence=0.5):
        """
        HandTracker sınıfını başlatır.

        Args:
            static_image_mode (bool): Görüntülerin statik mi yoksa video akışı mı olduğunu belirtir.
                                      Varsayılan: False.
            max_num_hands (int): Algılanacak maksimum el sayısı. Varsayılan: 2.
            min_detection_confidence (float): El algılamanın başarılı sayılması için minimum güven skoru.
                                             Varsayılan: 0.7.
            min_tracking_confidence (float): El takibinin başarılı sayılması için minimum güven skoru.
                                            Varsayılan: 0.5.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Parmak ucu landmark ID'leri (Başparmak, İşaret, Orta, Yüzük, Serçe)
        self.tip_ids = [4, 8, 12, 16, 20]  # DÜZELTME: Değer atandı

        # Çizim stilleri
        self.landmark_drawing_spec = self.mp_drawing_styles.get_default_hand_landmarks_style()
        self.connection_drawing_spec = self.mp_drawing_styles.get_default_hand_connections_style()
        # İsteğe bağlı özel çizim stilleri:
        # self.landmark_drawing_spec_custom = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
        # self.connection_drawing_spec_custom = self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)


    def process_frame(self, image):
        """
        Verilen görüntü karesinde el takibini gerçekleştirir ve landmarkları çizer.

        Args:
            image (numpy.ndarray): İşlenecek BGR formatındaki görüntü karesi.

        Returns:
            tuple: (İşlenmiş görüntü, MediaPipe el landmark sonuçları)
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False  # Performans optimizasyonu
        results = self.hands.process(image_rgb)
        image.flags.writeable = True  # Çizim için geri yazılabilir yap

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Varsayılan veya özel stillerle çizim
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.landmark_drawing_spec, # veya self.landmark_drawing_spec_custom
                    self.connection_drawing_spec # veya self.connection_drawing_spec_custom
                )
        return image, results

    def get_finger_status(self, hand_landmarks, handedness_label):
        """
        Belirli bir elin parmaklarının açık/kapalı durumunu belirler.

        Args:
            hand_landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
                Bir elin tespit edilmiş landmarkları.
            handedness_label (str): Elin "Left" ya da "Right" olduğunu belirten string.

        Returns:
            list: 5 elemanlı bir liste, her eleman bir parmağın durumunu (1: açık, 0: kapalı) temsil eder.
                  Sırasıyla: Başparmak, İşaret, Orta, Yüzük, Serçe.
        """
        fingers = []  # DÜZELTME: Liste başlatıldı
        landmarks = hand_landmarks.landmark

        thumb_tip = landmarks[self.tip_ids[0]]  # Başparmak ucu (Landmark 4)
        thumb_mcp = landmarks[2]  # Başparmak MCP eklemi (Landmark 2)

        if handedness_label == "Right":
            # Sağ el için, başparmak ucu MCP'den daha sağdaysa (x değeri büyükse) açık (görüntü çevrildiği için).
            if thumb_tip.x > thumb_mcp.x: 
                fingers.append(1)
            else:
                fingers.append(0)
        elif handedness_label == "Left":
            # Sol el için, başparmak ucu MCP'den daha soldaysa (x değeri küçükse) açık (görüntü çevrildiği için).
            if thumb_tip.x < thumb_mcp.x: 
                fingers.append(1)
            else:
                fingers.append(0)
        else:  # Handedness bilinmiyorsa
            fingers.append(0)


        # Diğer dört parmak (İşaret, Orta, Yüzük, Serçe)
        # Parmak ucu (tip_ids[i]), bir önceki eklemden (tip_ids[i]-2) daha yukarıdaysa (y değeri küçükse) açık.
        for i in range(1, 5):  # İşaret, Orta, Yüzük, Serçe parmakları (indeks 1'den 4'e)
            finger_tip = landmarks[self.tip_ids[i]]
            finger_pip = landmarks[self.tip_ids[i] - 2]  # PIP eklemi
            if finger_tip.y < finger_pip.y:
                fingers.append(1)  # Açık
            else:
                fingers.append(0)  # Kapalı
        return fingers

    def _calculate_angle(self, p1_coords, p2_coords, p3_coords):
        """
        Üç nokta (p1, p2, p3) arasındaki açıyı hesaplar. p2 köşe noktasıdır.
        Noktalar (x,y) koordinatları olarak verilir.

        Args:
            p1_coords (tuple): İlk noktanın (x, y) koordinatları.
            p2_coords (tuple): Köşe noktasının (x, y) koordinatları.
            p3_coords (tuple): Üçüncü noktanın (x, y) koordinatları.

        Returns:
            float: Derece cinsinden hesaplanan açı. NaN dönebilir eğer hesaplanamazsa.
        """
        p1 = np.array(p1_coords)
        p2 = np.array(p2_coords)
        p3 = np.array(p3_coords)

        vector_ba = p1 - p2
        vector_bc = p3 - p2

        norm_ba = np.linalg.norm(vector_ba)
        norm_bc = np.linalg.norm(vector_bc)

        if norm_ba == 0 or norm_bc == 0:  # Vektörlerden biri sıfırsa açı tanımsız
            return np.nan 

        dot_product = np.dot(vector_ba, vector_bc)
        # Sıfıra bölme hatasını önlemek için küçük bir epsilon değeri eklenebilir, ancak norm kontrolü zaten var.
        # Ancak, norm_ba * norm_bc çarpımı çok küçükse yine de sorun olabilir.
        denominator = norm_ba * norm_bc
        if denominator < 1e-6:  # Çok küçük paydayı önle
             return np.nan
        
        cosine_angle = dot_product / denominator
        
        # Sayısal hatalardan dolayı aralık dışına çıkmayı önle
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

def main():
    """
    Ana fonksiyon: Kamera akışını başlatır, el takibini yapar ve sonuçları gösterir.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Hata: Kamera başlatılamadı.")
        return

    tracker = HandTracker()
    # Görüntü boyutlarını bir kez al
    ret, frame_for_size = cap.read()
    if not ret:
        print("Hata: Kameradan ilk kare okunamadı.")
        cap.release()
        return
    image_height, image_width, _ = frame_for_size.shape
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Kamerayı başa sar (eğer destekleniyorsa) veya yeniden aç


    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Hata: Kameradan kare okunamadı.")
            break

        # Görüntüyü yatayda çevir (ayna efekti)
        image = cv2.flip(image, 1)

        # El takibini yap ve landmarkları çiz
        image, results = tracker.process_frame(image)

        total_fingers_count_display = 0  # Tüm ellerdeki başparmak hariç toplam parmak sayısı

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Elin Sağ/Sol bilgisini al
                handedness_classification = results.multi_handedness[hand_idx]
                # DÜZELTME: 'classification' bir liste olduğu için ilk elemanına erişiliyor
                handedness_label = MessageToDict(handedness_classification)['classification'][0]['label']

                # Parmak durumlarını al
                finger_statuses = tracker.get_finger_status(hand_landmarks, handedness_label)
                
                # Hareket tanıma
                finger_statuses_tuple = tuple(finger_statuses)
                gesture_name = GESTURE_MAP.get(finger_statuses_tuple, "Bilinmiyor")

                # Başparmak hariç açık parmak sayısını hesapla (sayı sayma için)
                current_hand_finger_count = sum(finger_statuses[1:])  # Başparmak hariç

                # Bilgileri ekrana yazdır
                wrist_landmark = hand_landmarks.landmark[0]
                text_x = int(wrist_landmark.x * image_width) - 50
                text_y = int(wrist_landmark.y * image_height) - 50
                
                cv2.putText(image, f"{handedness_label} El: {gesture_name}", (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 50, 50), 2, cv2.LINE_AA)
                cv2.putText(image, f"Parmaklar: {current_hand_finger_count}", (text_x, text_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 50, 50), 2, cv2.LINE_AA)

                total_fingers_count_display += current_hand_finger_count

                # İşaret parmağı açısını hesapla ve göster (landmark 5, 6, 8)
                # INDEX_FINGER_MCP (5), INDEX_FINGER_PIP (6), INDEX_FINGER_TIP (8)
                # Bu landmarkların varlığını kontrol et
                required_landmarks_for_angle = [5, 6, 8]
                if len(hand_landmarks.landmark) > max(required_landmarks_for_angle):
                    
                    p1_mcp = hand_landmarks.landmark[5]
                    p2_pip = hand_landmarks.landmark[6]
                    p3_tip = hand_landmarks.landmark[8]

                    angle = tracker._calculate_angle(
                        (p1_mcp.x * image_width, p1_mcp.y * image_height),
                        (p2_pip.x * image_width, p2_pip.y * image_height),
                        (p3_tip.x * image_width, p3_tip.y * image_height)
                    )
                    if not np.isnan(angle):
                         cv2.putText(image, f"Isaret Acisi: {int(angle)}", 
                                     (int(p2_pip.x * image_width) - 20, int(p2_pip.y * image_height) - 20),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)

            # Tüm ellerdeki toplam (başparmak hariç) parmak sayısını göster
            cv2.putText(image, f"Toplam Sayi: {total_fingers_count_display}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Parmak sayısına göre renk (orijinal script'teki mantık, isteğe bağlı kullanılabilir)
            text_color_by_count = (50, 50, 50)  # Varsayılan renk
            if total_fingers_count_display == 1: 
                text_color_by_count = (255, 0, 0)  # Mavi (BGR)
            elif total_fingers_count_display == 2: 
                text_color_by_count = (0, 255, 0)  # Yeşil
            elif total_fingers_count_display == 3: 
                text_color_by_count = (0, 0, 255)  # Kırmızı
            elif total_fingers_count_display == 4: 
                text_color_by_count = (255, 255, 0)  # Cyan
            elif total_fingers_count_display == 5: 
                text_color_by_count = (255, 0, 255)  # Magenta
            #... diğer sayılar için eklenebilir
            # Örneğin, yukarıdaki "Toplam Sayi" metninin rengini değiştirmek için:
            # cv2.putText(image, f"Toplam Sayi: {total_fingers_count_display}", (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, text_color_by_count, 2, cv2.LINE_AA)


        cv2.imshow('Gelismis El Takibi', image)

        if cv2.waitKey(5) & 0xFF == ord('z'):  # 'z' tuşuna basılınca çık
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()