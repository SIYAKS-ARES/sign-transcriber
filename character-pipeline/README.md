# Character Pipeline (Hand Sign Detection) Rehberi

Bu klasör, kamera/videodan **el işareti (hand sign)** tespiti ve basit jest tanıma üzerine deneysel çalışmaları içerir. Amaç, işaret dili kelime sınıflandırıcılarından daha önce, **tek kare/sekans el şekli** seviyesinde ön denemeler yapmaktır.

---

## Klasör Yapısı

```
character-pipeline/
├── hand/                    # MediaPipe Hands tabanlı jest tespiti
│   ├── hand.py              # Gerçek zamanlı el takibi + jest sınıflandırma
│   └── El İşaretleri Algılama İyileştirmesi_.docx
│
├── Hand Sign Detection/     # Toplanan el işareti görüntüleri (ham veri)
│   └── *.jpg                # Deneysel el pozları
│
└── hand-sign-detection/     # cvzone tabanlı basit el tespiti (eski)
    ├── data-collection.py   # Veri toplama scripti (kamera + crop)
    └── test.py              # Hızlı test (HandDetector)
```

---

## 1. MediaPipe Hands Jest Tanıma (`hand/hand.py`)

### Özeti

- **Kütüphaneler**: OpenCV, MediaPipe Hands, NumPy  
- **Çıktı**: Ekranda el landmarkları çizili görüntü + basit jest ismi (örn. \"Begen\", \"Iki\", \"Bes\", \"Yumruk\")  
- **Mantık**:
  - MediaPipe ile el landmark’ları çıkarılır.
  - Başparmak ve diğer 4 parmağın açık/kapalı durumu hesaplanır (`get_finger_status`).
  - 5 bitlik parmak durumu vektörü (`(B, İ, O, Y, S)`) `GESTURE_MAP` sözlüğünde karşılık bulursa jest etiketi döner.

### Çalıştırma

```bash
cd character-pipeline/hand
python hand.py
```

Beklenti:
- Kamera açılır.
- Elinizi çerçeveye soktuğunuzda MediaPipe landmarkları görünür.
- Parmak kombinasyonuna göre jest etiketi terminalde / ekranda gösterilir (kod içinde).

> Not: Jest haritası (`GESTURE_MAP`) genişletilebilir; yeni jestler için parmak kombinasyonları eklenebilir.

---

## 2. Eski cvzone Tabanlı Denemeler (`hand-sign-detection/`)

Bu klasör, **cvzone.HandTrackingModule.HandDetector** kullanılarak yapılmış ilk el tespiti ve veri toplama denemelerini içerir.

### 2.1. Veri Toplama (`data-collection.py`)

Amaç:
- Kameradan gelen görüntülerden, belirli el pozlarını kırpıp **görüntü dataseti** oluşturmak (örneğin karakter tahmini için).

Tipik kullanım (varsayılan kamera 0):

```bash
cd character-pipeline/hand-sign-detection
python data-collection.py
```

Klasörde `Hand Sign Detection/` altında `.jpg` dosyaları oluşur (her bir kare veya crop).

### 2.2. Hızlı Test (`test.py`)

```bash
python test.py
```

Bu script:
- Kamerayı açar.
- `HandDetector` ile el bbox’unu bulur.
- Basit bir görüntüleme döngüsü çalıştırır.

Bu kod, daha gelişmiş MediaPipe tabanlı `hand/hand.py` için bir başlangıç prototip olarak düşünülmelidir.

---

## Kullanım Önerileri

- **Model eğitimi için**:  
  - `Hand Sign Detection/` altındaki kareleri etiketleyip basit bir CNN/Transformer ile harf/jest sınıflandırma çalışmaları yapabilirsiniz.
  - Daha olgun pipeline için, `transformer-signlang` ve `anlamlandirma-sistemi` projeleriyle entegrasyon düşünebilirsiniz.

- **MediaPipe -> Keypoint Pipeline**:  
  - `local_model_handler.py` içinde kullanılan MediaPipe keypoint akışına benzer şekilde, bu klasördeki denemeler el seviyesinde daha hafif prototiplerdir.

---

## Notlar

- Bu klasördeki kodlar **deneysel** ve araştırma sırasında hızlı prototip amaçlıdır; üretim kalitesinde hata yönetimi ve dokümantasyon `anlamlandirma-sistemi` ve `transformer-signlang` klasörlerinde daha olgundur.
- Yeni jestler veya karakterler eklerken:
  - Jest haritasını (`GESTURE_MAP`) güncelleyin.
  - Gerekirse parmak açıları/landmark pozisyonlarına dayalı ek sezgisel kurallar ekleyin.

