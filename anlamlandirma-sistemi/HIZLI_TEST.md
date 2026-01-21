# Video Oynatma - Hızlı Test Rehberi

## Yapılan Değişiklikler

### ✅ 1. Video Oynatma Basitleştirildi
- Karmaşık kontroller kaldırıldı
- Video seçildiğinde otomatik olarak gösteriliyor
- Başlat tuşuna basıldığında oynatılıyor

### ✅ 2. LLM Hatası Yönetimi
- API kotası dolduğunda sistem artık çökmüyor
- LLM olmadan da çalışabiliyor (sadece model tahmini gösteriyor)
- Hata mesajları kullanıcı dostu

### ✅ 3. Veritabanı Eklendi
- SQLite ile video kayıtları saklanıyor
- İstatistikler tutulabiliyor
- `anlamlandirma.db` dosyasında kayıtlar

## Hızlı Test Adımları

### 1. Conda Environment Aktif Et
```bash
conda activate anlamlandirma
```

### 2. Uygulamayı Başlat
```bash
cd /Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/anlamlandirma-sistemi
python app.py
```

### 3. Tarayıcıda Aç
```
http://localhost:5005/demo
```

### 4. Test Et
1. **Video Yükle** sekmesini seç
2. Bir test videosu seç (test_videos/acikmak_1.mp4)
3. Video ekranda görünecek
4. **Başlat** butonuna bas
5. Video oynatılacak ve aynı anda API'ye gönderilecek

## Beklenen Davranış

### Video Oynatma
- ✅ Video yüklendiğinde preview gösterilir
- ✅ Başlat tuşuna basınca video oynar
- ✅ Video kontrolleri (play/pause/seek) kullanılabilir

### Model Tahmini (LLM Olmadan)
- ✅ Video analiz edilir
- ✅ Model tahmini gösterilir (örn: "Model tahmini: acikmak")
- ✅ LLM kotası dolduğu için çeviri yapılmaz (şimdilik)

### Veritabanı
- ✅ Her işlem veritabanına kaydedilir
- ✅ İstatistikler tutulur

## LLM'i Tekrar Aktif Etmek İçin

Eğer LLM kotanız yenilendiyse veya başka bir provider kullanmak isterseniz:

1. `demo.html` dosyasında şu satırı bulun (605. satır):
   ```javascript
   formData.append('use_llm', 'false');
   ```

2. Şöyle değiştirin:
   ```javascript
   formData.append('use_llm', 'true');
   ```

3. Provider seçimini yapın (OpenAI, Anthropic, Gemini)

## Veritabanı İstatistikleri Görüntüleme

```bash
python database.py
```

## API Endpoint'leri

- `/api/process_video` - Video işleme (POST)
- `/api/history` - İşlem geçmişi (GET)
- `/api/test_model` - Model testi (POST)

## Sorun Giderme

### Video Oynatılmıyor
- Tarayıcı konsolunu açın (F12)
- Console'da hata var mı kontrol edin
- Video formatı destekleniyor mu kontrol edin (MP4 önerilir)

### API Hatası
- LLM kotası dolmuşsa normal
- `use_llm: false` ile çalıştırın
- Model tahmini yine de gösterilir

### Port Meşgul
```bash
pkill -f "python.*app.py"
python app.py
```

## Başarı Kriterleri

✅ Video seçilebiliyor
✅ Video ekranda görüntüleniyor
✅ Başlat tuşuna basınca video oynatılıyor
✅ API'ye gönderiliyor
✅ Sonuç gösteriliyor (LLM olmadan da)
✅ Hata mesajları anlaşılır
✅ Veritabanına kaydediliyor

## Notlar

- Şu an LLM kotası dolduğu için `use_llm: false` modunda çalışıyor
- Model tahmini yine de gösteriliyor
- Video oynatma tamamen bağımsız çalışıyor
- Sistem artık daha basit ve kararlı

