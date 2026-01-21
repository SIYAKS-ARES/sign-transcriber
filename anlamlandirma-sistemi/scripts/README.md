# Scripts Rehberi (`anlamlandirma-sistemi/scripts/`)

Bu klasörde, `anlamlandirma-sistemi` içindeki RAG altyapısı için yardımcı script’ler bulunur.

---

## `init_vectorstore.py`

Amaç:
- `anlamlandirma-sistemi/vectorstore/` altında ChromaDB vektör deposunun varlığını kontrol etmek,
- İstatistikleri göstermek,
- Gerekirse `TID_Sozluk_Verileri/` üzerinden sıfırdan oluşturmak.

### Kullanım

```bash
cd anlamlandirma-sistemi

# 1) Durum kontrolü (varsayılan)
python scripts/init_vectorstore.py --check

# 2) İstatistikler
python scripts/init_vectorstore.py --stats

# 3) Sıfırdan kurulum (TID_Sozluk_Verileri gerekli)
python scripts/init_vectorstore.py --init
```

### Beklenen Çıktılar

- Sozluk koleksiyonu: `tid_sozluk` kayıt sayısı
- Hafıza koleksiyonu: `tid_hafiza` kayıt sayısı

### Notlar

- `--init` için `TID_Sozluk_Verileri/` klasörü bulunmalı.  
  Yoksa script, hazır vektörstore kopyalama yolunu önerir:
  `cp -r ../TRANSKRIPSIYON-RAG-VDB/vectorstore .`

- Vektörstore yolu `VECTORSTORE_PATH` env değişkeni ile override edilebilir (bkz. `rag/config.py`).

---

**Son Güncelleme:** 2026-01-21

