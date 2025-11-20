"""
Utility fonksiyon: AUTSL dataset'indeki 226 sınıf ismini yükle

Author: AI Assistant
Date: 7 Ekim 2025
"""

import pandas as pd
import os
from pathlib import Path


def load_all_class_names(csv_path=None):
    """
    AUTSL dataset'indeki tüm 226 sınıf ismini CSV'den yükle.
    
    Args:
        csv_path (str, optional): SignList_ClassId_TR_EN.csv dosyasının yolu.
                                  Belirtilmezse varsayılan konumdan yükler.
    
    Returns:
        list: 226 elemanlı sınıf isimleri listesi (Türkçe)
              ClassId sırasına göre (0-225)
    
    Example:
        >>> class_names = load_all_class_names()
        >>> len(class_names)
        226
        >>> class_names[0]
        'abla'
        >>> class_names[1]
        'acele'
        >>> class_names[225]
        'zor'
    """
    
    # Varsayılan CSV yolu
    if csv_path is None:
        # transformer-signlang/ dizininden Data/ dizinine git
        current_dir = Path(__file__).parent.parent  # transformer-signlang/
        project_root = current_dir.parent  # klassifier-sign-language/
        csv_path = project_root / "Data" / "Class ID" / "SignList_ClassId_TR_EN.csv"
    
    # CSV dosyasını kontrol et
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"SignList_ClassId_TR_EN.csv bulunamadı: {csv_path}\n"
            f"Lütfen dosya yolunu kontrol edin."
        )
    
    # CSV'yi yükle
    df = pd.read_csv(csv_path)
    
    # Sütun kontrolü
    required_columns = ['ClassId', 'TR', 'EN']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            f"CSV dosyası beklenen sütunları içermiyor!\n"
            f"Beklenen: {required_columns}\n"
            f"Bulunan: {df.columns.tolist()}"
        )
    
    # ClassId'ye göre sırala (0-225)
    df = df.sort_values('ClassId')
    
    # ClassId doğrulama
    expected_class_ids = list(range(226))
    actual_class_ids = df['ClassId'].tolist()
    
    if actual_class_ids != expected_class_ids:
        raise ValueError(
            f"ClassId sırası beklenen ile uyuşmuyor!\n"
            f"Beklenen: 0-225 arası kesintisiz\n"
            f"Bulunan: {len(actual_class_ids)} sınıf"
        )
    
    # Türkçe isimleri al
    class_names = df['TR'].tolist()
    
    # Final doğrulama
    if len(class_names) != 226:
        raise ValueError(
            f"Beklenen 226 sınıf, bulunan {len(class_names)} sınıf!"
        )
    
    return class_names


def get_class_name_mappings(csv_path=None):
    """
    Class ID → İsim ve İsim → Class ID mapping'lerini döndür.
    
    Returns:
        tuple: (id_to_tr, id_to_en, tr_to_id, en_to_id)
    
    Example:
        >>> id_to_tr, id_to_en, tr_to_id, en_to_id = get_class_name_mappings()
        >>> id_to_tr[0]
        'abla'
        >>> tr_to_id['acele']
        1
    """
    
    # Varsayılan CSV yolu
    if csv_path is None:
        current_dir = Path(__file__).parent.parent
        project_root = current_dir.parent
        csv_path = project_root / "Data" / "Class ID" / "SignList_ClassId_TR_EN.csv"
    
    # CSV yükle
    df = pd.read_csv(csv_path)
    df = df.sort_values('ClassId')
    
    # Mapping'leri oluştur
    id_to_tr = dict(zip(df['ClassId'], df['TR']))
    id_to_en = dict(zip(df['ClassId'], df['EN']))
    tr_to_id = dict(zip(df['TR'], df['ClassId']))
    en_to_id = dict(zip(df['EN'], df['ClassId']))
    
    return id_to_tr, id_to_en, tr_to_id, en_to_id


if __name__ == "__main__":
    """Test fonksiyonları"""
    
    print("="*70)
    print("TEST: load_all_class_names()")
    print("="*70)
    
    # Sınıf isimlerini yükle
    class_names = load_all_class_names()
    
    print(f"\n✅ Toplam sınıf: {len(class_names)}")
    print(f"\n📋 İlk 10 sınıf (0-9):")
    for i in range(10):
        print(f"   {i:3d}: {class_names[i]}")
    
    print(f"\n📋 Son 10 sınıf (216-225):")
    for i in range(216, 226):
        print(f"   {i:3d}: {class_names[i]}")
    
    print("\n" + "="*70)
    print("TEST: get_class_name_mappings()")
    print("="*70)
    
    id_to_tr, id_to_en, tr_to_id, en_to_id = get_class_name_mappings()
    
    print(f"\n✅ Mapping'ler oluşturuldu:")
    print(f"   id_to_tr: {len(id_to_tr)} eleman")
    print(f"   id_to_en: {len(id_to_en)} eleman")
    print(f"   tr_to_id: {len(tr_to_id)} eleman")
    print(f"   en_to_id: {len(en_to_id)} eleman")
    
    print(f"\n📋 Örnek mapping'ler:")
    print(f"   ClassId 0 → TR: {id_to_tr[0]}, EN: {id_to_en[0]}")
    print(f"   ClassId 1 → TR: {id_to_tr[1]}, EN: {id_to_en[1]}")
    print(f"   'acele' → ClassId: {tr_to_id['acele']}")
    print(f"   'hurry' → ClassId: {en_to_id['hurry']}")
    
    print("\n" + "="*70)
    print("✅ TÜM TESTLER BAŞARILI!")
    print("="*70)

