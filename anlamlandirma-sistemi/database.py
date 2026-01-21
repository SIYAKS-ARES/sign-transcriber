"""
Basit SQLite veritabanı - Video kayıtlarını saklar
"""
import sqlite3
import os
from datetime import datetime
import json

DB_PATH = 'anlamlandirma.db'


def init_db():
    """Veritabanını başlat ve tabloları oluştur"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Video işleme kayıtları tablosu
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS video_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filesize INTEGER,
            duration REAL,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            process_time TIMESTAMP,
            transcription TEXT,
            translation TEXT,
            confidence REAL,
            provider TEXT,
            status TEXT DEFAULT 'pending',
            error_message TEXT
        )
    ''')
    
    # İndeks oluştur
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_upload_time 
        ON video_records(upload_time DESC)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_status 
        ON video_records(status)
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ Veritabanı hazır: {DB_PATH}")


def save_video_record(filename, filesize=None, duration=None):
    """Yeni video kaydı ekle"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO video_records (filename, filesize, duration, status)
        VALUES (?, ?, ?, 'processing')
    ''', (filename, filesize, duration))
    
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return record_id


def update_video_record(record_id, transcription=None, translation=None, 
                       confidence=None, provider=None, status='completed', 
                       error_message=None):
    """Video işleme sonucunu güncelle"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE video_records 
        SET process_time = CURRENT_TIMESTAMP,
            transcription = ?,
            translation = ?,
            confidence = ?,
            provider = ?,
            status = ?,
            error_message = ?
        WHERE id = ?
    ''', (transcription, translation, confidence, provider, status, error_message, record_id))
    
    conn.commit()
    conn.close()


def get_recent_records(limit=10):
    """Son işlenen videoları getir"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Dict-like access
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM video_records 
        ORDER BY upload_time DESC 
        LIMIT ?
    ''', (limit,))
    
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return records


def get_statistics():
    """İstatistikleri getir"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Toplam video sayısı
    cursor.execute('SELECT COUNT(*) FROM video_records')
    total = cursor.fetchone()[0]
    
    # Başarılı işlemler
    cursor.execute('SELECT COUNT(*) FROM video_records WHERE status = "completed"')
    completed = cursor.fetchone()[0]
    
    # Hatalı işlemler
    cursor.execute('SELECT COUNT(*) FROM video_records WHERE status = "error"')
    errors = cursor.fetchone()[0]
    
    # Ortalama güven
    cursor.execute('SELECT AVG(confidence) FROM video_records WHERE confidence IS NOT NULL')
    avg_confidence = cursor.fetchone()[0] or 0
    
    conn.close()
    
    return {
        'total': total,
        'completed': completed,
        'errors': errors,
        'success_rate': (completed / total * 100) if total > 0 else 0,
        'avg_confidence': avg_confidence
    }


def clear_old_records(days=30):
    """Eski kayıtları temizle"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        DELETE FROM video_records 
        WHERE upload_time < datetime('now', '-' || ? || ' days')
    ''', (days,))
    
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    
    return deleted


if __name__ == '__main__':
    # Test: Veritabanını başlat
    init_db()
    print("Veritabanı başlatıldı!")
    
    # Test: İstatistikleri göster
    stats = get_statistics()
    print(f"\n📊 İstatistikler:")
    print(f"  Toplam: {stats['total']}")
    print(f"  Tamamlanan: {stats['completed']}")
    print(f"  Hatalı: {stats['errors']}")
    print(f"  Başarı oranı: {stats['success_rate']:.1f}%")
    print(f"  Ortalama güven: {stats['avg_confidence']:.2f}")
    
    # Son kayıtları göster
    records = get_recent_records(5)
    if records:
        print(f"\n📹 Son {len(records)} kayıt:")
        for r in records:
            print(f"  - {r['filename']} ({r['status']}) - {r['upload_time']}")

