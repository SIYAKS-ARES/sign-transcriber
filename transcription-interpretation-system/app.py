import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from local_model_handler import (
    load_model,
    get_transcription_from_local_model,
    predict_from_frames,
)
from preprocessor import preprocess_text_for_llm, create_final_prompt
from llm_services import translate_with_llm
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image


load_dotenv()

app = Flask(__name__)

# Uygulama başında yerel modeli yükle (şimdilik placeholder)
LOCAL_MODEL = load_model()


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/demo', methods=['GET'])
def demo():
    selected_provider = 'openai'  # Varsayılan olarak OpenAI
    return render_template(
        'demo.html',
        selected_provider=selected_provider,
        result=None,
        original_transcription=None,
    )


@app.route('/translate', methods=['POST'])
def translate():
    selected_provider = request.form.get('provider') or 'openai'
    input_source = request.form.get('input_source') or 'sample'

    # Yerel modelden (placeholder) transkripsiyon al
    transcription = get_transcription_from_local_model(input_source)

    # RAG-REGEX ile ön işle
    processed_transcription = preprocess_text_for_llm(transcription)

    # LLM için nihai promptu oluştur
    final_prompt = create_final_prompt(processed_transcription)

    # Seçilen sağlayıcı ile çeviriyi yap (artık dict döner)
    try:
        result_data = translate_with_llm(selected_provider, final_prompt)
    except Exception as exc:
        result_data = {
            "translation": "",
            "confidence": 0,
            "explanation": f"LLM çağrısı sırasında hata oluştu: {exc}",
            "error": str(exc),
        }

    return render_template(
        'demo.html',
        result=result_data,
        original_transcription=transcription,
        selected_provider=selected_provider,
    )


# Yeni API endpoint'leri model entegrasyonu için
@app.route('/api/process_frames', methods=['POST'])
def process_frames():
    """Kamera veya video frame'lerini işler ve transkripsiyon üretir."""
    try:
        data = request.get_json()
        frames_data = data.get('frames', [])
        provider = data.get('provider', 'openai')
        
        if not frames_data:
            return jsonify({'success': False, 'error': 'Frame verisi bulunamadı'}), 400
        
        # Frame'leri decode et
        frames = []
        for frame_data in frames_data:
            # Base64'ten image'e dönüştür
            image_data = base64.b64decode(frame_data.split(',')[1])
            image = Image.open(BytesIO(image_data))
            # PIL'i OpenCV formatına çevir
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            frames.append(frame)
        
        # Frame sequence'i modele gönder (placeholder)
        transcription = process_frame_sequence(frames)
        
        # Eğer transkripsiyon boş ise LLM çağrısı yapma
        if not transcription:
            result_data = {
                'translation': '',
                'confidence': 0,
                'explanation': 'Model bir işaret tespit edemedi veya güven eşiğinin altında kaldı.',
                'error': None,
            }
            processed_transcription = ''
        else:
            # RAG-REGEX ile ön işle
            processed_transcription = preprocess_text_for_llm(transcription)
            # LLM prompt oluştur
            final_prompt = create_final_prompt(processed_transcription)
            # LLM ile çevir
            result_data = translate_with_llm(provider, final_prompt)
        
        return jsonify({
            'success': True,
            'original_transcription': transcription,
            'processed_transcription': processed_transcription,
            'result': result_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/process_video', methods=['POST'])
def process_video():
    """Yüklenen video dosyasını frame'lere ayırır ve işler."""
    temp_path = None
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'Video dosyası bulunamadı'}), 400
        
        video_file = request.files['video']
        provider = request.form.get('provider', 'openai')
        
        if not video_file.filename:
            return jsonify({'success': False, 'error': 'Video dosya adı boş'}), 400
        
        # Güvenli dosya adı oluştur
        import uuid
        safe_filename = f"{uuid.uuid4().hex}_{video_file.filename}"
        temp_path = f"/tmp/{safe_filename}"
        
        print(f"Video kaydediliyor: {temp_path}")
        video_file.save(temp_path)
        
        # Dosya varlığını kontrol et
        if not os.path.exists(temp_path):
            raise ValueError(f"Video dosyası kaydedilemedi: {temp_path}")
        
        # Video'dan frame'leri çıkar
        frames = extract_frames_from_video(temp_path)
        
        if not frames:
            raise ValueError("Video'dan frame çıkarılamadı")
        
        # Frame sequence'i işle
        transcription = process_frame_sequence(frames)
        
        # Eğer transkripsiyon boş ise LLM çağrısı yapma
        if not transcription:
            result_data = {
                'translation': '',
                'confidence': 0,
                'explanation': 'Model bir işaret tespit edemedi veya güven eşiğinin altında kaldı.',
                'error': None,
            }
            processed_transcription = ''
        else:
            # RAG-REGEX ile ön işle
            processed_transcription = preprocess_text_for_llm(transcription)
            # LLM prompt oluştur
            final_prompt = create_final_prompt(processed_transcription)
            # LLM ile çevir
            result_data = translate_with_llm(provider, final_prompt)
        
        return jsonify({
            'success': True,
            'original_transcription': transcription,
            'processed_transcription': processed_transcription,
            'result': result_data
        })
        
    except Exception as e:
        print(f"Video işleme hatası: {e}")
        return jsonify({'success': False, 'error': f'Video işleme hatası: {str(e)}'}), 500
    finally:
        # Geçici dosyayı temizle
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Geçici dosya silindi: {temp_path}")
            except Exception as cleanup_error:
                print(f"Geçici dosya silinirken hata: {cleanup_error}")


def extract_frames_from_video(video_path, target_fps=5):
    """Video dosyasından belirtilen FPS'te frame'leri çıkarır."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Video dosyası açılamadı: {video_path}")
        raise ValueError("Video dosyası açılamadı")
    
    # Video bilgilerini al
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video bilgileri - FPS: {fps}, Toplam frame: {total_frames}, Süre: {duration:.2f}s")
    
    frame_interval = int(fps / target_fps) if fps > target_fps else 1
    
    frame_count = 0
    extracted_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Belirtilen aralıkta frame'leri topla
        if frame_count % frame_interval == 0:
            frames.append(frame)
            extracted_count += 1
            
        # Maksimum 120 frame al (24 saniye @ 5fps)
        if extracted_count >= 120:
            break
        
        frame_count += 1
    
    cap.release()
    print(f"Video'dan {len(frames)} frame çıkarıldı")
    return frames


def process_frame_sequence(frames):
    """Frame sequence'ini yerel modele gönderip ham transkripsiyon üretir.
    Şimdilik tek pencere tahmini: top-1 etiket veya eşik altı ise boş string.
    """
    try:
        result = predict_from_frames(LOCAL_MODEL, frames, confidence_threshold=0.3)
        transcription = result.get('pred_name', '') if result.get('threshold_met') else ''
        return transcription
    except Exception as exc:
        print(f"Model işlem hatası: {exc}")
        return ""


@app.route('/api/test_model', methods=['POST'])
def test_model():
    """Model testini LLM olmadan yapar - sadece raw model çıktısı döner."""
    temp_path = None
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'Video dosyası bulunamadı'}), 400
        
        video_file = request.files['video']
        
        if not video_file.filename:
            return jsonify({'success': False, 'error': 'Video dosya adı boş'}), 400
        
        # Güvenli dosya adı oluştur
        import uuid
        safe_filename = f"{uuid.uuid4().hex}_{video_file.filename}"
        temp_path = f"/tmp/{safe_filename}"
        
        print(f"Test video kaydediliyor: {temp_path}")
        video_file.save(temp_path)
        
        # Dosya varlığını kontrol et
        if not os.path.exists(temp_path):
            raise ValueError(f"Video dosyası kaydedilemedi: {temp_path}")
        
        # Video'dan frame'leri çıkar
        frames = extract_frames_from_video(temp_path)
        
        if not frames:
            raise ValueError("Video'dan frame çıkarılamadı")
        
        # Frame sequence'i işle - direkt model sonucu döndür
        result = predict_from_frames(LOCAL_MODEL, frames, confidence_threshold=0.1)
        
        return jsonify({
            'success': True,
            'frame_count': len(frames),
            'model_result': result,
            'raw_transcription': result.get('pred_name', '') if result.get('threshold_met') else 'Eşik altı'
        })
        
    except Exception as e:
        print(f"Model test hatası: {e}")
        return jsonify({'success': False, 'error': f'Model test hatası: {str(e)}'}), 500
    finally:
        # Geçici dosyayı temizle
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Test geçici dosya silindi: {temp_path}")
            except Exception as cleanup_error:
                print(f"Test geçici dosya silinirken hata: {cleanup_error}")


if __name__ == '__main__':
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', '5005'))
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    app.run(host=host, port=port, debug=debug)


