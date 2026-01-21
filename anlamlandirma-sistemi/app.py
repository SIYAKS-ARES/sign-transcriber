import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import base64
import numpy as np
from io import BytesIO
from PIL import Image

# OpenCV import
try:
    import cv2
except ImportError:
    print("❌ OpenCV yüklenemedi!")
    cv2 = None

from local_model_handler import (
    load_model,
    get_transcription_from_local_model,
    predict_from_frames,
)
from preprocessor import preprocess_text_for_llm, create_final_prompt, is_rag_available, translate_with_rag
from llm_services import translate_with_llm

# Veritabanı import
from database import init_db, save_video_record, update_video_record, get_recent_records, get_statistics


load_dotenv()

app = Flask(__name__)

# Veritabanını başlat
init_db()

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


@app.route('/test-video', methods=['GET'])
def test_video():
    """Basit video oynatma test sayfası"""
    return render_template('test_video.html')


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
    record_id = None
    
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'Video dosyası bulunamadı'}), 400
        
        video_file = request.files['video']
        provider = request.form.get('provider', 'gemini')
        
        if not video_file.filename:
            return jsonify({'success': False, 'error': 'Video dosya adı boş'}), 400
        
        # Dosya boyutunu al
        video_file.seek(0, 2)  # Dosya sonuna git
        filesize = video_file.tell()
        video_file.seek(0)  # Başa dön
        
        # Veritabanına kayıt ekle
        record_id = save_video_record(video_file.filename, filesize=filesize)
        print(f"✅ Veritabanı kaydı oluşturuldu: ID={record_id}")
        
        # Güvenli dosya adı oluştur
        import uuid
        safe_filename = f"{uuid.uuid4().hex}_{video_file.filename}"
        temp_path = f"/tmp/{safe_filename}"
        
        print(f"📹 Video kaydediliyor: {temp_path}")
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
        
        # LLM kullanımı opsiyonel
        use_llm = request.form.get('use_llm', 'true').lower() == 'true'
        
        # Transkripsiyon sonucuna göre işle
        if not transcription:
            result_data = {
                'translation': 'Tespit edilemedi',
                'confidence': 0,
                'explanation': 'Model bir işaret tespit edemedi veya güven eşiğinin altında kaldı.',
                'error': None,
            }
            processed_transcription = ''
        elif not use_llm:
            # LLM kullanma, sadece model tahmini
            # Kelimeyi capitalize et
            display_word = transcription.capitalize() if transcription else 'Bilinmiyor'
            result_data = {
                'translation': display_word,
                'confidence': 5,
                'explanation': f'İşaret dili kelimesi: {display_word}',
                'error': None,
            }
            processed_transcription = transcription
        else:
            # LLM ile çevir
            try:
                processed_transcription = preprocess_text_for_llm(transcription)
                final_prompt = create_final_prompt(processed_transcription)
                result_data = translate_with_llm(provider, final_prompt)
            except Exception as llm_error:
                error_msg = str(llm_error)
                display_word = transcription.capitalize() if transcription else 'Bilinmiyor'
                
                if '429' in error_msg or 'quota' in error_msg.lower() or 'exhausted' in error_msg.lower():
                    result_data = {
                        'translation': display_word,
                        'confidence': 5,
                        'explanation': f'⚠️ LLM API kotası aşıldı. Model tahmini: {display_word}',
                        'error': 'API quota exceeded'
                    }
                else:
                    result_data = {
                        'translation': display_word,
                        'confidence': 3,
                        'explanation': f'⚠️ LLM çevirisi başarısız. Model tahmini: {display_word}',
                        'error': error_msg
                    }
                processed_transcription = transcription
        
        # Veritabanı kaydını güncelle
        update_video_record(
            record_id,
            transcription=transcription,
            translation=result_data.get('translation', ''),
            confidence=result_data.get('confidence', 0),
            provider=provider,
            status='completed'
        )
        print(f"✅ Veritabanı kaydı güncellendi: ID={record_id}")
        
        return jsonify({
            'success': True,
            'record_id': record_id,
            'original_transcription': transcription,
            'processed_transcription': processed_transcription,
            'result': result_data
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Video işleme hatası: {e}")
        print(f"Detaylı hata:\n{error_details}")
        
        # Hata durumunda veritabanını güncelle
        if record_id:
            update_video_record(record_id, status='error', error_message=str(e))
        
        return jsonify({'success': False, 'error': f'Video işleme hatası: {str(e)}'}), 500
    finally:
        # Geçici dosyayı temizle
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"🗑️ Geçici dosya silindi: {temp_path}")
            except Exception as cleanup_error:
                print(f"⚠️ Geçici dosya silinirken hata: {cleanup_error}")


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
        # Eşiği biraz düşür (0.30 → 0.10) ki modelin
        # makul ama düşük güvenli tahminleri de kaçmasın.
        result = predict_from_frames(LOCAL_MODEL, frames, confidence_threshold=0.10)
        transcription = result.get('pred_name', '') if result.get('threshold_met') else ''
        return transcription
    except Exception as exc:
        print(f"Model işlem hatası: {exc}")
        return ""


@app.route('/api/history', methods=['GET'])
def get_history():
    """Son işlenen videoları döndür"""
    try:
        limit = request.args.get('limit', 10, type=int)
        records = get_recent_records(limit)
        stats = get_statistics()
        
        return jsonify({
            'success': True,
            'records': records,
            'statistics': stats
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


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
        import traceback
        error_details = traceback.format_exc()
        print(f"Model test hatası: {e}")
        print(f"Detaylı hata:\n{error_details}")
        return jsonify({'success': False, 'error': f'Model test hatası: {str(e)}'}), 500
    finally:
        # Geçici dosyayı temizle
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Test geçici dosya silindi: {temp_path}")
            except Exception as cleanup_error:
                print(f"Test geçici dosya silinirken hata: {cleanup_error}")


# =============================================================================
# RAG System Status Check
# =============================================================================

def check_rag_status():
    """Check if RAG system is available and ready."""
    try:
        rag_ready = is_rag_available()
        if rag_ready:
            print("RAG sistemi hazir.")
        else:
            print("RAG sistemi kulanilamiyor, fallback modunda.")
        return rag_ready
    except Exception as e:
        print(f"RAG sistem kontrolu hatasi: {e}")
        return False

# Check RAG status at startup
RAG_READY = check_rag_status()


# =============================================================================
# Experiment Routes
# =============================================================================

@app.route('/experiments', methods=['GET'])
def experiments_page():
    """Experiments dashboard page."""
    return render_template('experiments.html', rag_ready=RAG_READY)


@app.route('/api/rag_status', methods=['GET'])
def rag_status():
    """Return RAG system status."""
    return jsonify({
        'success': True,
        'rag_ready': RAG_READY,
        'rag_available': is_rag_available(),
    })


@app.route('/api/run_experiment', methods=['POST'])
def run_experiment():
    """Run a translation experiment."""
    try:
        data = request.get_json()
        word_count = data.get('word_count', 3)
        limit = data.get('limit')
        provider = data.get('provider', 'gemini')
        use_rag = data.get('use_rag', True)
        
        from experiments.experiment_runner import ExperimentRunner
        
        runner = ExperimentRunner(provider=provider, use_rag=use_rag)
        batch_result = runner.run_batch(word_count, limit=limit, verbose=False)
        
        return jsonify({
            'success': True,
            'result': batch_result.to_dict(),
        })
        
    except Exception as e:
        import traceback
        print(f"Experiment error: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/run_all_experiments', methods=['POST'])
def run_all_experiments():
    """Run experiments for all word counts."""
    try:
        data = request.get_json() or {}
        limit = data.get('limit')
        provider = data.get('provider', 'gemini')
        use_rag = data.get('use_rag', True)
        word_counts = data.get('word_counts', [3, 4, 5])
        
        from experiments.experiment_runner import ExperimentRunner
        
        runner = ExperimentRunner(provider=provider, use_rag=use_rag)
        results = runner.run_all(word_counts=word_counts, limit=limit, verbose=False)
        
        return jsonify({
            'success': True,
            'results': {str(k): v.to_dict() for k, v in results.items()},
        })
        
    except Exception as e:
        import traceback
        print(f"Experiment error: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/translate_rag', methods=['POST'])
def translate_rag_endpoint():
    """Translate using full RAG pipeline."""
    try:
        data = request.get_json()
        gloss = data.get('gloss', '')
        
        if not gloss:
            return jsonify({'success': False, 'error': 'Gloss required'}), 400
        
        result = translate_with_rag(gloss)
        
        return jsonify({
            'success': True,
            'result': result,
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/approve_translation', methods=['POST'])
def approve_translation():
    """Kullanici onayladigi ceviriyi TID_Hafiza koleksiyonuna kaydet."""
    try:
        data = request.get_json()
        gloss = data.get('gloss', '')
        translation = data.get('translation', '')
        reference = data.get('reference', '')
        
        if not gloss or not translation:
            return jsonify({'success': False, 'error': 'Gloss ve translation zorunlu'}), 400
        
        # ChromaDB'ye kaydet
        from rag.tid_collections.hafiza_collection import HafizaCollection
        hafiza = HafizaCollection()
        doc_id = hafiza.add_translation(
            transkripsiyon=gloss,
            ceviri=translation,
            provider="user_approved",
            confidence=1.0,
        )
        
        print(f"Ceviri onaylandi ve kaydedildi: {gloss} -> {translation} (ID: {doc_id})")
        
        return jsonify({
            'success': True,
            'message': 'Ceviri basariyla kaydedildi',
            'doc_id': doc_id,
            'hafiza_count': hafiza.get_count(),
        })
        
    except Exception as e:
        import traceback
        print(f"Ceviri onaylama hatasi: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# Manual LLM Response Endpoints (for API rate limit workaround)
# =============================================================================

@app.route('/api/get_prompt', methods=['POST'])
def get_prompt_endpoint():
    """Generate and return the prompt for a given gloss (for manual LLM input)."""
    try:
        data = request.get_json()
        gloss = data.get('gloss', '')
        use_rag = data.get('use_rag', True)
        
        if not gloss:
            return jsonify({'success': False, 'error': 'Gloss required'}), 400
        
        # Preprocess the gloss
        processed = preprocess_text_for_llm(gloss)
        
        # Create the final prompt
        prompt = create_final_prompt(processed)
        
        return jsonify({
            'success': True,
            'prompt': prompt,
            'processed_gloss': processed,
            'rag_used': is_rag_available() and use_rag,
        })
        
    except Exception as e:
        import traceback
        print(f"Get prompt error: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/process_manual_response', methods=['POST'])
def process_manual_response():
    """Process a manually entered LLM response (for API rate limit workaround)."""
    try:
        data = request.get_json()
        gloss = data.get('gloss', '')
        reference = data.get('reference', '')
        llm_response = data.get('llm_response', '')
        provider = data.get('provider', 'unknown')
        word_count = data.get('word_count', 0)
        index = data.get('index', 0)
        
        if not gloss or not llm_response:
            return jsonify({'success': False, 'error': 'Gloss and llm_response required'}), 400
        
        # Parse the LLM response using existing parsing functions
        from llm_services import parse_multi_alternative_output, parse_structured_output
        
        # Try multi-alternative parsing first, then structured output
        parsed = parse_multi_alternative_output(llm_response)
        
        # If parsing didn't find structured output, try to extract basic info
        if not parsed.get('translation') and not parsed.get('alternatives'):
            # Fallback: treat entire response as translation
            parsed = {
                'translation': llm_response.strip()[:500],  # Limit length
                'confidence': 5,  # Default confidence for unstructured responses
                'explanation': 'Manuel giris - yapilandirilmamis cevap',
                'alternatives': [],
            }
        
        # Build result object compatible with experiment results
        result = {
            'gloss': gloss,
            'reference': reference,
            'translation': parsed.get('translation', ''),
            'confidence': parsed.get('confidence', 5),
            'explanation': parsed.get('explanation', ''),
            'alternatives': parsed.get('alternatives', []),
            'error': '',  # Clear error since we have a manual response
            'manual_provider': provider,  # Flag this as manually entered
            'latency_ms': 0,  # No API latency for manual input
        }
        
        return jsonify({
            'success': True,
            'result': result,
            'provider': provider,
            'word_count': word_count,
            'index': index,
        })
        
    except Exception as e:
        import traceback
        print(f"Process manual response error: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', '5005'))
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    app.run(host=host, port=port, debug=debug)


