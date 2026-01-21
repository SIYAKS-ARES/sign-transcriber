❯ python scripts/init_vectorstore.py --check
Vectorstore path: /Users/siyaksares/Developer/GitHub/sign-transcriber/anlamlandirma-sistemi/vectorstore
Path exists: True

Sozluk koleksiyonu: 2867 kayit
Hafiza koleksiyonu: 2845 kayit

Vectorstore hazir!
❯ python experiments/run_experiments.py --all --limit 5
RAG sistemi basariyla baslatildi.

============================================================
TID Translation Experiment Runner
=================================

Provider: gemini
RAG: Enabled
Word counts: [3, 4, 5]
Limit: 5 samples per word count
===============================

============================================================
Running 3-word experiments (5 samples)
Provider: gemini, RAG: True
===========================

[1/5] BEN OKUL GITMEK
/Users/siyaksares/Developer/GitHub/sign-transcriber/anlamlandirma-sistemi/rag/llm/llm_client.py:98: FutureWarning:

All support for the `google.generativeai` package has ended. It will no longer be receiving
updates or bug fixes. Please switch to the `google.genai` package as soon as possible.
See README for more details:

https://github.com/google-gemini/deprecated-generative-ai-python/blob/main/README.md

  import google.generativeai as genai
  -> ... [0/10] (ERROR: No Gemini API key found. Set GEMINI_API_KEY or GEMINI_API_KEY_2)

[2/5] SEN YEMEK YEMEK
  -> ... [0/10] (ERROR: No Gemini API key found. Set GEMINI_API_KEY or GEMINI_API_KEY_2)

[3/5] BEN KITAP OKUMAK
  -> ... [0/10] (ERROR: No Gemini API key found. Set GEMINI_API_KEY or GEMINI_API_KEY_2)

[4/5] ANNE YEMEK PISIRMEK
  -> ... [0/10] (ERROR: No Gemini API key found. Set GEMINI_API_KEY or GEMINI_API_KEY_2)

[5/5] COCUK PARK OYNAMAK
  -> ... [0/10] (ERROR: No Gemini API key found. Set GEMINI_API_KEY or GEMINI_API_KEY_2)

============================================================
Results: 0/5 successful
Avg Confidence: 0/10
Avg Latency: 289ms
==================

============================================================
Running 4-word experiments (5 samples)
Provider: gemini, RAG: True
===========================

[1/5] BEN CAY ICMEK ISTEMEK
  -> ... [0/10] (ERROR: No Gemini API key found. Set GEMINI_API_KEY or GEMINI_API_KEY_2)

[2/5] BEN UYANMAK ERKEN OLMAK
  -> ... [0/10] (ERROR: No Gemini API key found. Set GEMINI_API_KEY or GEMINI_API_KEY_2)

[3/5] OTOMOBIL HIZLI GITMEK TEHLIKELI
  -> ... [0/10] (ERROR: No Gemini API key found. Set GEMINI_API_KEY or GEMINI_API_KEY_2)

[4/5] BEN SU ICMEK ISTEMEK
  -> ... [0/10] (ERROR: No Gemini API key found. Set GEMINI_API_KEY or GEMINI_API_KEY_2)

[5/5] GUN GUZEL OLMAK BUGUN
  -> ... [0/10] (ERROR: No Gemini API key found. Set GEMINI_API_KEY or GEMINI_API_KEY_2)

============================================================
Results: 0/5 successful
Avg Confidence: 0/10
Avg Latency: 164ms
==================

============================================================
Running 5-word experiments (5 samples)
Provider: gemini, RAG: True
===========================

[1/5] DUN BEN ARKADAS BULUSMAK KONUSMAK
  -> ... [0/10] (ERROR: No Gemini API key found. Set GEMINI_API_KEY or GEMINI_API_KEY_2)

[2/5] YARIN SEN OKUL SINAV OLMAK
  -> ... [0/10] (ERROR: No Gemini API key found. Set GEMINI_API_KEY or GEMINI_API_KEY_2)

[3/5] ANNE MUTFAK YEMEK PISIRMEK BITMEK
  -> ... [0/10] (ERROR: No Gemini API key found. Set GEMINI_API_KEY or GEMINI_API_KEY_2)

[4/5] BEN ISTANBUL UCAK GITMEK ISTEMEK
  -> ... [0/10] (ERROR: No Gemini API key found. Set GEMINI_API_KEY or GEMINI_API_KEY_2)

[5/5] COCUK PARK TOP OYNAMAK SEVMEK
  -> ... [0/10] (ERROR: No Gemini API key found. Set GEMINI_API_KEY or GEMINI_API_KEY_2)

============================================================
Results: 0/5 successful
Avg Confidence: 0/10
Avg Latency: 178ms
==================

============================================================
SUMMARY
=======

3-word: 0/5 (Avg conf: 0/10, Latency: 289ms)
4-word: 0/5 (Avg conf: 0/10, Latency: 164ms)
5-word: 0/5 (Avg conf: 0/10, Latency: 178ms)

Total: 0/15 successful
======================

❯ python app.py
✅ Veritabanı hazır: anlamlandirma.db

======================================================================
🔧 TRANSFORMER MODEL YÜKLENİYOR
=================================

✅ Config yüklendi

- Input dim: 258
- Sequence length: 200
- Num classes: 226
  ✅ Device: CPU (MPS mask issue nedeniyle)
  ✅ Model checkpoint yüklendi:
- Epoch: 98
- Val Accuracy: 0.8787
- Val F1: 0.8756
  /Users/siyaksares/opt/miniconda3/envs/sign-transcriber/lib/python3.11/site-packages/sklearn/base.py:463: InconsistentVersionWarning: Trying to unpickle estimator StandardScaler from version 1.7.2 when using version 1.8.0. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
  https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
  warnings.warn(
  ✅ Scaler yüklendi
  ✅ Sınıf isimleri yüklendi: 226 sınıf
  ======================================================================

✅ Model yüklendi:

- Device: cpu
- Sınıf sayısı: 226
- Model type: Transformer (PyTorch)
  RAG sistemi basariyla baslatildi.
  RAG sistemi hazir.

* Serving Flask app 'app'
* Debug mode: on
  WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
* Running on all addresses (0.0.0.0)
* Running on http://127.0.0.1:5005
* Running on http://10.196.3.211:5005
  Press CTRL+C to quit
* Restarting with stat
  ✅ Veritabanı hazır: anlamlandirma.db

======================================================================
🔧 TRANSFORMER MODEL YÜKLENİYOR
=================================

✅ Config yüklendi

- Input dim: 258
- Sequence length: 200
- Num classes: 226
  ✅ Device: CPU (MPS mask issue nedeniyle)
  ✅ Model checkpoint yüklendi:
- Epoch: 98
- Val Accuracy: 0.8787
- Val F1: 0.8756
  /Users/siyaksares/opt/miniconda3/envs/sign-transcriber/lib/python3.11/site-packages/sklearn/base.py:463: InconsistentVersionWarning: Trying to unpickle estimator StandardScaler from version 1.7.2 when using version 1.8.0. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
  https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
  warnings.warn(
  ✅ Scaler yüklendi
  ✅ Sınıf isimleri yüklendi: 226 sınıf
  ======================================================================

✅ Model yüklendi:

- Device: cpu
- Sınıf sayısı: 226
- Model type: Transformer (PyTorch)
  RAG sistemi basariyla baslatildi.
