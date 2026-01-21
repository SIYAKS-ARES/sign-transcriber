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
/Users/siyaksares/Developer/GitHub/sign-transcriber/anlamlandirma-sistemi/rag/llm/llm_client.py:159: FutureWarning:

All support for the `google.generativeai` package has ended. It will no longer be receiving
updates or bug fixes. Please switch to the `google.genai` package as soon as possible.
See README for more details:

https://github.com/google-gemini/deprecated-generative-ai-python/blob/main/README.md

  import google.generativeai as genai
  -> Ben okula gidiyorum.... [9/10] (OK)

[2/5] SEN YEMEK YEMEK
  -> Sen çok yemek yedin.... [9/10] (OK)

[3/5] BEN KITAP OKUMAK
  -> Kitap okuyorum.... [10/10] (OK)

[4/5] ANNE YEMEK PISIRMEK
  -> Anne yemek pişiriyor.... [10/10] (OK)

[5/5] COCUK PARK OYNAMAK
  -> Çocuk parkta oynuyor.... [9/10] (OK)

============================================================
Results: 5/5 successful
Avg Confidence: 9.4/10
Avg Latency: 9882ms
===================

============================================================
Running 4-word experiments (5 samples)
Provider: gemini, RAG: True
===========================

[1/5] BEN CAY ICMEK ISTEMEK
  -> Ben çay içmek istiyorum.... [10/10] (OK)

[2/5] BEN UYANMAK ERKEN OLMAK
  -> Ben erken uyandım.... [9/10] (OK)

[3/5] OTOMOBIL HIZLI GITMEK TEHLIKELI
  -> Arabayı hızlı kullanmak tehlikelidir.... [9/10] (OK)

[4/5] BEN SU ICMEK ISTEMEK
  -> Ben su içmek istiyorum.... [10/10] (OK)

[5/5] GUN GUZEL OLMAK BUGUN
  -> Bugün gün güzel.... [10/10] (OK)

============================================================
Results: 5/5 successful
Avg Confidence: 9.6/10
Avg Latency: 10883ms
====================

============================================================
Running 5-word experiments (5 samples)
Provider: gemini, RAG: True
===========================

[1/5] DUN BEN ARKADAS BULUSMAK KONUSMAK
  [API Key rotated to key #1]
  -> Dün arkadaşımla buluşup konuştum.... [10/10] (OK)

[2/5] YARIN SEN OKUL SINAV OLMAK
  [API Key rotated to key #1]
  -> Yarın senin okulda sınavın olacak mı?... [9/10] (OK)

[3/5] ANNE MUTFAK YEMEK PISIRMEK BITMEK
  [API Key rotated to key #1]
  -> Annem mutfakta yemeği pişirdi.... [10/10] (OK)

[4/5] BEN ISTANBUL UCAK GITMEK ISTEMEK
  [API Key rotated to key #1]
  -> Ben İstanbul'a uçakla gitmek istiyorum.... [10/10] (OK)

[5/5] COCUK PARK TOP OYNAMAK SEVMEK
  -> Çocuk parkta top oynamayı sever.... [10/10] (OK)

============================================================
Results: 5/5 successful
Avg Confidence: 9.8/10
Avg Latency: 16206ms
====================

============================================================
SUMMARY
=======

3-word: 5/5 (Avg conf: 9.4/10, Latency: 9882ms)
4-word: 5/5 (Avg conf: 9.6/10, Latency: 10883ms)
5-word: 5/5 (Avg conf: 9.8/10, Latency: 16206ms)

Total: 15/15 successful
=======================

╭─    ~/Dev/G/sign-transcriber/anlamlandirma-sistemi    main !2 ?12 ▓▒░
╰─
