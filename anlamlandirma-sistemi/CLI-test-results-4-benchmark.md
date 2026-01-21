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

❯ python experiments/run_experiments.py --all --output results.json --report benchmark.md
RAG sistemi basariyla baslatildi.

============================================================
TID Translation Experiment Runner
=================================

Provider: gemini
RAG: Enabled
Word counts: [3, 4, 5]
======================

============================================================
Running 3-word experiments (20 samples)
Provider: gemini, RAG: True
===========================

[1/20] BEN OKUL GITMEK
/Users/siyaksares/Developer/GitHub/sign-transcriber/anlamlandirma-sistemi/rag/llm/llm_client.py:159: FutureWarning:

All support for the `google.generativeai` package has ended. It will no longer be receiving
updates or bug fixes. Please switch to the `google.genai` package as soon as possible.
See README for more details:

https://github.com/google-gemini/deprecated-generative-ai-python/blob/main/README.md

  import google.generativeai as genai
  [API Key rotated to key #1]
  -> Ben okula gidiyorum.... [9/10] (OK)

[2/20] O ISTANBUL CALISMAK
  -> O İstanbul'da çalışıyor.... [10/10] (OK)

[3/20] BUGUN HAVA GUZEL
  [API Key rotated to key #1]
  -> Bugün hava güzel.... [10/10] (OK)

[4/20] BU KITAP BENIM
  [API Key rotated to key #1]
  -> Bu kitap benim.... [10/10] (OK)

[5/20] BIZ FILM SEYRETMEK
  -> Biz filmi seyrediyoruz.... [10/10] (OK)

[6/20] BEN ANLAMAK DEGIL
  [API Key rotated to key #1]
  -> Ben anlamıyorum.... [10/10] (OK)

[7/20] BU DOGRU DEGIL
  [API Key rotated to key #2]
  [API Key rotated to key #1]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 51.025844553s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 20
  }
  , retry_delay {
  seconds: 51
  }
  ])

[8/20] EVDE KIMSE YOK
  -> Evde kimse yok.... [10/10] (OK)

[9/20] PARA HIC YOK
  -> Param hiç yok.... [10/10] (OK)

[10/20] O GELMEK ISTEMEMEK
  [API Key rotated to key #1]
  -> O gelmek istemiyor.... [10/10] (OK)

[11/20] SEN ADIN NE
  [API Key rotated to key #1]
  -> Adın ne?... [10/10] (OK)

[12/20] SEN NEREDE OTURMAK
  -> Nerede oturuyorsun?... [10/10] (OK)

[13/20] YARIN OKUL VAR
  [API Key rotated to key #1]
  -> Yarın okul var.... [10/10] (OK)

[14/20] BU KIMIN KALEM
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 5, model: gemini-2.5-flash
  Please retry in 54.410865336s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 5
  }
  , retry_delay {
  seconds: 54
  }
  ])

[15/20] SAAT KAC SIMDI
  [API Key rotated to key #1]
  -> Şimdi saat kaç?... [10/10] (OK)

[16/20] SEN KAPIYI KAPAT
  [API Key rotated to key #1]
  -> Sen kapıyı kapat.... [10/10] (OK)

[17/20] LUTFEN BANA BAK
  [API Key rotated to key #2]
  [API Key rotated to key #1]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 21.299591014s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 20
  }
  , retry_delay {
  seconds: 21
  }
  ])

[18/20] HEMEN BURAYA GEL
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 16.315746801s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 20
  }
  , retry_delay {
  seconds: 16
  }
  ])

[19/20] ORAYA GITMEK YOK
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 11.468592617s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 20
  }
  , retry_delay {
  seconds: 11
  }
  ])

[20/20] KITABI BANA VER
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 6.579856991s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 20
  }
  , retry_delay {
  seconds: 6
  }
  ])

============================================================
Results: 14/20 successful
Avg Confidence: 9.93/10
Avg Latency: 8590ms
===================

============================================================
Running 4-word experiments (20 samples)
Provider: gemini, RAG: True
===========================

[1/20] PIKNIK ICIN PLAN YAPMAK
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 1.744118896s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 20
  }
  , retry_delay {
  seconds: 1
  }
  ])

[2/20] BIZ HEMEN HASTANE GITMEK
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 56.656916135s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 20
  }
  , retry_delay {
  seconds: 56
  }
  ])

[3/20] AHMET KENDI KENDI CALISMAK
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 51.581326444s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 20
  }
  , retry_delay {
  seconds: 51
  }
  ])

[4/20] BENIM KALEM ALDI O
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 46.704506108s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 20
  }
  , retry_delay {
  seconds: 46
  }
  ])

[5/20] ZEYNEP ARABA SURMEK OLABILIR
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 41.351977145s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 20
  }
  , retry_delay {
  seconds: 41
  }
  ])

[6/20] KUTUPHANE SAGIR HAYAT VAR
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 36.198987175s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 20
  }
  , retry_delay {
  seconds: 36
  }
  ])

[7/20] BEN SENI DAVET ETMEK
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 31.350121805s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 20
  }
  , retry_delay {
  seconds: 31
  }
  ])

[8/20] PARA BU YETER DEGIL
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 26.55697382s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 20
  }
  , retry_delay {
  seconds: 26
  }
  ])

[9/20] KADIN ADAM OTURMAK GORMEK
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 21.52934373s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 20
  }
  , retry_delay {
  seconds: 21
  }
  ])

[10/20] BABA BULASIK YIKAMAK TABAK
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 16.562768646s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 20
  }
  , retry_delay {
  seconds: 16
  }
  ])

[11/20] KIZ KUCUK ARABA ITMEK
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 11.564594107s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 20
  }
  , retry_delay {
  seconds: 11
  }
  ])

[12/20] BU EV YUZ KOTU
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 6.691371851s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 20
  }
  , retry_delay {
  seconds: 6
  }
  ])

[13/20] DUN ISTANBUL GELMEK BEN
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 1.602531289s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 20
  }
  , retry_delay {
  seconds: 1
  }
  ])

[14/20] ELVAN KENDI KENDI SEVMEK
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 56.368940604s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 20
  }
  , retry_delay {
  seconds: 56
  }
  ])

[15/20] COCUK SUT ISTEMEK DEGIL
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 51.46344131s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 20
  }
  , retry_delay {
  seconds: 51
  }
  ])

[16/20] SENIN EVIN NEREDE VAR
  [API Key rotated to key #1]
^CTraceback (most recent call last):
  File "/Users/siyaksares/Developer/GitHub/sign-transcriber/anlamlandirma-sistemi/rag/llm/llm_client.py", line 228, in _generate_gemini
    response = self._model.generate_content(user_prompt)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/siyaksares/opt/miniconda3/envs/sign-transcriber/lib/python3.11/site-packages/google/generativeai/generative_models.py", line 331, in generate_content
    response = self._client.generate_content(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/siyaksares/opt/miniconda3/envs/sign-transcriber/lib/python3.11/site-packages/google/ai/generativelanguage_v1beta/services/generative_service/client.py", line 835, in generate_content
    response = rpc(
               ^^^^
  File "/Users/siyaksares/opt/miniconda3/envs/sign-transcriber/lib/python3.11/site-packages/google/api_core/gapic_v1/method.py", line 131, in __call__
    return wrapped_func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/siyaksares/opt/miniconda3/envs/sign-transcriber/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 294, in retry_wrapped_func
    return retry_target(
           ^^^^^^^^^^^^^
  File "/Users/siyaksares/opt/miniconda3/envs/sign-transcriber/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 156, in retry_target
    next_sleep = _retry_error_helper(
                 ^^^^^^^^^^^^^^^^^^^^
  File "/Users/siyaksares/opt/miniconda3/envs/sign-transcriber/lib/python3.11/site-packages/google/api_core/retry/retry_base.py", line 214, in _retry_error_helper
    raise final_exc from source_exc
  File "/Users/siyaksares/opt/miniconda3/envs/sign-transcriber/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 147, in retry_target
    result = target()
             ^^^^^^^^
  File "/Users/siyaksares/opt/miniconda3/envs/sign-transcriber/lib/python3.11/site-packages/google/api_core/timeout.py", line 130, in func_with_timeout
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/siyaksares/opt/miniconda3/envs/sign-transcriber/lib/python3.11/site-packages/google/api_core/grpc_helpers.py", line 77, in error_remapped_callable
    raise exceptions.from_grpc_error(exc) from exc
google.api_core.exceptions.ResourceExhausted: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 49.020976908s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
  }
  , violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
  key: "model"
  value: "gemini-2.5-flash"
  }
  quota_dimensions {
  key: "location"
  value: "global"
  }
  quota_value: 20
  }
  , retry_delay {
  seconds: 49
  }
  ]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/siyaksares/Developer/GitHub/sign-transcriber/anlamlandirma-sistemi/experiments/run_experiments.py", line 197, in `<module>`
    main()
  File "/Users/siyaksares/Developer/GitHub/sign-transcriber/anlamlandirma-sistemi/experiments/run_experiments.py", line 146, in main
    results = runner.run_all(
              ^^^^^^^^^^^^^^^
  File "/Users/siyaksares/Developer/GitHub/sign-transcriber/anlamlandirma-sistemi/experiments/experiment_runner.py", line 271, in run_all
    batch_result = self.run_batch(word_count, limit=limit, verbose=verbose)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/siyaksares/Developer/GitHub/sign-transcriber/anlamlandirma-sistemi/experiments/experiment_runner.py", line 213, in run_batch
    result = self.run_single_experiment(gloss, reference)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/siyaksares/Developer/GitHub/sign-transcriber/anlamlandirma-sistemi/experiments/experiment_runner.py", line 129, in run_single_experiment
    result = translate_with_rag(gloss)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/siyaksares/Developer/GitHub/sign-transcriber/anlamlandirma-sistemi/preprocessor.py", line 230, in translate_with_rag
    result = pipeline.translate(transcription)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/siyaksares/Developer/GitHub/sign-transcriber/anlamlandirma-sistemi/rag/pipeline/translation_pipeline.py", line 229, in translate
    translation_result = self.llm_client.translate(user_prompt)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/siyaksares/Developer/GitHub/sign-transcriber/anlamlandirma-sistemi/rag/llm/llm_client.py", line 276, in translate
    raw_response = self.generate(user_prompt)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/siyaksares/Developer/GitHub/sign-transcriber/anlamlandirma-sistemi/rag/llm/llm_client.py", line 207, in generate
    return self._generate_gemini(user_prompt)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/siyaksares/Developer/GitHub/sign-transcriber/anlamlandirma-sistemi/rag/llm/llm_client.py", line 240, in _generate_gemini
    time.sleep(RATE_LIMIT_RETRY_DELAY)
KeyboardInterrupt

╭─    ~/Dev/G/sign-transcriber/anlamlandirma-sistemi    main !2 ?12 ▓▒░
╰─
