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
  -> Ben okula gidiyorum.... [9/10] (OK)

[2/5] SEN YEMEK YEMEK
  -> Çok yemek yedin.... [9/10] (OK)

[3/5] BEN KITAP OKUMAK
  -> Ben kitap okuyorum.... [9/10] (OK)

[4/5] ANNE YEMEK PISIRMEK
  -> Anne yemeği pişiriyor.... [9/10] (OK)

[5/5] COCUK PARK OYNAMAK
  -> Çocuk parkta oynuyor.... [10/10] (OK)

============================================================
Results: 5/5 successful
Avg Confidence: 9.2/10
Avg Latency: 8144ms
===================

============================================================
Running 4-word experiments (5 samples)
Provider: gemini, RAG: True
===========================

[1/5] BEN CAY ICMEK ISTEMEK
  -> Çay içmek istiyorum.... [10/10] (OK)

[2/5] BEN UYANMAK ERKEN OLMAK
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 5, model: gemini-2.5-flash
  Please retry in 31.5320098s. [links {
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
  seconds: 31
  }
  ])

[3/5] OTOMOBIL HIZLI GITMEK TEHLIKELI
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 5, model: gemini-2.5-flash
  Please retry in 31.014402583s. [links {
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
  seconds: 31
  }
  ])

[4/5] BEN SU ICMEK ISTEMEK
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 5, model: gemini-2.5-flash
  Please retry in 30.465762225s. [links {
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
  seconds: 30
  }
  ])

[5/5] GUN GUZEL OLMAK BUGUN
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 5, model: gemini-2.5-flash
  Please retry in 29.38624019s. [links {
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
  seconds: 29
  }
  ])

============================================================
Results: 1/5 successful
Avg Confidence: 10.0/10
Avg Latency: 2027ms
===================

============================================================
Running 5-word experiments (5 samples)
Provider: gemini, RAG: True
===========================

[1/5] DUN BEN ARKADAS BULUSMAK KONUSMAK
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 5, model: gemini-2.5-flash
  Please retry in 28.518775905s. [links {
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
  seconds: 28
  }
  ])

[2/5] YARIN SEN OKUL SINAV OLMAK
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 5, model: gemini-2.5-flash
  Please retry in 27.79831404s. [links {
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
  seconds: 27
  }
  ])

[3/5] ANNE MUTFAK YEMEK PISIRMEK BITMEK
  -> Annem mutfakta yemek pişirdi.... [9/10] (OK)

[4/5] BEN ISTANBUL UCAK GITMEK ISTEMEK
  -> Ben İstanbul'a uçakla gitmek istiyorum.... [10/10] (OK)

[5/5] COCUK PARK TOP OYNAMAK SEVMEK
  -> Çocuk parkta top oynamayı sever.... [9/10] (OK)

============================================================
Results: 3/5 successful
Avg Confidence: 9.33/10
Avg Latency: 6732ms
===================

============================================================
SUMMARY
=======

3-word: 5/5 (Avg conf: 9.2/10, Latency: 8144ms)
4-word: 1/5 (Avg conf: 10.0/10, Latency: 2027ms)
5-word: 3/5 (Avg conf: 9.33/10, Latency: 6732ms)

Total: 9/15 successful
======================
