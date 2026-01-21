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
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 2.07355508s. [links {
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
  seconds: 2
  }
  ])

[2/5] O ISTANBUL CALISMAK
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 56.987874746s. [links {
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

[3/5] BUGUN HAVA GUZEL
  [API Key rotated to key #1]
  [API Key rotated to key #2]
  -> ... [0/10] (ERROR: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit.

* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
  Please retry in 51.802815813s. [links {
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

[4/5] BU KITAP BENIM
  [API Key rotated to key #1]
  [API Key rotated to key #2]
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
  Please retry in 47.054200231s. [links {
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
  seconds: 47
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
