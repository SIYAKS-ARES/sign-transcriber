"""RAG-augmented prompt building with TID linguistic support."""

from .augmented_prompt import AugmentedPromptBuilder, create_augmented_prompt
from .system_instructions import (
    TID_SYSTEM_INSTRUCTION,
    TID_SYSTEM_INSTRUCTION_COMPACT,
    build_dynamic_system_instruction,
    get_tense_context,
    get_question_context,
    get_negation_context,
    get_repetition_context,
)
from .few_shot_builder import (
    FewShotBuilder,
    FewShotExample,
    build_few_shot_examples,
)
from .templates import (
    OUTPUT_FORMAT_TEMPLATE,
    USER_PROMPT_TEMPLATE,
    build_linguistic_context,
    build_user_prompt,
    build_simple_prompt,
    PromptComponents,
)

__all__ = [
    # Prompt builder
    "AugmentedPromptBuilder",
    "create_augmented_prompt",
    # System instructions
    "TID_SYSTEM_INSTRUCTION",
    "TID_SYSTEM_INSTRUCTION_COMPACT",
    "build_dynamic_system_instruction",
    "get_tense_context",
    "get_question_context",
    "get_negation_context",
    "get_repetition_context",
    # Few-shot builder
    "FewShotBuilder",
    "FewShotExample",
    "build_few_shot_examples",
    # Templates
    "OUTPUT_FORMAT_TEMPLATE",
    "USER_PROMPT_TEMPLATE",
    "build_linguistic_context",
    "build_user_prompt",
    "build_simple_prompt",
    "PromptComponents",
]
