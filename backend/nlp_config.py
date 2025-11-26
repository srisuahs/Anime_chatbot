"""
NLP Pipeline Configuration
Handles tokenizer max lengths and model paths
"""

# Model Paths (local or HuggingFace)
INTENT_CLASSIFIER_MODEL_PATH = "backend/models/intent_classifier"
RESPONSE_GENERATOR_MODEL_PATH = "backend/models/distilgpt2_finetuned"

# Token Limits
DISTILGPT2_MAX_TOKENS = 512  # DistilGPT-2 context window
CONTEXT_RESERVE_TOKENS = 100  # Reserved for prompt structure + response generation
CONTEXT_WINDOW_TOKENS = DISTILGPT2_MAX_TOKENS - CONTEXT_RESERVE_TOKENS  # 412 tokens for history

# Intent IDs (maps to intent_training_data.json)
INTENT_MAP = {
    "get_recommendation": 0,
    "compare_anime": 1,
    "search_anime": 2,
    "set_watching": 3,
    "set_completed": 4,
    "get_info": 5,
    "get_schedule": 6,
    "general_chat": 7,
    "update_status": 8,
    "list_anime": 9
}

INTENT_ID_TO_NAME = {v: k for k, v in INTENT_MAP.items()}

# DistilGPT-2 Generation Parameters
GPT2_GENERATION_CONFIG = {
    "max_new_tokens": 80,  # Max response length
    "min_length": 10,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "num_beams": 1,
}

# Tokenizer Parameters
TOKENIZER_KWARGS = {
    "padding": "max_length",
    "truncation": True,
    "max_length": DISTILGPT2_MAX_TOKENS,
    "return_tensors": "pt"
}
