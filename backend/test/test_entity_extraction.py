#!/usr/bin/env python3
"""
TEST: Entity Extraction with Phi-3-mini
Run this to test if anime name and status extraction works
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json

# ===== Setup =====
CHATBOT_ROOT = Path(__file__).resolve().parent.parent
MODELS_CACHE_DIR = Path.home() / ".anime_chatbot_cache"
PHI_MODEL_PATH = MODELS_CACHE_DIR / "phi-3-mini-4k-instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Loading Phi-3 from: {PHI_MODEL_PATH}")
print("=" * 70)

# ===== Load Phi-3 =====
try:
    from transformers import BitsAndBytesConfig
    
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        str(PHI_MODEL_PATH),
        local_files_only=True,
        trust_remote_code=False
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        str(PHI_MODEL_PATH),
        quantization_config=nf4_config,
        device_map="cuda:0",
        trust_remote_code=False,
        attn_implementation="eager"
    )
    model.eval()
    
    print(f"✅ Phi-3 loaded in 4-bit mode!")
    print("=" * 70)
    
except Exception as e:
    print(f"❌ Failed to load: {e}")
    exit(1)

# ===== Test Cases =====
test_cases = [
    ("I'm watching Attack on Titan", "update_status", "Attack on Titan", "watching"),
    ("I completed Death Note", "update_status", "Death Note", "completed"),
    ("Add Naruto to my plan to watch", "update_status", "Naruto", "plan_to_watch"),
    ("Drop Bleach", "update_status", "Bleach", "dropped"),
    ("I'm on hold with One Piece", "update_status", "One Piece", "on_hold"),
    ("Recommend me anime", "recommend", None, None),
    ("Show my list", "general", None, None),
]

print("\n🧪 TESTING ENTITY EXTRACTION\n")

for user_message, intent, expected_anime, expected_status in test_cases:
    extraction_prompt = f"""You are a data extraction assistant.

Task: Extract anime title and status from the user message.

User Message: "{user_message}"

Intent: {intent}

Return ONLY a JSON object. Format: {{"anime_name": "Title", "status": "watching/completed/dropped/on_hold/plan_to_watch"}}

If no anime is found, set anime_name to null.

JSON:

{{"""

    inputs = tokenizer(
        extraction_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON
    start_idx = generated_text.find('{')
    end_idx = generated_text.rfind('}')
    
    extracted = {"anime_name": None, "status": None}
    if start_idx != -1 and end_idx != -1:
        try:
            extracted = json.loads(generated_text[start_idx:end_idx+1])
        except:
            pass
    
    anime_match = extracted.get("anime_name") == expected_anime if expected_anime else extracted.get("anime_name") is None
    status_match = extracted.get("status") == expected_status if expected_status else extracted.get("status") is None
    match = "✅" if (anime_match and status_match) else "❌"
    
    print(f"{match} Message: \"{user_message}\"")
    print(f"   Intent: {intent}")
    print(f"   Expected: anime='{expected_anime}', status='{expected_status}'")
    print(f"   Extracted: anime='{extracted.get('anime_name')}', status='{extracted.get('status')}'")
    print()

print("=" * 70)
print("✅ Entity Extraction Test Complete!")
