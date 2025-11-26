#!/usr/bin/env python3
"""
TEST: Intent Classifier
Run this to test if DistilBERT intent classification works correctly
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import json

# ===== Setup Paths =====
CHATBOT_ROOT = Path(__file__).resolve().parent.parent
INTENT_CLASSIFIER_PATH = CHATBOT_ROOT / "backend" / "models" / "intent_classifier"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Loading from: {INTENT_CLASSIFIER_PATH}")
print("=" * 70)

# ===== Load Models =====
try:
    tokenizer = AutoTokenizer.from_pretrained(str(INTENT_CLASSIFIER_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(INTENT_CLASSIFIER_PATH)).to(DEVICE)
    model.eval()
    
    # Load labels
    label_map_path = INTENT_CLASSIFIER_PATH / "label_map.json"
    with open(label_map_path, "r") as f:
        intent_label_map = json.load(f)
    id_to_intent = {v: k for k, v in intent_label_map.items()}
    
    print(f"✅ Models loaded!")
    print(f"   Intent labels: {list(id_to_intent.values())}")
    print("=" * 70)
    
except Exception as e:
    print(f"❌ Failed to load: {e}")
    exit(1)

# ===== Test Cases =====
test_cases = [
    ("Can you recommend anime", "recommend"),
    ("Show me my anime list", "get_user_list"),
    ("I'm watching Attack on Titan", "update_status"),
    ("What should I watch next", "recommend"),
    ("Add Naruto to my list", "update_status"),
    ("I completed Death Note", "update_status"),
    ("Hello, how are you?", "general"),
    ("Can you help me", "general"),
]

print("\n🧪 TESTING INTENT CLASSIFICATION\n")

for user_message, expected_intent in test_cases:
    inputs = tokenizer(
        user_message,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, idx].item()
    
    predicted_intent = id_to_intent.get(idx, "general")
    match = "✅" if predicted_intent == expected_intent else "❌"
    
    print(f"{match} Message: \"{user_message}\"")
    print(f"   Expected: {expected_intent}")
    print(f"   Predicted: {predicted_intent} (confidence: {confidence:.2f})")
    print()

print("=" * 70)
print("✅ Intent Classifier Test Complete!")
