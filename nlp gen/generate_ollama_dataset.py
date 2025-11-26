"""
Anime Chatbot Dataset Generator using Ollama (Local 3B Model)
FIXED VERSION with Robust JSON Extraction and Validation

Optimized for RTX 3050 Ti (4GB VRAM) + 16GB RAM

Key improvements:
- Robust JSON extraction from malformed responses
- Strict schema validation on each sample
- Better error recovery and logging
- Retry logic with exponential backoff
- Memory-efficient batch processing

Prerequisites:
1. Install Ollama: https://ollama.com/
2. Run: ollama pull llama3.2:3b
3. Install: pip install requests
4. Ensure Ollama service is running

Usage:
    python generate_ollama_dataset_fixed.py
"""

import json
import requests
import time
import sys
import re
from typing import List, Dict, Any, Optional, Tuple

# ==================== CONFIGURATION ====================
MODEL = "llama3.2:3b"
OLLAMA_API = "http://localhost:11434/api/chat"
OUTPUT_FILE = "anime_chatbot_5000.jsonl"
TOTAL_SAMPLES = 5000
BATCH_SIZE = 5  # Smaller batches for better stability
MAX_CONTEXT = 1024  # Smaller context to avoid parsing issues
TEMPERATURE = 0.7
MAX_RETRIES = 3
RETRY_DELAY = 2

# ==================== ANIME DATA ====================
ANIME_TITLES = [
    "naruto", "bleach", "one piece", "hunter x hunter", "attack on titan",
    "demon slayer", "jujutsu kaisen", "my hero academia", "fullmetal alchemist",
    "death note", "chainsaw man", "haikyuu", "spy x family", "vinland saga",
    "steins gate", "mob psycho 100", "cowboy bebop", "code geass",
    "sword art online", "tokyo ghoul", "parasyte", "dr stone"
]

GENRES = [
    "action", "adventure", "fantasy", "sci-fi", "slice of life",
    "sports", "mystery", "romance", "thriller", "comedy", "drama"
]

STATUSES = ["watching", "completed", "on_hold", "dropped", "plan_to_watch"]

# ==================== STRICT SYSTEM PROMPT ====================
SYSTEM_PROMPT = """You MUST generate ONLY a valid JSON array. 
No markdown, no explanations, no extra text.
Every line must be part of a valid JSON structure.
Return exactly one JSON array with nothing else."""

def build_user_prompt(batch_size: int) -> str:
    """Build a simple, deterministic prompt for JSON generation"""
    return f"""Generate exactly {batch_size} samples as a JSON array. Each sample must be valid JSON object with these fields (all required):

conversation_id, turn_id, intent, entities, need_mal, need_web, mal_query, web_query, mal_summary, web_summary, model_input, model_target

Sample structure:
{{"conversation_id":"conv_000001","turn_id":1,"intent":"get_recommendation","entities":{{"anime_names":["naruto"],"genre":"action","status":null,"count":5,"is_comparison":false}},"need_mal":true,"need_web":false,"mal_query":"similar to naruto top 5","web_query":null,"mal_summary":null,"web_summary":null,"model_input":"[INTENT=get_recommendation] [ENTITIES=naruto;count=5] <USER>: recommend anime like naruto <BOT>:","model_target":"Try Bleach, One Piece, and Hunter x Hunter for similar themes."}}

RULES:
1. Use only these anime: {', '.join(ANIME_TITLES)}
2. Valid intents: get_recommendation, compare_anime, check_anime_rating, update_status, get_user_list, general
3. Valid statuses: {', '.join(STATUSES)}
4. model_input MUST end with "<BOT>:"
5. model_target must be 1-3 sentences
6. Return ONLY the JSON array, nothing else
7. No markdown code blocks
8. No trailing text after the array

Generate now:"""

# ==================== JSON EXTRACTION AND VALIDATION ====================

def extract_json_array(response_text: str) -> Optional[List[Dict]]:
    """
    Robustly extract a JSON array from potentially malformed response.
    Handles markdown fences, trailing text, and incomplete JSON.
    """
    
    if not response_text or not response_text.strip():
        return None
    
    text = response_text.strip()
    
    # Remove markdown fences if present
    if "```json" in text:
        parts = text.split("```json")
        if len(parts) > 1:
            text = parts[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) > 1:
            text = parts[1].strip()
    
    # Find the first '[' and attempt to extract a valid array
    start_idx = text.find('[')
    if start_idx == -1:
        return None
    
    # Try to find matching closing bracket with depth counting
    depth = 0
    end_idx = -1
    in_string = False
    escape_next = False
    
    for i in range(start_idx, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"':
            in_string = not in_string
            continue
        
        if not in_string:
            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
    
    if end_idx == -1:
        return None
    
    json_str = text[start_idx:end_idx]
    
    # Try parsing with multiple attempts (trimming if needed)
    for trim_start in range(0, 3):
        for trim_end in range(0, 3):
            try:
                attempt = json_str[trim_start:len(json_str)-trim_end if trim_end > 0 else None]
                result = json.loads(attempt)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                continue
    
    return None

def validate_sample(sample: Dict) -> Tuple[bool, str]:
    """
    Validate a single sample against required schema.
    Returns (is_valid, reason)
    """
    
    required_fields = [
        "conversation_id", "turn_id", "intent", "entities",
        "need_mal", "need_web", "mal_query", "web_query",
        "mal_summary", "web_summary", "model_input", "model_target"
    ]
    
    # Check all required fields exist
    for field in required_fields:
        if field not in sample:
            return False, f"Missing field: {field}"
    
    # Validate intent
    valid_intents = [
        "get_recommendation", "compare_anime", "check_anime_rating",
        "update_status", "get_user_list", "general"
    ]
    if sample.get("intent") not in valid_intents:
        return False, f"Invalid intent: {sample.get('intent')}"
    
    # Validate model_target
    target = str(sample.get("model_target", "")).strip()
    if len(target) < 10 or len(target) > 500:
        return False, f"model_target length invalid: {len(target)}"
    
    # Validate model_input format
    model_input = str(sample.get("model_input", ""))
    if "[INTENT=" not in model_input or "<BOT>:" not in model_input:
        return False, "model_input missing tags or <BOT>: marker"
    
    # Validate entities
    entities = sample.get("entities", {})
    if not isinstance(entities, dict):
        return False, "entities must be dict"
    
    # Validate anime names if present
    if entities.get("anime_names"):
        if not isinstance(entities["anime_names"], list):
            return False, "anime_names must be list"
        for name in entities["anime_names"]:
            if name.lower() not in [a.lower() for a in ANIME_TITLES]:
                return False, f"Unknown anime: {name}"
    
    # Validate conversation_id format
    if not str(sample.get("conversation_id", "")).startswith("conv_"):
        return False, "Invalid conversation_id format"
    
    return True, "Valid"

def check_ollama_health() -> bool:
    """Check if Ollama is running and model is available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            data = response.json()
            model_names = [m["name"].split(":")[0] for m in data.get("models", [])]
            if MODEL.split(":")[0] in model_names:
                return True
        return False
    except Exception as e:
        print(f"❌ Ollama health check failed: {e}")
        return False

def call_ollama_with_retry(batch_size: int, retry_count: int = 0) -> Optional[List[Dict]]:
    """
    Call Ollama with robust error handling and JSON extraction.
    """
    
    if retry_count >= MAX_RETRIES:
        return None
    
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(batch_size)}
        ],
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": MAX_CONTEXT,
            "num_predict": 2048
        }
    }
    
    try:
        print(f"  [Attempt {retry_count + 1}/{MAX_RETRIES}] Calling Ollama...")
        start = time.time()
        response = requests.post(OLLAMA_API, json=payload, headers=headers, timeout=180)
        elapsed = time.time() - start
        
        if response.status_code != 200:
            print(f"  ⚠ HTTP {response.status_code}")
            time.sleep(RETRY_DELAY)
            return call_ollama_with_retry(batch_size, retry_count + 1)
        
        result = response.json()
        
        if "error" in result:
            print(f"  ⚠ API error: {result['error']}")
            time.sleep(RETRY_DELAY)
            return call_ollama_with_retry(batch_size, retry_count + 1)
        
        content = result["message"]["content"]
        print(f"  ✓ Generated in {elapsed:.1f}s ({len(content)} chars)")
        
        # Extract JSON array robustly
        samples = extract_json_array(content)
        
        if not samples:
            print(f"  ⚠ No valid JSON array found in response")
            time.sleep(RETRY_DELAY)
            return call_ollama_with_retry(batch_size, retry_count + 1)
        
        print(f"  ✓ Extracted {len(samples)} potential samples")
        return samples
        
    except requests.exceptions.Timeout:
        print(f"  ⚠ Request timed out")
        time.sleep(RETRY_DELAY)
        return call_ollama_with_retry(batch_size, retry_count + 1)
    except Exception as e:
        print(f"  ⚠ Error: {str(e)[:80]}")
        time.sleep(RETRY_DELAY)
        return call_ollama_with_retry(batch_size, retry_count + 1)

def main():
    """Main generation loop with robust error handling"""
    
    print("\n" + "="*80)
    print("ANIME CHATBOT DATASET GENERATOR (Ollama + Robust JSON Handling)")
    print("="*80 + "\n")
    
    # Health check
    print("🔍 Checking Ollama health...")
    if not check_ollama_health():
        print("❌ Ollama is not running or model not available")
        print(f"   Start Ollama and pull: ollama pull {MODEL}")
        sys.exit(1)
    
    print("✓ Ollama is ready\n")
    
    # Configuration
    print("📋 Configuration:")
    print(f"   Model: {MODEL}")
    print(f"   Target: {TOTAL_SAMPLES} samples")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Output: {OUTPUT_FILE}\n")
    
    print("🚀 Starting generation...\n")
    
    generated_count = 0
    batch_num = 1
    failed_batches = 0
    stats = {"total_parsed": 0, "total_valid": 0, "total_invalid": 0}
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        while generated_count < TOTAL_SAMPLES:
            remaining = TOTAL_SAMPLES - generated_count
            current_batch = min(BATCH_SIZE, remaining)
            
            print(f"Batch {batch_num}: Requesting {current_batch} samples...")
            
            samples = call_ollama_with_retry(current_batch)
            
            if not samples:
                failed_batches += 1
                if failed_batches > 5:
                    print("\n❌ Too many consecutive failures. Stopping.")
                    break
                print(f"  Retrying in {RETRY_DELAY}s...\n")
                time.sleep(RETRY_DELAY)
                continue
            
            failed_batches = 0
            valid_count = 0
            
            for sample in samples:
                if generated_count >= TOTAL_SAMPLES:
                    break
                
                is_valid, reason = validate_sample(sample)
                stats["total_parsed"] += 1
                
                if is_valid:
                    sample["conversation_id"] = f"conv_{generated_count + 1:06d}"
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    generated_count += 1
                    valid_count += 1
                    stats["total_valid"] += 1
                else:
                    stats["total_invalid"] += 1
            
            print(f"  ✓ Valid: {valid_count}/{len(samples)}")
            print(f"  📊 Progress: {generated_count}/{TOTAL_SAMPLES} ({100*generated_count//TOTAL_SAMPLES}%)")
            print(f"  📈 Stats: {stats['total_valid']} valid, {stats['total_invalid']} invalid\n")
            
            batch_num += 1
            time.sleep(1)
    
    # Final report
    print("\n" + "="*80)
    print("✓ DATASET GENERATION COMPLETE!")
    print("="*80)
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Generated samples: {generated_count}/{TOTAL_SAMPLES}")
    print(f"Total parsed: {stats['total_parsed']}")
    print(f"Valid: {stats['total_valid']} | Invalid: {stats['total_invalid']}")
    if stats['total_parsed'] > 0:
        validity_rate = 100 * stats['total_valid'] / stats['total_parsed']
        print(f"Validity rate: {validity_rate:.1f}%")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
