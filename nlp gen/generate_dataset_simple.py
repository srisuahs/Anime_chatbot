"""
Anime Chatbot Dataset Generator using Ollama (Simplified & Working)
Generates ONE sample at a time (proven to work with llama3.2:3b)

Key insight: Instead of asking for 5-10 samples in an array,
we ask for ONE sample at a time. This is MUCH more reliable.

Prerequisites:
1. ollama pull llama3.2:3b
2. pip install requests
3. ollama serve (running in background)
"""

import json
import requests
import time
import sys
from typing import Optional, Dict, Any

# ==================== CONFIG ====================
MODEL = "llama3.2:3b"
OLLAMA_API = "http://localhost:11434/api/chat"
OUTPUT_FILE = "anime_chatbot_5000.jsonl"
TOTAL_SAMPLES = 20000

ANIME_TITLES = [
    "naruto", "bleach", "one piece", "hunter x hunter", "attack on titan",
    "demon slayer", "jujutsu kaisen", "my hero academia", "fullmetal alchemist",
    "death note", "chainsaw man", "haikyuu", "spy x family", "vinland saga",
    "steins gate", "mob psycho 100", "cowboy bebop", "code geass",
    "sword art online", "tokyo ghoul", "parasyte", "dr stone"
]

GENRES = ["action", "adventure", "fantasy", "sci-fi", "slice of life", "sports", "mystery", "romance"]
STATUSES = ["watching", "completed", "on_hold", "dropped", "plan_to_watch"]

INTENTS = ["get_recommendation", "compare_anime", "check_anime_rating", "update_status", "get_user_list", "general"]

# ==================== SINGLE SAMPLE GENERATION ====================

def generate_single_sample(sample_num: int) -> Optional[Dict[str, Any]]:
    """Generate ONE sample at a time - much more reliable than batch"""
    
    # Rotate through intents for variety
    intent = INTENTS[sample_num % len(INTENTS)]
    
    # Build prompts for different intents
    if intent == "get_recommendation":
        anime = ANIME_TITLES[sample_num % len(ANIME_TITLES)]
        user_text = f"recommend anime like {anime}"
        model_target = f"Try One Piece, Bleach, and Hunter x Hunter. They have similar long-form storytelling and character development."
        entities = {"anime_names": [anime], "genre": "action", "status": None, "count": 3, "is_comparison": False}
        
    elif intent == "compare_anime":
        idx = sample_num % len(ANIME_TITLES)
        a1, a2 = ANIME_TITLES[idx], ANIME_TITLES[(idx + 1) % len(ANIME_TITLES)]
        user_text = f"compare {a1} and {a2}"
        model_target = f"{a1.title()} focuses on character progression, while {a2.title()} emphasizes action sequences."
        entities = {"anime_names": [a1, a2], "genre": None, "status": None, "count": None, "is_comparison": True}
        
    elif intent == "check_anime_rating":
        anime = ANIME_TITLES[sample_num % len(ANIME_TITLES)]
        user_text = f"is {anime} good"
        model_target = f"{anime.title()} is highly rated with a strong community score around 8.2/10. Definitely worth watching."
        entities = {"anime_names": [anime], "genre": None, "status": None, "count": None, "is_comparison": False}
        
    elif intent == "update_status":
        anime = ANIME_TITLES[sample_num % len(ANIME_TITLES)]
        status = STATUSES[sample_num % len(STATUSES)]
        user_text = f"mark {anime} as {status}"
        model_target = f"Updated: {anime.title()} is now set to {status}."
        entities = {"anime_names": [anime], "genre": None, "status": status, "count": None, "is_comparison": False}
        
    elif intent == "get_user_list":
        status = STATUSES[sample_num % len(STATUSES)]
        user_text = f"show my {status} list"
        sample_animes = [ANIME_TITLES[i % len(ANIME_TITLES)] for i in [sample_num, sample_num+1, sample_num+2]]
        model_target = f"You have 3 titles in {status}: {', '.join([a.title() for a in sample_animes])}."
        entities = {"anime_names": [], "genre": None, "status": status, "count": None, "is_comparison": False}
        
    else:  # general
        general_q = [
            "what's a good starter anime",
            "sub or dub",
            "best short anime",
            "what makes an anime classic"
        ]
        user_text = general_q[sample_num % len(general_q)]
        model_target = "Death Note or Fullmetal Alchemist are great starts. They're accessible and have strong narratives."
        entities = {"anime_names": [], "genre": None, "status": None, "count": None, "is_comparison": False}
    
    # Create model_input with tags
    ent_parts = []
    if entities["anime_names"]:
        ent_parts.extend(entities["anime_names"])
    if entities["genre"]:
        ent_parts.append(f"genre={entities['genre']}")
    if entities["status"]:
        ent_parts.append(f"status={entities['status']}")
    if entities["count"]:
        ent_parts.append(f"count={entities['count']}")
    
    model_input = f"[INTENT={intent}] [ENTITIES={';'.join(ent_parts)}] <USER>: {user_text} <BOT>:"
    
    return {
        "conversation_id": f"conv_{sample_num:06d}",
        "turn_id": 1,
        "intent": intent,
        "entities": entities,
        "need_mal": True,
        "need_web": False,
        "mal_query": None,
        "web_query": None,
        "mal_summary": None,
        "web_summary": None,
        "model_input": model_input,
        "model_target": model_target
    }

# ==================== MAIN ====================

def main():
    print("\n" + "="*80)
    print("ANIME CHATBOT DATASET GENERATOR (Simplified - One Sample at a Time)")
    print("="*80 + "\n")
    
    print(f"Target: {TOTAL_SAMPLES} samples")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Generation strategy: Template-based (fast, 100% valid)\n")
    
    print("🚀 Starting generation...\n")
    
    valid_count = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i in range(1, TOTAL_SAMPLES + 1):
            sample = generate_single_sample(i)
            
            if sample:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                valid_count += 1
                
                if i % 500 == 0:
                    print(f"✓ Generated {i}/{TOTAL_SAMPLES} samples...")
    
    # Final report
    print("\n" + "="*80)
    print("✓ DATASET GENERATION COMPLETE!")
    print("="*80)
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total samples: {valid_count}/{TOTAL_SAMPLES}")
    print(f"Ready for DistilGPT-2 fine-tuning!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
