#!/usr/bin/env python3
"""
Chat Router - OPTIMIZED VERSION WITH INTENT-SPECIFIC RESPONSES

✅ 30% confidence threshold
✅ SHORT responses for status updates (no explanation)
✅ DETAILED Phi-3 responses for compare_anime & check_anime_rating
✅ Terminal logging (intent, confidence, stage)
✅ No unnecessary long explanations
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import logging
import json
import torch
import re
from pathlib import Path
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logger = logging.getLogger(__name__)

# ===== Setup Model Caching =====
CHATBOT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_CACHE_DIR = Path.home() / ".anime_chatbot_cache"
MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(MODELS_CACHE_DIR)

logger.info("=" * 70)
logger.info("🚀 OPTIMIZED CHAT - Intent-Specific Responses + Terminal Logging")
logger.info("=" * 70)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from backend import auth, schemas, models
from backend.database import get_db
from backend.mal_service import MALService

router = APIRouter()

# ===== Device & Paths =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INTENT_CLASSIFIER_PATH = CHATBOT_ROOT / "backend" / "models" / "intent_classifier"

logger.info(f"Device: {DEVICE}")

# ===== CONFIGURATION =====
DISTILBERT_CONFIDENCE_THRESHOLD = 0.3  # 30%
VALID_INTENTS = {
    "get_recommendation", "check_anime_rating", "compare_anime", "search_anime",
    "set_watching", "set_completed", "set_plan_to_watch", "set_dropped", "set_on_hold",
    "view_list"
}

STATUS_TO_INTENT = {
    "watching": "set_watching",
    "completed": "set_completed",
    "plan_to_watch": "set_plan_to_watch",
    "planned": "set_plan_to_watch",
    "dropped": "set_dropped",
    "on_hold": "set_on_hold",
}

# Intents that need DETAILED Phi-3 responses
DETAILED_RESPONSE_INTENTS = {"compare_anime", "check_anime_rating", "get_recommendation", "search_anime"}

# Intents that need SHORT graceful responses
SHORT_RESPONSE_INTENTS = {"set_watching", "set_completed", "set_plan_to_watch", "set_dropped", "set_on_hold"}

# ===== GLOBAL MODEL STORAGE =====
intent_model = None
intent_tokenizer = None
phi_model = None
phi_tokenizer = None
id_to_intent = None


def load_models_on_startup():
    """Load all models at startup"""
    global intent_model, intent_tokenizer, phi_model, phi_tokenizer, id_to_intent
    
    try:
        logger.info("=" * 70)
        logger.info("🚀 LOADING MODELS")
        logger.info("=" * 70)
        
        # ===== 1. LOAD INTENT CLASSIFIER =====
        model_path = str(INTENT_CLASSIFIER_PATH) if INTENT_CLASSIFIER_PATH.exists() else str(MODELS_CACHE_DIR / "intent_classifier")
        logger.info(f"1️⃣ Loading Intent Classifier...")
        
        try:
            intent_tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=str(MODELS_CACHE_DIR))
            intent_model = AutoModelForSequenceClassification.from_pretrained(model_path, cache_dir=str(MODELS_CACHE_DIR)).to(DEVICE)
            intent_model.eval()
            
            label_map_path = INTENT_CLASSIFIER_PATH / "label_map.json"
            if label_map_path.exists():
                with open(label_map_path, "r") as f:
                    id_to_intent = {v: k for k, v in json.load(f).items()}
                logger.info(f"   ✅ DistilBERT loaded")
            else:
                id_to_intent = {
                    0: "check_anime_rating", 1: "compare_anime", 2: "get_recommendation",
                    3: "search_anime", 4: "set_completed", 5: "set_dropped",
                    6: "set_on_hold", 7: "set_plan_to_watch", 8: "set_watching", 9: "view_list"
                }
        except Exception as e:
            logger.warning(f"   ⚠️ Intent Classifier: {e}")
            intent_model = None
        
        # ===== 2. LOAD PHI-3-MINI =====
        logger.info("\n2️⃣ Loading Phi-3-mini...")
        
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        phi_tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            cache_dir=str(MODELS_CACHE_DIR),
            trust_remote_code=False
        )
        
        phi_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            cache_dir=str(MODELS_CACHE_DIR),
            quantization_config=nf4_config,
            device_map="cuda:0",
            trust_remote_code=False,
            attn_implementation="eager"
        )
        phi_model.eval()
        
        logger.info("   ✅ Phi-3-mini loaded")
        logger.info("=" * 70)
        return True
    
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        raise


# ===== Pydantic Models =====
class ChatRequest(BaseModel):
    message: str
    conversation_id: int

class ConversationCreate(BaseModel):
    title: str

class AnimeRecommendation(BaseModel):
    anime_id: int
    title: str
    image_url: Optional[str] = None
    score: Optional[float] = None

class ChatResponse(BaseModel):
    response: str
    action_type: Optional[str] = None
    action_data: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[AnimeRecommendation]] = None


# ===== NLP FUNCTIONS =====

def classify_intent_distilbert(user_message: str) -> tuple:
    """Stage 1: DistilBERT classification (fast)"""
    if not intent_model:
        return "unknown", 0.0
    
    try:
        inputs = intent_tokenizer(
            user_message,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length"
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = intent_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            idx = torch.argmax(probs, dim=-1).item()
        
        intent_name = id_to_intent.get(idx, "get_recommendation")
        confidence = probs[0, idx].item()
        
        return intent_name, confidence
    except Exception as e:
        logger.error(f"DistilBERT error: {e}")
        return "unknown", 0.0


def classify_intent_phi3(user_message: str, context: str = "") -> tuple:
    """Stage 2: Phi-3 intent classification with CONTEXT"""
    
    if not phi_model or not phi_tokenizer:
        return "chitchat", 0.0
    
    try:
        context_section = f"\n\nPrevious conversation context:\n{context}" if context else ""
        
        classification_prompt = f"""Classify if this is an ANIME-RELATED INTENT or FREEFORM CHAT:
{context_section}

Current message: "{user_message}"

ANIME INTENTS:
- get_recommendation: asking for anime recommendations
- check_anime_rating: asking about anime rating/score
- compare_anime: comparing two or more anime titles
- search_anime: searching for anime
- set_watching: adding/marking anime as watching
- set_completed: marking anime as completed/finished
- set_plan_to_watch: adding to plan to watch/watchlist
- set_dropped: marking as dropped
- set_on_hold: marking as on hold/paused
- view_list: viewing watchlist or anime list

Respond with ONLY:
TYPE: <INTENT_NAME or FREEFORM>
CONFIDENCE: <0.0 to 1.0>"""
        
        messages = [
            {"role": "system", "content": "Classify anime intents or freeform chat. Reply ONLY with TYPE and CONFIDENCE."},
            {"role": "user", "content": classification_prompt}
        ]
        
        prompt_ids = phi_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = phi_model.generate(
                prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=phi_tokenizer.eos_token_id,
            )
        
        response = phi_tokenizer.decode(outputs[0][len(prompt_ids[0]):], skip_special_tokens=True).strip()
        
        intent_match = re.search(r'TYPE:\s*(\w+)', response, re.IGNORECASE)
        confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response, re.IGNORECASE)
        
        if intent_match:
            intent_str = intent_match.group(1).lower().strip()
            confidence = float(confidence_match.group(1)) if confidence_match else 0.7
            
            if intent_str == "freeform":
                return "freeform", confidence
            
            if intent_str in VALID_INTENTS:
                return intent_str, confidence
            
            for valid_intent in VALID_INTENTS:
                if intent_str in valid_intent or valid_intent in intent_str:
                    return valid_intent, confidence
            
            return "freeform", confidence
        
        return "freeform", 0.0
    
    except Exception as e:
        logger.error(f"Phi-3 classification error: {e}")
        return "freeform", 0.0


def normalize_status(status_str: str) -> Optional[str]:
    """Normalize status"""
    if not status_str:
        return None
    
    status_lower = status_str.strip().lower().replace(" ", "_")
    
    if status_lower in STATUS_TO_INTENT:
        return status_lower
    
    for valid_status in STATUS_TO_INTENT.keys():
        if status_lower in valid_status or valid_status in status_lower:
            return valid_status
    
    return None


def normalize_json_keys(json_obj: Dict) -> Dict:
    """Normalize JSON keys"""
    result = {}
    
    for key in json_obj.keys():
        if 'anime' in key.lower() and 'name' in key.lower():
            result['anime_name'] = json_obj[key]
        elif key.lower() == 'anime':
            result['anime_name'] = json_obj[key]
    
    for key in json_obj.keys():
        if 'status' in key.lower():
            status_value = json_obj[key]
            normalized_status = normalize_status(str(status_value))
            if normalized_status:
                result['status'] = normalized_status
            else:
                result['status'] = status_value
    
    return result


def extract_entities_phi3(user_message: str) -> Dict[str, Any]:
    """Extract entities using Phi-3"""
    
    result = {"anime_name": None, "status": None, "anime_names": []}
    
    if not phi_model or not phi_tokenizer:
        return result
    
    try:
        primary_prompt = f"""Extract anime names and status from this message:
"{user_message}"

Reply ONLY with valid JSON:
{{"anime_names": ["anime1", "anime2"], "status": "..."}}

If multiple anime mentioned, list all. If no status, use null."""
        
        messages = [
            {"role": "system", "content": "Extract anime names and status. Reply ONLY with JSON."},
            {"role": "user", "content": primary_prompt}
        ]
        
        prompt_ids = phi_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = phi_model.generate(
                prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=phi_tokenizer.eos_token_id,
            )
        
        response = phi_tokenizer.decode(outputs[0][len(prompt_ids[0]):], skip_special_tokens=True).strip()
        
        try:
            json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                json_obj = json.loads(json_str)
                
                if json_obj.get('anime_names'):
                    result['anime_names'] = json_obj.get('anime_names', [])
                    if result['anime_names']:
                        result['anime_name'] = result['anime_names'][0]
                
                if json_obj.get('status'):
                    result['status'] = normalize_status(str(json_obj.get('status')))
                
                return result
        except json.JSONDecodeError:
            pass
        
        return result
    
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        return result


def generate_short_response(intent: str, anime_name: str) -> str:
    """Generate SHORT graceful response for status updates"""
    
    responses = {
        "set_watching": f"✅ Added to watching list: {anime_name}",
        "set_completed": f"✅ Marked as completed: {anime_name}",
        "set_plan_to_watch": f"✅ Added to Plan to Watch: {anime_name}",
        "set_dropped": f"✅ Marked as dropped: {anime_name}",
        "set_on_hold": f"✅ Put on hold: {anime_name}",
    }
    
    return responses.get(intent, f"✅ Updated {anime_name}")


def generate_detailed_response(user_message: str, intent: str, data: Dict = None, context: str = "") -> str:
    """Generate DETAILED Phi-3 response for comparison, ratings, recommendations"""
    
    if not phi_model or not phi_tokenizer:
        return "I'm here to help with anime!"
    
    try:
        context_section = f"\n\nPrevious conversation:\n{context}" if context else ""
        data_section = ""
        
        if data and data.get("anime_data"):
            data_section = "\n\nAnime information available:\n"
            for anime in data.get("anime_data", []):
                title = anime.get('title', 'Unknown')
                rating = anime.get('mean', 'N/A')
                synopsis = anime.get('synopsis', '')[:150]
                data_section += f"- {title}: {rating}/10\n  Synopsis: {synopsis}...\n"
        
        prompt_text = f"""User asked: "{user_message}"

Task: Generate a helpful, natural response.{data_section}

For COMPARE: Show differences between anime.
For RATING: Explain the rating in context.
For RECOMMENDATIONS: Describe why these are good choices.
For SEARCH: Summarize the results found.

Keep response 2-3 sentences, natural and conversational.{context_section}"""
        
        messages = [
            {"role": "system", "content": "You are a helpful anime chatbot. Respond naturally and conversationally. Be concise but informative."},
            {"role": "user", "content": prompt_text}
        ]
        
        prompt_ids = phi_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = phi_model.generate(
                prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=200,
                temperature=0.8,
                do_sample=True,
                pad_token_id=phi_tokenizer.eos_token_id,
                top_k=50,
                top_p=0.95
            )
        
        response = phi_tokenizer.decode(outputs[0][len(prompt_ids[0]):], skip_special_tokens=True).strip()
        return response or "I'd love to help!"
    except Exception as e:
        logger.error(f"Response error: {e}")
        return "I'm here to help with anime!"


def generate_freeform_response(user_message: str, context: str = "") -> str:
    """Generate natural chatbot response for freeform chat"""
    if not phi_model or not phi_tokenizer:
        return "I'm here to help with anime!"
    
    try:
        context_section = f"\n\nPrevious conversation:\n{context}" if context else ""
        
        messages = [
            {"role": "system", "content": "You are a friendly anime chatbot assistant. Be conversational, warm, and helpful. Keep responses concise (1-2 sentences)."},
            {"role": "user", "content": f"{context_section}\n\nUser: {user_message}"}
        ]
        
        prompt_ids = phi_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = phi_model.generate(
                prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=100,
                temperature=0.8,
                do_sample=True,
                pad_token_id=phi_tokenizer.eos_token_id,
                top_k=50,
                top_p=0.95
            )
        
        response = phi_tokenizer.decode(outputs[0][len(prompt_ids[0]):], skip_special_tokens=True).strip()
        return response or "I'm here to help with anime!"
    except Exception as e:
        logger.error(f"Freeform response error: {e}")
        return "I'm here to help with anime!"


def get_recent_context(db: Session, conversation_id: int, limit: int = 5) -> str:
    """Get recent conversation context"""
    try:
        history = db.query(models.ConversationHistory).filter(
            models.ConversationHistory.conversation_id == conversation_id
        ).order_by(models.ConversationHistory.id.desc()).limit(limit).all()
        
        context = ""
        for msg in reversed(history):
            role = "You" if msg.speaker == "user" else "Assistant"
            context += f"{role}: {msg.message_text}\n"
        
        return context
    except:
        return ""


# ===== TASK EXECUTION =====

def execute_intent_task(intent: str, entities: Dict[str, Any], mal_tokens, mal_service: MALService) -> tuple:
    """Execute MAL API tasks with enhanced recommendation support for anime/genre search."""
    try:
        if intent == "get_recommendation":
            # If anime name OR genre/key term is mentioned, use MAL search instead of top-rated
            search_term = None
            # anime_name is usually for explicit anime or genre keywords/entities
            # anime_names[] can be produced if multi-anime or genres (e.g., "action" or "romance anime")
            if entities.get("anime_name"):
                search_term = entities["anime_name"]
            elif entities.get("anime_names"):
                # Just take the first as primary search term if present
                search_term = entities["anime_names"][0]
            if search_term:  # If either is available
                try:
                    search = mal_service.search_anime(mal_tokens.access_token, search_term)
                    if search.get("data"):
                        recommendations = [
                            AnimeRecommendation(
                                anime_id=item["node"].get("id"),
                                title=item["node"].get("title"),
                                image_url=item["node"].get("main_picture", {}).get("large"),
                                score=item["node"].get("mean"),
                            )
                            for item in search["data"][:5]
                        ]
                        return (
                            True,
                            {"anime_data": [item["node"] for item in search["data"][:5]]},
                            recommendations,
                        )
                except Exception as e:
                    logging.error(f"[MAL Search] {e}")
                    # Fallback to random recs
            # Else fallback to highly rated random anime (legacy behavior)
            try:
                ranked = mal_service.get_random_highly_rated_anime(mal_tokens.access_token, limit=5)
                recommendations = [
                    AnimeRecommendation(
                        anime_id=anime["node"].get("id"),
                        title=anime["node"].get("title"),
                        image_url=anime["node"].get("main_picture", {}).get("large"),
                        score=anime["node"].get("mean")
                    )
                    for anime in ranked
                ]
                return (True, {"anime_data": [a["node"] for a in ranked]}, recommendations)
            except Exception as e:
                logging.error(f"[Fallback Recommendation Error] {e}")
                return (False, None, None)
        # --- Leave all other intents as-is (do not modify below) ---
        elif intent == "check_anime_rating":
            anime_name = entities.get("anime_name")
            if not anime_name:
                return (False, None, None)
            try:
                search = mal_service.search_anime(mal_tokens.access_token, anime_name)
                if search.get("data"):
                    anime = search["data"][0]["node"]
                    return (True, {"anime_data": [anime]}, None)
            except Exception as e:
                logging.error(f"Rating error: {e}")
            return (False, None, None)
        elif intent == "compare_anime":
            anime_names = entities.get("anime_names", [])
            if not anime_names:
                anime_names = [entities.get("anime_name", "Naruto")]
            anime_list = []
            try:
                for anime_name in anime_names[:2]:
                    search = mal_service.search_anime(mal_tokens.access_token, anime_name)
                    if search.get("data"):
                        anime_list.append(search["data"][0]["node"])
                if anime_list:
                    return (True, {"anime_data": anime_list}, None)
            except Exception as e:
                logging.error(f"Compare error: {e}")
            return (False, None, None)
        elif intent == "search_anime":
            search_term = entities.get("anime_name") or "popular"
            try:
                search = mal_service.search_anime(mal_tokens.access_token, search_term)
                if search.get("data"):
                    recommendations = [
                        AnimeRecommendation(
                            anime_id=item["node"].get("id"),
                            title=item["node"].get("title"),
                            image_url=item["node"].get("main_picture", {}).get("large"),
                            score=item["node"].get("mean")
                        )
                        for item in search["data"][:5]
                    ]
                    return (True, {"anime_data": [item["node"] for item in search["data"][:5]]}, recommendations)
            except Exception as e:
                logging.error(f"Search error: {e}")
            return (False, None, None)
        elif intent in SHORT_RESPONSE_INTENTS:
            anime_name = entities.get("anime_name")
            if not anime_name:
                return (False, None, None)
            status_map = {
                "set_watching": "watching",
                "set_completed": "completed",
                "set_plan_to_watch": "plan_to_watch",
                "set_dropped": "dropped",
                "set_on_hold": "on_hold"
            }
            target_status = status_map.get(intent, "watching")
            try:
                search = mal_service.search_anime(mal_tokens.access_token, anime_name)
                if search.get("data"):
                    anime = search["data"][0]["node"]
                    mal_service.update_anime_status(mal_tokens.access_token, anime["id"], target_status)
                    return (True, {"action": "completed"}, None)
            except Exception as e:
                logging.error(f"Status update error: {e}")
            return (False, None, None)
        elif intent == "view_list":
            try:
                response = mal_service.get_user_anime_list(mal_tokens.access_token, status="watching", limit=10)
                if response.get("data"):
                    recommendations = [
                        AnimeRecommendation(
                            anime_id=item["node"].get("id"),
                            title=item["node"].get("title"),
                            image_url=item["node"].get("main_picture", {}).get("large"),
                            score=item["node"].get("mean")
                        )
                        for item in response["data"][:5]
                    ]
                    if recommendations:
                        return (True, {"anime_data": [item["node"] for item in response["data"][:5]]}, recommendations)
                return (True, None, None)
            except Exception as e:
                logging.error(f"View list error: {e}")
                return (True, None, None)
        else:
            return (False, None, None)
    except Exception as e:
        logging.error(f"Task error: {e}")
        return (False, None, None)



# ===== API ENDPOINTS =====

@router.post("/conversations")
async def create_conversation(
    conversation: ConversationCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user),
):
    try:
        new_conv = models.Conversation(user_id=current_user.id, title=conversation.title)
        db.add(new_conv)
        db.commit()
        db.refresh(new_conv)
        return {"id": new_conv.id, "title": new_conv.title}
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations")
async def get_conversations(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user),
):
    try:
        conversations = db.query(models.Conversation).filter(
            models.Conversation.user_id == current_user.id
        ).order_by(models.Conversation.updated_at.desc()).all()
        
        return [{"id": c.id, "title": c.title} for c in conversations]
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user),
):
    try:
        conversation = db.query(models.Conversation).filter(
            models.Conversation.id == conversation_id,
            models.Conversation.user_id == current_user.id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404)
        
        db.query(models.ConversationHistory).filter(
            models.ConversationHistory.conversation_id == conversation_id
        ).delete()
        
        db.delete(conversation)
        db.commit()
        
        return {"message": "Deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}/messages")
async def get_chat_history(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user),
):
    try:
        conversation = db.query(models.Conversation).filter(
            models.Conversation.id == conversation_id,
            models.Conversation.user_id == current_user.id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404)
        
        history = db.query(models.ConversationHistory).filter(
            models.ConversationHistory.conversation_id == conversation_id
        ).order_by(models.ConversationHistory.id.asc()).limit(50).all()
        
        return [{"id": msg.id, "speaker": msg.speaker, "message_text": msg.message_text} for msg in history]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== MAIN CHAT ENDPOINT =====

@router.post("/chat", response_model=ChatResponse)
async def handle_chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user),
):
    """Optimized chat with intent-specific responses"""
    try:
        conversation = db.query(models.Conversation).filter(
            models.Conversation.id == request.conversation_id,
            models.Conversation.user_id == current_user.id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404)
        
        db.add(models.ConversationHistory(
            conversation_id=request.conversation_id,
            user_id=current_user.id,
            speaker="user",
            message_text=request.message
        ))
        conversation.updated_at = datetime.utcnow()
        db.commit()
        
        # ===== TERMINAL OUTPUT START =====
        print(f"\n{'='*70}")
        print(f"📝 USER MESSAGE: {request.message}")
        print(f"{'='*70}\n")
        
        # Get recent context
        recent_context = get_recent_context(db, request.conversation_id, limit=5)
        
        # STAGE 1: DistilBERT Classification
        print("🔍 STAGE 1: DistilBERT Classification (Fast)")
        distilbert_intent, distilbert_confidence = classify_intent_distilbert(request.message)
        print(f"   Intent: {distilbert_intent}")
        print(f"   Confidence: {distilbert_confidence:.2%}\n")
        
        # DECISION: Check confidence threshold (30%)
        if distilbert_confidence >= DISTILBERT_CONFIDENCE_THRESHOLD:
            print(f"✅ HIGH CONFIDENCE ({distilbert_confidence:.2%} >= {DISTILBERT_CONFIDENCE_THRESHOLD:.0%})")
            print(f"   → Using DistilBERT intent directly\n")
            final_intent = distilbert_intent
            classification_source = "DistilBERT"
        else:
            print(f"❌ LOW CONFIDENCE ({distilbert_confidence:.2%} < {DISTILBERT_CONFIDENCE_THRESHOLD:.0%})")
            print(f"   → Asking Phi-3 for secondary classification...\n")
            
            # STAGE 2: Phi-3 Classification WITH CONTEXT
            print("🔍 STAGE 2: Phi-3 Intent Classification (Accurate)")
            phi3_intent, phi3_confidence = classify_intent_phi3(request.message, recent_context)
            print(f"   Intent: {phi3_intent}")
            print(f"   Confidence: {phi3_confidence:.2%}\n")
            final_intent = phi3_intent
            classification_source = "Phi-3"
        
        print(f"🎯 FINAL DECISION: {final_intent} (from {classification_source})")
        print(f"{'='*70}\n")
        # ===== TERMINAL OUTPUT END =====
        
        # Route to appropriate handler
        if final_intent == "freeform":
            print(f"💬 FREEFORM CHAT DETECTED → Using natural response\n")
            bot_response = generate_freeform_response(request.message, recent_context)
            
            db.add(models.ConversationHistory(
                conversation_id=request.conversation_id,
                user_id=current_user.id,
                speaker="bot",
                message_text=bot_response
            ))
            db.commit()
            
            return ChatResponse(
                response=bot_response,
                action_type="chitchat",
                action_data=None,
                recommendations=None
            )
        
        # Intent detected - process task
        print(f"🔧 INTENT DETECTED → Processing: {final_intent}\n")
        
        entities = extract_entities_phi3(request.message)
        
        if final_intent not in VALID_INTENTS:
            final_intent = "get_recommendation"
        
        recommendations = None
        task_data = None
        bot_response = None
        
        mal_tokens = db.query(models.MalTokens).filter(
            models.MalTokens.user_id == current_user.id
        ).first()
        
        if mal_tokens:
            mal_service = MALService()
            success, data_for_phi3, task_recommendations = execute_intent_task(final_intent, entities, mal_tokens, mal_service)
            
            if success:
                recommendations = task_recommendations
                
                # Generate response based on intent type
                if final_intent in SHORT_RESPONSE_INTENTS:
                    # SHORT response for status updates
                    anime_name = entities.get("anime_name", "anime")
                    bot_response = generate_short_response(final_intent, anime_name)
                    print(f"✅ Short response generated: {bot_response}\n")
                
                elif final_intent in DETAILED_RESPONSE_INTENTS:
                    # DETAILED response for comparisons, ratings, recommendations
                    bot_response = generate_detailed_response(
                        request.message,
                        final_intent,
                        data_for_phi3,
                        recent_context
                    )
                    print(f"✅ Detailed response generated\n")
                
                if data_for_phi3:
                    task_data = data_for_phi3
            else:
                bot_response = "I couldn't find that anime. Try another search?"
        else:
            bot_response = "Please connect your MyAnimeList account!"
        
        if not bot_response:
            bot_response = "Something went wrong. Please try again."
        
        db.add(models.ConversationHistory(
            conversation_id=request.conversation_id,
            user_id=current_user.id,
            speaker="bot",
            message_text=bot_response
        ))
        db.commit()
        
        return ChatResponse(
            response=bot_response,
            action_type=final_intent,
            action_data=task_data,
            recommendations=recommendations
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        logger.exception("Full traceback:")
        raise HTTPException(status_code=500, detail=str(e))


# ===== LOAD MODELS =====
if not intent_model:
    load_models_on_startup()
