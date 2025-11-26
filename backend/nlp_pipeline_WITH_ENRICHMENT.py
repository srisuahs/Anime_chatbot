"""
Updated NLP Pipeline with Data Enrichment
Integrates MAL + Web search data into prompt generation
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from typing import Tuple, Dict, List, Optional
import os
import asyncio

from backend.nlp_config import (
    INTENT_CLASSIFIER_MODEL_PATH,
    RESPONSE_GENERATOR_MODEL_PATH,
    DISTILGPT2_MAX_TOKENS,
    CONTEXT_WINDOW_TOKENS,
    INTENT_MAP,
    INTENT_ID_TO_NAME,
    GPT2_GENERATION_CONFIG,
)
from backend.data_enricher import get_data_enricher

logger = logging.getLogger(__name__)

class NLPPipelineWithEnrichment:
    """
    Enhanced NLP pipeline with MAL + Web data enrichment
    """
    
    def __init__(self, mal_access_token: Optional[str] = None):
        """Initialize models and enricher"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"NLP Pipeline initialized on device: {self.device}")
        
        # Intent Classifier (DistilBERT)
        try:
            self.intent_tokenizer = AutoTokenizer.from_pretrained(
                INTENT_CLASSIFIER_MODEL_PATH if os.path.exists(INTENT_CLASSIFIER_MODEL_PATH)
                else "distilbert-base-uncased"
            )
            self.intent_model = AutoModelForSequenceClassification.from_pretrained(
                INTENT_CLASSIFIER_MODEL_PATH if os.path.exists(INTENT_CLASSIFIER_MODEL_PATH)
                else "distilbert-base-uncased",
                num_labels=len(INTENT_MAP)
            ).to(self.device)
            self.intent_model.eval()
            logger.info("Intent classifier loaded successfully")
        except Exception as e:
            logger.error(f"Error loading intent classifier: {e}")
            raise
        
        # Response Generator (DistilGPT-2)
        try:
            self.gpt_tokenizer = AutoTokenizer.from_pretrained(
                RESPONSE_GENERATOR_MODEL_PATH if os.path.exists(RESPONSE_GENERATOR_MODEL_PATH)
                else "distilgpt2"
            )
            self.gpt_model = AutoModelForCausalLM.from_pretrained(
                RESPONSE_GENERATOR_MODEL_PATH if os.path.exists(RESPONSE_GENERATOR_MODEL_PATH)
                else "distilgpt2"
            ).to(self.device)
            self.gpt_model.eval()
            
            if self.gpt_tokenizer.pad_token is None:
                self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
            
            logger.info("Response generator loaded successfully")
        except Exception as e:
            logger.error(f"Error loading response generator: {e}")
            raise
        
        # Data Enricher
        self.enricher = get_data_enricher(mal_access_token)
        logger.info("Data enricher initialized")
    
    def classify_intent(self, user_input: str) -> Tuple[str, float]:
        """Classify user intent using DistilBERT"""
        try:
            inputs = self.intent_tokenizer(
                user_input,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.intent_model(**inputs)
                logits = outputs.logits
            
            probabilities = torch.softmax(logits, dim=-1)
            intent_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][intent_id].item()
            intent_name = INTENT_ID_TO_NAME.get(intent_id, "general_chat")
            
            logger.info(f"Intent: {intent_name} (confidence: {confidence:.2f})")
            return intent_name, confidence
        
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return "general_chat", 0.0
    
    async def build_enriched_prompt_with_context(
        self,
        user_message: str,
        intent: str,
        chat_history: List[Dict[str, str]],
        mal_access_token: Optional[str] = None
    ) -> str:
        """
        Build prompt with MAL + Web data enrichment + chat history
        """
        try:
            # Get enriched data
            enriched = await self.enricher.enrich_prompt(
                user_message,
                mal_access_token
            )
            
            logger.info(f"Enrichment context: {enriched['enriched_context'][:100]}...")
            
            # Build prompt parts
            prompt_parts = []
            
            # Add enrichment data
            if enriched["enriched_context"]:
                prompt_parts.append(enriched["enriched_context"])
            
            # Add intent and entities
            entities_str = ", ".join(enriched["extracted_entities"])
            if entities_str:
                prompt_parts.append(f"[INTENT={intent}] [ENTITIES={entities_str}]")
            else:
                prompt_parts.append(f"[INTENT={intent}]")
            
            # Add genres if present
            if enriched["genres"]:
                prompt_parts.append(f"[GENRES={', '.join(enriched['genres'])}]")
            
            # Add chat history (limited to token window)
            context_tokens = 0
            for msg in reversed(chat_history):
                msg_text = f"<USER>: {msg['user']}\n<BOT>: {msg['bot']}\n"
                msg_tokens = len(self.gpt_tokenizer.encode(msg_text))
                
                if context_tokens + msg_tokens > CONTEXT_WINDOW_TOKENS:
                    logger.info("Context limit reached")
                    break
                
                prompt_parts.append(msg_text)
                context_tokens += msg_tokens
            
            # Add current user message
            prompt_parts.append(f"<USER>: {user_message}\n<BOT>:")
            
            final_prompt = "".join(reversed(prompt_parts))
            
            # Validate token count
            final_tokens = len(self.gpt_tokenizer.encode(final_prompt))
            if final_tokens > DISTILGPT2_MAX_TOKENS:
                logger.warning(f"Prompt exceeds max ({final_tokens}). Truncating...")
                final_prompt = final_prompt[:int(len(final_prompt) * (DISTILGPT2_MAX_TOKENS / final_tokens))]
            
            return final_prompt
        
        except Exception as e:
            logger.error(f"Prompt building error: {e}")
            # Fallback to simple prompt
            return f"[INTENT={intent}] <USER>: {user_message}\n<BOT>:"
    
    def generate_response(self, prompt: str, max_new_tokens: int = 80) -> str:
        """Generate response using DistilGPT-2"""
        try:
            inputs = self.gpt_tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=DISTILGPT2_MAX_TOKENS,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.gpt_model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    temperature=GPT2_GENERATION_CONFIG["temperature"],
                    top_p=GPT2_GENERATION_CONFIG["top_p"],
                    do_sample=GPT2_GENERATION_CONFIG["do_sample"],
                    pad_token_id=self.gpt_tokenizer.eos_token_id
                )
            
            new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            response = self.gpt_tokenizer.decode(new_tokens, skip_special_tokens=True)
            response = response.strip()
            if response.endswith("<BOT>:"):
                response = response[:-6].strip()
            
            logger.info(f"Generated: {response[:100]}...")
            return response
        
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "I'm having trouble processing that. Could you rephrase?"
    
    async def process_message(
        self,
        user_message: str,
        chat_history: List[Dict[str, str]],
        mal_access_token: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Complete pipeline: classify + enrich + generate
        """
        # Step 1: Classify intent
        intent, confidence = self.classify_intent(user_message)
        
        # Step 2: Build enriched prompt with context
        prompt = await self.build_enriched_prompt_with_context(
            user_message=user_message,
            intent=intent,
            chat_history=chat_history,
            mal_access_token=mal_access_token
        )
        
        # Step 3: Generate response
        response = self.generate_response(prompt)
        
        return {
            "intent": intent,
            "confidence": confidence,
            "prompt": prompt,
            "response": response
        }


# Global pipeline instance
_nlp_pipeline = None

def get_nlp_pipeline(mal_token: Optional[str] = None) -> NLPPipelineWithEnrichment:
    """Get or create NLP pipeline"""
    global _nlp_pipeline
    if _nlp_pipeline is None:
        _nlp_pipeline = NLPPipelineWithEnrichment(mal_token)
    return _nlp_pipeline
