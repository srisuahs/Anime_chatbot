"""
NLP Pipeline: Intent Classification + Response Generation
Handles DistilBERT intent classification and DistilGPT-2 response generation
with proper context windowing
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from typing import Tuple, Dict, List, Optional
import os

from backend.nlp_config import (
    INTENT_CLASSIFIER_MODEL_PATH,
    RESPONSE_GENERATOR_MODEL_PATH,
    DISTILGPT2_MAX_TOKENS,
    CONTEXT_WINDOW_TOKENS,
    INTENT_MAP,
    INTENT_ID_TO_NAME,
    GPT2_GENERATION_CONFIG,
    TOKENIZER_KWARGS
)

logger = logging.getLogger(__name__)

class NLPPipeline:
    """
    Combined NLP pipeline for intent classification and response generation
    """
    
    def __init__(self):
        """Initialize models and tokenizers"""
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
            
            # Set pad token
            if self.gpt_tokenizer.pad_token is None:
                self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
            
            logger.info("Response generator loaded successfully")
        except Exception as e:
            logger.error(f"Error loading response generator: {e}")
            raise
    
    def classify_intent(self, user_input: str) -> Tuple[str, float]:
        """
        Classify user intent using DistilBERT
        
        Args:
            user_input: Raw user message
            
        Returns:
            (intent_name, confidence_score)
        """
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
            
            logger.info(f"Intent classified: {intent_name} (confidence: {confidence:.2f})")
            return intent_name, confidence
        
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return "general_chat", 0.0
    
    def _build_prompt_with_context(
        self,
        user_message: str,
        intent: str,
        chat_history: List[Dict[str, str]],
        extracted_entities: Optional[str] = None
    ) -> str:
        """
        Build prompt with chat history context (with token limit)
        
        Args:
            user_message: Current user message
            intent: Classified intent
            chat_history: List of previous messages from database
            extracted_entities: Optional entities extracted from message
            
        Returns:
            Formatted prompt with context
        """
        # Start with current message and intent
        prompt_parts = []
        
        if extracted_entities:
            prompt_parts.append(f"[INTENT={intent}] [ENTITIES={extracted_entities}]")
        else:
            prompt_parts.append(f"[INTENT={intent}]")
        
        # Add chat history in reverse (most recent first) until token limit
        context_tokens = 0
        
        # Tokenize to estimate size
        for msg in reversed(chat_history):
            msg_text = f"<USER>: {msg['user']}\n<BOT>: {msg['bot']}\n"
            msg_tokens = len(self.gpt_tokenizer.encode(msg_text))
            
            # Check if adding this would exceed context window
            if context_tokens + msg_tokens > CONTEXT_WINDOW_TOKENS:
                logger.info(f"Context limit reached. Including {len(prompt_parts)-1} history messages")
                break
            
            prompt_parts.append(msg_text)
            context_tokens += msg_tokens
        
        # Add current user message
        prompt_parts.append(f"<USER>: {user_message}\n<BOT>:")
        
        final_prompt = "".join(reversed(prompt_parts))
        
        # Final validation
        final_tokens = len(self.gpt_tokenizer.encode(final_prompt))
        if final_tokens > DISTILGPT2_MAX_TOKENS:
            logger.warning(f"Prompt exceeds max tokens ({final_tokens} > {DISTILGPT2_MAX_TOKENS}). Truncating...")
            final_prompt = final_prompt[:int(len(final_prompt) * (DISTILGPT2_MAX_TOKENS / final_tokens))]
        
        return final_prompt
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 80
    ) -> str:
        """
        Generate response using DistilGPT-2
        
        Args:
            prompt: Formatted prompt with context
            max_new_tokens: Max tokens for response
            
        Returns:
            Generated response
        """
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
            
            # Extract only the new tokens generated
            new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            response = self.gpt_tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean up response
            response = response.strip()
            if response.endswith("<BOT>:"):
                response = response[:-6].strip()
            
            logger.info(f"Response generated: {response[:100]}...")
            return response
        
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I'm having trouble processing that. Could you rephrase?"
    
    def process_message(
        self,
        user_message: str,
        chat_history: List[Dict[str, str]],
        extracted_entities: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Complete NLP pipeline: classify intent + generate response
        
        Args:
            user_message: Current user input
            chat_history: Previous messages for context
            extracted_entities: Optional entity extraction
            
        Returns:
            {
                "intent": str,
                "confidence": float,
                "prompt": str,
                "response": str
            }
        """
        # Step 1: Classify intent
        intent, confidence = self.classify_intent(user_message)
        
        # Step 2: Build prompt with context
        prompt = self._build_prompt_with_context(
            user_message=user_message,
            intent=intent,
            chat_history=chat_history,
            extracted_entities=extracted_entities
        )
        
        # Step 3: Generate response
        response = self.generate_response(prompt)
        
        return {
            "intent": intent,
            "confidence": confidence,
            "prompt": prompt,
            "response": response
        }


# Global pipeline instance (lazy loaded)
_nlp_pipeline = None

def get_nlp_pipeline() -> NLPPipeline:
    """Get or create NLP pipeline instance"""
    global _nlp_pipeline
    if _nlp_pipeline is None:
        _nlp_pipeline = NLPPipeline()
    return _nlp_pipeline
